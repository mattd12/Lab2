/* widthA=heightB for valid matrix multiplication */
__kernel void simpleMultiply(
    __global float *outputD,
    int WorkItems) 
{

__local float localsum[1]; 

 /* get global position in X direction */
    int x = get_global_id (0);

    float sum = 0.0f; 

/* calculate result of pi/4, with 8 elements in one work item */
sum = (1.0/(16 * x + 1)) - (1.0/(16 * x + 3)) + (1.0/(16 * x + 5)) - 
	(1.0/(16 * x + 7)) +(1.0/(16 * x + 9)) - (1.0/(16 * x + 11)) +
	 (1.0/(16 * x + 13)) - (1.0/(16 * x + 15));


/* Add the previous x index to the current x sum */
	if(x != 0) {  
	sum += localsum[0];
	} 

	if(x == (WorkItems - 1)) {
	outputD[0] = sum;
	}

/* Put the sum into local memory for the next x index */ 
	localsum[0] = sum;  

barrier(CLK_LOCAL_MEM_FENCE);  
}
