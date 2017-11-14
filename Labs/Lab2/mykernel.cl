/* This kernel calculates and passes back Pi/4. */ 
/* FinalSum is the overall result of Pi/4. */
/* WorkItemMem is to store the sum of each work item in local memory. */
/* WorkGroupMem is to store the sum of each work group into global memory. */
/* WorkGroups is to iterate through WorkGroupMem for global reduction. */
/* localws is to iterate through WorkItemMem for local reduction. */
/* WorkGroups and localws are passed in to the kernel for convienence with */
/* multiple test cases with varying Work Items, Work Groups, and work items */
/* per work group. */  

__kernel void PiOverFour(
    __global float *FinalSum,
    __local float *WorkItemMem,
    __global float *WorkGroupMem,
    int WorkGroups,
    int localws)
{

 /* get 1 dimensional global position, local position,
    and work group position */
    int x = get_global_id (0); /*Global_Position, x for convienence */
    int Local_Position = get_local_id (0);
    int WorkGroup_Position = get_group_id (0);  
   
/* Declare accumulating and reduction variables */
    float sum = 0.0f;
    float localsum = 0.0f;
    float globalsum = 0.0f; 
 

/* calculate result of pi/4, with 8 elements in one work item */
sum = (1.0/(16 * x + 1)) - (1.0/(16 * x + 3)) + (1.0/(16 * x + 5)) - 
	(1.0/(16 * x + 7)) +(1.0/(16 * x + 9)) - (1.0/(16 * x + 11)) +
	 (1.0/(16 * x + 13)) - (1.0/(16 * x + 15));

/*Store all Work Item Sum Results into Local Memory */ 
WorkItemMem[Local_Position] = sum; 

/* Wait for all local processing to be complete */
barrier(CLK_LOCAL_MEM_FENCE); 

/* Perform Local Reduction */
for(int i = 0; i < (localws - 1); i++){   
localsum += WorkItemMem[i]; 
}

/* Store all Work Group Sum Results into Global Memory */
WorkGroupMem[WorkGroup_Position] = localsum; 

/*Wait for all global processing to be complete */
barrier(CLK_GLOBAL_MEM_FENCE);  

/* Perform Global Reduction */
for(int j = 0; j < (WorkGroups - 1); j++){
globalsum += WorkGroupMem[j];
} 

/*Output Final Sum */  
FinalSum[0] = globalsum; 
  
}
