// RUN: %libomp-compile && env OMP_MAX_TASK_PRIORITY=42 %libomp-run
// Test OMP 4.5 task priorities
// Currently only API function and envirable parsing implemented.
// Test environment sets envirable: OMP_MAX_TASK_PRIORITY=42 as tested below.
#include <stdio.h>
#include <omp.h>

int main (void) {
    int passed;

    passed = (omp_get_max_task_priority() == 42);    
    printf("Got %d\n", omp_get_max_task_priority());

    if (passed) {
       printf("passed\n");
       return 0;
    }

    printf("failed\n");
    return 1;
}

