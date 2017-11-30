// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run
// XFAIL: gcc
// Clang had a bug until version 4.0.1 which resulted in a hang.
// UNSUPPORTED: clang-3, clang-4.0.0

// Regression test for a bug in cancellation to cover effect of `#pragma omp cancel`
// in a loop construct, on sections construct.
// Pass condition: Cancellation status from `for` does not persist
// to `sections`.

#include <stdio.h>
#include <omp.h>

int result[2] = {0, 0};

void cq416850_for_sections() {

    unsigned i;
     // 1) loop
    #pragma omp for
    for (i = 0; i < 1; i++) {
        result[0] = 1;
        #pragma omp cancel for
        result[0] = 2;
    }

//        printf("thread %d: result[0] = %d, result[1] = %d \n",  omp_get_thread_num(), result[0], result[1]);


    // 2) sections
    #pragma omp sections
    {
        #pragma omp section
        {
            result[1] = 1;
            #pragma omp cancellation point sections
            result[1] = 2;
        }
    }
}

int main(void) {
    if(!omp_get_cancellation()) {
        printf("Cancellation not enabled!\n");
        return 2;
    }

    #pragma omp parallel num_threads(4)
    {
        cq416850_for_sections();
    }

    if (result[0] != 1 || result[1] != 2) {
        printf("Incorrect values. "
               "result[0] = %d (expected 1), "
               "result[1] = %d (expected 2).\n",
               result[0], result[1]);
        printf("FAILED\n");
        return 1;
    }

    printf("PASSED\n");
    return 0;
}
