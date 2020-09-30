// RUN: %gdb-compile-and-run 2>&1 | tee %t.out | FileCheck %s

#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include "../ompt_plugin.h"

int main()
{
  printf("Application: Process %d started.\n", getpid());

    int number;
    int i;
    omp_set_num_threads(3);
    omp_set_nested(1); // 1 - enables nested parallelism; 0 - disables nested parallelism.

    #pragma omp parallel // parallel region begins
    {
        printf("outer parallel region Thread ID == %d\n", omp_get_thread_num());
        /* Code for work to be done by outer parallel region threads over here. */

        if (omp_get_thread_num() == 2) sleep(1);

        #pragma omp parallel num_threads(2) // nested parallel region
        {
                    /* Code for work to be done by inner parallel region threads over here. */
                    printf("inner parallel region thread id %d\n", omp_get_thread_num());

                //if (omp_get_thread_num() == 1) sleep(1000);

                #pragma omp parallel num_threads(2) //
                {

                            #pragma omp for
                            for(i=0;i<20;i++)
                            {
                                // Some independent iterative computation to be done.
                                 printf("");
                                ompd_tool_test(0);
                            }
                }
        }
    }

    //sleep(1000);

    return 0;
}

// CHECK-NOT: OMPT-OMPD mismatch
// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
