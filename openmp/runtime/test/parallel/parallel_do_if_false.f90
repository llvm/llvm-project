!RUN: %flang %s -o %t -fopenmp  && %t | FileCheck %s
PROGRAM parallel_do_if

        USE OMP_LIB

        INTEGER :: I

        INTEGER :: N = 0

        INTEGER, PARAMETER :: NI = 1

        !$OMP PARALLEL DO IF(.false.)

        DO I = 1, NI

        IF (omp_in_parallel() .EQV. .false.) THEN               
                !CHECK: PASS
                PRINT *, "PASS"
        ELSE
                PRINT *, "FAIL"

        END IF

        END DO
        !$OMP END PARALLEL DO

END PROGRAM
