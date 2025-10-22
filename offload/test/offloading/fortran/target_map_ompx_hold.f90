! Basic test that checks that when ompx_hold is in use we cannot delete the data
! until the ompx_hold falls out of scope, and verifies this via the utilisation of
! present.
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-fail-generic 2>&1 \
! RUN: | %fcheck-generic

program ompx_hold
    implicit none
    integer :: presence_check

!CHECK-NOT: omptarget message: device mapping required by 'present' map type modifier does not exist for host address{{.*}}
!$omp target data map(ompx_hold, tofrom: presence_check)
!$omp target exit data map(delete: presence_check)
!$omp target map(present, tofrom: presence_check)
    presence_check = 10
!$omp end target
!$omp end target data

!CHECK: omptarget message: device mapping required by 'present' map type modifier does not exist for host address{{.*}}
!$omp target data map(tofrom: presence_check)
!$omp target exit data map(delete: presence_check)
!$omp target map(present, tofrom: presence_check)
presence_check = 20
!$omp end target
!$omp end target data

end program
