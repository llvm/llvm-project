! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=VALUE

program target_firstprivate_info
  implicit none
  integer :: factor, base, res
  factor = 7; base = 5; res = 0

  ! 'factor' is explicitly privatized, 'base' is an implicit scalar capture.
  !$omp target map(tofrom: res) firstprivate(factor)
    res = factor + base
  !$omp end target
  print *, "res1 =", res
end program target_firstprivate_info

! CHECK: Entering OpenMP kernel {{.*}} with 3 arguments:
! CHECK-DAG: tofrom(res)[4]
! CHECK-DAG: firstprivate(base)[4] (implicit)
! CHECK-DAG: to(factor)[4]
! VALUE: res1 = 12
