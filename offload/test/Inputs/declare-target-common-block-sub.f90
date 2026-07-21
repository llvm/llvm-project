! Not compiled and verified through this file, utilised by the
! declare-target-common-block-main.f90 test to create a multiple
! file reproducer.

subroutine foo()
  common /dxyz/ arr(10)
  !$omp declare target (/dxyz/)
  !$omp target map(always, tofrom: /dxyz/)
  arr(1) = 1.0
  !$omp end target
end subroutine
