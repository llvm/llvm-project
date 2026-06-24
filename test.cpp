! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

subroutine test_release()
  integer :: x, y

  !$omp atomic capture release
    y = x
    x = x + 1
  !$omp end atomic
end subroutine

subroutine test_acq_rel()
  integer :: x, y

  !$omp atomic capture acq_rel
    y = x
    x = x + 1
  !$omp end atomic
end subroutine

! CHECK-LABEL: func.func @_QPtest_release
! CHECK: omp.atomic.capture
! CHECK-NOT: memory_order(release)

! CHECK-LABEL: func.func @_QPtest_acq_rel
! CHECK: omp.atomic.capture
! CHECK-NOT: memory_order(acq_rel)