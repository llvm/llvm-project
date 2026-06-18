! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

program map_present_pointer
  implicit none
  integer, target :: src(4) = [10, 20, 30, 40]
  integer, pointer :: p(:) => null()
  integer :: out

  out = -1
!$omp target map(present, to: p) map(tofrom: out)
  if (associated(p)) out = p(1)
!$omp end target

  if (out /= -1) stop 1
  print *, "null pointer ok"

  p => src
  out = -1

!$omp target enter data map(to: src)
!$omp target map(present, to: p) map(from: out)
  out = p(2)
!$omp end target
!$omp target exit data map(delete: src)

  if (out /= 20) stop 2
  print *, "associated pointer ok"
end program

! CHECK-NOT: omptarget
! CHECK: null pointer ok
! CHECK: associated pointer ok
