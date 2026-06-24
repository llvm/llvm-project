! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program map_present_pointer
  implicit none
  integer, target :: src(4) = [10, 20, 30, 40]
  integer, pointer :: p(:)
  integer :: out

  p => src
  out = -1

!$omp target enter data map(to: src)
!$omp target map(present, to: p) map(from: out)
  out = p(2)
!$omp end target
!$omp target exit data map(delete: src)

  if (out /= 20) stop 1
  print *, "associated pointer ok"
end program

! CHECK-NOT: omptarget
! CHECK: associated pointer ok
