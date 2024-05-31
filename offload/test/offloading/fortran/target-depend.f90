! Offloading test checking the use of the depend clause on
! the target construct
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  integer :: a = 0
  call foo(5, a)
  print*, "======= FORTRAN Test passed! ======="
  print*, "foo(5) returned ", a, ", expected 6\n"
  !       stop 0
end program main
subroutine foo(N, r)
  integer, intent(in) :: N
  integer, intent(out) :: r
  integer :: z

  z = 1
  !$omp task depend(out: z) shared(z)
  z = N
  !$omp end task

  !$omp target map(tofrom: z) depend(in: z)
  z = z + 1
  !$omp end target

  r = z
end subroutine foo

!CHECK: ======= FORTRAN Test passed! =======
!CHECK: foo(5) returned 6 , expected 6
