! Test target private
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
  call target_private(a)
  print*, "======= FORTRAN Test passed! ======="
  print*, "foo(a) should not return 20, got " , a
  if (a /= 20) then
     stop 0
  else
     stop 1
  end if
  
  !       stop 0
end program main
subroutine target_private(r)
  implicit none
  integer, dimension(2) :: simple_vars
  integer :: a, r
  ! set a to 10
  a = 5
  simple_vars(1) = a
  simple_vars(2) = a
  !$omp target map(tofrom: simple_vars) private(a)
  ! Without private(a), a would be firstprivate, meaning it's value would be 5
  ! with private(a), it's value would be uninitialized, which means it'd have
  ! a very small chance of being 5.
  ! So, simple_vars(1|2) should be 5 + a_private
  simple_vars(1) = simple_vars(1) + a
  simple_vars(2) = simple_vars(2) + a
  !$omp end target
  r = simple_vars(1) + simple_vars(2)
end subroutine target_private
