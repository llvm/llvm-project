! Offloading test checking mapping of the pointee of allocatable with
! PRIVATE map-type.
! REQUIRES: flang, amdgpu
! TODO:Remove usupported once https://github.com/llvm/llvm-project/issues/244545
! is fixed.
! UNSUPPORTED: amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program main
  type t
    integer :: n
    integer, allocatable :: a(:)
  end type
  type(t) :: d
  integer, allocatable :: x(:)
  logical :: ok

  d%n = 5
  ok = .false.
  !$omp target firstprivate(d, x) map(from: ok)
    ok = (d%n == 5) .and. .not. allocated(d%a) .and. .not. allocated(x)
  !$omp end target

  print *, ok
end program main

! CHECK: T
