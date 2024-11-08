! Offloading test which maps a specific element of a
! derived type to the device and then accesses the
! element alongside an individual element of an array
! that the derived type contains. In particular, this
! test helps to check that we can replace the constants
! within the kernel with instructions and then replace
! these instructions with the kernel parameters.
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-unknown-linux-gnu
! UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test_0
    type dtype
      integer elements(20)
      integer value
    end type dtype

    type (dtype) array_dtype(5)
  contains

  subroutine assign()
    implicit none
!$omp target map(tofrom: array_dtype(5))
    array_dtype(5)%elements(5) = 500
!$omp end target
  end subroutine

  subroutine add()
    implicit none

!$omp target map(tofrom: array_dtype(5))
    array_dtype(5)%elements(5) = array_dtype(5)%elements(5) + 500
!$omp end target
  end subroutine
end module test_0

program main
   use test_0

  call assign()
  call add()

  print *, array_dtype(5)%elements(5)
end program

! CHECK: 1000