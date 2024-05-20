! Offloading test checking interaction of an
! explicit derived type member mapping of
! an array with bounds when mapped to
! target using a combination of enter and
! exit directives.
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: scalar_array
        integer(4) :: array(10)
    end type scalar_array

    type(scalar_array) :: scalar_arr

    do I = 1, 10
        scalar_arr%array(I) = I + I
    end do

    !$omp target enter data map(to: scalar_arr%array(3:6))

    ! Shouldn't overwrite data already locked in
    ! on target via enter, which will then be
    ! overwritten by our exit
    do I = 1, 10
        scalar_arr%array(I) = 10
    end do

  ! The compiler/runtime is less friendly about read/write out of
  ! bounds when using enter and exit, we have to specifically loop
  ! over the correct range
   !$omp target
    do i=3,6
        scalar_arr%array(i) = scalar_arr%array(i) + i
    end do
  !$omp end target

  !$omp target exit data map(from: scalar_arr%array(3:6))

  print*, scalar_arr%array
end program

!CHECK: 10 10 9 12 15 18 10 10 10 10
