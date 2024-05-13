! Offloading test checking interaction of an
! enter and exit map of an array of scalars
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer :: array(10)

    do I = 1, 10
      array(I) = I + I
    end do

    !$omp target enter data map(to: array)

    ! Shouldn't overwrite data already locked in
    ! on target via enter, this will then be 
    ! overwritten by our exit
    do I = 1, 10
      array(I) = 10
    end do

   !$omp target
    do i=1,10
      array(i) = array(i) + i
    end do
  !$omp end target 

  !$omp target exit data map(from: array)

  print*, array
end program

!CHECK: 3 6 9 12 15 18 21 24 27 30
