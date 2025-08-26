! Offloading test checking interaction of an enter and exit map of an array of
! scalars with specified bounds
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program main
    integer :: array(10)

    do I = 1, 10
      array(I) = I + I
    end do

    !$omp target enter data map(to: array(3:6))
    ! Shouldn't overwrite data already locked in
    ! on target via enter, which will then be
    ! overwritten by our exit
    do I = 1, 10
      array(I) = 10
    end do

  ! The compiler/runtime is less lenient about read/write out of
  ! bounds when using enter and exit, we have to specifically loop
  ! over the correctly mapped range
   !$omp target
    do i=3,6
      array(i) = array(i) + i
    end do
  !$omp end target

  !$omp target exit data map(from: array(3:6))
  print *, array
end program

!CHECK: 10 10 9 12 15 18 10 10 10 10
