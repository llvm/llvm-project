! Test the implicit `declare target to` interaction with `target update from`
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test
    implicit none
    integer :: array(10)
    !$omp declare target(array)
end module test

PROGRAM main
    use test
    implicit none
    integer :: i

  do i = 1, 10
        array(i) = 0
  end do

  !$omp target
    do i = 1, 10
        array(i) = i
    end do
  !$omp end target

  !$omp target
    do i = 1, 10
        array(i) = array(i) + i
    end do
  !$omp end target

   print *, array

  !$omp target update from(array)

   print *, array
END PROGRAM

! CHECK: 0 0 0 0 0 0 0 0 0 0
! CHECK: 2 4 6 8 10 12 14 16 18 20
