! Test that checks an allocatable array can be marked implicit
! `declare target to` and functions without issue.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test
  implicit none
  integer, allocatable, dimension(:) :: alloca_arr
  !$omp declare target(alloca_arr)
end module test

program main
  use test
  implicit none
  integer :: cycle, i

  allocate(alloca_arr(10))

  do i = 1, 10
      alloca_arr(i) = 0
  end do

  !$omp target data map(to:alloca_arr)
    do cycle = 1, 2
      !$omp target
          do i = 1, 10
              alloca_arr(i) = alloca_arr(i) + i
          end do
      !$omp end target

      ! NOTE: Technically doesn't affect the results, but there is a
      ! regression case that'll cause a runtime crash if this is
      ! invoked more than once, so this checks for that.
      !$omp target update from(alloca_arr)
    end do
  !$omp end target data

  print *, alloca_arr
end program

! CHECK: 2 4 6 8 10 12 14 16 18 20
