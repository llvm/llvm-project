! Offload test that checks an allocatable array can be mapped with a specified
! 3-D bounds when contained within a derived type and mapped via member mapping.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: top_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:,:,:)
    integer(4) :: k
    end type top_layer

    type(top_layer) :: one_l
    integer :: inArray(3,3,3)

    allocate(one_l%array_j(3,3,3))

    do i = 1, 3
        do j = 1, 3
          do k = 1, 3
              inArray(i, j, k) = 42
              one_l%array_j(i, j, k) = 0
          end do
         end do
      end do

!$omp target map(tofrom: one_l%array_j(1:3, 1:3, 2:2)) map(to: inArray(1:3, 1:3, 1:3))
    do j = 1, 3
        do k = 1, 3
            one_l%array_j(k, j, 2) = inArray(k, j, 2)
        end do
      end do
!$omp end target

    print *, one_l%array_j
end program main

!CHECK: 0 0 0 0 0 0 0 0 0 42 42 42 42 42 42 42 42 42 0 0 0 0 0 0 0 0 0
