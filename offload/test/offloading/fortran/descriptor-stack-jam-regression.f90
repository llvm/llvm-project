! This test doesn't expect any results, the pass condition is running to completion
! without any memory access errors on device or mapping issues from descriptor
! collisions due to local descriptors being placed on device and not being unampped
! before a subsequent local descriptor residing at the same address is mapped to
! device.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
module test
contains
    subroutine kernel_1d(array)
        implicit none
        real, dimension(:) :: array
        integer :: i

        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do
        do i=1, ubound(array, 1)
            array(i) = 42.0
        end do
        !$omp target update from(array)
    end subroutine

    subroutine kernel_2d(array)
        implicit none
        real, dimension(:,:) :: array
        integer :: i, j

        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(2)
        do j=1, ubound(array, 2)
            do i=1, ubound(array, 1)
                array(i,j) = 42.0
            end do
        end do
        !$omp target update from(array)
    end subroutine

    subroutine kernel_3d(array)
        implicit none
        real, dimension(:,:,:) :: array
        integer :: i, j, k

        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(3)
        do k=1, ubound(array, 3)
            do j=1, ubound(array, 2)
                do i=1, ubound(array, 1)
                    array(i,j,k) = 42.0
                end do
            end do
        end do
        !$omp target update from(array)
    end subroutine

    subroutine kernel_4d(array)
        implicit none
        real, dimension(:,:,:,:) :: array
        integer :: i, j, k, l

        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(4)
        do l=1, ubound(array, 4)
            do k=1, ubound(array, 3)
                do j=1, ubound(array, 2)
                    do i=1, ubound(array, 1)
                        array(i,j,k,l) = 42.0
                    end do
                end do
            end do
        enddo
        !$omp target update from(array)
    end subroutine
end module

program main
    use test
    implicit none
    integer, parameter :: n = 2
    real :: array1(n)
    real :: array2(n,n)
    real :: array3(n,n,n)
    real :: array4(n,n,n,n)

    call kernel_1d(array1)
    call kernel_2d(array2)
    call kernel_3d(array3)
    call kernel_4d(array4)

    print *, array1
    print *, array2
    print *, array3
    print *, array4
    print *, "PASS"
end program

! CHECK: 42. 42.
! CHECK: 42. 42. 42. 42.
! CHECK: 42. 42. 42. 42. 42. 42. 42. 42.
! CHECK: 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42.
! CHECK: PASS
