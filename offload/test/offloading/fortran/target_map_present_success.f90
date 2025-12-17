! This checks that the basic functionality of map type present functions as
! expected, no-op'ng when present
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

 subroutine target_data_present()
    double precision, dimension(:), allocatable :: arr
    integer, parameter :: N = 16
    integer :: i

    allocate(arr(N))

    arr(:) = 10.0d0

!$omp target data map(tofrom:arr)

!$omp target data map(present,alloc:arr)

!$omp target
    do i = 1,N
        arr(i) = 42.0d0
    end do
!$omp end target

!$omp end target data

!$omp end target data

    print *, arr

    deallocate(arr)

    return
end subroutine

program map_present
    implicit none
   call target_data_present()
end program

!CHECK: 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42. 42.
