! This checks that the basic functionality of map type present functions as
! expected, emitting an omptarget error when the data is not present.
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-fail-generic 2>&1 \
! RUN: | %fcheck-generic

! NOTE: This should intentionally fatal error in omptarget as it's not
! present, as is intended.
 subroutine target_data_not_present()
    double precision, dimension(:), allocatable :: arr
    integer, parameter :: N = 16
    integer :: i

    allocate(arr(N))

!$omp target data map(present,alloc:arr)

!$omp target
    do i = 1,N
        arr(i) = 42.0d0
    end do
!$omp end target

!$omp end target data

    deallocate(arr)
    return
end subroutine

program map_present
    implicit none
    call target_data_not_present()
end program

!CHECK: omptarget message: device mapping required by 'present' map type modifier does not exist for host address{{.*}}
