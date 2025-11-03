! Offloading test checking the use of the depend clause on the target construct
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-unknown-linux-gnu
! UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
subroutine defaultmap_allocatable_present()
    implicit none
    integer, dimension(:), allocatable :: arr
    integer :: N = 16
    integer :: i

    allocate(arr(N))

!$omp target enter data map(to: arr)

!$omp target defaultmap(present: allocatable)
    do i = 1,N
        arr(i) = N + 40
    end do
!$omp end target

!$omp target exit data map(from: arr)

    print *, arr
    deallocate(arr)

    return
end subroutine

subroutine defaultmap_scalar_tofrom()
    implicit none
    integer :: scalar_int
    scalar_int = 10

   !$omp target defaultmap(tofrom: scalar)
        scalar_int = 20
   !$omp end target

    print *, scalar_int
    return
end subroutine

subroutine defaultmap_all_default()
    implicit none
    integer, dimension(:), allocatable :: arr
    integer :: aggregate(16)
    integer :: N = 16
    integer :: i, scalar_int

    allocate(arr(N))

    scalar_int = 10
    aggregate = scalar_int

   !$omp target defaultmap(default: all)
        scalar_int = 20
        do i = 1,N
            arr(i) = scalar_int + aggregate(i)
        end do
   !$omp end target

    print *, scalar_int
    print *, arr

    deallocate(arr)
    return
end subroutine

subroutine defaultmap_pointer_to()
    implicit none
    integer, dimension(:), pointer :: arr_ptr(:)
    integer :: scalar_int, i
    allocate(arr_ptr(10))
    arr_ptr = 10
    scalar_int = 20

    !$omp target defaultmap(to: pointer)
        do i = 1,10
            arr_ptr(i) = scalar_int + 20
        end do
    !$omp end target

    print *, arr_ptr
    deallocate(arr_ptr)
    return
end subroutine

subroutine defaultmap_scalar_from()
    implicit none
    integer :: scalar_test
    scalar_test = 10
    !$omp target defaultmap(from: scalar)
        scalar_test = 20
    !$omp end target

    print *, scalar_test
    return
end subroutine

subroutine defaultmap_aggregate_to()
    implicit none
    integer :: aggregate_arr(16)
    integer :: i, scalar_test = 0
    aggregate_arr = 0
    !$omp target map(tofrom: scalar_test) defaultmap(to: aggregate)
        do i = 1,16
            aggregate_arr(i) = i
            scalar_test = scalar_test + aggregate_arr(i)
        enddo
    !$omp end target

    print *, scalar_test
    print *, aggregate_arr
    return
end subroutine

subroutine defaultmap_dtype_aggregate_to()
    implicit none
    type :: dtype
        real(4) :: i
        real(4) :: j
        integer(4) :: array_i(10)
        integer(4) :: k
        integer(4) :: array_j(10)
    end type dtype

    type(dtype) :: aggregate_type

    aggregate_type%k = 20
    aggregate_type%array_i = 30

    !$omp target defaultmap(to: aggregate)
        aggregate_type%k = 40
        aggregate_type%array_i(1) = 50
    !$omp end target

    print *, aggregate_type%k
    print *, aggregate_type%array_i(1)
    return
end subroutine

program map_present
    implicit none
! CHECK: 56 56 56 56 56 56 56 56 56 56 56 56 56 56 56 56
    call defaultmap_allocatable_present()
! CHECK: 20
    call defaultmap_scalar_tofrom()
! CHECK: 10
! CHECK: 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30
    call defaultmap_all_default()
! CHECK: 10 10 10 10 10 10 10 10 10 10
    call defaultmap_pointer_to()
! CHECK: 20
    call defaultmap_scalar_from()
! CHECK: 136
! CHECK: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    call defaultmap_aggregate_to()
! CHECK: 20
! CHECK: 30
    call defaultmap_dtype_aggregate_to()
end program
