! REQUIRES: flang, amdgpu, flang-rt
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! A scalar dummy and an array element designate the same component storage.
! The unmapped sibling makes the record larger than the component descriptor.
! Check all four enter/exit access-path combinations, device execution,
! copy-back, and removal of both descriptor and allocation mappings.
module component_mapping
  use omp_lib, only: omp_is_initial_device
  implicit none
  type :: item
    real, allocatable :: a(:)
    integer :: k = 123
  end type
contains
  subroutine enter_item(x)
    type(item), intent(inout) :: x
    !$omp target enter data map(to: x%a)
  end subroutine

  subroutine exit_item(x)
    type(item), intent(inout) :: x
    !$omp target exit data map(from: x%a)
  end subroutine

  subroutine update_item(x)
    type(item), intent(inout) :: x
    logical :: on_host
    !$omp target map(present, tofrom: x%a) map(from: on_host)
    on_host = omp_is_initial_device()
    x%a = 7.0
    !$omp end target
    if (on_host) stop 1
  end subroutine
end module

program test_component_mapping
  use component_mapping
  implicit none
  type(item), allocatable :: w(:), scalar
  integer :: i
  interface
    subroutine ompx_dump_mapping_tables() bind(c)
    end subroutine
  end interface

  allocate(w(4))
  do i = 1, 4
    allocate(w(i)%a(100))
    w(i)%a = 1.0
  end do

  call enter_item(w(1))
  call update_item(w(1))
  !$omp target exit data map(from: w(1)%a)

  !$omp target enter data map(to: w(2)%a)
  call update_item(w(2))
  call exit_item(w(2))

  call enter_item(w(3))
  call update_item(w(3))
  call exit_item(w(3))

  !$omp target enter data map(to: w(4)%a)
  call update_item(w(4))
  !$omp target exit data map(from: w(4)%a)

  do i = 1, 4
    if (any(w(i)%a /= 7.0)) stop 2
    if (w(i)%k /= 123) stop 3
  end do

  ! Both paths must update the same reference count. One release must leave
  ! the component present until the second exit copies it back and removes it.
  w(1)%a = 1.0
  call enter_item(w(1))
  !$omp target enter data map(to: w(1)%a)
  !$omp target exit data map(release: w(1)%a)
  call update_item(w(1))
  call exit_item(w(1))
  if (any(w(1)%a /= 7.0)) stop 6

  ! An allocatable scalar parent must not introduce a descriptor mapping
  ! either: only its explicitly mapped component needs device storage.
  allocate(scalar)
  allocate(scalar%a(100))
  scalar%a = 1.0
  call enter_item(scalar)
  call update_item(scalar)
  !$omp target exit data map(from: scalar%a)
  if (any(scalar%a /= 7.0)) stop 4

  scalar%a = 1.0
  !$omp target enter data map(to: scalar%a)
  call update_item(scalar)
  call exit_item(scalar)
  if (any(scalar%a /= 7.0)) stop 5

  call ompx_dump_mapping_tables()
  deallocate(w, scalar)
end program

! CHECK-NOT: Host Ptr
! CHECK: omptarget device {{[0-9]+}} info: OpenMP Host-Device pointer mappings table empty
! CHECK-NOT: Host Ptr
