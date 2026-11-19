! REQUIRES: flang, amdgpu, flang-rt
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! Enter data maps only the component. Subsequent structured regions must
! reuse that storage while retaining descriptors for direct device access
! through an array element or an allocatable scalar parent.
module direct_component_mapping
  use omp_lib, only: omp_is_initial_device
  implicit none
  type :: item
    integer :: before = 123
    real, allocatable :: a(:)
    integer :: after = 456
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

  subroutine check_item(x, expected)
    type(item), intent(in) :: x
    real, intent(in) :: expected
    if (any(x%a /= expected)) stop 2
    if (x%before /= 123 .or. x%after /= 456) stop 3
  end subroutine

  subroutine update_array(x)
    type(item), intent(inout) :: x(:)
    logical :: on_host
    !$omp target map(present, tofrom: x(2)%a) map(from: on_host)
      on_host = omp_is_initial_device()
      x(2)%a = real(lbound(x(2)%a, 1) + size(x(2)%a))
    !$omp end target
    if (on_host) stop 1
  end subroutine

  subroutine update_array_data(x)
    type(item), intent(inout) :: x(:)
    logical :: on_host
    !$omp target data map(present, tofrom: x(3)%a)
      !$omp target map(present, tofrom: x(3)%a) map(from: on_host)
        on_host = omp_is_initial_device()
        x(3)%a = real(ubound(x(3)%a, 1) + size(x(3)%a))
      !$omp end target
      if (on_host) stop 1
    !$omp end target data
  end subroutine
end module

program test_direct_component_mapping
  use direct_component_mapping
  implicit none
  type(item), allocatable :: w(:), scalar
  logical :: on_host
  interface
    subroutine ompx_dump_mapping_tables() bind(c)
    end subroutine
  end interface

  allocate(w(-1:1), scalar)
  allocate(w(0)%a(-3:4), w(1)%a(-3:4), scalar%a(-3:4))
  w(0)%a = 1.0
  w(1)%a = 1.0
  scalar%a = 1.0

  ! A non-first array element, with non-default parent and component bounds.
  call enter_item(w(0))
  !$omp target map(present, tofrom: w(0)%a) map(from: on_host)
    on_host = omp_is_initial_device()
    w(0)%a = real(lbound(w(0)%a, 1) + size(w(0)%a))
  !$omp end target
  if (on_host) stop 1
  !$omp target exit data map(from: w(0)%a)
  call check_item(w(0), 5.0)

  ! Both the target data region and its nested target access the element
  ! directly. Enter/exit through the opposite paths to the previous case.
  !$omp target enter data map(to: w(1)%a)
  !$omp target data map(present, tofrom: w(1)%a)
    !$omp target map(present, tofrom: w(1)%a) map(from: on_host)
      on_host = omp_is_initial_device()
      w(1)%a = real(ubound(w(1)%a, 1) + size(w(1)%a))
    !$omp end target
    if (on_host) stop 1
  !$omp end target data
  call exit_item(w(1))
  call check_item(w(1), 12.0)

  call enter_item(scalar)
  !$omp target map(present, tofrom: scalar%a) map(from: on_host)
    on_host = omp_is_initial_device()
    scalar%a = real(lbound(scalar%a, 1) + size(scalar%a))
  !$omp end target
  if (on_host) stop 1
  !$omp target exit data map(from: scalar%a)
  call check_item(scalar, 5.0)

  scalar%a = 1.0
  !$omp target enter data map(to: scalar%a)
  !$omp target data map(present, tofrom: scalar%a)
    !$omp target map(present, tofrom: scalar%a) map(from: on_host)
      on_host = omp_is_initial_device()
      scalar%a = real(ubound(scalar%a, 1) + size(scalar%a))
    !$omp end target
    if (on_host) stop 1
  !$omp end target data
  call exit_item(scalar)
  call check_item(scalar, 12.0)

  ! Assumed-shape dummies have different bounds and temporary descriptors.
  ! The array's base address is outside the mapped component in both cases.
  w(0)%a = 1.0
  !$omp target enter data map(to: w(0)%a)
  call update_array(w)
  call exit_item(w(0))
  call check_item(w(0), 5.0)

  w(1)%a = 1.0
  call enter_item(w(1))
  call update_array_data(w)
  !$omp target exit data map(from: w(1)%a)
  call check_item(w(1), 12.0)

  call ompx_dump_mapping_tables()
  deallocate(w, scalar)
end program

! CHECK-NOT: Host Ptr
! CHECK: omptarget device {{[0-9]+}} info: OpenMP Host-Device pointer mappings table empty
! CHECK-NOT: Host Ptr
