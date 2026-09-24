! Offload test for dynamic dispatch through mapped polymorphic array
! descriptors.
!
! This is the array analogue of the polymorphic class-pointer type-bound
! dispatch test. It verifies that dispatch follows the dynamic element type and
! that extended dynamic-type components are copied to and from the device for
! polymorphic arrays.
!
! This verifies that polymorphic array descriptor maps use runtime elem_len
! rather than the declared element type size when copying array elements.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_array_type_bound_dispatch_mod
  implicit none

  type :: base_t
    integer :: base_value
  contains
    procedure :: value => base_value_fn
    procedure :: adjust => base_adjust
  end type base_t

  type, extends(base_t) :: child_a_t
    integer :: a_value
  contains
    procedure :: value => child_a_value_fn
    procedure :: adjust => child_a_adjust
  end type child_a_t

  type, extends(base_t) :: child_b_t
    integer :: b_value
  contains
    procedure :: value => child_b_value_fn
    procedure :: adjust => child_b_adjust
  end type child_b_t

contains

  integer function base_value_fn(self)
    class(base_t), intent(in) :: self
    base_value_fn = self%base_value
  end function base_value_fn

  subroutine base_adjust(self, amount)
    class(base_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%base_value = self%base_value + amount
  end subroutine base_adjust

  integer function child_a_value_fn(self)
    class(child_a_t), intent(in) :: self
    child_a_value_fn = self%base_value + self%a_value * 10
  end function child_a_value_fn

  subroutine child_a_adjust(self, amount)
    class(child_a_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%base_value = self%base_value + amount
    self%a_value = self%a_value + 1
  end subroutine child_a_adjust

  integer function child_b_value_fn(self)
    class(child_b_t), intent(in) :: self
    child_b_value_fn = self%base_value + self%b_value * 100
  end function child_b_value_fn

  subroutine child_b_adjust(self, amount)
    class(child_b_t), intent(inout) :: self
    integer, intent(in) :: amount
    self%base_value = self%base_value + amount * 2
    self%b_value = self%b_value + 2
  end subroutine child_b_adjust

end module polymorphic_array_type_bound_dispatch_mod

program main
  use polymorphic_array_type_bound_dispatch_mod
  implicit none

  class(base_t), allocatable :: a_arr(:)
  class(base_t), allocatable :: b_arr(:)
  integer :: result_a
  integer :: result_b
  logical :: failed

  allocate(child_a_t :: a_arr(2))
  allocate(child_b_t :: b_arr(2))

  a_arr(1)%base_value = 5
  a_arr(2)%base_value = 11
  select type (a_arr)
  type is (child_a_t)
    a_arr(1)%a_value = 3
    a_arr(2)%a_value = 6
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  b_arr(1)%base_value = 7
  b_arr(2)%base_value = 13
  select type (b_arr)
  type is (child_b_t)
    b_arr(1)%b_value = 2
    b_arr(2)%b_value = 5
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  result_a = -1
  result_b = -1

  !$omp target map(tofrom: a_arr, result_a)
    result_a = a_arr(1)%value()
    call a_arr(2)%adjust(4)
  !$omp end target

  !$omp target map(tofrom: b_arr, result_b)
    result_b = b_arr(1)%value()
    call b_arr(2)%adjust(4)
  !$omp end target

  failed = .false.

  if (result_a /= 35) failed = .true.
  if (result_b /= 207) failed = .true.

  if (a_arr(2)%base_value /= 15) failed = .true.
  select type (a_arr)
  type is (child_a_t)
    if (a_arr(2)%a_value /= 7) failed = .true.
  class default
    failed = .true.
  end select

  if (b_arr(2)%base_value /= 21) failed = .true.
  select type (b_arr)
  type is (child_b_t)
    if (b_arr(2)%b_value /= 7) failed = .true.
  class default
    failed = .true.
  end select

  if (failed) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
