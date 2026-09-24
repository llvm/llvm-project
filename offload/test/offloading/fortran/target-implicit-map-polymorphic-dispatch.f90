! Offload tests for polymorphic objects captured by implicit target maps.
!
! These cases mirror explicit map polymorphic tests, but intentionally omit
! the polymorphic list items from the map clauses so lowering must create
! implicit maps for the captured class descriptors.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module implicit_map_polymorphic_dispatch_mod
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

  type :: wrapper_t
    integer :: marker
    class(base_t), allocatable :: item
  end type wrapper_t

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

  subroutine init_wrapper(w)
    type(wrapper_t), intent(out) :: w

    w%marker = 5
    allocate(child_b_t :: w%item)
    w%item%base_value = 12
    select type (item => w%item)
    type is (child_b_t)
      item%b_value = 3
    class default
      stop 1
    end select
  end subroutine init_wrapper

end module implicit_map_polymorphic_dispatch_mod

program main
  use implicit_map_polymorphic_dispatch_mod
  implicit none

  class(base_t), allocatable :: section_arr(:)
  class(base_t), allocatable :: whole_arr(:)
  type(child_a_t), target :: ptr_a
  type(child_b_t), target :: ptr_b
  class(base_t), pointer :: ptr
  type(wrapper_t) :: wrapped
  type(base_t) :: base_mold
  integer :: section_result
  integer :: whole_result
  integer :: pointer_result
  logical :: extends_base
  logical :: base_extends_item
  logical :: failed

  failed = .false.

  allocate(child_a_t :: section_arr(10))
  section_arr(:)%base_value = -100
  section_arr(4)%base_value = 5
  section_arr(10)%base_value = 11
  select type (section_arr)
  type is (child_a_t)
    section_arr(:)%a_value = -10
    section_arr(4)%a_value = 3
    section_arr(10)%a_value = 6
  class default
    failed = .true.
  end select

  section_result = -1
  ! The polymorphic array is deliberately not present in the map clause. This
  ! checks that implicit capture uses runtime element-size bounds for the
  ! dynamically typed array storage, as the explicit section-dispatch test does.
  !$omp target map(tofrom: section_result)
    section_result = section_arr(4)%value()
    call section_arr(10)%adjust(4)
  !$omp end target

  if (section_result /= 35) failed = .true.
  if (section_arr(10)%base_value /= 15) failed = .true.
  select type (section_arr)
  type is (child_a_t)
    if (section_arr(10)%a_value /= 7) failed = .true.
  class default
    failed = .true.
  end select

  allocate(child_b_t :: whole_arr(2))
  whole_arr(1)%base_value = 7
  whole_arr(2)%base_value = 13
  select type (whole_arr)
  type is (child_b_t)
    whole_arr(1)%b_value = 2
    whole_arr(2)%b_value = 5
  class default
    failed = .true.
  end select

  whole_result = -1
  ! Whole polymorphic array implicit capture: exercises dynamic dispatch and
  ! copy-back of extended dynamic-type components without an explicit map.
  !$omp target map(tofrom: whole_result)
    whole_result = whole_arr(1)%value()
    call whole_arr(2)%adjust(4)
  !$omp end target

  if (whole_result /= 207) failed = .true.
  if (whole_arr(2)%base_value /= 21) failed = .true.
  select type (whole_arr)
  type is (child_b_t)
    if (whole_arr(2)%b_value /= 7) failed = .true.
  class default
    failed = .true.
  end select

  ptr_a%base_value = 5
  ptr_a%a_value = 3
  ptr_b%base_value = 7
  ptr_b%b_value = 2

  ptr => ptr_a
  pointer_result = -1
  ! Polymorphic pointer implicit capture: verifies pointer descriptor RTTI and
  ! dispatch still work when the pointer itself is not explicitly mapped.
  !$omp target map(tofrom: pointer_result)
    pointer_result = ptr%value()
    call ptr%adjust(4)
  !$omp end target

  if (pointer_result /= 35) failed = .true.
  if (ptr_a%base_value /= 9) failed = .true.
  if (ptr_a%a_value /= 4) failed = .true.

  call init_wrapper(wrapped)
  extends_base = .false.
  base_extends_item = .true.

  ! Derived type with a nested allocatable polymorphic component. The wrapper is
  ! implicitly mapped, so its generated default mapper must use runtime
  ! element-size bounds for the nested class component while preserving RTTI
  ! intrinsics and copy-back of the extended dynamic-type storage.
  !$omp target map(tofrom: extends_base, base_extends_item) map(to: base_mold)
    extends_base = extends_type_of(wrapped%item, base_mold)
    base_extends_item = extends_type_of(base_mold, wrapped%item)
    wrapped%marker = wrapped%marker + 1
    wrapped%item%base_value = wrapped%item%base_value + 1
  !$omp end target

  if (.not. extends_base) failed = .true.
  if (base_extends_item) failed = .true.
  if (wrapped%marker /= 6) failed = .true.
  if (wrapped%item%base_value /= 13) failed = .true.

  if (failed) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
