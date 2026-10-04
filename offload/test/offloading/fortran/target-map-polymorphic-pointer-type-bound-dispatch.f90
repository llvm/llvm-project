! Offload test for dynamic dispatch through a mapped polymorphic pointer
! descriptor.
!
! This verifies that a class pointer descriptor's derived_type addendum is
! attached to the canonical device RTTI global, and that dispatch follows the
! pointer's current dynamic type across separate target regions.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_pointer_type_bound_dispatch_mod
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

end module polymorphic_pointer_type_bound_dispatch_mod

program main
  use polymorphic_pointer_type_bound_dispatch_mod
  implicit none

  type(child_a_t), target :: a_obj
  type(child_b_t), target :: b_obj
  class(base_t), pointer :: obj
  integer :: result_a
  integer :: result_b

  a_obj%base_value = 5
  a_obj%a_value = 3
  b_obj%base_value = 7
  b_obj%b_value = 2

  result_a = -1
  result_b = -1

  obj => a_obj
  !$omp target map(tofrom: obj, result_a)
    result_a = obj%value()
    call obj%adjust(4)
  !$omp end target

  obj => b_obj
  !$omp target map(tofrom: obj, result_b)
    result_b = obj%value()
    call obj%adjust(4)
  !$omp end target

  if (result_a /= 35) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (result_b /= 207) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (a_obj%base_value /= 9) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (a_obj%a_value /= 4) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (b_obj%base_value /= 15) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (b_obj%b_value /= 4) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
