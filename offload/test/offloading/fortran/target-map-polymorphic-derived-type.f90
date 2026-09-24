! Offload test for mapping a polymorphic Fortran derived type whose dynamic
! type extends its declared type.
!
! The initial target enter data map is performed while the selector has the
! extended dynamic type.  Later target regions use the original base-class
! polymorphic variable in map clauses.  This follows the OpenMP rule that a
! polymorphic list item whose dynamic type differs from its declared type must
! already have a corresponding list item in the device data environment when it
! is encountered by a map-entering construct.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module polymorphic_map_mod
  implicit none

  type :: base_t
    integer :: base_value
    integer :: base_array(4)
  end type base_t

  type, extends(base_t) :: child_t
    integer :: child_value
    integer :: child_array(4)
  end type child_t

contains

  subroutine init_child(obj)
    class(base_t), allocatable, intent(out) :: obj
    integer :: i

    allocate(child_t :: obj)

    select type (obj)
    type is (child_t)
      obj%base_value = 10
      obj%child_value = 20
      do i = 1, 4
        obj%base_array(i) = i
        obj%child_array(i) = 10 * i
      end do
    end select
  end subroutine init_child

end module polymorphic_map_mod

program main
  use polymorphic_map_mod
  implicit none

  class(base_t), allocatable :: obj
  integer :: i
  logical :: logic

  call init_child(obj)

  logic = .false.

  ! Map the full dynamic object. The list item in later regions is the
  ! base-class polymorphic variable, which requires the corresponding dynamic
  ! type allocation to already exist in the device data environment.
  select type (obj)
  type is (child_t)
    !$omp target enter data map(to: obj)
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  ! Verify that the dynamic type is preserved and that the extended components
  ! can also be accessed after mapping the base-class polymorphic list item.
  !$omp target map(tofrom: logic)
    obj%base_value = obj%base_value + 1
    do i = 1, 4
      obj%base_array(i) = obj%base_array(i) + 1
    end do

    select type (obj)
    type is (child_t)
      logic = .true.
      obj%child_value = obj%child_value + obj%base_value
      do i = 1, 4
        obj%child_array(i) = obj%child_array(i) + obj%base_array(i)
      end do
    class default
      obj%base_value = -999
    end select
  !$omp end target

  ! Copy the full dynamic object back.
  select type (obj)
  type is (child_t)
    !$omp target exit data map(from: obj)
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  if (.not. logic) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  if (obj%base_value /= 11) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  do i = 1, 4
    if (obj%base_array(i) /= i + 1) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
  end do

  select type (obj)
  type is (child_t)
    if (obj%child_value /= 31) then
      print *, "======= Test Failed! ======="
      stop 1
    end if

    do i = 1, 4
      if (obj%child_array(i) /= 10 * i + i + 1) then
        print *, "======= Test Failed! ======="
        stop 1
      end if
    end do
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  print *, "======= Test Passed! ======="
end program main

! CHECK: ======= Test Passed! =======
