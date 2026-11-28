! Offload regression test for a contiguous ranged map of a multidimensional
! polymorphic array whose dynamic element type is larger than the declared type.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_dynamic_type_contiguous_range_mod
  implicit none

  type :: base_t
    integer :: x = 0
  end type base_t

  type, extends(base_t) :: child_t
    integer :: y = 0
  end type child_t
end module target_map_polymorphic_dynamic_type_contiguous_range_mod

program main
  use target_map_polymorphic_dynamic_type_contiguous_range_mod
  implicit none

  class(base_t), allocatable :: obj(:,:)
  integer :: i, j
  logical :: failed

  allocate(child_t :: obj(5,5))

  do j = 1, 5
    do i = 1, 5
      obj(i,j)%x = 100 * i + j
    end do
  end do

  select type (obj)
  type is (child_t)
    do j = 1, 5
      do i = 1, 5
        obj(i,j)%y = 1000 * i + 10 * j
      end do
    end do
  class default
    print *, "======= Test Failed! ======="
    stop 1
  end select

  !$omp target map(tofrom: obj(3:4,2))
    obj(3,2)%x = 320
    obj(4,2)%x = 420
    select type (obj)
    type is (child_t)
      obj(3,2)%y = 3020
      obj(4,2)%y = 4020
    class default
      obj(3,2)%x = -1
    end select
  !$omp end target

  failed = .false.
  if (obj(3,2)%x /= 320) failed = .true.
  if (obj(4,2)%x /= 420) failed = .true.

  select type (obj)
  type is (child_t)
    if (obj(3,2)%y /= 3020) failed = .true.
    if (obj(4,2)%y /= 4020) failed = .true.
    if (obj(2,2)%y /= 2020) failed = .true.
    if (obj(5,2)%y /= 5020) failed = .true.
    if (obj(3,3)%y /= 3030) failed = .true.
  class default
    failed = .true.
  end select

  if (obj(2,2)%x /= 202) failed = .true.
  if (obj(5,2)%x /= 502) failed = .true.
  if (obj(3,3)%x /= 303) failed = .true.

  if (failed) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
