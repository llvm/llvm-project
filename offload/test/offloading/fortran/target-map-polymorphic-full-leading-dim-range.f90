! Offload regression test for a larger contiguous slice of a multidimensional
! polymorphic array with array sectioning.
!
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61 && %libomptarget-run-generic | %fcheck-generic

module target_map_polymorphic_full_leading_dim_range_mod
  implicit none

  type :: base_t
    integer :: x = 0
  end type base_t
end module target_map_polymorphic_full_leading_dim_range_mod

program main
  use target_map_polymorphic_full_leading_dim_range_mod
  implicit none

  class(base_t), allocatable :: obj(:,:)
  integer :: i, j
  logical :: failed

  allocate(base_t :: obj(5,5))

  do j = 1, 5
    do i = 1, 5
      obj(i,j)%x = 100 * i + j
    end do
  end do

  !$omp target map(tofrom: obj(:,2:3))
    do j = 2, 3
      do i = 1, 5
        obj(i,j)%x = obj(i,j)%x + 1000
      end do
    end do
  !$omp end target

  failed = .false.
  do j = 2, 3
    do i = 1, 5
      if (obj(i,j)%x /= 100 * i + j + 1000) failed = .true.
    end do
  end do

  do i = 1, 5
    if (obj(i,1)%x /= 100 * i + 1) failed = .true.
    if (obj(i,4)%x /= 100 * i + 4) failed = .true.
  end do

  if (failed) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(obj)
end program main

! CHECK: ======= Test Passed! =======
