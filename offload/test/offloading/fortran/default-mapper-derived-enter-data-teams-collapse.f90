! Regression test for default mappers on nested derived types with allocatable
! members when mapping a parent object and running an optimized target region.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -O3
! RUN: %libomptarget-run-generic | %fcheck-generic

program test_default_mapper_enter_data_teams_collapse
  implicit none

  type inner_type
    real, allocatable :: data(:)
  end type inner_type

  type outer_type
    type(inner_type) :: inner
    character(len=19) :: desc = ' '
  end type outer_type

  type(outer_type) :: obj
  integer, parameter :: n = 10
  integer :: i, j
  real :: expected, actual

  allocate(obj%inner%data(n))
  obj%inner%data = 0.0

  !$omp target enter data map(to: obj)

  !$omp target teams distribute parallel do collapse(2)
  do i = 1, n
    do j = 1, n
      obj%inner%data(i) = real(i)
    end do
  end do
  !$omp end target teams distribute parallel do

  !$omp target exit data map(from: obj)

  expected = real(n * (n + 1)) / 2.0
  actual = sum(obj%inner%data)

  if (abs(actual - expected) < 1.0e-6) then
    print *, "PASS"
  else
    print *, "FAIL", actual, expected
  end if

  deallocate(obj%inner%data)
end program test_default_mapper_enter_data_teams_collapse

! CHECK: PASS
