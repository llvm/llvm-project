! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

module enter_exit_mapper_mod
  implicit none

  type :: field_type
    real, allocatable :: values(:)
  end type field_type

  type :: tile_type
    type(field_type) :: field
    integer, allocatable :: neighbors(:)
  end type tile_type

contains
  subroutine init_tile(tile)
    type(tile_type), intent(inout) :: tile
    integer :: j

    allocate(tile%field%values(4))
    allocate(tile%neighbors(4))
    do j = 1, 4
      tile%field%values(j) = 10.0 * j
      tile%neighbors(j) = j
    end do
  end subroutine init_tile

end module enter_exit_mapper_mod

program implicit_enter_exit
  use enter_exit_mapper_mod
  implicit none
  integer :: j
  type(tile_type) :: tile

  call init_tile(tile)

  !$omp target enter data map(alloc: tile%field%values)

  !$omp target
  do j = 1, size(tile%field%values)
    tile%field%values(j) = 5.0 * j
  end do
  !$omp end target

  !$omp target exit data map(from: tile%field%values)

  do j = 1, size(tile%field%values)
    if (tile%field%values(j) /= 5.0 * j) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
    if (tile%neighbors(j) /= j) then
      print *, "======= Test Failed! ======="
      stop 1
    end if
  end do

  print *, "======= Test Passed! ======="
end program implicit_enter_exit

! CHECK: ======= Test Passed! =======
