! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
module test
implicit none

TYPE field_type
  REAL, DIMENSION(:,:), ALLOCATABLE :: density0, density1
END TYPE field_type

TYPE tile_type
  TYPE(field_type) :: field
  INTEGER          :: tile_neighbours(4)
END TYPE tile_type

TYPE chunk_type
  INTEGER                                    :: filler
  TYPE(tile_type), DIMENSION(:), ALLOCATABLE :: tiles
END TYPE chunk_type

end module test

program reproducer
  use test
  implicit none
  integer          :: i, j
  TYPE(chunk_type) :: chunk

  allocate(chunk%tiles(2))
  do i = 1, 2
    allocate(chunk%tiles(i)%field%density0(2, 2))
    allocate(chunk%tiles(i)%field%density1(2, 2))
    do j = 1, 4
      chunk%tiles(i)%tile_neighbours(j) = j * 10
    end do
  end do

  !$omp target enter data map(alloc:       &
  !$omp  chunk%tiles(2)%field%density0)

  !$omp target
    chunk%tiles(2)%field%density0(1,1) = 25
    chunk%tiles(2)%field%density0(1,2) = 50
    chunk%tiles(2)%field%density0(2,1) = 75
    chunk%tiles(2)%field%density0(2,2) = 100
  !$omp end target

  !$omp target exit data map(from:         &
  !$omp  chunk%tiles(2)%field%density0)

  if (chunk%tiles(2)%field%density0(1,1) /= 25) then
    print*, "======= Test Failed! ======="
    stop 1
  end if

  if (chunk%tiles(2)%field%density0(1,2) /= 50) then
    print*, "======= Test Failed! ======="
    stop 1
  end if

  if (chunk%tiles(2)%field%density0(2,1) /= 75) then
    print*, "======= Test Failed! ======="
    stop 1
  end if

  if (chunk%tiles(2)%field%density0(2,2) /= 100) then
    print*, "======= Test Failed! ======="
    stop 1
  end if

  do j = 1, 4
    if (chunk%tiles(2)%tile_neighbours(j) /= j * 10) then
      print*, "======= Test Failed! ======="
      stop 1
    end if
  end do

  print *, "======= Test Passed! ======="
end program reproducer

! CHECK: "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK: ======= Test Passed! =======
