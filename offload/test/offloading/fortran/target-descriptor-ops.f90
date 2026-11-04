! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  implicit none
  integer :: result

  ! CHECK: 100
  result = 0
  !$omp target map(from: result)
  block
    integer, allocatable :: arr(:)
    integer :: i
    allocate(arr(4))
    do i = 1, 4
      arr(i) = i * 10
    end do
    result = arr(1) + arr(2) + arr(3) + arr(4)
    deallocate(arr)
  end block
  !$omp end target
  print *, result

  ! CHECK: 21
  result = 0
  !$omp target map(from: result)
  block
    integer, allocatable :: mat(:,:)
    allocate(mat(2, 3))
    mat(1,1) = 1; mat(2,1) = 2
    mat(1,2) = 3; mat(2,2) = 4
    mat(1,3) = 5; mat(2,3) = 6
    result = mat(1,1) + mat(2,1) + mat(1,2) + mat(2,2) + mat(1,3) + mat(2,3)
    deallocate(mat)
  end block
  !$omp end target
  print *, result

  ! CHECK: 17
  result = 0
  !$omp target map(from: result)
  block
    integer, allocatable :: arr(:)
    allocate(arr(8))
    result = size(arr) + lbound(arr, 1) + ubound(arr, 1)
    deallocate(arr)
  end block
  !$omp end target
  print *, result
end program main
