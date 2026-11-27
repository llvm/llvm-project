! Check whole-array and noncontiguous updates of a character-array component.
! REQUIRES: flang, flang-rt, libc, amdgpu
! UNSUPPORTED: apu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  implicit none
  type char_t
    CHARACTER(LEN=16), dimension(10,10) :: char_arr
  end type char_t
  type(char_t) :: dtype_char
  character(len=16) :: actual(10,10), expected(10,10), host_values(10,10)
  character(len=16), parameter :: host_sentinel = 'host   sentinel '
  character(len=16), parameter :: device_sentinel = 'device sentinel '
  integer :: i, j

  dtype_char%char_arr = host_sentinel
!$omp target enter data map(alloc:dtype_char%char_arr)

  ! Check the entire string, including embedded and trailing blanks.
!$omp target map(present,alloc:dtype_char%char_arr)
  dtype_char%char_arr = 'whole from GPU  '
!$omp end target
!$omp target update from(dtype_char%char_arr)
  if (any(dtype_char%char_arr /= 'whole from GPU  ')) stop 1

  do j = 1, 10
    do i = 1, 10
      write(host_values(i,j), '(i2.2,a,i2.2,a)') i, '/', j, ': ab cd xy '
    end do
  end do
  dtype_char%char_arr = host_sentinel
  dtype_char%char_arr(3:6,:) = host_values(3:6,:)
!$omp target map(present,alloc:dtype_char%char_arr)
  dtype_char%char_arr = device_sentinel
!$omp end target

  ! Four complete strings per column, separated by six unselected strings.
!$omp target update to(dtype_char%char_arr(3:6,:))
!$omp target map(present,alloc:dtype_char%char_arr) map(from:actual)
  actual = dtype_char%char_arr
  dtype_char%char_arr = 'from GPU value  '
!$omp end target
  expected = device_sentinel
  expected(3:6,:) = host_values(3:6,:)
  if (any(actual /= expected)) stop 2

  dtype_char%char_arr = host_sentinel
!$omp target update from(dtype_char%char_arr(3:6,:))
  expected = host_sentinel
  expected(3:6,:) = 'from GPU value  '
  if (any(dtype_char%char_arr /= expected)) stop 3
!$omp target exit data map(delete:dtype_char%char_arr)

  print *, 'PASS'
end program

! CHECK: PASS
