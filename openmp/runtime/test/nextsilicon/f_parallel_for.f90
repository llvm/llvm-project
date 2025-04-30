! XFAIL: *
! RUN: %flang -cpp  -g -fopenmp -std=f2018 -l nsapi_core -l omp -l stdc++ -o %t %s && %t | FileCheck %s
#define TEST_LOG          print *, '[..]',
#define TEST_LOG_ERROR    print *, '[***ERROR***]',

module constants

implicit none

  integer(kind=8), parameter :: random_high = 2**16
  integer(kind=8), parameter :: random_low  = -(2**16)

end module constants

function fetch_random() result(r)
  use constants
  implicit none

  integer(kind=8)  :: r
  real(kind=8) :: x

  call random_number(x)

  x = x*(random_high - random_low) + random_low

  r = INT(A=x, KIND=8)

  return

end function fetch_random


module openmp_parallel_for_test
  use constants
  implicit none

  integer :: super_iterations, matrix_size, chunk_size

  integer(kind=8) :: expected_sum

  contains
  subroutine run_test(super_iterations_parm, matrix_size_parm, chunk_size_parm)
    integer :: super_iterations_parm
    integer :: matrix_size_parm
    integer :: chunk_size_parm
    integer :: super_counter
    integer(kind=8) :: parallel_sum

    integer(kind=8), dimension(matrix_size, matrix_size) :: mat

    super_iterations = super_iterations_parm
    matrix_size = matrix_size_parm
    chunk_size = chunk_size_parm

    TEST_LOG 'super iterations=', super_iterations, 'matrix_size=', matrix_size, 'chunk_size=', chunk_size

    call build_random_matrix(mat)

    expected_sum = summarize_serial(mat)

    do super_counter = 1, super_iterations_parm
      TEST_LOG 'loop #', super_counter
      parallel_sum = summarize_parallel(mat)

      if (check_result(parallel_sum, expected_sum)) then
        TEST_LOG 'test ok! diff=', abs(parallel_sum - expected_sum)
      else
        TEST_LOG_ERROR 'test failed!, expected=', expected_sum
        TEST_LOG_ERROR 'test: actual=', parallel_sum
        TEST_LOG_ERROR 'diff:', abs(parallel_sum - expected_sum)
      end if
    end do
  end subroutine run_test

  function summarize_serial(mat) result(sum)
    integer(kind=8), dimension(:,:) :: mat
    integer(kind=8)                 :: sum
    integer :: nr, nc, i, j

    nr = size(mat, 1)
    nc = size(mat, 2)

    sum = 0
    do i=1, nr
      do j=1, nc
        sum = sum + mat(i,j)
      end do
    end do

  end function summarize_serial

  pure function check_result(expected, actual) result(ok)
    integer(kind=8), intent(in) :: expected, actual
    logical :: ok

    if (expected == actual) then
      ok = .true.
    else
      ok = .false.
    end if
  end function check_result

  subroutine build_random_matrix(mat)
    integer(kind=8), dimension(:,:), intent(out) :: mat
    interface
      function fetch_random() result(r)
        integer(kind=8) :: r
      end function fetch_random
    end interface
    integer :: nr, nc, i, j

    nr = size(mat,1)
    nc = size(mat,2)

    do i=1, nr
      do j=1, nc
        mat(i,j) = fetch_random()
      end do
    end do

  end subroutine build_random_matrix

  function summarize_parallel(mat) result(sum)
    integer(kind=8), dimension(:,:), intent(in) :: mat
    integer(kind=8)                             :: sum
    integer :: nr, nc, i, j

    nr = size(mat, 1)
    nc = size(mat, 2)

    sum = 0
    !$omp parallel reduction(+:sum) private(j)
      !$omp do schedule(static, chunk_size)
        do i=1, nr
          do j=1, nc
            sum = sum + mat(i,j)
          end do
        end do
      !$omp end do
    !$omp end parallel

  end function summarize_parallel


end module openmp_parallel_for_test


program f_parallel_for

  use constants
  use openmp_parallel_for_test
  implicit none
  integer, allocatable :: seed(:)
  integer              :: random_n

  TEST_LOG 'starting'

  call random_seed(size=random_n)
  allocate(seed(random_n))
  call random_seed(get=seed)
  TEST_LOG 'seed=', seed

  TEST_LOG 'setting dynamic'
  call omp_set_dynamic(.true.)

  TEST_LOG 'dynamic test commences'

  call run_test(super_iterations_parm=10, matrix_size_parm=400, chunk_size_parm=1)
  call run_test(super_iterations_parm=10, matrix_size_parm=400, chunk_size_parm=10)
  call run_test(super_iterations_parm=10, matrix_size_parm=400, chunk_size_parm=20)
  call run_test(super_iterations_parm=10, matrix_size_parm=400, chunk_size_parm=30)

  TEST_LOG 'ending'

end program f_parallel_for




! CHECK:      [..]
! CHECK-NOT:  [***ERROR***]
