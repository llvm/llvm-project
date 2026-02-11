! Offload test for parameter (constant) arrays and character scalars accessed
! with dynamic indices/substrings in OpenMP target regions.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program test_parameter_mapping
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)
  logical :: all_tests_pass

  all_tests_pass = .true.

  ! Test 1: Parameter array with dynamic index
  call test_param_array_dynamic_index(all_tests_pass)

  ! Test 2: Integer parameter array
  call test_int_param_array(all_tests_pass)

  ! Test 3: Character scalar with dynamic substring
  call test_char_substring(all_tests_pass)

  ! Test 4: Verify scalar parameters work (inlined)
  call test_scalar_param(all_tests_pass)

  if (all_tests_pass) then
    print *, "PASS"
  else
    print *, "FAIL"
  endif

contains

! Test 1: Parameter array with dynamic index in target region
subroutine test_param_array_dynamic_index(test_pass)
  logical, intent(inout) :: test_pass
  real(dp), parameter :: const_array(3) = [1.0_dp, 2.0_dp, 3.0_dp]
  integer :: idx
  real(dp) :: result
  real(dp), parameter :: expected = 2.0_dp
  real(dp), parameter :: tolerance = 1.0e-10_dp

  idx = 2
  result = 0.0_dp

  !$omp target map(tofrom:result) map(to:idx)
    ! Access parameter array with dynamic index
    result = const_array(idx)
  !$omp end target

  if (abs(result - expected) > tolerance) then
    print *, "Test 1 FAILED: expected", expected, "got", result
    test_pass = .false.
  endif
end subroutine test_param_array_dynamic_index

! Test 2: Integer parameter array with different indices
subroutine test_int_param_array(test_pass)
  logical, intent(inout) :: test_pass
  integer, parameter :: int_array(4) = [10, 20, 30, 40]
  integer :: idx1, idx2
  integer :: result1, result2

  idx1 = 1
  idx2 = 4
  result1 = 0
  result2 = 0

  !$omp target map(tofrom:result1, result2) map(to:idx1, idx2)
    ! Access parameter array with different dynamic indices
    result1 = int_array(idx1)
    result2 = int_array(idx2)
  !$omp end target

  if (result1 /= 10 .or. result2 /= 40) then
    print *, "Test 2 FAILED: expected 10, 40 got", result1, result2
    test_pass = .false.
  endif
end subroutine test_int_param_array

! Test 3: Character scalar parameter with dynamic substring access
subroutine test_char_substring(test_pass)
  logical, intent(inout) :: test_pass
  character(len=20), parameter :: char_scalar = "constant_string_data"
  integer :: start_idx, end_idx
  character(len=8) :: result
  character(len=8), parameter :: expected = "string_d"

  start_idx = 10
  end_idx = 17
  result = ""

  !$omp target map(tofrom:result) map(to:start_idx, end_idx)
    ! Dynamic substring access - character scalar must be mapped
    result = char_scalar(start_idx:end_idx)
  !$omp end target

  if (result /= expected) then
    print *, "Test 3 FAILED: expected '", expected, "' got '", result, "'"
    test_pass = .false.
  endif
end subroutine test_char_substring

! Test 4: Scalar parameter (can be inlined, no mapping needed)
subroutine test_scalar_param(test_pass)
  logical, intent(inout) :: test_pass
  integer, parameter :: scalar_const = 42
  real(dp), parameter :: real_const = 3.14159_dp
  integer :: int_result
  real(dp) :: real_result
  real(dp), parameter :: tolerance = 1.0e-5_dp

  int_result = 0
  real_result = 0.0_dp

  !$omp target map(tofrom:int_result, real_result)
    ! Scalar parameters should be inlined (no mapping needed)
    int_result = scalar_const
    real_result = real_const
  !$omp end target

  if (int_result /= 42 .or. abs(real_result - real_const) > tolerance) then
    print *, "Test 4 FAILED: expected 42, 3.14159 got", int_result, real_result
    test_pass = .false.
  endif
end subroutine test_scalar_param

end program test_parameter_mapping

! CHECK: PASS
