! Test implicit mapping of alloctable record fields.

! REQUIRES: flang, amdgpu

! This fails only because it needs the Fortran runtime built for device. If this
! is available, this test succeeds when run.
! XFAIL: *

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program test_implicit_field_mapping
  implicit none

  type record_t
    real, allocatable :: not_to_implicitly_map(:)
    real, allocatable :: to_implicitly_map(:)
  end type

  type(record_t) :: dst_record
  real :: src_array(10)
  real :: dst_sum, src_sum
  integer :: i

  call random_number(src_array)
  dst_sum = 0
  src_sum = 0

  do i=1,10
    src_sum = src_sum + src_array(i)
  end do
  print *, "src_sum=", src_sum

  !$omp target map(from: dst_sum)
    dst_record%to_implicitly_map = src_array
    dst_sum = 0

    do i=1,10
      dst_sum = dst_sum + dst_record%to_implicitly_map(i)
    end do
  !$omp end target

  print *, "dst_sum=", dst_sum

  if (src_sum == dst_sum) then
    print *, "Test succeeded!"
  else
    print *, "Test failed!", " dst_sum=", dst_sum, "vs. src_sum=", src_sum
  endif
end program

! CHECK: "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK: Test succeeded!
