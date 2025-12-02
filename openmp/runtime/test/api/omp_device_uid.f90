! RUN: %flang %flags %openmp_flags -fopenmp-version=60 %s -o %t
! RUN: %t | FileCheck %s

program test_omp_device_uid_main
  use omp_lib
  use, intrinsic :: iso_c_binding
  implicit none

  integer(kind=omp_integer_kind) :: num_devices, i, num_failed
  logical :: success

  num_devices = omp_get_num_devices()
  num_failed = 0

  ! Test all devices plus the initial device (num_devices)
  do i = 0, num_devices
    success = test_omp_device_uid(i)
    if (.not. success) then
      print '("FAIL for device ", I0)', i
      num_failed = num_failed + 1
    end if
  end do

  if (num_failed /= 0) then
    print *, "FAIL"
    stop 1
  end if

  print *, "PASS"
  stop 0

contains

  logical function test_omp_device_uid(device_num)
    use omp_lib
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=omp_integer_kind), intent(in) :: device_num
    character(:), pointer :: device_uid => null()
    integer(kind=omp_integer_kind) :: device_num_from_uid

    device_uid => omp_get_uid_from_device(device_num)

    ! Check if device_uid is NULL
    if (.not. associated(device_uid)) then
      print '("FAIL for device ", I0, ": omp_get_uid_from_device returned NULL")', device_num
      test_omp_device_uid = .false.
      return
    end if

    device_num_from_uid = omp_get_device_from_uid(device_uid)
    if (device_num_from_uid /= device_num) then
      print '("FAIL for device ", I0, ": omp_get_device_from_uid returned ", I0)', &
            device_num, device_num_from_uid
      test_omp_device_uid = .false.
      return
    end if

    test_omp_device_uid = .true.

    if (associated(device_uid)) then
      deallocate(device_uid)
      nullify(device_uid)
    end if
  end function test_omp_device_uid

end program test_omp_device_uid_main

! CHECK: PASS
