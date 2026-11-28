! Test that every host thread has its own copy of the device selection ICVs
! (OpenACC 3.4, section 2.3.1): "If an ICV is modified, then a unique copy of
! that ICV must be created for the modifying thread". Two threads select
! different devices, or different device types, and read their selection back:
! each one has to get its own.
!
! The selections can only be told apart if there is something to choose
! between: a device type with at least two devices for the device number, and
! at least two device types with devices for the device type. A part of the test
! that cannot be run on the system reports that it is skipped.

! RUN: %libacctarget-compile-fortran-openmp-generic
! RUN: %libacctarget-run-generic | %fcheck-generic
! REQUIRES: flang, libacctarget, host

! CHECK-NOT: FAIL
! CHECK: PASS

program t
  use openacc
  use libacctarget_test
  implicit none

  integer(acc_device_kind), parameter :: all_types(4) = &
      [acc_device_nvidia, acc_device_radeon, acc_device_intelgpu, &
       acc_device_host]

  call libacctarget_init_all_devices()

  call test_device_num()
  call test_device_type()

  call libacctarget_deinit()
  print '(A)', 'PASS'

contains

  ! The first thread selects the first device of the first device type that has
  ! at least two devices, and the second thread the second one.
  subroutine test_device_num()
    use omp_lib
    integer(acc_device_kind) :: type
    integer :: i, tid

    type = acc_device_none
    do i = 1, size(all_types)
      if (acc_get_num_devices(all_types(i)) >= 2) then
        type = all_types(i)
        exit
      end if
    end do
    if (type == acc_device_none) then
      print '(A)', 'SKIP: the device number needs a type with two devices'
      return
    end if

    !$omp parallel num_threads(2) private(tid)
    tid = omp_get_thread_num()
    call expect('there are two threads', omp_get_num_threads() == 2)
    call acc_set_device_num(tid, type)
    !$omp barrier

    !$omp critical
    call expect('own device number is kept', acc_get_device_num(type) == tid)
    !$omp end critical
    call acc_set_device_num(0, type)
    !$omp end parallel
    call acc_set_device_type(acc_device_default)
  end subroutine

  ! The first thread selects the first device type that has devices, and the
  ! second thread the second one.
  subroutine test_device_type()
    use omp_lib
    integer(acc_device_kind) :: avail(4)
    integer :: i, nt, tid

    nt = 0
    do i = 1, size(all_types)
      if (acc_get_num_devices(all_types(i)) > 0) then
        nt = nt + 1
        avail(nt) = all_types(i)
      end if
    end do
    if (nt < 2) then
      print '(A)', 'SKIP: the device type needs two types with devices'
      return
    end if

    !$omp parallel num_threads(2) private(tid)
    tid = omp_get_thread_num()
    call expect('there are two threads', omp_get_num_threads() == 2)
    call acc_set_device_type(avail(tid + 1))
    !$omp barrier

    !$omp critical
    call expect('own device type is kept', &
                acc_get_device_type() == avail(tid + 1))
    call expect('current follows this thread', &
                acc_get_num_devices(acc_device_current) == &
                acc_get_num_devices(avail(tid + 1)))
    ! avail(1) is the first type with devices, which is the implementation
    ! default (nvidia, then radeon, then intelgpu, then the host).
    call expect('default is the implementation type', &
                acc_get_num_devices(acc_device_default) == &
                acc_get_num_devices(avail(1)))
    !$omp end critical
    call acc_set_device_type(acc_device_default)
    !$omp end parallel
  end subroutine

end program
