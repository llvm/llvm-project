! RUN: %libacctarget-compile-fortran-generic
! RUN: env CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
! RUN:   %libacctarget-run-generic | %fcheck-generic
! REQUIRES: flang, libacctarget, host

! CHECK-NOT: FAIL
! CHECK: PASS

program t
  use openacc
  use libacctarget_test
  implicit none
  integer(acc_device_kind) :: def

  call libacctarget_init_all_devices()

  ! The default device type is the first one with devices: the Intel one if
  ! there are Intel devices, and otherwise the host.
  def = acc_device_host
  if (acc_get_num_devices(acc_device_intelgpu) > 0) def = acc_device_intelgpu

  call expect('no nvidia devices', acc_get_num_devices(acc_device_nvidia) == 0)
  call expect('no radeon devices', acc_get_num_devices(acc_device_radeon) == 0)
  call expect('no nvidia device number', &
              acc_get_device_num(acc_device_nvidia) == -1)
  call expect('no radeon device number', &
              acc_get_device_num(acc_device_radeon) == -1)
  call expect('not_host counts the Intel devices', &
              acc_get_num_devices(acc_device_not_host) == &
              acc_get_num_devices(acc_device_intelgpu))
  call expect('the host has devices', acc_get_num_devices(acc_device_host) >= 1)
  call expect('host device number', acc_get_device_num(acc_device_host) == 0)
  call expect('default counts the default type', &
              acc_get_num_devices(acc_device_default) == acc_get_num_devices(def))
  call expect('current counts the default type', &
              acc_get_num_devices(acc_device_current) == &
              acc_get_num_devices(def))
  call expect('default device number', &
              acc_get_device_num(acc_device_default) == 0)
  call expect('current device number', &
              acc_get_device_num(acc_device_current) == 0)
  call expect('the default type is the device type', &
              acc_get_device_type() == def)

  call libacctarget_deinit()
  print '(A)', 'PASS'
end program
