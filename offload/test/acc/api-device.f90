! RUN: %libacctarget-compile-fortran-generic
! RUN: %libacctarget-run-generic | %fcheck-generic
! REQUIRES: flang, libacctarget, host

! CHECK-NOT: FAIL
! CHECK: PASS

program t
  use openacc
  use libacctarget_test
  implicit none
  integer(acc_device_kind) :: types(4), def, other
  integer :: i, j, n, total, not_host, nmin

  types = [acc_device_nvidia, acc_device_radeon, acc_device_intelgpu, &
           acc_device_host]

  call libacctarget_init_all_devices()

  ! There is a device, and acc_device_not_host counts the devices of the types
  ! that are not the host.
  total = 0
  not_host = 0
  do i = 1, size(types)
    total = total + acc_get_num_devices(types(i))
    if (types(i) /= acc_device_host) &
      not_host = not_host + acc_get_num_devices(types(i))
  end do
  call expect('there is a device', total >= 1)
  call expect('not_host counts the devices that are not the host', &
              acc_get_num_devices(acc_device_not_host) == not_host)

  ! The device number is 0 where there are devices, and -1 where there are none.
  do i = 1, size(types)
    if (acc_get_num_devices(types(i)) > 0) then
      call expect('initial device number of a type', &
                  acc_get_device_num(types(i)) == 0)
    else
      call expect('absent type has no device number', &
                  acc_get_device_num(types(i)) == -1)
    end if
  end do

  ! The initial device type is a concrete type, and never acc_device_default.
  def = acc_get_device_type()
  call expect('initial device type has devices', &
              any(def == types) .and. acc_get_num_devices(def) >= 1)
  n = acc_get_num_devices(def)
  call expect('default counts the implementation default type', &
              acc_get_num_devices(acc_device_default) == n)
  call expect('current counts the current type', &
              acc_get_num_devices(acc_device_current) == n)
  call expect('initial device number', acc_get_device_num(def) == 0)
  call expect('initial default device number', &
              acc_get_device_num(acc_device_default) == 0)
  call expect('initial current device number', &
              acc_get_device_num(acc_device_current) == 0)

  ! Select every device of the current type in turn, and then the first again.
  ! The current type is still the implementation default, so default and
  ! current agree.
  do j = 0, n - 1
    call acc_set_device_num(j, def)
    call expect('device number after set', acc_get_device_num(def) == j)
    call expect('device type after set_device_num', &
                acc_get_device_type() == def)
    call expect('default device number after set', &
                acc_get_device_num(acc_device_default) == j)
    call expect('current device number after set', &
                acc_get_device_num(acc_device_current) == j)
  end do
  call acc_set_device_num(0, def)
  call expect('device number after reset', acc_get_device_num(def) == 0)

  ! acc_device_none selects the device number for all device types that have
  ! devices, and does not change the device type. A number that is a device of
  ! every one of them is the last one of the type with the fewest devices.
  nmin = acc_get_num_devices(def)
  do i = 1, size(types)
    if (acc_get_num_devices(types(i)) > 0) &
      nmin = min(nmin, acc_get_num_devices(types(i)))
  end do
  call acc_set_device_num(nmin - 1, acc_device_none)
  do i = 1, size(types)
    if (acc_get_num_devices(types(i)) > 0) then
      call expect('device number of a type after selecting all types', &
                  acc_get_device_num(types(i)) == nmin - 1)
    else
      call expect('absent type has no device number after selecting all', &
                  acc_get_device_num(types(i)) == -1)
    end if
  end do
  call expect('device type after selecting all types', &
              acc_get_device_type() == def)
  ! A negative number selects the default one, the first device.
  call acc_set_device_num(-1, acc_device_none)
  do i = 1, size(types)
    if (acc_get_num_devices(types(i)) > 0) &
      call expect('device number after a negative number for all types', &
                  acc_get_device_num(types(i)) == 0)
  end do

  ! Select every device type that has devices, the host included.
  do i = 1, size(types)
    if (acc_get_num_devices(types(i)) > 0) then
      call acc_set_device_type(types(i))
      call expect('device type after set', &
                  acc_get_device_type() == types(i))
    end if
  end do
  call acc_set_device_type(acc_device_default)
  call expect('device type after set default', acc_get_device_type() == def)

  ! default vs current only diverge when another type has devices, which the
  ! host is on a system with an accelerator.
  other = acc_device_none
  do i = 1, size(types)
    if (types(i) /= def .and. acc_get_num_devices(types(i)) > 0) then
      other = types(i)
      exit
    end if
  end do
  if (other == acc_device_none) then
    print '(A)', 'SKIP: default vs current needs two types with devices'
  else
    if (n >= 2) call acc_set_device_num(1, def)
    call acc_set_device_type(other)
    call expect('current type is the selected type', &
                acc_get_device_type() == other)
    call expect('current counts the selected type', &
                acc_get_num_devices(acc_device_current) == &
                acc_get_num_devices(other))
    call expect('default still counts the initial type', &
                acc_get_num_devices(acc_device_default) == n)
    call expect('current device number is the selected type', &
                acc_get_device_num(acc_device_current) == &
                acc_get_device_num(other))
    call expect('default device number is the initial type', &
                acc_get_device_num(acc_device_default) == &
                acc_get_device_num(def))
    if (n >= 2) then
      call expect('default and current device numbers differ', &
                  acc_get_device_num(acc_device_default) == 1 .and. &
                  acc_get_device_num(acc_device_current) == 0)
    end if
    call acc_set_device_type(acc_device_default)
    call acc_set_device_num(0, def)
  end if

  call libacctarget_deinit()
  print '(A)', 'PASS'
end program
