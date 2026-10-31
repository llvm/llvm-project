! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_data_use_device_addr_common_block
  use iso_c_binding, only : c_associated, c_f_pointer, c_loc, c_ptr
  use omp_lib, only : omp_get_default_device, omp_get_mapped_ptr
  implicit none

  integer, target :: aggregate_x, aggregate_y, member_x, member_y
  integer, pointer :: device_aggregate_x, device_aggregate_y
  integer, pointer :: device_member_x, device_member_y
  type(c_ptr) :: host_aggregate_x, host_aggregate_y
  type(c_ptr) :: host_member_x, host_member_y
  type(c_ptr) :: uda_aggregate_x, uda_aggregate_y
  type(c_ptr) :: uda_member_x, uda_member_y
  type(c_ptr) :: mapped_aggregate_x, mapped_aggregate_y
  type(c_ptr) :: mapped_member_x, mapped_member_y
  integer :: device

  common /aggregate/ aggregate_x, aggregate_y
  common /members/ member_x, member_y

  aggregate_x = 10
  aggregate_y = 20
  member_x = 30
  member_y = 40
  device = omp_get_default_device()

  host_aggregate_x = c_loc(aggregate_x)
  host_aggregate_y = c_loc(aggregate_y)
  host_member_x = c_loc(member_x)
  host_member_y = c_loc(member_y)

  !$omp target enter data map(to: member_x, member_y)

  ! The aggregate COMMON returns one device base whose second member requires
  ! an offset. The pre-mapped COMMON returns one device address per member.
  !$omp target data map(tofrom: /aggregate/) &
  !$omp& use_device_addr(/aggregate/, /members/)
    uda_aggregate_x = c_loc(aggregate_x)
    uda_aggregate_y = c_loc(aggregate_y)
    uda_member_x = c_loc(member_x)
    uda_member_y = c_loc(member_y)

    mapped_aggregate_x = omp_get_mapped_ptr(host_aggregate_x, device)
    mapped_aggregate_y = omp_get_mapped_ptr(host_aggregate_y, device)
    mapped_member_x = omp_get_mapped_ptr(host_member_x, device)
    mapped_member_y = omp_get_mapped_ptr(host_member_y, device)

    if (.not. c_associated(uda_aggregate_x, mapped_aggregate_x)) then
      print *, "FAIL: aggregate first member address"
      stop 1
    end if
    if (.not. c_associated(uda_aggregate_y, mapped_aggregate_y)) then
      print *, "FAIL: aggregate second member address"
      stop 1
    end if
    if (.not. c_associated(uda_member_x, mapped_member_x)) then
      print *, "FAIL: expanded first member address"
      stop 1
    end if
    if (.not. c_associated(uda_member_y, mapped_member_y)) then
      print *, "FAIL: expanded second member address"
      stop 1
    end if

    call c_f_pointer(uda_aggregate_x, device_aggregate_x)
    call c_f_pointer(uda_aggregate_y, device_aggregate_y)
    call c_f_pointer(uda_member_x, device_member_x)
    call c_f_pointer(uda_member_y, device_member_y)

    !$omp target has_device_addr(device_aggregate_x, device_aggregate_y, &
    !$omp& device_member_x, device_member_y)
      device_aggregate_x = device_aggregate_x + 1
      device_aggregate_y = device_aggregate_y + 2
      device_member_x = device_member_x + 3
      device_member_y = device_member_y + 4
    !$omp end target
  !$omp end target data

  !$omp target exit data map(from: member_x, member_y)

  if (aggregate_x /= 11 .or. aggregate_y /= 22 .or. &
      member_x /= 33 .or. member_y /= 44) then
    print *, "FAIL: incorrect values", aggregate_x, aggregate_y, member_x, &
        member_y
    stop 2
  end if

  print *, "PASS"
end program

! CHECK: PASS
