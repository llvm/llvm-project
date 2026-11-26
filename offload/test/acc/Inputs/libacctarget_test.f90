! Test-only helper for the Fortran tests of the OpenACC runtime.
module libacctarget_test
  implicit none

  interface
    ! Bring the runtime up, which initializes all of its devices.
    subroutine libacctarget_init_all_devices() bind(c, name="__tgt_acc_rtl_init")
    end subroutine

    ! Bring the runtime down.
    subroutine libacctarget_deinit() bind(c, name="__tgt_acc_rtl_deinit")
    end subroutine
  end interface

contains
  ! Report a failed check. A test prints PASS at the end, and the output is
  ! checked for FAIL.
  subroutine expect(label, cond)
    character(*), intent(in) :: label
    logical, intent(in) :: cond
    if (.not. cond) print '(2A)', 'FAIL: ', label
  end subroutine
end module
