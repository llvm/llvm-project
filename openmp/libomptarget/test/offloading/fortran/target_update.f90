! Offloading test for the `target update` directive.

! XFAIL: amdgcn-amd-amdhsa

! REQUIRES: flang, amdgcn-amd-amdhsa

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program target_update
    implicit none
    integer :: x(1)
    integer :: host_id
    integer :: device_id(1)

    INTERFACE
        FUNCTION omp_get_device_num() BIND(C)
            USE, INTRINSIC :: iso_c_binding, ONLY: C_INT
            integer :: omp_get_device_num
        END FUNCTION omp_get_device_num
    END INTERFACE

    x(1) = 5
    host_id = omp_get_device_num()

!$omp target enter data map(to:x, device_id)
!$omp target
    x(1) = 42
!$omp end target

    ! Test that without a `target update` directive, the target update to x is
    ! not yet seen by the host.
    ! CHECK: After first target regions and before target update: x = 5
    print *, "After first target regions and before target update: x =", x(1)

!$omp target
    x(1) = 84
    device_id(1) = omp_get_device_num()
!$omp end target
!$omp target update from(x, device_id)

    ! Test that after the `target update`, the host can see the new x value.
    ! CHECK: After second target regions and target update: x = 84
    print *, "After second target regions and target update: x =", x(1)

    ! Make sure that offloading to the device actually happened. This way we
    ! verify that we didn't take the fallback host execution path.
    ! CHECK: Offloading succeeded!
    if (host_id /= device_id(1)) then
        print *, "Offloading succeeded!"
    else
        print *, "Offloading failed!"
    end if
end program target_update
