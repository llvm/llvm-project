! Offloading test checking interaction of a
! two 1-D allocatable arrays with a target region
! while providing the map upper and lower bounds
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer,  allocatable :: sp_read(:), sp_write(:)
    allocate(sp_read(10))
    allocate(sp_write(10))

    do i = 1, 10
        sp_read(i) = i
        sp_write(i) = 0
    end do

    !$omp target map(tofrom:sp_read(2:6)) map(tofrom:sp_write(2:6))
        do i = 1, 10
            sp_write(i) = sp_read(i)
        end do
    !$omp end target

    do i = 1, 10
        print *, sp_write(i)
    end do

    deallocate(sp_read)
    deallocate(sp_write)
end program

! CHECK: 0
! CHECK: 2
! CHECK: 3
! CHECK: 4
! CHECK: 5
! CHECK: 6
! CHECK: 0
! CHECK: 0
! CHECK: 0
! CHECK: 0
