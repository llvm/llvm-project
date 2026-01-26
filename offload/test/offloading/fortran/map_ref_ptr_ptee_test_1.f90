! This checks that we can specify ref_ptee and ref_ptr, not encounter
! an error and correctly map data to and from device.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=61
! RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=0  %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
    implicit none
    integer,  pointer :: map_ptr(:)
    integer, target :: b(10)
    integer :: index

    map_ptr => b

    ! Should have auto attach applied if my reading is
    ! correct and automatically attach to ref_ptee. So
    ! internally we implicitly apply the attach map
    ! type.
    !$omp target enter data map(ref_ptee, to: map_ptr)
    !$omp target enter data map(ref_ptr, to: map_ptr)

    ! should in theory memory access fault if we haven't attached
    ! correctly above. But if all went well should go fine.
    !$omp target map(to: index)
        do index = 1, 10
            map_ptr(index) = index
        end do
    !$omp end target

    ! Don't care about the descriptor, but we do want to
    ! deallocate it and only it and then map the data
    ! back. Doing it in a weird-ish order to test we can
    ! delete the descriptor separately and still pull the
    ! data back.
    !$omp target exit data map(ref_ptee, from: map_ptr)
    !$omp target exit data map(ref_ptr, delete: map_ptr)

    do index = 1, 10
        if (map_ptr(index) /= index) then
            print*, "Failed!"
            stop 1
        endif
    end do

    print*, "Passed!"
end program

! CHECK: Passed!
