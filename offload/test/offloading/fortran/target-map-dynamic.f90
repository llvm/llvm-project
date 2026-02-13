! Offloading test checking lowering of arrays with dynamic extents.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

subroutine test_array_target_enter_data(dims)
    integer, intent(in) :: dims(2)
    double precision :: A(2, dims(2))
    !$omp target enter data map(to: A)

    A(2,2) = 1.0
    !$omp target
         A(1,1) = 10
         A(2,1) = 20
         A(1,2) = 30
         A(2,2) = 40
    !$omp end target

    !$omp target exit data map(from: A)

    print *, A
end subroutine test_array_target_enter_data

program main
    integer :: dimensions(2)
    dimensions(1) = 1
    dimensions(2) = 2

call test_array_target_enter_data(dimensions)
end program


! CHECK:  10. 20. 30. 40.
