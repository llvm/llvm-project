! Offloading test checking interaction of fixed size
! arrays with enter, exit and target
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer :: A(10)

   !$omp target enter data map(alloc: A)

    !$omp target
        do I = 1, 10
            A(I) = I
        end do
    !$omp end target

    !$omp target exit data map(from: A)

    !$omp target exit data map(delete: A)

    do i = 1, 10
        print *, A(i)
    end do
end program

! CHECK: 1
! CHECK: 2
! CHECK: 3
! CHECK: 4
! CHECK: 5
! CHECK: 6
! CHECK: 7
! CHECK: 8
! CHECK: 9
! CHECK: 10
