! Offloading test checking interaction of implicit captured of a pointer
! targeting an allocatable member of a derived type, alongside the explicit
! map of the derived type and allocatable data via target enter and exit
! directives
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module dtype
    type :: my_dtype
            integer :: s, e
            integer,dimension(:),allocatable :: values
    end type
end module

program offload_types
    use dtype

    type(my_dtype),target :: my_instance
    integer,dimension(:),pointer :: values_ptr
    integer :: i

    allocate(my_instance%values(20))
    my_instance%s=1
    my_instance%e=20

    values_ptr => my_instance%values

    !$omp target enter data map(to:my_instance, my_instance%values)

    !$omp target
      do i = 1,20
             values_ptr(i) = i
      end do
    !$omp end target

    !$omp target exit data map(from:my_instance%values)

    write(*,*) my_instance%values

    !$omp target exit data map(release:my_instance)

    deallocate(my_instance%values)
end program

!CHECK: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
