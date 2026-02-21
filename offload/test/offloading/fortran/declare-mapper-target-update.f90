! Test we can correctly utilise declare mapper with update to/from
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program declare_mapper_update
   type my_type
      integer              :: x, y
   end type

   !$omp declare mapper (map_x : my_type :: var) map (var%x)
   !$omp declare mapper (map_y : my_type :: var) map (var%y)

   type(my_type) :: instance

   instance%x = 10
   instance%y = 20

   !$omp target enter data map(to: instance)

   instance%x = 30
   instance%y = 40

   ! Only update x
   !$omp target update to(mapper(map_x) : instance)

   ! Pull both x and y back seperately, and print, we 
   ! should have 30 for x, and 20 for y. As the previous
   ! update should only have written to x.
   !$omp target update from(mapper(map_x) : instance)
   !$omp target update from(mapper(map_y) : instance)

    print *, instance%x
    print *, instance%y
end program

!CHECK: 30
!CHECK: 20
