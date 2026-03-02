! Test custom mappers with target update to/from clauses
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_update_mapper_test
   implicit none
   integer, parameter :: n = 100
   
   type :: my_type
      integer :: a(n)
      integer :: b(n)
   end type my_type

   ! Declare custom mapper that only maps field 'a'
   !$omp declare mapper(custom : my_type :: t) map(t%a)
   
   ! Declare default mapper that maps both fields
   !$omp declare mapper(my_type :: t) map(t%a, t%b)

   type(my_type) :: obj1, obj2
   integer :: i, sum_a, sum_b

   ! ========== Test 1: Custom mapper (field 'a' only) ==========
   
   ! Initialize data on host
   do i = 1, n
      obj1%a(i) = i
      obj1%b(i) = i * 2
   end do

   ! Allocate and update using custom mapper (only 'a')
   !$omp target enter data map(mapper(custom), alloc: obj1)

   obj1%a = 10
   !$omp target update to(mapper(custom): obj1)

   obj1%a = 0
   !$omp target update from(mapper(custom): obj1)
   
   sum_a = sum(obj1%a)
   sum_b = sum(obj1%b)

   ! CHECK: Sum of a (custom mapper): 1000
   print *, "Sum of a (custom mapper):", sum_a
   
   ! Field 'b' was never mapped with custom mapper
   ! CHECK: Sum of b (never mapped): 10100
   print *, "Sum of b (never mapped):", sum_b

   !$omp target exit data map(mapper(custom), delete: obj1)

   ! ========== Test 2: Default mapper (both fields) ==========
   
   ! Initialize separate object for default mapper test
   do i = 1, n
      obj2%a(i) = 20
      obj2%b(i) = 30
   end do
   
   ! Allocate and update using default mapper (both 'a' and 'b')
   !$omp target enter data map(mapper(default), alloc: obj2)
   
   !$omp target update to(mapper(default): obj2)
   
   obj2%a = 0
   obj2%b = 0
   
   !$omp target update from(mapper(default): obj2)
   
   sum_a = sum(obj2%a)
   sum_b = sum(obj2%b)
   
   ! CHECK: Sum of a (default mapper): 2000
   print *, "Sum of a (default mapper):", sum_a
   
   ! CHECK: Sum of b (default mapper): 3000
   print *, "Sum of b (default mapper):", sum_b

   !$omp target exit data map(mapper(default), delete: obj2)

   ! CHECK: Test passed!
   print *, "Test passed!"

end program target_update_mapper_test
