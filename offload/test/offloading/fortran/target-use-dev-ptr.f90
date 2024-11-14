! Basic test of use_device_ptr, checking if the appropriate
! addresses are maintained across target boundaries
! REQUIRES: clang, flang, amdgcn-amd-amdhsa

! RUN: %clang -c -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   %S/../../Inputs/target-use-dev-ptr.c -o target-use-dev-ptr_c.o
! RUN: %libomptarget-compile-fortran-generic target-use-dev-ptr_c.o
! RUN: %t | %fcheck-generic

program use_device_test
   use iso_c_binding
   interface
      type(c_ptr) function get_ptr() BIND(C)
         USE, intrinsic :: iso_c_binding
         implicit none
      end function get_ptr

      integer(c_int) function check_result(host, dev) BIND(C)
         USE, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in) :: host, dev
      end function check_result
   end interface

   type(c_ptr) :: device_ptr, x

   x = get_ptr()
   device_ptr = x

   !$omp target data map(tofrom: x) use_device_ptr(x)
   device_ptr = x
   !$omp end target data

   print *, check_result(x, device_ptr)
end program use_device_test

! CHECK: SUCCESS
