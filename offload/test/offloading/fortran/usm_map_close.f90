! Test for map type close, verifying it appropriately places memory
! near/on device when utilised in USM mode.
! REQUIRES: clang, flang, amdgpu

! RUN: %clang -c -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
! RUN:   %S/../../Inputs/target-use-dev-ptr.c -o target-use-dev-ptr_c.o
! RUN: %libomptarget-compile-fortran-generic target-use-dev-ptr_c.o
! RUN: env HSA_XNACK=1 \
! RUN: %libomptarget-run-generic | %fcheck-generic

program use_device_test
    use iso_c_binding
    implicit none
    interface
       type(c_ptr) function get_ptr() BIND(C)
          USE, intrinsic :: iso_c_binding
          implicit none
       end function get_ptr
 
       integer(c_int) function check_equality(host, dev) BIND(C)
          USE, intrinsic :: iso_c_binding
          implicit none
          type(c_ptr), value, intent(in) :: host, dev
       end function check_equality
    end interface
    type(c_ptr) :: host_alloc, device_alloc
    integer, pointer :: a
  !$omp requires unified_shared_memory

    allocate(a)
    host_alloc = C_LOC(a)

! map + target no close
device_alloc = c_null_ptr
!$omp target data map(tofrom: a, device_alloc)
!$omp target map(tofrom: device_alloc)
    device_alloc = C_LOC(a)
!$omp end target
!$omp end target data

! CHECK: a used from unified memory
if (check_equality(host_alloc, device_alloc) == 1) then
    print*, "a used from unified memory"
end if

! map + target with close
device_alloc = c_null_ptr
!$omp target data map(close, tofrom: a) map(tofrom: device_alloc)
!$omp target map(tofrom: device_alloc)
    device_alloc = C_LOC(a)
!$omp end target
!$omp end target data

! CHECK: a copied to device
if (check_equality(host_alloc, device_alloc) == 0) then
    print *, "a copied to device"
end if

! map + use_device_ptr no close
device_alloc = c_null_ptr
!$omp target data map(tofrom: a) use_device_ptr(a)
    device_alloc = C_LOC(a)
!$omp end target data

! CHECK: a used from unified memory with use_device_ptr
if (check_equality(host_alloc, device_alloc) == 1) then
    print *, "a used from unified memory with use_device_ptr"
end if

! map enter/exit + close
device_alloc = c_null_ptr
!$omp target enter data map(close, to: a)

!$omp target map(from: device_alloc)
    device_alloc = C_LOC(a)
!$omp end target

!$omp target exit data map(from: a)

! CHECK: a has been mapped to the device
if (check_equality(host_alloc, device_alloc) == 0) then
    print *, "a has been mapped to the device"
end if

end program use_device_test
