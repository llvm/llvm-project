! OpenMP offloading test that checks we correctly map zero sized arrays
! allowing subsequent presence checks to pass. This is legal in Fortran
! as zero sized arrays are a little bit different to the scenario where
! we have a not present argument or something that's not been allocated
! at all.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
 integer, allocatable :: array(:)
 allocate(array(0))

 !$omp target enter data map(to: array) 

 ! Despite there technically being no data, the
 ! presence check should legally return true here
 ! and not fail.
 !$omp target map(present, alloc: array)
 !$omp end target

 print *, "PASS"
end program

! CHECK: PASS
