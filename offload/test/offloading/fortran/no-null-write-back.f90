! OpenMP offloading regression test that checks we do not write-back data
! in cases where the host descriptor and data have been deallocated during the
! time the data was on device. This predominantly crops up as an issue in the
! UMT benchmark.
!
! The key component of this test is that we map back elements of an array inside
! of a derived type, and then want to deallocate them on the host, but subsequent
! maps back from device of other elements re-allocate/re-map back the old data we
! just went deallocated on host.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program prog
   implicit none

   type :: dtype_a
      real(8),      pointer, contiguous :: array2d(:,:) => null()
      integer :: scalar_int
   end type dtype_a

   type :: dtype_b
      type(dtype_a),      pointer     :: array1d(:) => null()
   end type dtype_b

   integer :: i, j = 4
   type(dtype_b), pointer :: alloca_dtype => null()

   allocate(alloca_dtype)
   allocate(alloca_dtype%array1d(j))
   do i = 1, j
      allocate(alloca_dtype%array1d(i)%array2d(10,10))
   enddo

    !$omp target enter data map(to:alloca_dtype)
    !$omp target enter data map(always,to:alloca_dtype%array1d)
   do i = 1, j
    !$omp target enter data map(always,to:alloca_dtype%array1d(i)%array2d)
   end do

    ! Check everything is all still associated as we'd expect.
    do i = 1, j
        if (associated(alloca_dtype%array1d(i)%array2d)) then
          write (*,*) "iteration", i, "is associated"
       endif
    enddo

! In this loop, we map back scalar_int and array2d of the corresponding
! array1d iteration, however, subsequent map backs of scalar_int, will
! re-allocate and re-associate the previous iteration of array2d.
    do i = 1, j
!$omp target update from(alloca_dtype%array1d(i)%scalar_int)
!$omp target update from(alloca_dtype%array1d(i)%array2d)
        deallocate(alloca_dtype%array1d(i)%array2d)
    enddo

    do i = 1, j
        if (associated(alloca_dtype%array1d(i)%array2d) .NEQV. .TRUE.) then
            write (*,*) "iteration", i, "is unassociated"
       endif
    enddo
end program prog

! CHECK:  iteration  1  is associated
! CHECK:  iteration  2  is associated
! CHECK:  iteration  3  is associated
! CHECK:  iteration  4  is associated
! CHECK:  iteration  1  is unassociated
! CHECK:  iteration  2  is unassociated
! CHECK:  iteration  3  is unassociated
! CHECK:  iteration  4  is unassociated
