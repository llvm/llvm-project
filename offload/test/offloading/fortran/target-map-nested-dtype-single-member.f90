! Offloading test checking interaction of an single explicit member map from a
! nested derived type.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: bottom_layer
      real(8) :: i2
      real(4) :: array_i2(10)
      real(4) :: array_j2(10)
    end type bottom_layer

    type :: top_layer
      real(4) :: i
      integer(4) :: array_i(10)
      real(4) :: j
      type(bottom_layer) :: nested
      integer, allocatable :: array_j(:)
      integer(4) :: k
    end type top_layer

    type(top_layer) :: top_dtype

!$omp target map(tofrom: top_dtype%nested%array_i2)
    do i = 1, 10
      top_dtype%nested%array_i2(i) = i * 2
    end do
!$omp end target

  print *, top_dtype%nested%array_i2
end program main

!CHECK: 2. 4. 6. 8. 10. 12. 14. 16. 18. 20.
