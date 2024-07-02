! Offloading test checking interaction of an
! nested derived type member map with the
! inclusion of an entire nested derived
! type being mapped
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
      type(bottom_layer) :: nested2
    end type top_layer

    type(top_layer) :: top_dtype

!$omp target map(tofrom: top_dtype%k, top_dtype%nested2%array_i2, top_dtype%nested)
    do i = 1, 10
      top_dtype%nested2%array_i2(i) = i * 2
      top_dtype%nested%array_i2(i) = i * 2
    end do

    top_dtype%nested%i2 = 30.30
    top_dtype%k = 74
!$omp end target

  print *, top_dtype%nested%i2
  print *, top_dtype%k
  print *, top_dtype%nested%array_i2
  print *, top_dtype%nested2%array_i2
end program main

!CHECK: 30.299999237060547
!CHECK: 74
!CHECK: 2. 4. 6. 8. 10. 12. 14. 16. 18. 20.
!CHECK: 2. 4. 6. 8. 10. 12. 14. 16. 18. 20.
