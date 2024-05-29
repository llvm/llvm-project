! Offloading test checking interaction of an
! explicit member map from two large nested
! derived types
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: bottom_layer1
    real(4) :: i4
    real(4) :: j4
    real(4) :: k4
    end type bottom_layer1

    type :: bottom_layer2
      integer(4) :: i3
      integer(4) :: j3
      integer(4) :: k3
    end type bottom_layer2

    type :: middle_layer
     real(4) :: array_i2(10)
     real(4) :: i2
     real(4) :: array_j2(10)
     type(bottom_layer1) :: nest
     type(bottom_layer2) :: nest2
    end type middle_layer

    type :: top_layer
    real(4) :: i
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    type(middle_layer) :: nested
    end type top_layer

    type(top_layer) :: top_dtype
    type(top_layer) :: top_dtype2

    top_dtype2%nested%nest%i4 = 10
    top_dtype2%nested%nest%j4 = 12
    top_dtype2%nested%nest%k4 = 54

    top_dtype2%nested%nest2%i3 = 20
    top_dtype2%nested%nest2%j3 = 40
    top_dtype2%nested%nest2%k3 = 60

    top_dtype2%nested%i2 = 200

      do i = 1, 10
        top_dtype2%array_i(i) = i
      end do

!$omp target map(from: top_dtype%nested%nest%j4, top_dtype%nested%nest%i4, top_dtype%nested%nest%k4) &
!$omp map(from: top_dtype%array_i, top_dtype%nested%nest2%i3, top_dtype%nested%i2) &
!$omp map(from: top_dtype%nested%nest2%k3, top_dtype%nested%nest2%j3) &
!$omp map(to: top_dtype2%nested%nest%j4, top_dtype2%nested%nest%i4, top_dtype2%nested%nest%k4) &
!$omp map(to: top_dtype2%array_i, top_dtype2%nested%nest2%i3, top_dtype2%nested%i2) &
!$omp map(to: top_dtype2%nested%nest2%k3, top_dtype2%nested%nest2%j3)
    top_dtype%nested%nest%i4 = top_dtype2%nested%nest%i4
    top_dtype%nested%nest%j4 = top_dtype2%nested%nest%j4
    top_dtype%nested%nest%k4 = top_dtype2%nested%nest%k4

    top_dtype%nested%nest2%i3 = top_dtype2%nested%nest2%i3
    top_dtype%nested%nest2%j3 = top_dtype2%nested%nest2%j3
    top_dtype%nested%nest2%k3 = top_dtype2%nested%nest2%k3

    top_dtype%nested%i2 = top_dtype2%nested%i2

    do i = 1, 10
      top_dtype%array_i(i) = top_dtype2%array_i(i)
    end do
!$omp end target

  print *, top_dtype%nested%nest%i4
  print *, top_dtype%nested%nest%j4
  print *, top_dtype%nested%nest%k4

  print *, top_dtype%nested%nest2%i3
  print *, top_dtype%nested%nest2%j3
  print *, top_dtype%nested%nest2%k3

  print *, top_dtype%nested%i2

  print *, top_dtype%array_i
end program main

!CHECK: 10.
!CHECK: 12.
!CHECK: 54.
!CHECK: 20
!CHECK: 40
!CHECK: 60
!CHECK: 200.
!CHECK: 1 2 3 4 5 6 7 8 9 10
