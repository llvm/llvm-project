! Offloading test checking interaction of an explicit member map allocatable
! components of two large nested derived types. NOTE: Unfortunately this test
! loses a bit of its bite as we do not currently support linking against an
! offload compiled fortran runtime library which means allocatable scalar
! assignment isn't going to work in target regions.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: bottom_layer1
    real(4), allocatable :: i4
    real(4), allocatable :: j4
    integer, pointer :: array_ptr(:)
    real(4), allocatable :: k4
    end type bottom_layer1

    type :: bottom_layer2
      integer(4), allocatable :: i3
      integer(4), allocatable :: j3
      integer, allocatable :: scalar
      real, allocatable :: array_j(:)
      integer(4), allocatable :: k3
    end type bottom_layer2

    type :: middle_layer
     real(4) :: array_i2(10)
     real(4), allocatable :: i2
     integer, pointer :: scalar_ptr
     real(4) :: array_j2(10)
     type(bottom_layer1), allocatable :: nest
     type(bottom_layer2), allocatable :: nest2
    end type middle_layer

    type :: top_layer
    real(4) :: i
    integer(4), allocatable :: array_i(:)
    real(4) :: j
    integer(4) :: k
    type(middle_layer), allocatable :: nested
    end type top_layer

    type(top_layer), allocatable :: top_dtype
    type(top_layer), allocatable :: top_dtype2
    integer, target :: array_target(10)
    integer, target :: array_target2(10)

    allocate(top_dtype)
    allocate(top_dtype2)
    allocate(top_dtype%nested)
    allocate(top_dtype2%nested)
    allocate(top_dtype%nested%nest)
    allocate(top_dtype2%nested%nest)
    allocate(top_dtype%nested%nest2)
    allocate(top_dtype2%nested%nest2)
    allocate(top_dtype%array_i(10))
    allocate(top_dtype2%array_i(10))

    top_dtype%nested%nest%array_ptr => array_target
    allocate(top_dtype%nested%nest2%array_j(10))

    top_dtype2%nested%nest%array_ptr => array_target2
    allocate(top_dtype2%nested%nest2%array_j(10))

!$omp target map(tofrom: top_dtype%array_i, top_dtype%nested%nest2%array_j, top_dtype%nested%nest%array_ptr) &
!$omp map(tofrom: top_dtype2%array_i, top_dtype2%nested%nest2%array_j, top_dtype2%nested%nest%array_ptr)
    do i = 1, 10
      top_dtype%nested%nest%array_ptr(i) = i
      top_dtype%nested%nest2%array_j(i) = i
      top_dtype%array_i(i) = i
      top_dtype2%nested%nest%array_ptr(i) = i
      top_dtype2%nested%nest2%array_j(i) = i
      top_dtype2%array_i(i) = i
    end do
!$omp end target

  print *, top_dtype%nested%nest%array_ptr
  print *, top_dtype%nested%nest2%array_j
  print *, top_dtype%array_i

  print *, top_dtype2%nested%nest%array_ptr
  print *, top_dtype2%nested%nest2%array_j
  print *, top_dtype2%array_i
end program main

!CHECK: 1 2 3 4 5 6 7 8 9 10
!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
!CHECK: 1 2 3 4 5 6 7 8 9 10
!CHECK: 1 2 3 4 5 6 7 8 9 10
!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
!CHECK:  1 2 3 4 5 6 7 8 9 10
