! This test checks a number of more complex derived type member mapping
! syntaxes utilising a non-allocatable parent derived type.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type dtype2
    integer int
    real float
    real float_elements(10)
    end type dtype2

    type dtype1
    character (LEN=30) characters
    character (LEN=1) character
    type(dtype2) number
    end type dtype1

    type nonallocatabledtype
    integer elements(20)
    type(dtype1) num_chars
    integer value
    type(dtype2) internal_dtypes(5)
    end type nonallocatabledtype

    type (nonallocatabledtype) array_dtype(5)

  !$omp target map(tofrom: array_dtype(5))
      do i = 1, 20
        array_dtype(5)%elements(i) = 20 + i
      end do

      array_dtype(5)%num_chars%number%float_elements(5) = 10
      array_dtype(5)%value = 40
  !$omp end target

    print *, array_dtype(5)%elements
    print *, array_dtype(5)%num_chars%number%float_elements(5)
    print *, array_dtype(5)%value

  !$omp target map(tofrom: array_dtype(4)%elements(3))
    array_dtype(4)%elements(3) = 74
  !$omp end target

   print *, array_dtype(4)%elements(3)

  !$omp target map(tofrom: array_dtype(5)%elements(3:5))
    do i = 3, 5
       array_dtype(5)%elements(i) = i + 1
    end do
  !$omp end target

   do i = 3, 5
      print *, array_dtype(5)%elements(i)
   end do

  !$omp target map(tofrom: array_dtype(3:5))
    do i = 3, 5
      array_dtype(i)%value = i + 2
    end do
  !$omp end target

    do i = 3, 5
        print *, array_dtype(i)%value
    end do

  !$omp target map(tofrom: array_dtype(4)%num_chars%number%float_elements(8))
    array_dtype(4)%num_chars%number%float_elements(8) = 250
  !$omp end target

  print *, array_dtype(4)%num_chars%number%float_elements(8)

  !$omp target map(tofrom: array_dtype(4)%num_chars%number%float_elements(5:10))
    do i = 5, 10
      array_dtype(4)%num_chars%number%float_elements(i) = i + 3
    end do
  !$omp end target

  do i = 5, 10
    print *, array_dtype(4)%num_chars%number%float_elements(i)
  end do

  !$omp target map(tofrom: array_dtype(4)%internal_dtypes(3)%float_elements(4))
    array_dtype(4)%internal_dtypes(3)%float_elements(4) = 200
  !$omp end target

  print *, array_dtype(4)%internal_dtypes(3)%float_elements(4)

end program main

! CHECK: 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
! CHECK: 10.
! CHECK: 40
! CHECK: 74
! CHECK: 4
! CHECK: 5
! CHECK: 6
! CHECK: 5
! CHECK: 6
! CHECK: 7
! CHECK: 250.
! CHECK: 8.
! CHECK: 9.
! CHECK: 10.
! CHECK: 11.
! CHECK: 12.
! CHECK: 13.
! CHECK: 200
