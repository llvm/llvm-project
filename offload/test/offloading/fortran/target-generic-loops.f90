! Offloading test for generic target regions containing different kinds of
! loop constructs inside.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  integer :: i1, i2, n1, n2, counter

  n1 = 100
  n2 = 50

  counter = 0
  !$omp target map(tofrom:counter)
    !$omp teams distribute reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
  !$omp end target

  ! CHECK: 1 100
  print '(I2" "I0)', 1, counter

  counter = 0
  !$omp target map(tofrom:counter)
    !$omp parallel do reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
    !$omp parallel do reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
  !$omp end target

  ! CHECK: 2 200
  print '(I2" "I0)', 2, counter

  counter = 0
  !$omp target map(tofrom:counter)
    counter = counter + 1
    !$omp parallel do reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
    counter = counter + 1
    !$omp parallel do reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
    counter = counter + 1
  !$omp end target

  ! CHECK: 3 203
  print '(I2" "I0)', 3, counter

  counter = 0
  !$omp target map(tofrom: counter)
    counter = counter + 1
    !$omp parallel do reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
    counter = counter + 1
  !$omp end target

  ! CHECK: 4 102
  print '(I2" "I0)', 4, counter


  counter = 0
  !$omp target teams distribute reduction(+:counter)
  do i1=1, n1
    !$omp parallel do reduction(+:counter)
    do i2=1, n2
      counter = counter + 1
    end do
  end do

  ! CHECK: 5 5000
  print '(I2" "I0)', 5, counter

  counter = 0
  !$omp target teams distribute reduction(+:counter)
  do i1=1, n1
    counter = counter + 1
    !$omp parallel do reduction(+:counter)
    do i2=1, n2
      counter = counter + 1
    end do
    counter = counter + 1
  end do

  ! CHECK: 6 5200
  print '(I2" "I0)', 6, counter

  counter = 0
  !$omp target teams distribute reduction(+:counter)
  do i1=1, n1
    !$omp parallel do reduction(+:counter)
    do i2=1, n2
      counter = counter + 1
    end do
    !$omp parallel do reduction(+:counter)
    do i2=1, n2
      counter = counter + 1
    end do
  end do

  ! CHECK: 7 10000
  print '(I2" "I0)', 7, counter

  counter = 0
  !$omp target teams distribute reduction(+:counter)
  do i1=1, n1
    counter = counter + 1
    !$omp parallel do reduction(+:counter)
    do i2=1, n2
      counter = counter + 1
    end do
    counter = counter + 1
    !$omp parallel do reduction(+:counter)
    do i2=1, n2
      counter = counter + 1
    end do
    counter = counter + 1
  end do

  ! CHECK: 8 10300
  print '(I2" "I0)', 8, counter
end program
