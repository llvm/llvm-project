! Offloading test for generic target regions containing different kinds of
! loop constructs inside.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  integer :: i1, n1, counter

  n1 = 100

  counter = 0
  !$omp target parallel do reduction(+:counter)
  do i1=1, n1
    counter = counter + 1
  end do

  ! CHECK: 1 100
  print '(I2" "I0)', 1, counter

  counter = 0
  !$omp target map(tofrom:counter)
    !$omp parallel do reduction(+:counter)
    do i1=1, n1
      counter = counter + 1
    end do
  !$omp end target

  ! CHECK: 2 100
  print '(I2" "I0)', 2, counter

  counter = 0
  !$omp target teams distribute parallel do reduction(+:counter)
  do i1=1, n1
    counter = counter + 1
  end do

  ! CHECK: 3 100
  print '(I2" "I0)', 3, counter
end program
