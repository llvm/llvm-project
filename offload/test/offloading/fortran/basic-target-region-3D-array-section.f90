! Basic offloading test of a regular array explicitly passed within a target
! region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    integer :: inArray(3,3,3)
    integer :: outArray(3,3,3)
    integer :: i, j, k
    integer :: j2 = 3, k2 = 3

    do i = 1, 3
      do j = 1, 3
        do k = 1, 3
            inArray(i, j, k) = 42
            outArray(i, j, k) = 0
        end do
       end do
    end do

j = 1
k = 1
!$omp target map(tofrom:inArray(1:3, 1:3, 2:2), outArray(1:3, 1:3, 1:3), j, k, j2, k2)
    do while (j <= j2)
      k = 1
      do while (k <= k2)
        outArray(k, j, 2) = inArray(k, j, 2)
        k = k + 1
      end do
      j = j + 1
    end do
!$omp end target

 print *, outArray

end program

! CHECK:  0 0 0 0 0 0 0 0 0 42 42 42 42 42 42 42 42 42 0 0 0 0 0 0 0 0 0
