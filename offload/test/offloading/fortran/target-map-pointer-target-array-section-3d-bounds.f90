! Offloading test checking interaction of pointer and target with target where
! 3-D bounds have been specified
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer, pointer :: inArray(:,:,:)
    integer, pointer :: outArray(:,:,:)
    integer, target :: in(3,3,3)
    integer, target :: out(3,3,3)

    inArray => in
    outArray => out

    do i = 1, 3
      do j = 1, 3
        do k = 1, 3
            inArray(i, j, k) = 42
            outArray(i, j, k) = 0
        end do
       end do
    end do

!$omp target map(tofrom:inArray(1:3, 1:3, 2:2), outArray(1:3, 1:3, 1:3))
    do j = 1, 3
      do k = 1, 3
        outArray(k, j, 2) = inArray(k, j, 2)
      end do
    end do
!$omp end target

 print *, outArray

end program

! CHECK: 0 0 0 0 0 0 0 0 0 42 42 42 42 42 42 42 42 42 0 0 0 0 0 0 0 0 0
