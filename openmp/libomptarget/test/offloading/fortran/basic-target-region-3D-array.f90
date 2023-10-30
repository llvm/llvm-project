! Basic offloading test of a regular array explicitly
! passed within a target region
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    integer :: x(2,2,2)
    integer :: i = 1, j = 1, k = 1
    integer :: counter = 1
    do i = 1, 2
        do j = 1, 2
          do k = 1, 2
            x(i, j, k) = 0
          end do
        end do
    end do

!$omp target map(tofrom:x, i, j, k, counter)
    do i = 1, 2
        do j = 1, 2
          do k = 1, 2
            x(i, j, k) = counter
            counter = counter + 1
          end do
        end do
    end do
!$omp end target

     do i = 1, 2
        do j = 1, 2
          do k = 1, 2
            print *, x(i, j, k)
          end do
        end do
    end do
end program main
  
! CHECK: 1 2 3 4 5 6 7 8
