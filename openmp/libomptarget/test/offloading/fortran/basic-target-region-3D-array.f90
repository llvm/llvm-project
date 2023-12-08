! Basic offloading test of a regular array explicitly
! passed within a target region
! REQUIRES: flang, amdgcn-amd-amdhsa, nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer :: x(2,2,2)
    integer :: i, j, k
    integer :: i2 = 2, j2 = 2, k2 = 2
    integer :: counter = 1
    do i = 1, 2
        do j = 1, 2
          do k = 1, 2
            x(i, j, k) = 0
          end do
        end do
    end do

i = 1
j = 1
k = 1

!$omp target map(tofrom:x, counter) map(to: i, j, k, i2, j2, k2)
    do while (i <= i2)
      j = 1
        do while (j <= j2)
          k = 1
          do while (k <= k2)
            x(i, j, k) = counter
            counter = counter + 1
            k = k + 1
          end do
          j = j + 1
        end do
        i = i + 1
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

! CHECK: 1
! CHECK: 2
! CHECK: 3
! CHECK: 4
! CHECK: 5
! CHECK: 6
! CHECK: 7
! CHECK: 8
