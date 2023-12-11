! Basic offloading test of arrays with provided lower 
! and upper bounds as specified by OpenMP's sectioning
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    integer :: write_arr(10) =  (/0,0,0,0,0,0,0,0,0,0/)
    integer :: read_arr(10) = (/1,2,3,4,5,6,7,8,9,10/)
    integer :: i = 2
    integer :: j = 5
    !$omp target map(to:read_arr(2:5)) map(from:write_arr(2:5)) map(to:i,j)
        do while (i <= j)
            write_arr(i) = read_arr(i)
            i = i + 1
        end do
    !$omp end target
    
    print *, write_arr(:)
end program

! CHECK: 0 2 3 4 5 0 0 0 0 0
