! Offloading test with runtine call to ompx_dump_mapping_tables Fortran array
! writing some values and printing the variable mapped to device correctly
! receives the updates made on the device.
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-unknown-linux-gnu
! UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program map_dump_example
  INTERFACE
    SUBROUTINE ompx_dump_mapping_tables() BIND(C)
    END SUBROUTINE ompx_dump_mapping_tables
  END INTERFACE

  integer i,j,k,N
  integer async_q(4)
  real :: A(5000000)
  N=5000000
  do i=1, N
    A(i)=0
  enddo
! clang-format off
! CHECK: omptarget device 0 info: OpenMP Host-Device pointer mappings after block
! CHECK-NEXT: omptarget device 0 info: Host Ptr Target Ptr Size (B) DynRefCount HoldRefCount Declaration
! CHECK-NEXT: omptarget device 0 info: {{(0x[0-9a-f]{16})}} {{(0x[0-9a-f]{16})}}  20000000 1 0 {{.*}} at a(:n):21:11
! clang-format on
!$omp target enter data map(to:A(:N))
  call ompx_dump_mapping_tables()
!$omp target parallel do
  do i=1, N
    A(i)=A(i)*2
  enddo
!$omp target exit data map(from:A)
end program
