! Offloading test with a target region and chunks
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic

program main
  use omp_lib
        integer :: A(100)
!$omp target map(from:A)
!$omp parallel do schedule(static,2) num_threads(10)
        do index_ = 1, 100
          A(index_) = omp_get_team_num() * 1000 + omp_get_thread_num()
        end do
!$omp end target
        write(*,"(A)"), "omp target parallel for thread chunk size 2"
        call printArray(A)

end program main

subroutine printArray(Array)
        integer :: Array(*)
        do i = 1, 100
            write(*, "(A, I0, A, I0, A)", advance="no") "B",Array(i)/1000,"T",modulo(Array(i),1000)," "
        end do
        write(*,'(/)')
end subroutine printArray

!CHECK:      omp target parallel for thread chunk size 2

!CHECK-NEXT: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
!CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
!CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
!CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
!CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
!CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
!CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
!CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9
!CHECK-SAME: B0T0 B0T0 B0T1 B0T1 B0T2 B0T2 B0T3 B0T3 B0T4 B0T4
!CHECK-SAME: B0T5 B0T5 B0T6 B0T6 B0T7 B0T7 B0T8 B0T8 B0T9 B0T9

