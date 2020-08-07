; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -verify-misched -debug-only=machine-scheduler -aarch64-enable-stp-suppress=false -o - 2>&1 > /dev/null | FileCheck %s

; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: stp_i64_scale:%bb.0
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(4):   STRXui %1:gpr64, %0:gpr64common, 1
; CHECK:SU(3):   STRXui %1:gpr64, %0:gpr64common, 2
; CHECK:SU(2):   STRXui %1:gpr64, %0:gpr64common, 3
; CHECK:SU(5):   STRXui %1:gpr64, %0:gpr64common, 4
define i64 @stp_i64_scale(i64* nocapture %P, i64 %v) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %P, i64 3
  store i64 %v, i64* %arrayidx
  %arrayidx1 = getelementptr inbounds i64, i64* %P, i64 2
  store i64 %v, i64* %arrayidx1
  %arrayidx2 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx2
  %arrayidx3 = getelementptr inbounds i64, i64* %P, i64 4
  store i64 %v, i64* %arrayidx3
  ret i64 %v
}

; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: stp_i32_scale:%bb.0
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(4):   STRWui %1:gpr32, %0:gpr64common, 1
; CHECK:SU(3):   STRWui %1:gpr32, %0:gpr64common, 2
; CHECK:SU(2):   STRWui %1:gpr32, %0:gpr64common, 3
; CHECK:SU(5):   STRWui %1:gpr32, %0:gpr64common, 4
define i32 @stp_i32_scale(i32* nocapture %P, i32 %v) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %P, i32 3
  store i32 %v, i32* %arrayidx
  %arrayidx1 = getelementptr inbounds i32, i32* %P, i32 2
  store i32 %v, i32* %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, i32* %P, i32 1
  store i32 %v, i32* %arrayidx2
  %arrayidx3 = getelementptr inbounds i32, i32* %P, i32 4
  store i32 %v, i32* %arrayidx3
  ret i32 %v
}

; CHECK:********** MI Scheduling **********
; CHECK-LABEL:stp_i64_unscale:%bb.0 entry
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:SU(2):   STURXi %1:gpr64, %0:gpr64common, -24
; CHECK:SU(3):   STURXi %1:gpr64, %0:gpr64common, -8
; CHECK:SU(4):   STURXi %1:gpr64, %0:gpr64common, -16
; CHECK:SU(5):   STURXi %1:gpr64, %0:gpr64common, -32
define void @stp_i64_unscale(i64* nocapture %P, i64 %v) #0 {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %P, i64 -3
  store i64 %v, i64* %arrayidx
  %arrayidx1 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx1
  %arrayidx2 = getelementptr inbounds i64, i64* %P, i64 -2
  store i64 %v, i64* %arrayidx2
  %arrayidx3 = getelementptr inbounds i64, i64* %P, i64 -4
  store i64 %v, i64* %arrayidx3
  ret void
}

; CHECK:********** MI Scheduling **********
; CHECK-LABEL:stp_i32_unscale:%bb.0 entry
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:SU(2):   STURWi %1:gpr32, %0:gpr64common, -12
; CHECK:SU(3):   STURWi %1:gpr32, %0:gpr64common, -4
; CHECK:SU(4):   STURWi %1:gpr32, %0:gpr64common, -8
; CHECK:SU(5):   STURWi %1:gpr32, %0:gpr64common, -16
define void @stp_i32_unscale(i32* nocapture %P, i32 %v) #0 {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %P, i32 -3
  store i32 %v, i32* %arrayidx
  %arrayidx1 = getelementptr inbounds i32, i32* %P, i32 -1
  store i32 %v, i32* %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, i32* %P, i32 -2
  store i32 %v, i32* %arrayidx2
  %arrayidx3 = getelementptr inbounds i32, i32* %P, i32 -4
  store i32 %v, i32* %arrayidx3
  ret void
}

; CHECK:********** MI Scheduling **********
; CHECK-LABEL:stp_double:%bb.0
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(3):   STRDui %1:fpr64, %0:gpr64common, 1
; CHECK:SU(4):   STRDui %1:fpr64, %0:gpr64common, 2
; CHECK:SU(2):   STRDui %1:fpr64, %0:gpr64common, 3
; CHECK:SU(5):   STRDui %1:fpr64, %0:gpr64common, 4
define void @stp_double(double* nocapture %P, double %v)  {
entry:
  %arrayidx = getelementptr inbounds double, double* %P, i64 3
  store double %v, double* %arrayidx
  %arrayidx1 = getelementptr inbounds double, double* %P, i64 1
  store double %v, double* %arrayidx1
  %arrayidx2 = getelementptr inbounds double, double* %P, i64 2
  store double %v, double* %arrayidx2
  %arrayidx3 = getelementptr inbounds double, double* %P, i64 4
  store double %v, double* %arrayidx3
  ret void
}

; CHECK:********** MI Scheduling **********
; CHECK-LABEL:stp_float:%bb.0
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(3):   STRSui %1:fpr32, %0:gpr64common, 1
; CHECK:SU(4):   STRSui %1:fpr32, %0:gpr64common, 2
; CHECK:SU(2):   STRSui %1:fpr32, %0:gpr64common, 3
; CHECK:SU(5):   STRSui %1:fpr32, %0:gpr64common, 4
define void @stp_float(float* nocapture %P, float %v)  {
entry:
  %arrayidx = getelementptr inbounds float, float* %P, i64 3
  store float %v, float* %arrayidx
  %arrayidx1 = getelementptr inbounds float, float* %P, i64 1
  store float %v, float* %arrayidx1
  %arrayidx2 = getelementptr inbounds float, float* %P, i64 2
  store float %v, float* %arrayidx2
  %arrayidx3 = getelementptr inbounds float, float* %P, i64 4
  store float %v, float* %arrayidx3
  ret void
}

; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: stp_volatile:%bb.0
; CHECK-NOT: Cluster ld/st
; CHECK:SU(2):   STRXui %1:gpr64, %0:gpr64common, 3 :: (volatile
; CHECK:SU(3):   STRXui %1:gpr64, %0:gpr64common, 2 :: (volatile
; CHECK:SU(4):   STRXui %1:gpr64, %0:gpr64common, 1 :: (volatile
; CHECK:SU(5):   STRXui %1:gpr64, %0:gpr64common, 4 :: (volatile
define i64 @stp_volatile(i64* nocapture %P, i64 %v) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %P, i64 3
  store volatile i64 %v, i64* %arrayidx
  %arrayidx1 = getelementptr inbounds i64, i64* %P, i64 2
  store volatile i64 %v, i64* %arrayidx1
  %arrayidx2 = getelementptr inbounds i64, i64* %P, i64 1
  store volatile i64 %v, i64* %arrayidx2
  %arrayidx3 = getelementptr inbounds i64, i64* %P, i64 4
  store volatile i64 %v, i64* %arrayidx3
  ret i64 %v
}

; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: stp_i64_with_ld:%bb.0
; CHECK:Cluster ld/st SU(5) - SU(10)
; CHECK:Cluster ld/st SU(15) - SU(20)
; CHECK:SU(5):   STRXui %7:gpr64, %0:gpr64common, 0 ::
; CHECK:SU(10):   STRXui %12:gpr64, %0:gpr64common, 1 ::
; CHECK:SU(15):   STRXui %17:gpr64, %0:gpr64common, 2 ::
; CHECK:SU(20):   STRXui %22:gpr64, %0:gpr64common, 3 ::
define void @stp_i64_with_ld(i64* noalias nocapture %a, i64* noalias nocapture readnone %b, i64* noalias nocapture readnone %c) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 8
  %0 = load i64, i64* %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds i64, i64* %a, i64 16
  %1 = load i64, i64* %arrayidx3, align 8
  %mul = mul nsw i64 %1, %0
  %2 = load i64, i64* %a, align 8
  %add6 = add nsw i64 %2, %mul
  store i64 %add6, i64* %a, align 8
  %arrayidx.1 = getelementptr inbounds i64, i64* %a, i64 9
  %3 = load i64, i64* %arrayidx.1, align 8
  %arrayidx3.1 = getelementptr inbounds i64, i64* %a, i64 17
  %4 = load i64, i64* %arrayidx3.1, align 8
  %mul.1 = mul nsw i64 %4, %3
  %arrayidx5.1 = getelementptr inbounds i64, i64* %a, i64 1
  %5 = load i64, i64* %arrayidx5.1, align 8
  %add6.1 = add nsw i64 %5, %mul.1
  store i64 %add6.1, i64* %arrayidx5.1, align 8
  %arrayidx.2 = getelementptr inbounds i64, i64* %a, i64 10
  %6 = load i64, i64* %arrayidx.2, align 8
  %arrayidx3.2 = getelementptr inbounds i64, i64* %a, i64 18
  %7 = load i64, i64* %arrayidx3.2, align 8
  %mul.2 = mul nsw i64 %7, %6
  %arrayidx5.2 = getelementptr inbounds i64, i64* %a, i64 2
  %8 = load i64, i64* %arrayidx5.2, align 8
  %add6.2 = add nsw i64 %8, %mul.2
  store i64 %add6.2, i64* %arrayidx5.2, align 8
  %arrayidx.3 = getelementptr inbounds i64, i64* %a, i64 11
  %9 = load i64, i64* %arrayidx.3, align 8
  %arrayidx3.3 = getelementptr inbounds i64, i64* %a, i64 19
  %10 = load i64, i64* %arrayidx3.3, align 8
  %mul.3 = mul nsw i64 %10, %9
  %arrayidx5.3 = getelementptr inbounds i64, i64* %a, i64 3
  %11 = load i64, i64* %arrayidx5.3, align 8
  %add6.3 = add nsw i64 %11, %mul.3
  store i64 %add6.3, i64* %arrayidx5.3, align 8
  ret void
}

; Verify that the SU(2) and SU(4) are the preds of SU(3)
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: stp_missing_preds_edges:%bb.0
; CHECK:Cluster ld/st SU(3) - SU(5)
; CHECK: Copy Pred SU(4)
; CHECK: Copy Pred SU(2)
; CHECK:SU(2):   %0:gpr64common = COPY $x0
; CHECK:SU(3):   STRWui %1:gpr32, %0:gpr64common, 0
; CHECK:SU(4):   %3:gpr32common = nsw ADDWri %2:gpr32common, 5, 0
; CHECK:SU(5):   STRWui %3:gpr32common, %0:gpr64common, 1
define void @stp_missing_preds_edges(i32* %p, i32 %m, i32 %n) {
entry:
  store i32 %m, i32* %p, align 4
  %add = add nsw i32 %n, 5
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i64 1
  store i32 %add, i32* %arrayidx1, align 4
  ret void
}
