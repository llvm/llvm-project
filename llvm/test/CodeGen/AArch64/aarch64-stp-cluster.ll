; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -verify-misched -debug-only=machine-scheduler -aarch64-enable-stp-suppress=false -o - 2>&1 > /dev/null | FileCheck %s

; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: stp_i64_scale:BB#0
; CHECK:Cluster ld/st SU(4) - SU(3)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(4):   STRXui %1, %0, 1
; CHECK:SU(3):   STRXui %1, %0, 2
; CHECK:SU(2):   STRXui %1, %0, 3
; CHECK:SU(5):   STRXui %1, %0, 4
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
; CHECK-LABEL: stp_i32_scale:BB#0
; CHECK:Cluster ld/st SU(4) - SU(3)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(4):   STRWui %1, %0, 1
; CHECK:SU(3):   STRWui %1, %0, 2
; CHECK:SU(2):   STRWui %1, %0, 3
; CHECK:SU(5):   STRWui %1, %0, 4
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
; CHECK-LABEL:stp_i64_unscale:BB#0 entry
; CHECK:Cluster ld/st SU(5) - SU(2)
; CHECK:Cluster ld/st SU(4) - SU(3)
; CHECK:SU(5):   STURXi %1, %0, -32
; CHECK:SU(2):   STURXi %1, %0, -24
; CHECK:SU(4):   STURXi %1, %0, -16
; CHECK:SU(3):   STURXi %1, %0, -8
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
; CHECK-LABEL:stp_i32_unscale:BB#0 entry
; CHECK:Cluster ld/st SU(5) - SU(2)
; CHECK:Cluster ld/st SU(4) - SU(3)
; CHECK:SU(5):   STURWi %1, %0, -16
; CHECK:SU(2):   STURWi %1, %0, -12
; CHECK:SU(4):   STURWi %1, %0, -8
; CHECK:SU(3):   STURWi %1, %0, -4
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
; CHECK-LABEL:stp_double:BB#0
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(3):   STRDui %1, %0, 1
; CHECK:SU(4):   STRDui %1, %0, 2
; CHECK:SU(2):   STRDui %1, %0, 3
; CHECK:SU(5):   STRDui %1, %0, 4
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
; CHECK-LABEL:stp_float:BB#0
; CHECK:Cluster ld/st SU(3) - SU(4)
; CHECK:Cluster ld/st SU(2) - SU(5)
; CHECK:SU(3):   STRSui %1, %0, 1
; CHECK:SU(4):   STRSui %1, %0, 2
; CHECK:SU(2):   STRSui %1, %0, 3
; CHECK:SU(5):   STRSui %1, %0, 4
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
; CHECK-LABEL: stp_volatile:BB#0
; CHECK-NOT: Cluster ld/st
; CHECK:SU(2):   STRXui %1, %0, 3; mem:Volatile
; CHECK:SU(3):   STRXui %1, %0, 2; mem:Volatile
; CHECK:SU(4):   STRXui %1, %0, 1; mem:Volatile
; CHECK:SU(5):   STRXui %1, %0, 4; mem:Volatile
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

