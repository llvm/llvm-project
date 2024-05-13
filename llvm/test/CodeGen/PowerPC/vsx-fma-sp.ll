; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mattr=+vsx | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -mattr=+vsx -fast-isel -O0 | FileCheck -check-prefix=CHECK-FISL %s
; XFAIL: *

define void @test1sp(float %a, float %b, float %c, float %e, ptr nocapture %d) #0 {
entry:
  %0 = tail call float @llvm.fma.f32(float %b, float %c, float %a)
  store float %0, ptr %d, align 4
  %1 = tail call float @llvm.fma.f32(float %b, float %e, float %a)
  %arrayidx1 = getelementptr inbounds float, ptr %d, i64 1
  store float %1, ptr %arrayidx1, align 4
  ret void

; CHECK-LABEL: @test1sp
; CHECK-DAG: li [[C1:[0-9]+]], 4
; CHECK-DAG: xsmaddmsp 3, 2, 1
; CHECK-DAG: xsmaddasp 1, 2, 4
; CHECK-DAG: stxsspx 3, 0, 7
; CHECK-DAG: stxsspx 1, 7, [[C1]]
; CHECK: blr

; CHECK-FISL-LABEL: @test1sp
; CHECK-FISL-DAG: fmr 0, 1
; CHECK-FISL-DAG: xsmaddasp 0, 2, 3
; CHECK-FISL-DAG: stxsspx 0, 0, 7
; CHECK-FISL-DAG: xsmaddasp 1, 2, 4
; CHECK-FISL-DAG: li [[C1:[0-9]+]], 4
; CHECK-FISL-DAG: stxsspx 1, 7, [[C1]]
; CHECK-FISL: blr
}

define void @test2sp(float %a, float %b, float %c, float %e, float %f, ptr nocapture %d) #0 {
entry:
  %0 = tail call float @llvm.fma.f32(float %b, float %c, float %a)
  store float %0, ptr %d, align 4
  %1 = tail call float @llvm.fma.f32(float %b, float %e, float %a)
  %arrayidx1 = getelementptr inbounds float, ptr %d, i64 1
  store float %1, ptr %arrayidx1, align 4
  %2 = tail call float @llvm.fma.f32(float %b, float %f, float %a)
  %arrayidx2 = getelementptr inbounds float, ptr %d, i64 2
  store float %2, ptr %arrayidx2, align 4
  ret void

; CHECK-LABEL: @test2sp
; CHECK-DAG: li [[C1:[0-9]+]], 4
; CHECK-DAG: li [[C2:[0-9]+]], 8
; FIXME: We now miss this because of copy ordering at the MI level.
; CHECX-DAG: xsmaddmsp 3, 2, 1
; CHECX-DAG: xsmaddmsp 4, 2, 1
; CHECX-DAG: xsmaddasp 1, 2, 5
; CHECX-DAG: stxsspx 3, 0, 8
; CHECX-DAG: stxsspx 4, 8, [[C1]]
; CHECX-DAG: stxsspx 1, 8, [[C2]]
; CHECK: blr

; CHECK-FISL-LABEL: @test2sp
; CHECK-FISL-DAG: fmr 0, 1
; CHECK-FISL-DAG: xsmaddasp 0, 2, 3
; CHECK-FISL-DAG: stxsspx 0, 0, 8
; CHECK-FISL-DAG: fmr 0, 1
; CHECK-FISL-DAG: xsmaddasp 0, 2, 4
; CHECK-FISL-DAG: li [[C1:[0-9]+]], 4
; CHECK-FISL-DAG: stxsspx 0, 8, [[C1]]
; CHECK-FISL-DAG: xsmaddasp 1, 2, 5
; CHECK-FISL-DAG: li [[C2:[0-9]+]], 8
; CHECK-FISL-DAG: stxsspx 1, 8, [[C2]]
; CHECK-FISL: blr
}

define void @test3sp(float %a, float %b, float %c, float %e, float %f, ptr nocapture %d) #0 {
entry:
  %0 = tail call float @llvm.fma.f32(float %b, float %c, float %a)
  store float %0, ptr %d, align 4
  %1 = tail call float @llvm.fma.f32(float %b, float %e, float %a)
  %2 = tail call float @llvm.fma.f32(float %b, float %c, float %1)
  %arrayidx1 = getelementptr inbounds float, ptr %d, i64 3
  store float %2, ptr %arrayidx1, align 4
  %3 = tail call float @llvm.fma.f32(float %b, float %f, float %a)
  %arrayidx2 = getelementptr inbounds float, ptr %d, i64 2
  store float %3, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds float, ptr %d, i64 1
  store float %1, ptr %arrayidx3, align 4
  ret void

; CHECK-LABEL: @test3sp
; CHECK-DAG: fmr [[F1:[0-9]+]], 1
; CHECK-DAG: li [[C1:[0-9]+]], 12
; CHECK-DAG: li [[C2:[0-9]+]], 8
; CHECK-DAG: li [[C3:[0-9]+]], 4
; CHECK-DAG: xsmaddmsp 4, 2, 1
; CHECK-DAG: xsmaddasp 1, 2, 5

; Note: We could convert this next FMA to M-type as well, but it would require
; re-ordering the instructions.
; CHECK-DAG: xsmaddasp [[F1]], 2, 3

; CHECK-DAG: xsmaddmsp 3, 2, 4
; CHECK-DAG: stxsspx [[F1]], 0, 8
; CHECK-DAG: stxsspx 3, 8, [[C1]]
; CHECK-DAG: stxsspx 1, 8, [[C2]]
; CHECK-DAG: stxsspx 4, 8, [[C3]]
; CHECK: blr

; CHECK-FISL-LABEL: @test3sp
; CHECK-FISL-DAG: fmr [[F1:[0-9]+]], 1
; CHECK-FISL-DAG: xsmaddasp [[F1]], 2, 4
; CHECK-FISL-DAG: fmr 4, [[F1]]
; CHECK-FISL-DAG: xsmaddasp 4, 2, 3
; CHECK-FISL-DAG: li [[C1:[0-9]+]], 12
; CHECK-FISL-DAG: stxsspx 4, 8, [[C1]]
; CHECK-FISL-DAG: xsmaddasp 1, 2, 5
; CHECK-FISL-DAG: li [[C2:[0-9]+]], 8
; CHECK-FISL-DAG: stxsspx 1, 8, [[C2]]
; CHECK-FISL-DAG: li [[C3:[0-9]+]], 4
; CHECK-FISL-DAG: stxsspx 0, 8, [[C3]]
; CHECK-FISL: blr
}

define void @test4sp(float %a, float %b, float %c, float %e, float %f, ptr nocapture %d) #0 {
entry:
  %0 = tail call float @llvm.fma.f32(float %b, float %c, float %a)
  store float %0, ptr %d, align 4
  %1 = tail call float @llvm.fma.f32(float %b, float %e, float %a)
  %arrayidx1 = getelementptr inbounds float, ptr %d, i64 1
  store float %1, ptr %arrayidx1, align 4
  %2 = tail call float @llvm.fma.f32(float %b, float %c, float %1)
  %arrayidx3 = getelementptr inbounds float, ptr %d, i64 3
  store float %2, ptr %arrayidx3, align 4
  %3 = tail call float @llvm.fma.f32(float %b, float %f, float %a)
  %arrayidx4 = getelementptr inbounds float, ptr %d, i64 2
  store float %3, ptr %arrayidx4, align 4
  ret void

; CHECK-LABEL: @test4sp
; CHECK-DAG: fmr [[F1:[0-9]+]], 1
; CHECK-DAG: li [[C1:[0-9]+]], 4
; CHECK-DAG: li [[C2:[0-9]+]], 8
; CHECK-DAG: xsmaddmsp 4, 2, 1

; Note: We could convert this next FMA to M-type as well, but it would require
; re-ordering the instructions.
; CHECK-DAG: xsmaddasp 1, 2, 5

; CHECK-DAG: xsmaddasp [[F1]], 2, 3
; CHECK-DAG: stxsspx [[F1]], 0, 8
; CHECK-DAG: stxsspx 4, 8, [[C1]]
; CHECK-DAG: li [[C3:[0-9]+]], 12
; CHECK-DAG: xsmaddasp 4, 2, 3
; CHECK-DAG: stxsspx 4, 8, [[C3]]
; CHECK-DAG: stxsspx 1, 8, [[C2]]
; CHECK: blr

; CHECK-FISL-LABEL: @test4sp
; CHECK-FISL-DAG: fmr [[F1:[0-9]+]], 1
; CHECK-FISL-DAG: xsmaddasp [[F1]], 2, 3
; CHECK-FISL-DAG: stxsspx 0, 0, 8
; CHECK-FISL-DAG: fmr [[F1]], 1
; CHECK-FISL-DAG: xsmaddasp [[F1]], 2, 4
; CHECK-FISL-DAG: li [[C3:[0-9]+]], 4
; CHECK-FISL-DAG: stxsspx 0, 8, [[C3]]
; CHECK-FISL-DAG: xsmaddasp 0, 2, 3
; CHECK-FISL-DAG: li [[C1:[0-9]+]], 12
; CHECK-FISL-DAG: stxsspx 0, 8, [[C1]]
; CHECK-FISL-DAG: xsmaddasp 1, 2, 5
; CHECK-FISL-DAG: li [[C2:[0-9]+]], 8
; CHECK-FISL-DAG: stxsspx 1, 8, [[C2]]
; CHECK-FISL: blr
}

declare float @llvm.fma.f32(float, float, float) #0
