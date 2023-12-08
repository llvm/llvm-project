; Test moves between FPRs and GPRs for z196 and zEC12.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check that moves from i32s to floats can use high registers.
define float @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: llhh [[REG:%r[0-5]]], 0(%r2)
; CHECK: oihh [[REG]], 16256
; CHECK: ldgr %f0, [[REG]]
; CHECK: br %r14
  %base = load i16, ptr %ptr
  %ext = zext i16 %base to i32
  %full = or i32 %ext, 1065353216
  %res = bitcast i32 %full to float
  ret float %res
}

; Check that moves from floats to i32s can use high registers.
; This "store the low byte" technique is used by llvmpipe, for example.
define void @f2(float %val, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: lgdr [[REG:%r[0-5]]], %f0
; CHECK: stch [[REG]], 0(%r2)
; CHECK: br %r14
  %res = bitcast float %val to i32
  %trunc = trunc i32 %res to i8
  store i8 %trunc, ptr %ptr
  ret void
}

; Like f2, but with a conditional store.
define void @f3(float %val, ptr %ptr, i32 %which) {
; CHECK-LABEL: f3:
; CHECK: ciblh %r3, 0, 0(%r14)

; CHECK: lgdr [[REG:%r[0-5]]], %f0
; CHECK: stch [[REG]], 0(%r2)
; CHECK: br %r14
  %int = bitcast float %val to i32
  %trunc = trunc i32 %int to i8
  %old = load i8, ptr %ptr
  %cmp = icmp eq i32 %which, 0
  %res = select i1 %cmp, i8 %trunc, i8 %old
  store i8 %res, ptr %ptr
  ret void
}

; ...and again with 16-bit memory.
define void @f4(float %val, ptr %ptr, i32 %which) {
; CHECK-LABEL: f4:
; CHECK: ciblh %r3, 0, 0(%r14)
; CHECK: lgdr [[REG:%r[0-5]]], %f0
; CHECK: sthh [[REG]], 0(%r2)
; CHECK: br %r14
  %int = bitcast float %val to i32
  %trunc = trunc i32 %int to i16
  %old = load i16, ptr %ptr
  %cmp = icmp eq i32 %which, 0
  %res = select i1 %cmp, i16 %trunc, i16 %old
  store i16 %res, ptr %ptr
  ret void
}
