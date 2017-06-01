; Test moves between FPRs and GPRs for z13 and above.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Check that moves from i32s to floats use a low GR32 and vector operation.
define float @f1(i16 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: llh [[REG:%r[0-5]]], 0(%r2)
; CHECK: oilh [[REG]], 16256
; CHECK: vlvgf %v0, [[REG]], 0
; CHECK: br %r14
  %base = load i16, i16 *%ptr
  %ext = zext i16 %base to i32
  %full = or i32 %ext, 1065353216
  %res = bitcast i32 %full to float
  ret float %res
}

; Check that moves from floats to i32s use a low GR32 and vector operation.
define void @f2(float %val, i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vlgvf [[REG:%r[0-5]]], %v0, 0
; CHECK: stc [[REG]], 0(%r2)
; CHECK: br %r14
  %res = bitcast float %val to i32
  %trunc = trunc i32 %res to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Like f2, but with a conditional store.
define void @f3(float %val, i8 *%ptr, i32 %which) {
; CHECK-LABEL: f3:
; CHECK-DAG: ciblh %r3, 0, 0(%r14)
; CHECK-DAG: vlgvf [[REG:%r[0-5]]], %v0, 0
; CHECK: stc [[REG]], 0(%r2)
; CHECK: br %r14
  %int = bitcast float %val to i32
  %trunc = trunc i32 %int to i8
  %old = load i8, i8 *%ptr
  %cmp = icmp eq i32 %which, 0
  %res = select i1 %cmp, i8 %trunc, i8 %old
  store i8 %res, i8 *%ptr
  ret void
}

; ...and again with 16-bit memory.
define void @f4(float %val, i16 *%ptr, i32 %which) {
; CHECK-LABEL: f4:
; CHECK-DAG: ciblh %r3, 0, 0(%r14)
; CHECK-DAG: vlgvf [[REG:%r[0-5]]], %v0, 0
; CHECK: sth [[REG]], 0(%r2)
; CHECK: br %r14
  %int = bitcast float %val to i32
  %trunc = trunc i32 %int to i16
  %old = load i16, i16 *%ptr
  %cmp = icmp eq i32 %which, 0
  %res = select i1 %cmp, i16 %trunc, i16 %old
  store i16 %res, i16 *%ptr
  ret void
}
