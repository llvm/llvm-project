; RUN: llc -mtriple=arm64-apple-ios -mcpu=cyclone < %s | FileCheck %s
; rdar://11481771
; rdar://13713797

define void @t1() nounwind ssp {
entry:
; CHECK-LABEL: t1:
; CHECK-NOT: fmov
; CHECK: movi.2d v0, #0000000000000000
; CHECK: movi.2d v1, #0000000000000000
; CHECK: movi.2d v2, #0000000000000000
; CHECK: movi.2d v3, #0000000000000000
  tail call void @bar(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00) nounwind
  ret void
}

define void @t2() nounwind ssp {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: mov w0, wzr
; CHECK: mov w0, #0
; CHECK: mov w1, #0
  tail call void @bari(i32 0, i32 0) nounwind
  ret void
}

define void @t3() nounwind ssp {
entry:
; CHECK-LABEL: t3:
; CHECK-NOT: mov x0, xzr
; CHECK: mov x0, #0
; CHECK: mov x1, #0
  tail call void @barl(i64 0, i64 0) nounwind
  ret void
}

define void @t4() nounwind ssp {
; CHECK-LABEL: t4:
; CHECK-NOT: fmov
; CHECK: movi.2d v0, #0000000000000000
; CHECK: movi.2d v1, #0000000000000000
  tail call void @barf(float 0.000000e+00, float 0.000000e+00) nounwind
  ret void
}

declare void @bar(double, double, double, double)
declare void @bari(i32, i32)
declare void @barl(i64, i64)
declare void @barf(float, float)
