; RUN: llc < %s -mtriple=aarch64 -mattr=+neon | FileCheck %s
; RUN: llc < %s -global-isel -global-isel-abort=2 -pass-remarks-missed=gisel* -mtriple=aarch64 | FileCheck %s --check-prefixes=FALLBACK,CHECK

; CHECK-LABEL: testmsws:
; CHECK:       frintx  [[REG:s[0-9]]], s0
; CHECK-NEXT:  fcvtzs  x0, [[REG]]
; CHECK:       ret
; FALLBACK-NOT: remark{{.*}}testmsws
define i32 @testmsws(float %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f32(float %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxs:
; CHECK:       frintx  [[REG:s[0-9]]], s0
; CHECK-NEXT:  fcvtzs  x0, [[REG]]
; CHECK-NEXT:  ret
; FALLBACK-NOT: remark{{.*}}testmsxs
define i64 @testmsxs(float %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f32(float %x)
  ret i64 %0
}

; CHECK-LABEL: testmswd:
; CHECK:       frintx  [[REG:d[0-9]]], d0
; CHECK-NEXT:  fcvtzs  x0, [[REG]]
; CHECK:       ret
; FALLBACK-NOT: remark{{.*}}testmswd
define i32 @testmswd(double %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f64(double %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxd:
; CHECK:       frintx  [[REG:d[0-9]]], d0
; CHECK-NEXT:  fcvtzs  x0, [[REG]]
; CHECK-NEXT:  ret
; FALLBACK-NOT: remark{{.*}}testmsxd
define i64 @testmsxd(double %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f64(double %x)
  ret i64 %0
}

; CHECK-LABEL: testmswl:
; CHECK:       bl      lrintl
define dso_local i32 @testmswl(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f128(fp128 %x)
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsll:
; CHECK:       b       lrintl
define dso_local i64 @testmsll(fp128 %x) {
entry:
  %0 = tail call i64 @llvm.lrint.i64.f128(fp128 %x)
  ret i64 %0
}

declare i64 @llvm.lrint.i64.f32(float) nounwind readnone
declare i64 @llvm.lrint.i64.f64(double) nounwind readnone
declare i64 @llvm.lrint.i64.f128(fp128) nounwind readnone
