; RUN: llc < %s -O2 -mtriple=powerpc-linux-musl | FileCheck %s
; RUN: llc < %s -O2 -mtriple=powerpc64-linux-musl | FileCheck %s
; RUN: llc < %s -O2 -mtriple=powerpc64le-linux-musl | FileCheck %s

define fp128 @addkf3(fp128 %a, fp128 %b) {
; CHECK-LABEL: addkf3:
; CHECK: __addkf3
  %1 = fadd fp128 %a, %b
  ret fp128 %1
}

define fp128 @subkf3(fp128 %a, fp128 %b) {
; CHECK-LABEL: subkf3:
; CHECK: __subkf3
  %1 = fsub fp128 %a, %b
  ret fp128 %1
}

define fp128 @mulkf3(fp128 %a, fp128 %b) {
; CHECK-LABEL: mulkf3:
; CHECK: __mulkf3
  %1 = fmul fp128 %a, %b
  ret fp128 %1
}

define fp128 @divkf3(fp128 %a, fp128 %b) {
; CHECK-LABEL: divkf3:
; CHECK: __divkf3
  %1 = fdiv fp128 %a, %b
  ret fp128 %1
}

define fp128 @extendsfkf2(float %a) {
; CHECK-LABEL: extendsfkf2:
; CHECK: __extendsfkf2
  %1 = fpext float %a to fp128
  ret fp128 %1
}

define fp128 @extenddfkf2(double %a) {
; CHECK-LABEL: extenddfkf2:
; CHECK: __extenddfkf2
  %1 = fpext double %a to fp128
  ret fp128 %1
}

define float @trunckfsf2(fp128 %a) {
; CHECK-LABEL: trunckfsf2:
; CHECK: __trunckfsf2
  %1 = fptrunc fp128 %a to float
  ret float %1
}

define double @trunckfdf2(fp128 %a) {
; CHECK-LABEL: trunckfdf2:
; CHECK: __trunckfdf2
  %1 = fptrunc fp128 %a to double
  ret double %1
}

define i32 @fixkfsi(fp128 %a) {
; CHECK-LABEL: fixkfsi:
; CHECK: __fixkfsi
  %1 = fptosi fp128 %a to i32
  ret i32 %1
}

define i64 @fixkfdi(fp128 %a) {
; CHECK-LABEL: fixkfdi:
; CHECK: __fixkfdi
  %1 = fptosi fp128 %a to i64
  ret i64 %1
}

define i32 @fixunskfsi(fp128 %a) {
; CHECK-LABEL: fixunskfsi:
; CHECK: __fixunskfsi
  %1 = fptoui fp128 %a to i32
  ret i32 %1
}

define i64 @fixunskfdi(fp128 %a) {
; CHECK-LABEL: fixunskfdi:
; CHECK: __fixunskfdi
  %1 = fptoui fp128 %a to i64
  ret i64 %1
}

define fp128 @floatsikf(i32 %a) {
; CHECK-LABEL: floatsikf:
; CHECK: __floatsikf
  %1 = sitofp i32 %a to fp128
  ret fp128 %1
}

define fp128 @floatdikf(i64 %a) {
; CHECK-LABEL: floatdikf:
; CHECK: __floatdikf
  %1 = sitofp i64 %a to fp128
  ret fp128 %1
}

define fp128 @floatunsikf(i32 %a) {
; CHECK-LABEL: floatunsikf:
; CHECK: __floatunsikf
  %1 = uitofp i32 %a to fp128
  ret fp128 %1
}

define fp128 @floatundikf(i64 %a) {
; CHECK-LABEL: floatundikf:
; CHECK: __floatundikf
  %1 = uitofp i64 %a to fp128
  ret fp128 %1
}

define i1 @test_eqkf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_eqkf2:
; CHECK: __eqkf2
  %1 = fcmp oeq fp128 %a, %b
  ret i1 %1
}

define i1 @test_nekf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_nekf2:
; CHECK: __nekf2
  %1 = fcmp une fp128 %a, %b
  ret i1 %1
}

define i1 @test_gekf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_gekf2:
; CHECK: __gekf2
  %1 = fcmp oge fp128 %a, %b
  ret i1 %1
}

define i1 @test_ltkf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_ltkf2:
; CHECK: __ltkf2
  %1 = fcmp olt fp128 %a, %b
  ret i1 %1
}

define i1 @test_lekf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_lekf2:
; CHECK: __lekf2
  %1 = fcmp ole fp128 %a, %b
  ret i1 %1
}

define i1 @test_gtkf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_gtkf2:
; CHECK: __gtkf2
  %1 = fcmp ogt fp128 %a, %b
  ret i1 %1
}

define i1 @test_unordkf2(fp128 %a, fp128 %b) {
; CHECK-LABEL: test_unordkf2:
; CHECK: __unordkf2
  %1 = fcmp uno fp128 %a, %b
  ret i1 %1
}
