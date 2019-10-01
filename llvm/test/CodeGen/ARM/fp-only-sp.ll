; RUN: llc -mtriple=thumbv7em-apple-macho -mcpu=cortex-m4 %s -o - -O0 | FileCheck %s
; RUN: llc -mtriple=thumbv7em-apple-macho -mcpu=cortex-m4 %s -o - | FileCheck %s

; Note: vldr and vstr really do have 64-bit variants even with -fp64
define void @test_load_store(double* %addr) {
; CHECK-LABEL: test_load_store:
; CHECK: vldr [[TMP:d[0-9]+]], [r0]
; CHECK: vstr [[TMP]], [r0]
  %val = load volatile double, double* %addr
  store volatile double %val, double* %addr
  ret void
}

define void @test_cmp(double %l, double %r, i1* %addr.dst) {
; CHECK-LABEL: test_cmp:
; CHECK: bl ___eqdf2
  %res = fcmp oeq double %l, %r
  store i1 %res, i1* %addr.dst
  ret void
}

define void @test_ext(float %in, double* %addr) {
; CHECK-LABEL: test_ext:
; CHECK: bl ___extendsfdf2
  %res = fpext float %in to double
  store double %res, double* %addr
  ret void
}

define void @test_trunc(double %in, float* %addr) {
; CHECK-LABEL: test_trunc:
; CHECK: bl ___truncdfsf2
  %res = fptrunc double %in to float
  store float %res, float* %addr
  ret void
}

define void @test_itofp(i32 %in, double* %addr) {
; CHECK-LABEL: test_itofp:
; CHECK: bl ___floatsidf
  %res = sitofp i32 %in to double
  store double %res, double* %addr
;  %res = fptoui double %tmp to i32
  ret void
}

define i32 @test_fptoi(double* %addr) {
; CHECK-LABEL: test_fptoi:
; CHECK: bl ___fixunsdfsi
  %val = load double, double* %addr
  %res = fptoui double %val to i32
  ret i32 %res
}

define void @test_binop(double* %addr) {
; CHECK-LABEL: test_binop:
; CHECK: bl ___adddf3
  %in = load double, double* %addr
  %res = fadd double %in, %in
  store double %res, double* %addr
  ret void
}
