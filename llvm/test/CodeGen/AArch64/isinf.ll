; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon,+fullfp16 < %s -o -| FileCheck %s

declare half   @llvm.fabs.f16(half)
declare float  @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare fp128  @llvm.fabs.f128(fp128)

; INFINITY requires loading the constant for _Float16
define i32 @replace_isinf_call_f16(half %x) {
; CHECK-LABEL: replace_isinf_call_f16:
; CHECK:       adrp    [[ADDR:x[0-9]+]], [[CSTLABEL:.LCP.*]]
; CHECK:       ldr     [[INFINITY:h[0-9]+]], {{[[]}}[[ADDR]], :lo12:[[CSTLABEL]]{{[]]}}
; CHECK-NEXT:  fabs    [[ABS:h[0-9]+]], h0
; CHECK-NEXT:  fcmp    [[ABS]], [[INFINITY]]
; CHECK-NEXT:  cset    w0, eq
  %abs = tail call half @llvm.fabs.f16(half %x)
  %cmpinf = fcmp oeq half %abs, 0xH7C00
  %ret = zext i1 %cmpinf to i32
  ret i32 %ret
}

; Check if INFINITY for float is materialized
define i32 @replace_isinf_call_f32(float %x) {
; CHECK-LABEL: replace_isinf_call_f32:
; CHECK:       mov    [[INFSCALARREG:w[0-9]+]], #2139095040
; CHECK-NEXT:  fabs   [[ABS:s[0-9]+]], s0
; CHECK-NEXT:  fmov   [[INFREG:s[0-9]+]], [[INFSCALARREG]]
; CHECK-NEXT:  fcmp   [[ABS]], [[INFREG]]
; CHECK-NEXT:  cset   w0, eq
  %abs = tail call float @llvm.fabs.f32(float %x)
  %cmpinf = fcmp oeq float %abs, 0x7FF0000000000000
  %ret = zext i1 %cmpinf to i32
  ret i32 %ret
}

; Check if INFINITY for double is materialized
define i32 @replace_isinf_call_f64(double %x) {
; CHECK-LABEL: replace_isinf_call_f64:
; CHECK:       mov    [[INFSCALARREG:x[0-9]+]], #9218868437227405312
; CHECK-NEXT:  fabs   [[ABS:d[0-9]+]], d0
; CHECK-NEXT:  fmov   [[INFREG:d[0-9]+]], [[INFSCALARREG]]
; CHECK-NEXT:  fcmp   [[ABS]], [[INFREG]]
; CHECK-NEXT:  cset   w0, eq
  %abs = tail call double @llvm.fabs.f64(double %x)
  %cmpinf = fcmp oeq double %abs, 0x7FF0000000000000
  %ret = zext i1 %cmpinf to i32
  ret i32 %ret
}

; For long double it still requires loading the constant.
define i32 @replace_isinf_call_f128(fp128 %x) {
; CHECK-LABEL: replace_isinf_call_f128:
; CHECK:       adrp    [[ADDR:x[0-9]+]], [[CSTLABEL:.LCP.*]]
; CHECK:       ldr     q1, {{[[]}}[[ADDR]], :lo12:[[CSTLABEL]]{{[]]}}
; CHECK:       bl      __eqtf2
; CHECK:       cmp     w0, #0
; CHECK:       cset    w0, eq
  %abs = tail call fp128 @llvm.fabs.f128(fp128 %x)
  %cmpinf = fcmp oeq fp128 %abs, 0xL00000000000000007FFF000000000000
  %ret = zext i1 %cmpinf to i32
  ret i32 %ret
}
