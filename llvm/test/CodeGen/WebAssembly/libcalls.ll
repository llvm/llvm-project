; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test a subset of compiler-rt/libm libcalls expected to be emitted by the wasm backend

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare fp128 @llvm.sqrt.f128(fp128)
declare fp128 @llvm.floor.f128(fp128)
declare fp128 @llvm.trunc.f128(fp128)
declare fp128 @llvm.nearbyint.f128(fp128)
declare fp128 @llvm.pow.f128(fp128, fp128)

declare double @llvm.cos.f64(double)
declare double @llvm.log10.f64(double)
declare double @llvm.pow.f64(double, double)
declare double @llvm.log.f64(double)
declare double @llvm.exp.f64(double)
declare i32 @llvm.lround(double)



; CHECK-LABEL: fp128libcalls:
define fp128 @fp128libcalls(fp128 %x, fp128 %y) {
  ; compiler-rt call
  ; CHECK: call __addtf3
  %a = fadd fp128 %x, %y
  ; CHECK: call __multf3
  %b = fmul fp128 %a, %y
  ; CHECK: call __divtf3
  %c = fdiv fp128 %b, %y
  ; libm calls
  ; CHECK: call sqrtl
  %d = call fp128 @llvm.sqrt.f128(fp128 %c)
  ; CHECK: call floorl
  %e = call fp128 @llvm.floor.f128(fp128 %d)
  ; CHECK: call powl
  %f = call fp128 @llvm.pow.f128(fp128 %e, fp128 %y)
  ; CHECK: call truncl
  %g = call fp128 @llvm.trunc.f128(fp128 %f)
  ; CHECK: call nearbyintl
  %h = call fp128 @llvm.nearbyint.f128(fp128 %g)
  ret fp128 %h
}

; CHECK-LABEL: i128libcalls:
define i128 @i128libcalls(i128 %x, i128 %y) {
  ; Basic ops should be expanded
  ; CHECK_NOT: call
  %a = add i128 %x, %y
  ; CHECK: call __multi3
  %b = mul i128 %a, %y
  ; CHECK: call __umodti3
  %c = urem i128 %b, %y
  ret i128 %c
}

; CHECK-LABEL: f64libcalls:
define i32 @f64libcalls(double %x, double %y) {
 ; CHECK: f64.call $push{{[0-9]}}=, cos
 %a = call double @llvm.cos.f64(double %x)
 ; CHECK: f64.call $push{{[0-9]}}=, log10
 %b = call double @llvm.log10.f64(double %a)
 ; CHECK: f64.call $push{{[0-9]}}=, pow
 %c = call double @llvm.pow.f64(double %b, double %y)
 ; CHECK: f64.call $push{{[0-9]}}=, log
 %d = call double @llvm.log.f64(double %c)
 ; CHECK: f64.call $push{{[0-9]}}=, exp
 %e = call double @llvm.exp.f64(double %d)
 ; CHECK: i32.call $push{{[0-9]}}=, lround
 %f = call i32 @llvm.lround(double %e)
 ret i32 %f
}

; fcmp ord and unord (RTLIB::O_F32 / RTLIB::UO_F32 etc) are a special case (see
; comment in WebAssemblyRunimeLibcallSignatures.cpp) so check them separately.
; no libcalls are needed for f32 and f64

; CHECK-LABEL: unordd:
define i1 @unordd(double %x, double %y) {
 ; CHECK-NOT: call
 ; CHECK: f64.ne
 %a = fcmp uno double %x, %y
 ; CHECK-NOT: call
 ; CHECK: f64.eq
 %b = fcmp ord double %x, %y
 ; CHECK: i32.xor
 %c = xor i1 %a, %b
 ret i1 %c
}

; CHECK-LABEL: unordf:
define i1 @unordf(float %x, float %y) {
 ; CHECK-NOT: call
 ; CHECK: f32.ne
 %a = fcmp uno float %x, %y
 ; CHECK-NOT: call
 ; CHECK: f32.eq
 %b = fcmp ord float %x, %y
 ; CHECK: i32.xor
 %c = xor i1 %a, %b
 ret i1 %c
}

; CHECK-LABEL: unordt:
define i1 @unordt(fp128 %x, fp128 %y) {
 ; CHECK: i32.call $push[[CALL:[0-9]]]=, __unordtf2
 ; CHECK-NEXT: i32.const $push[[ZERO:[0-9]+]]=, 0
 ; CHECK-NEXT: i32.ne $push{{[0-9]}}=, $pop[[CALL]], $pop[[ZERO]]
 %a = fcmp uno fp128 %x, %y
 ret i1 %a
}

; CHECK-LABEL: ordt:
define i1 @ordt(fp128 %x, fp128 %y) {
 ; CHECK: i32.call $push[[CALL:[0-9]]]=, __unordtf2
 ; CHECK-NEXT: i32.eqz $push{{[0-9]}}=, $pop[[CALL]]
 %a = fcmp ord fp128 %x, %y
 ret i1 %a
}
