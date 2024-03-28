; RUN: not opt -S %s -passes=verify -verify-with-context 2>&1 | FileCheck %s

declare i32 @foo(i32)
define i32 @test_callbr_landingpad_not_first_inst() {
entry:
  %0 = callbr i32 asm "", "=r,!i"()
          to label %asm.fallthrough [label %landingpad]

asm.fallthrough:
  ret i32 42

landingpad:
  %foo = call i32 @foo(i32 42)
; CHECK: Verification Error: No other instructions may proceed intrinsic
; CHECK-NEXT: Verification Error: Context [Function 'test_callbr_landingpad_not_first_inst' -> BasicBlock 'landingpad' -> Instruction 'out' (number 2 inside BB)]
; CHECK-NEXT: Context [Function 'test_callbr_landingpad_not_first_inst' -> BasicBlock 'landingpad']
; CHECK-NEXT: %out = call i32 @llvm.callbr.landingpad.i32(i32 %0)
  %out = call i32 @llvm.callbr.landingpad.i32(i32 %0)
  ret i32 %out
}

declare <2 x double> @llvm.masked.load.v2f64.p0(ptr, i32, <2 x i1>, <2 x double>)

define <2 x double> @masked_load(<2 x i1> %mask, ptr %addr, <2 x double> %dst) {
; CHECK: Verification Error: masked_load: alignment must be a power of 2
; CHECK-NEXT: Verification Error: Context [Function 'masked_load' -> BasicBlock '' -> Instruction 'res' (number 1 inside BB)]
; CHECK-NEXT: Context [Function 'masked_load' -> BasicBlock '']
; CHECK-NEXT: %res = call <2 x double> @llvm.masked.load.v2f64.p0(ptr %addr, i32 3, <2 x i1> %mask, <2 x double> %dst)
  %res = call <2 x double> @llvm.masked.load.v2f64.p0(ptr %addr, i32 3, <2 x i1>%mask, <2 x double> %dst)
  ret <2 x double> %res
}
