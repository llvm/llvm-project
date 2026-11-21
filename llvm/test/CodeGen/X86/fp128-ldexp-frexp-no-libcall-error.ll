; RUN: not llc -mtriple=x86_64-pc-windows-msvc -filetype=null %s 2>&1 | FileCheck %s

; The Windows math runtime has no fp128 ldexpl/frexpl. fp128 is a legal type on
; x86-64 while i128 is not, so the generic software expansion cannot run;
; legalization must diagnose the missing libcall instead of crashing.

; CHECK: error: no libcall available for fldexp
define fp128 @test_ldexp_f128_i32(fp128 %val, i32 %a) {
  %call = call fp128 @llvm.ldexp.f128.i32(fp128 %val, i32 %a)
  ret fp128 %call
}

; CHECK: error: no libcall available for ffrexp
define { fp128, i32 } @test_frexp_f128_i32(fp128 %a) {
  %result = call { fp128, i32 } @llvm.frexp.f128.i32(fp128 %a)
  ret { fp128, i32 } %result
}
