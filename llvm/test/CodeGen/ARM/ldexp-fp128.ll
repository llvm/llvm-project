; RUN: not llc -mtriple=armv7-unknown-linux -filetype=null %s 2>&1 | FileCheck %s

; ARM long double is IEEE double, so there is no fp128 ldexpl libcall. The
; softening path diagnoses this rather than crashing.
; CHECK: error: do not know how to soften fpowi to fpow

; This is an improperly typed call, long double is not fp128.
define fp128 @testExpl(fp128 %val, i32 %a) {
  %call = tail call fp128 @ldexpl(fp128 %val, i32 %a)
  ret fp128 %call
}

declare fp128 @ldexpl(fp128, i32) memory(none)

define fp128 @test_ldexp_f128_i32(fp128 %val, i32 %a) {
  %call = tail call fp128 @llvm.ldexp.f128.i32(fp128 %val, i32 %a)
  ret fp128 %call
}

define <2 x fp128> @test_ldexp_v2f128_v2i32(<2 x fp128> %val, <2 x i32> %a) {
  %call = tail call <2 x fp128> @llvm.ldexp.v2f128.v2i32(<2 x fp128> %val, <2 x i32> %a)
  ret <2 x fp128> %call
}
