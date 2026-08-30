; REQUIRES: x86-registered-target
;
; RUN: llc -mtriple=x86_64-linux-gnu -O0 -global-isel \
; RUN:   -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s

declare target("llvm.test.tokenlike", i64) @llvm.ssa.copy.tllvm.test.tokenlike_i64t(target("llvm.test.tokenlike", i64) returned)
declare target("llvm.test.tokenlike") @llvm.ssa.copy.tllvm.test.tokenliket(target("llvm.test.tokenlike") returned)

define void @sized_token_like() {
  ; CHECK-LABEL: name: sized_token_like
  ; CHECK: [[POISON:%[0-9]+]]:_(s64) = G_IMPLICIT_DEF
  ; CHECK-NEXT: %{{[0-9]+}}:_(s64) = G_INTRINSIC intrinsic(@llvm.ssa.copy), [[POISON]](s64)
  %handle = call target("llvm.test.tokenlike", i64) @llvm.ssa.copy.tllvm.test.tokenlike_i64t(target("llvm.test.tokenlike", i64) poison)
  ret void
}

define void @unsized_token_like() {
  ; CHECK-LABEL: name: unsized_token_like
  ; CHECK: [[POISON:%[0-9]+]]:_(s0) = G_IMPLICIT_DEF
  ; CHECK-NEXT: %{{[0-9]+}}:_(s0) = G_INTRINSIC intrinsic(@llvm.ssa.copy), [[POISON]](s0)
  %mark = call target("llvm.test.tokenlike") @llvm.ssa.copy.tllvm.test.tokenliket(target("llvm.test.tokenlike") poison)
  ret void
}
