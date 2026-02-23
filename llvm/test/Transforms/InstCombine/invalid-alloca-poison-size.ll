; REQUIRES: asserts && x86_64-linux
; RUN: not --crash opt -passes="function(sroa,instcombine<no-verify-fixpoint>,gvn,simplifycfg,infer-address-spaces)" -S %s 2>&1 | FileCheck %s
; CHECK: Assertion `(isa<AllocaInst>(Arg) || isa<PoisonValue>(Arg)) && "Expected alloca instruction or poison value."' failed.

; InstCombineLoadStoreAlloca replaces alloca uses with null when the
; alloca size is undef (poison is undef as per spec). This causes invalid IR:
;
;   call void @llvm.lifetime.start(ptr %i_)
;   becomes:
;   call void @llvm.lifetime.start(ptr null)
;
; According to the lifetime intrinsic spec, the argument must be either:
;   - A pointer to an alloca instruction, or
;   - A poison value.
;
; Since null is neither an alloca pointer nor poison, this transformation
; produces invalid IR.

target triple = "x86_64-unknown-linux-gnu"

define noundef i32 @foo() {
entry:
  %i_ = alloca i32, i32 poison, align 4
  call void @llvm.lifetime.start(ptr %i_)
  store i32 0, ptr %i_, align 4
  ret i32 0
}

declare void @llvm.lifetime.start(ptr)

