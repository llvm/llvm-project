; REQUIRES: x86-registered-target
; Test that calls to aliases are instrumented, and the assembly references the
; aliased function.
;
; RUN: opt -passes=ctx-instr-gen,assign-guid,ctx-instr-lower -profile-context-root=an_entrypoint \
; RUN:   -profile-context-root=another_entrypoint_no_callees \
; RUN:   -S %s -o %t.ll
; RUN: llc < %t.ll | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

@foo_alias = weak_odr unnamed_addr alias void (), ptr @foo

define void @foo(i32) {
  ret void
}

define void @call_alias(ptr %a) {
entry:
  call void @foo(i32 0, ptr %a)
  ret void
}

; CHECK-LABEL:   call_alias:
; CHECK:         movq    foo@GOTPCREL(%rip), [[REG:%r[a-z0-9]+]]
; CHECK-NEXT:    movq    [[REG]], %fs:__llvm_ctx_profile_expected_callee@TPOFF{{.*}}
