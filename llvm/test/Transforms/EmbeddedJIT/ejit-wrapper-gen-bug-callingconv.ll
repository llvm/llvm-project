; RUN: opt -passes=ejit-wrapper-gen -S %s | FileCheck %s
;
; XFAIL: *
;
; KNOWN BUG (recorded now, fix tracked separately).
;
; The jit_dispatch block calls the JIT-specialized function pointer with
;     Builder.CreateCall(F->getFunctionType(), JitResult, Args)
; which uses the DEFAULT C calling convention. But ejit_compile_or_get returns
; a function compiled from F's own bitcode, i.e. with F's calling convention.
; When F is e.g. `fastcc`, the indirect dispatch call must also be `fastcc`,
; otherwise the call site and the callee disagree on argument/return register
; and stack usage -> runtime corruption.
;
; The fix should propagate F->getCallingConv() onto the dispatch CallInst.
; Until then the CHECK below (correct behavior) does not match -> XFAIL.

define fastcc i32 @fastcc_entry(i32 %a) !ejit.metadata !0 {
entry:
  ret i32 %a
}

!0 = distinct !{!{!"ejit_entry"}}

; The dispatch call must use the same calling convention as the function.
; CHECK-LABEL: define fastcc i32 @fastcc_entry(i32 %a)
; CHECK: jit_dispatch:
; CHECK: call fastcc i32 {{.*}}(i32 %a)
