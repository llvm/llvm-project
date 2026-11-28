; Regression test for a crash when compiling a trivial unused internal
; function with -enable-ipra on AArch64. Enabling IPRA causes
; `-enable-ipra` (via setRequiresCodeGenSCCOrder) to skip codegen for
; `@empty_internal_func` inside the CallGraphSCCPassManager. The
; MachineOutliner ModulePass (enabled by default on AArch64) then closed
; that inner FunctionPassManager and the post-outliner FPM visited
; `@empty_internal_func` anyway, created a fresh empty MachineFunction for
; it, and crashed in `Branch relaxation pass`. See
; https://github.com/llvm/llvm-project/issues/119556.

; RUN: llc -mtriple=aarch64    -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=aarch64_be -enable-ipra < %s | FileCheck %s

; CHECK-LABEL: empty_internal_func:
; CHECK: ret

define internal void @empty_internal_func() {
  ret void
}
