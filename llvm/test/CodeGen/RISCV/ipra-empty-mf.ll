; Regression test for a crash when compiling a trivial unused internal
; function with -enable-ipra on RISC-V. Enabling IPRA causes
; `-enable-ipra` (via setRequiresCodeGenSCCOrder) to skip codegen for
; `@empty_internal_func` inside the CallGraphSCCPassManager. The
; MachineOutliner ModulePass then closed that inner FunctionPassManager and
; the post-outliner FPM visited `@empty_internal_func` anyway, created a
; fresh empty MachineFunction for it, and crashed the RISC-V AsmPrinter at
; `MF->front()`.
;
; RUN: llc -mtriple=riscv64 -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=riscv32 -enable-ipra < %s | FileCheck %s

; CHECK-LABEL: empty_internal_func:
; CHECK: ret

define internal void @empty_internal_func() {
  ret void
}
