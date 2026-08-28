; Regression test for a crash when compiling a trivial unused internal
; function with -enable-ipra on ARM. ARM enables the MachineOutliner in its
; TargetMachine, which under IPRA triggers the same empty-MF crash as on
; RISC-V and AArch64. See https://github.com/llvm/llvm-project/issues/119556.

; RUN: llc -mtriple=armv7-unknown-linux -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=armv7eb-unknown-linux -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-unknown-linux -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=thumbv8-unknown-linux -enable-ipra < %s | FileCheck %s

; CHECK-LABEL: empty_internal_func:
; CHECK: bx lr

define internal void @empty_internal_func() {
  ret void
}
