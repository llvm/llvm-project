; Companion regression test for the IPRA empty-MachineFunction crash (see
; https://github.com/llvm/llvm-project/issues/119556). X86 targets do not
; enable the Machine Outliner by default (SupportsDefaultOutlining is false),
; so `-enable-ipra` alone does not trigger the crash — it silently drops the
; unreferenced function. Forcing the outliner reproduces the same pipeline
; split as on RISC-V/AArch64/Arm, so these RUN lines pin the fix on x86_64
; and i386.

; RUN: llc -mtriple=x86_64 -enable-ipra -enable-machine-outliner=always < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=x86_64 -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=i386   -enable-ipra -enable-machine-outliner=always < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=i386   -enable-ipra < %s | FileCheck %s

; CHECK-LABEL: empty_internal_func:
; CHECK: ret

define internal void @empty_internal_func() {
  ret void
}
