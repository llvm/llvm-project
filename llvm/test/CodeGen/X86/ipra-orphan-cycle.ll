; Companion regression test for the cyclic-orphan case of the IPRA empty-MF
; bug (see https://github.com/llvm/llvm-project/issues/119556). X86 does not
; schedule a mid-codegen ModulePass by default under IPRA, so -enable-ipra
; alone does not trigger the empty-MF crash; forcing the MachineOutliner
; reproduces the same pipeline split as on RISC-V/AArch64/Arm and exercises
; the cyclic-orphan path.

; RUN: llc -mtriple=x86_64 -enable-ipra -enable-machine-outliner=always < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=x86_64 -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=i386   -enable-ipra -enable-machine-outliner=always < %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=i386   -enable-ipra < %s | FileCheck %s

; CHECK-DAG: foo:
; CHECK-DAG: bar:
; CHECK: ret

define internal void @foo() {
  call void @bar()
  ret void
}

define internal void @bar() {
  call void @foo()
  ret void
}
