; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -fast-isel=true -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -global-isel=true -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

; Checkpoints produce no machine instructions in any instruction selector.
; CHECK-LABEL: checkpoint:
; CHECK:       # %bb.{{[0-9]+}}:
; CHECK-NEXT:    retq
define void @checkpoint() {
  call void @llvm.observable.checkpoint()
  ret void
}

; CHECK-LABEL: store_before_checkpoint:
; CHECK:       # %bb.{{[0-9]+}}:
; CHECK-NEXT:    movl $42, (%rdi)
; CHECK-NEXT:    retq
define void @store_before_checkpoint(ptr %p) {
  store i32 42, ptr %p
  call void @llvm.observable.checkpoint()
  ret void
}

declare void @llvm.observable.checkpoint()
