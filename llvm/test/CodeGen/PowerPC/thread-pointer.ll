; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-32
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-64
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-64

; Function Attrs: nounwind readnone
declare ptr @llvm.thread.pointer() #1

define ptr @thread_pointer() {
; CHECK-32-LABEL: @thread_pointer
; CHECK-32: mr 3, 2
; CHECK-32: blr
; CHECK-64-LABEL: @thread_pointer
; CHECK-64: mr 3, 13
; CHECK-64: blr
  %1 = tail call ptr @llvm.thread.pointer()
  ret ptr %1
}
