; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-32
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-64
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-64

; Function Attrs: nounwind readnone
declare i8* @llvm.thread.pointer() #1

define i8* @thread_pointer() {
; CHECK-32-LABEL: @thread_pointer
; CHECK-32: mr 3, 2
; CHECK-32: blr
; CHECK-64-LABEL: @thread_pointer
; CHECK-64: mr 3, 13
; CHECK-64: blr
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
