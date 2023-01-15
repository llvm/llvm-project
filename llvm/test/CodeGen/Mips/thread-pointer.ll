; RUN: llc -march=mips < %s | FileCheck %s
; RUN: llc -march=mips64 < %s | FileCheck %s
; RUN: llc -march=mipsel < %s | FileCheck %s
; RUN: llc -march=mips64el < %s | FileCheck %s

declare ptr @llvm.thread.pointer() nounwind readnone

define ptr @thread_pointer() {
; CHECK: rdhwr $3, $29
  %1 = tail call ptr @llvm.thread.pointer()
  ret ptr %1
}
