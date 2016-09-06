; RUN: llc -march=mips < %s | FileCheck %s
; RUN: llc -march=mips64 < %s | FileCheck %s
; RUN: llc -march=mipsel < %s | FileCheck %s
; RUN: llc -march=mips64el < %s | FileCheck %s

declare i8* @llvm.thread.pointer() nounwind readnone

define i8* @thread_pointer() {
; CHECK: rdhwr $3, $29
  %1 = tail call i8* @llvm.thread.pointer()
  ret i8* %1
}
