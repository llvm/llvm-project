; RUN: llc -march=mipsel -mcpu=mips32 < %s | FileCheck %s

declare void @llvm.trap()

define void @f1() {
entry:
  call void @llvm.trap()
  unreachable

; CHECK:        break
}
