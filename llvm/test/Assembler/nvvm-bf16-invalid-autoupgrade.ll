; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: intrinsic has incorrect return type!
declare half @llvm.nvvm.neg.bf16(bfloat)

define void @test() {
  %t = call half @llvm.nvvm.neg.bf16(bfloat 1.0)
  ret void
}
