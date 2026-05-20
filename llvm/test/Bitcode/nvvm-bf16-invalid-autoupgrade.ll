; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: intrinsic has incorrect return type!
declare half @llvm.nvvm.neg.bf16(bfloat)

define void @test() {
  %t = call half @llvm.nvvm.neg.bf16(bfloat 1.0)
  ret void
}

; CHECK: intrinsic has incorrect argument type!
declare bfloat @llvm.nvvm.fmax.bf16(bfloat, half)

define void @wrong_argument(bfloat %x, half %y) {
  %t = call bfloat @llvm.nvvm.fmax.bf16(bfloat %x, half %y)
  ret void
}

; CHECK: intrinsic has incorrect number of args. Expected 2, but got 1
declare bfloat @llvm.nvvm.fmin.bf16(bfloat)

define void @wrong_num_arguments(bfloat %x) {
  %t = call bfloat @llvm.nvvm.fmin.bf16(bfloat %x)
  ret void
}
