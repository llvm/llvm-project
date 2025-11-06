; RUN: opt -S -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define void @test_llvm_assume(i1 %0)  {
; CHECK-LABEL: test_llvm_assume
; CHECK-NEXT: ret void
tail call void @llvm.assume(i1 %0)
ret void
}

