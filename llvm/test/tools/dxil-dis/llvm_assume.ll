; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

define void @test_llvm_assume(i1 %0)  {
; CHECK-LABEL: test_llvm_assume
; CHECK-NEXT: tail call void @llvm.assume(i1 %0)
tail call void @llvm.assume(i1 %0)
ret void
}

