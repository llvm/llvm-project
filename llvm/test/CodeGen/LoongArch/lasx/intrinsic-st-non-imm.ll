; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lasx.xvst(<32 x i8>, i8*, i32)

define void @lasx_xvst(<32 x i8> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lasx.xvst(<32 x i8> %va, i8* %p, i32 %b)
  ret void
}
