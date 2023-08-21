; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lsx.vst(<16 x i8>, i8*, i32)

define void @lsx_vst(<16 x i8> %va, i8* %p, i32 %b) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  call void @llvm.loongarch.lsx.vst(<16 x i8> %va, i8* %p, i32 %b)
  ret void
}
