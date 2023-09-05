; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lsx.vst(<16 x i8>, i8*, i32)

define void @lsx_vst_lo(<16 x i8> %va, i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vst: argument out of range
entry:
  call void @llvm.loongarch.lsx.vst(<16 x i8> %va, i8* %p, i32 -2049)
  ret void
}

define void @lsx_vst_hi(<16 x i8> %va, i8* %p) nounwind {
; CHECK: llvm.loongarch.lsx.vst: argument out of range
entry:
  call void @llvm.loongarch.lsx.vst(<16 x i8> %va, i8* %p, i32 2048)
  ret void
}
