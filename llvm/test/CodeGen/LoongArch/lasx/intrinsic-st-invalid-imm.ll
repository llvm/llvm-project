; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare void @llvm.loongarch.lasx.xvst(<32 x i8>, i8*, i32)

define void @lasx_xvst_lo(<32 x i8> %va, i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvst: argument out of range
entry:
  call void @llvm.loongarch.lasx.xvst(<32 x i8> %va, i8* %p, i32 -2049)
  ret void
}

define void @lasx_xvst_hi(<32 x i8> %va, i8* %p) nounwind {
; CHECK: llvm.loongarch.lasx.xvst: argument out of range
entry:
  call void @llvm.loongarch.lasx.xvst(<32 x i8> %va, i8* %p, i32 2048)
  ret void
}
