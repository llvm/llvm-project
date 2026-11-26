; RUN: llc -mtriple=riscv64 -stop-after=codegenprepare -o %t %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64"

define <vscale x 2 x ptr> @kernel(ptr %a) {
entry:
  %0 = call <vscale x 2 x i64> @llvm.stepvector.nxv2i64()
  %1 = call i64 @llvm.vscale.i64()
  %2 = shl i64 %1, 1
  br label %vector.body

vector.body:
  %vec.ind = phi <vscale x 2 x i64> [ %0, %entry ], [ %vec.ind.next, %vector.body ]
  %broadcast.splatinsert = insertelement <vscale x 2 x i64> zeroinitializer, i64 %2, i64 0
  %broadcast.splat = shufflevector <vscale x 2 x i64> %broadcast.splatinsert, <vscale x 2 x i64> zeroinitializer, <vscale x 2 x i32> zeroinitializer
  %vec.ind.next = add <vscale x 2 x i64> %vec.ind, %broadcast.splat
  br i1 false, label %middle.block, label %vector.body

middle.block:
  %wide.gep = getelementptr [8 x i8], ptr %a, <vscale x 2 x i64> %vec.ind
  ret <vscale x 2 x ptr> %wide.gep
}
