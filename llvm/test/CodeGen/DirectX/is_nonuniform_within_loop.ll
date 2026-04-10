; RUN: opt -S  -dxil-intrinsic-expansion -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Regression test for llvm/llvm-project#189438.
; The goal of this test it to make sure compilation finishes successfully.


@Out.str = private unnamed_addr constant [4 x i8] c"Out\00", align 1
; CHECK-LABEL: define void @main
; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 %phi, i1 false) 

define void @main() local_unnamed_addr {
entry:
  %cmp = icmp eq i32 4, 0
  br i1 %cmp, label %_Z4mainj.exit, label %for.body.i

for.body.i:                                       ; preds = %entry, %for.body.i
  %phi = phi i32 [ %inc.loop, %for.body.i ], [ 0, %entry ]
  %binding = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 4, i32 %phi, ptr nonnull @Out.str)
  %inc.counter = tail call noundef i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i32, 1, 0) %binding, i8 1)
  %inc.loop = add nuw nsw i32 %phi, 1
  %exitcond = icmp eq i32 %inc.loop, 4
  br i1 %exitcond, label %_Z4mainj.exit, label %for.body.i

_Z4mainj.exit:                                    ; preds = %for.body.i, %entry
  ret void
}
