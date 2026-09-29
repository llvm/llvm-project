; RUN: opt -S  -dxil-intrinsic-expansion -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s


@Out.str = private unnamed_addr constant [4 x i8] c"Out\00", align 1

; CHECK-LABEL: define void @main
; CHECK: call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 %phi, i1 true) 
define void @main() local_unnamed_addr {
entry:
  br label %while.body.i

while.body.i:
  %phi = phi i32 [ 0, %entry ], [ %nuri, %while.body.i ]
  %binding = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 4, i32 %phi, ptr nonnull @Out.str)
  %update.counter = tail call noundef i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i32, 1, 0) %binding, i8 1)
  %inc = add nuw nsw i32 %phi, 1
  %nuri = tail call noundef i32 @llvm.dx.resource.nonuniformindex(i32 %inc)
  %cmp = icmp ult i32 %nuri, 4
  br i1 %cmp, label %while.body.i, label %exit

exit:
  ret void
}
