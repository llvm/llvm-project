; RUN: opt -S -dxil-resource-type -dxil-resource-access \
; RUN:   -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Regression test for a loop-carried resource pointer. Walking backwards from
; the load encounters the self-referential incoming value of %ptr.

@Buffer.str = internal unnamed_addr constant [7 x i8] c"Buffer\00"

; CHECK-LABEL: define i32 @loop_carried_pointer_phi(
; CHECK: call { i32, i1 } @llvm.dx.resource.load.rawbuffer
define i32 @loop_carried_pointer_phi(i1 %select, i32 %index, i32 %count) {
entry:
  %handle0 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 2, i32 0, ptr @Buffer.str)
  %base0 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", i32, 1, 0) %handle0, i32 %index)
  %handle1 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 2, i32 1, ptr @Buffer.str)
  %base1 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", i32, 1, 0) %handle1, i32 %index)
  br i1 %select, label %left, label %right

left:
  br label %loop

right:
  br label %loop

loop:
  %ptr = phi ptr [ %base0, %left ], [ %base1, %right ], [ %ptr, %loop ]
  %i = phi i32 [ 0, %left ], [ 0, %right ], [ %inc, %loop ]
  %value = load i32, ptr %ptr, align 4
  %inc = add nuw i32 %i, 1
  %done = icmp eq i32 %inc, %count
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %value
}
