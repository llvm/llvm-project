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

; An inner phi that only forwards the outer loop-carried phi caches the outer
; phi's placeholder indices. When those placeholders fold away, the cached
; entry must follow them so the later load from %q sees the final indices.
; CHECK-LABEL: define i32 @nested_pointer_phi(
; CHECK: head:
; CHECK-NEXT: %[[IDX:.*]] = phi i32 [ 0, %l ], [ 1, %r ], [ %[[IDX]], %latch ]
; CHECK: latch:
; CHECK-NEXT: %[[HANDLE:.*]] = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.RawBuffer_i32_1_0t(i32 2, i32 0, i32 2, i32 %[[IDX]], ptr @Buffer.str)
; CHECK-NEXT: call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0) %[[HANDLE]], i32 %index, i32 0)
define i32 @nested_pointer_phi(i1 %c, i1 %d, i32 %index, i32 %n) {
entry:
  %h0 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 2, i32 0, ptr @Buffer.str)
  %b0 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", i32, 1, 0) %h0, i32 %index)
  %h1 = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 2, i32 0, i32 2, i32 1, ptr @Buffer.str)
  %b1 = call ptr @llvm.dx.resource.getpointer(target("dx.RawBuffer", i32, 1, 0) %h1, i32 %index)
  br i1 %c, label %l, label %r

l:
  br label %head

r:
  br label %head

head:
  %p = phi ptr [ %b0, %l ], [ %b1, %r ], [ %q, %latch ]
  %v0 = load i32, ptr %p, align 4
  br i1 %d, label %a, label %b

a:
  br label %latch

b:
  br label %latch

latch:
  %q = phi ptr [ %p, %a ], [ %p, %b ]
  %v1 = load i32, ptr %q, align 4
  %s = add i32 %v0, %v1
  %done = icmp eq i32 %s, %n
  br i1 %done, label %exit, label %head

exit:
  ret i32 %s
}
