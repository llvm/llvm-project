; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s
; RUN: llc < %s -mtriple=i686-pc-win32 -O0

%struct.S = type { [1024 x i8] }
%struct.T = type { [3000 x i8] }
%struct.U = type { [10000 x i8] }

define void @basics() {
; CHECK-LABEL: basics:
entry:
  br label %bb1

; Allocation move sizes should have been removed.
; CHECK-NOT: movl $1024
; CHECK-NOT: movl $3000

bb1:
  %p0 = alloca %struct.S
; The allocation is small enough not to require stack probing, but the %esp
; offset after the prologue is not known, so the stack must be touched before
; the pointer is adjusted.
; CHECK: pushl %eax
; CHECK: subl $1020, %esp

  %saved_stack = tail call i8* @llvm.stacksave()

  %p1 = alloca %struct.S
; We know the %esp offset from above, so there is no need to touch the stack
; before adjusting it.
; CHECK: subl $1024, %esp

  %p2 = alloca %struct.T
; The offset is now 2048 bytes, so allocating a T must touch the stack again.
; CHECK: pushl %eax
; CHECK: subl $2996, %esp

  call void @f(%struct.S* %p0)
; CHECK: calll

  %p3 = alloca %struct.T
; The call above touched the stack, so there is room for a T object.
; CHECK: subl $3000, %esp

  %p4 = alloca %struct.U
; The U object is large enough to require stack probing.
; CHECK: movl $10000, %eax
; CHECK: calll __chkstk

  %p5 = alloca %struct.T
; The stack probing above touched the tip of the stack, so there's room for a T.
; CHECK: subl $3000, %esp

  call void @llvm.stackrestore(i8* %saved_stack)
  %p6 = alloca %struct.S
; The stack restore means we lose track of the stack pointer and must probe.
; CHECK: pushl %eax
; CHECK: subl $1020, %esp

; Use the pointers so they're not optimized away.
  call void @f(%struct.S* %p1)
  call void @g(%struct.T* %p2)
  call void @g(%struct.T* %p3)
  call void @h(%struct.U* %p4)
  call void @g(%struct.T* %p5)
  ret void
}

define void @loop() {
; CHECK-LABEL: loop:
entry:
  br label %bb1

bb1:
  %p1 = alloca %struct.S
; The entry offset is unknown; touch-and-sub.
; CHECK: pushl %eax
; CHECK: subl $1020, %esp
  br label %loop1

loop1:
  %i1 = phi i32 [ 10, %bb1 ], [ %dec1, %loop1 ]
  %p2 = alloca %struct.S
; We know the incoming offset from bb1, but from the back-edge, we assume the
; worst, and therefore touch-and-sub to allocate.
; CHECK: pushl %eax
; CHECK: subl $1020, %esp
  %dec1 = sub i32 %i1, 1
  %cmp1 = icmp sgt i32 %i1, 0
  br i1 %cmp1, label %loop1, label %end
; CHECK: decl
; CHECK: jg

end:
  call void @f(%struct.S* %p1)
  call void @f(%struct.S* %p2)
  ret void
}

define void @probe_size_attribute() "stack-probe-size"="512" {
; CHECK-LABEL: probe_size_attribute:
entry:
  br label %bb1

bb1:
  %p0 = alloca %struct.S
; The allocation would be small enough not to require probing, if it wasn't
; for the stack-probe-size attribute.
; CHECK: movl $1024, %eax
; CHECK: calll __chkstk
  call void @f(%struct.S* %p0)
  ret void
}

define void @cfg(i1 %x, i1 %y) {
; Test that the blocks are analyzed in the correct order.
; CHECK-LABEL: cfg:
entry:
  br i1 %x, label %bb1, label %bb2

bb1:
  %p1 = alloca %struct.S
; CHECK: pushl %eax
; CHECK: subl $1020, %esp
  br label %bb3
bb2:
  %p2 = alloca %struct.T
; CHECK: pushl %eax
; CHECK: subl $2996, %esp
  br label %bb3

bb3:
  br i1 %y, label %bb4, label %bb5

bb4:
  %p4 = alloca %struct.S
; CHECK: subl $1024, %esp
  call void @f(%struct.S* %p4)
  ret void

bb5:
  %p5 = alloca %struct.T
; CHECK: pushl %eax
; CHECK: subl $2996, %esp
  call void @g(%struct.T* %p5)
  ret void
}


declare void @f(%struct.S*)
declare void @g(%struct.T*)
declare void @h(%struct.U*)

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
