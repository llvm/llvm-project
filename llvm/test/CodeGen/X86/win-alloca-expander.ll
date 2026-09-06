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

  %saved_stack = tail call ptr @llvm.stacksave()

  %p1 = alloca %struct.S
  %p2 = alloca %struct.T
; MergeAllocas coalesces these two textually-adjacent allocas into a single
; 4024-byte allocation before instruction selection. Combined with the 1024
; bytes %p0 already used, that exceeds the stack probe size, so the merged
; allocation needs its own touch-and-sub -- still fewer total instructions
; than touching %p1 and %p2 separately would have been.
; CHECK: pushl %eax
; CHECK: subl $4020, %esp

  call void @f(ptr %p0)
; CHECK: calll

  %p3 = alloca %struct.T
  %p4 = alloca %struct.U
  %p5 = alloca %struct.T
; MergeAllocas coalesces these three textually-adjacent allocas into a
; single 16000-byte allocation, which is large enough to require stack
; probing on its own.
; CHECK: movl $16000, %eax
; CHECK: calll __chkstk

  call void @llvm.stackrestore(ptr %saved_stack)
  %p6 = alloca %struct.S
; The stack restore means we lose track of the stack pointer and must probe.
; CHECK: pushl %eax
; CHECK: subl $1020, %esp

; Use the pointers so they're not optimized away.
  call void @f(ptr %p1)
  call void @g(ptr %p2)
  call void @g(ptr %p3)
  call void @h(ptr %p4)
  call void @g(ptr %p5)
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
  call void @f(ptr %p1)
  call void @f(ptr %p2)
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
  call void @f(ptr %p0)
  ret void
}

define void @cfg(i1 %x, i1 %y) {
; Test that the blocks are analyzed in the correct order.
; CHECK-LABEL: cfg:
entry:
  br i1 %x, label %bb1, label %bb3

bb1:
  %p1 = alloca %struct.S
; CHECK: pushl %eax
; CHECK: subl $1020, %esp
  br label %bb4

bb2:
  %p5 = alloca %struct.T
; CHECK: pushl %eax
; CHECK: subl $2996, %esp
  call void @g(ptr %p5)
  ret void

bb3:
  %p2 = alloca %struct.T
; CHECK: pushl %eax
; CHECK: subl $2996, %esp
  br label %bb4

bb4:
  br i1 %y, label %bb5, label %bb2

bb5:
  %p4 = alloca %struct.S
; CHECK: subl $1024, %esp
  call void @f(ptr %p4)
  ret void

}


declare void @f(ptr)
declare void @g(ptr)
declare void @h(ptr)

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)
