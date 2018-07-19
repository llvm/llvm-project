; RUN: opt -basicaa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
;
; Currently, MemorySSA doesn't support invariant groups. So, we should ignore
; invariant.group.barrier intrinsics entirely. We'll need to pay attention to
; them when/if we decide to support invariant groups.

@g = external global i32

define i32 @foo(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

  %1 = bitcast i32* %a to i8*
; CHECK: MemoryUse(2)
; CHECK-NEXT: %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; This have to be MemoryUse(2), because we can't skip the barrier based on
; invariant.group.
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0
  ret i32 %2
}

define i32 @skipBarrier(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

  %1 = bitcast i32* %a to i8*
; CHECK: MemoryUse(1)
; CHECK-NEXT: %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(1)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0
  ret i32 %2
}

define i32 @skipBarrier2(i32* %a) {

; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = load i32
  %v = load i32, i32* %a, align 4, !invariant.group !0

  %1 = bitcast i32* %a to i8*
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v2 = load i32
  %v2 = load i32, i32* %a32, align 4, !invariant.group !0
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

; FIXME: based on invariant.group it should be MemoryUse(liveOnEntry)
; CHECK: MemoryUse(1)
; CHECK-NEXT: %v3 = load i32
  %v3 = load i32, i32* %a32, align 4, !invariant.group !0
  %add = add nsw i32 %v2, %v3
  %add2 = add nsw i32 %add, %v
  ret i32 %add2
}

define i32 @handleInvariantGroups(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4
  %1 = bitcast i32* %a to i8*
; CHECK: MemoryUse(2)
; CHECK-NEXT: %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a8 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: store i32 2
  store i32 2, i32* @g, align 4

; FIXME: This can be changed to MemoryUse(2)
; CHECK: MemoryUse(3)
; CHECK-NEXT: %3 = load i32
  %3 = load i32, i32* %a32, align 4, !invariant.group !0
  %add = add nsw i32 %2, %3
  ret i32 %add
}

define i32 @loop(i1 %a) {
entry:
  %0 = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, i32* %0, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(i32* %0)
  br i1 %a, label %Loop.Body, label %Loop.End

Loop.Body:
; FIXME: MemoryUse(1)
; CHECK: MemoryUse(2)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* %0, !invariant.group !0
  br i1 %a, label %Loop.End, label %Loop.Body

Loop.End:
; FIXME: MemoryUse(1)
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load
  %2 = load i32, i32* %0, align 4, !invariant.group !0
  br i1 %a, label %Ret, label %Loop.Body

Ret:
  ret i32 %2
}

define i8 @loop2(i8* %p) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, i8* %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(i8* %p)

; CHECK: MemoryUse(2)
; CHECK-NEXT: %after = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p)
  %after = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p)
  br i1 undef, label %Loop.Body, label %Loop.End

Loop.Body:
; 4 = MemoryPhi({entry,2},{Loop.Body,3},{Loop.End,5})
; CHECK: MemoryUse(4)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, i8* %after, !invariant.group !0

; FIXME: MemoryUse(1)
; CHECK: MemoryUse(4)
; CHECK-NEXT: %1 = load i8
  %1 = load i8, i8* %p, !invariant.group !0

; CHECK: 3 = MemoryDef(4)
  store i8 4, i8* %after, !invariant.group !0

  br i1 undef, label %Loop.End, label %Loop.Body

Loop.End:
; 5 = MemoryPhi({entry,2},{Loop.Body,3})
; CHECK: MemoryUse(5)
; CHECK-NEXT: %2 = load
  %2 = load i8, i8* %after, align 4, !invariant.group !0

; FIXME: MemoryUse(1)
; CHECK: MemoryUse(5)
; CHECK-NEXT: %3 = load
  %3 = load i8, i8* %p, align 4, !invariant.group !0
  br i1 undef, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}


define i8 @loop3(i8* %p) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, i8* %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(i8* %p)

; CHECK: MemoryUse(2)
; CHECK-NEXT: %after = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p)
  %after = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p)
  br i1 undef, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: 6 = MemoryPhi({entry,2},{Loop.Body,3},{Loop.next,4},{Loop.End,5})
; CHECK: MemoryUse(6)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, i8* %after, !invariant.group !0

; CHECK: 3 = MemoryDef(6)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(i8* %after)

; FIXME: MemoryUse(6)
; CHECK: MemoryUse(3)
; CHECK-NEXT: %1 = load i8
  %1 = load i8, i8* %after, !invariant.group !0

  br i1 undef, label %Loop.next, label %Loop.Body
Loop.next:
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(i8* %after)

; FIXME: MemoryUse(6)
; CHECK: MemoryUse(4)
; CHECK-NEXT: %2 = load i8
  %2 = load i8, i8* %after, !invariant.group !0

  br i1 undef, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: 7 = MemoryPhi({entry,2},{Loop.next,4})
; CHECK: MemoryUse(7)
; CHECK-NEXT: %3 = load
  %3 = load i8, i8* %after, align 4, !invariant.group !0

; CHECK: 5 = MemoryDef(7)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(i8* %after)

; FIXME: MemoryUse(7)
; CHECK: MemoryUse(5)
; CHECK-NEXT: %4 = load
  %4 = load i8, i8* %after, align 4, !invariant.group !0
  br i1 undef, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}

define i8 @loop4(i8* %p) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, i8* %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(i8* %p)
; CHECK: MemoryUse(2)
; CHECK-NEXT: %after = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p)
  %after = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p)
  br i1 undef, label %Loop.Pre, label %Loop.End

Loop.Pre:
; CHECK: MemoryUse(2)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, i8* %after, !invariant.group !0
  br label %Loop.Body
Loop.Body:
; CHECK: 4 = MemoryPhi({Loop.Pre,2},{Loop.Body,3},{Loop.End,5})
; CHECK-NEXT: MemoryUse(4)
; CHECK-NEXT: %1 = load i8
  %1 = load i8, i8* %after, !invariant.group !0

; FIXME: MemoryUse(2)
; CHECK: MemoryUse(4)
; CHECK-NEXT: %2 = load i8
  %2 = load i8, i8* %p, !invariant.group !0

; CHECK: 3 = MemoryDef(4)
  store i8 4, i8* %after, !invariant.group !0
  br i1 undef, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: 5 = MemoryPhi({entry,2},{Loop.Body,3})
; CHECK-NEXT: MemoryUse(5)
; CHECK-NEXT: %3 = load
  %3 = load i8, i8* %after, align 4, !invariant.group !0

; FIXME: MemoryUse(2)
; CHECK: MemoryUse(5)
; CHECK-NEXT: %4 = load
  %4 = load i8, i8* %p, align 4, !invariant.group !0
  br i1 undef, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}

declare i8* @llvm.invariant.group.barrier.p0i8(i8*)
declare void @clobber(i32*)
declare void @clobber8(i8*)


!0 = !{!"group1"}
