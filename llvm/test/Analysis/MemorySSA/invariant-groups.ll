; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa-walker>' -verify-memoryssa < %s 2>&1 | FileCheck %s

@g = external global i32

; CHECK-LABEL: define {{.*}} @global(
define i32 @global() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* @g, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(i32* @g)

; FIXME: this could be clobbered by 1 if we walked the instruction list for loads/stores to @g.
; But we can't look at the uses of @g in a function analysis.
; CHECK: MemoryUse(2) {{.*}} clobbered by 2
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* @g, align 4, !invariant.group !0
  ret i32 %1
}

; CHECK-LABEL: define {{.*}} @global2(
define i32 @global2() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* inttoptr (i64 ptrtoint (i32* @g to i64) to i32*), align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(i32* inttoptr (i64 ptrtoint (i32* @g to i64) to i32*))

; FIXME: this could be clobbered by 1 if we walked the instruction list for loads/stores to @g.
; But we can't look at the uses of @g in a function analysis.
; CHECK: MemoryUse(2) {{.*}} clobbered by 2
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* inttoptr (i64 ptrtoint (i32* @g to i64) to i32*), align 4, !invariant.group !0
  ret i32 %1
}

; CHECK-LABEL: define {{.*}} @foo(
define i32 @foo(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

  %1 = bitcast i32* %a to i8*
; CHECK:  3 = MemoryDef(2)
; CHECK-NEXT: %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; This have to be MemoryUse(2), because we can't skip the barrier based on
; invariant.group.
; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0
  ret i32 %2
}

; CHECK-LABEL: define {{.*}} @volatile1(
define void @volatile1(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(i32* %a)

; CHECK: 3 = MemoryDef(2){{.*}} clobbered by 2
; CHECK-NEXT: load volatile
  %b = load volatile i32, i32* %a, align 4, !invariant.group !0

  ret void
}

; CHECK-LABEL: define {{.*}} @volatile2(
define void @volatile2(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store volatile i32 0
  store volatile i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(i32* %a)

; CHECK: MemoryUse(2){{.*}} clobbered by 2
; CHECK-NEXT: load i32
  %b = load i32, i32* %a, align 4, !invariant.group !0

  ret void
}

; CHECK-LABEL: define {{.*}} @skipBarrier(
define i32 @skipBarrier(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

  %1 = bitcast i32* %a to i8*
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)  
  %a32 = bitcast i8* %a8 to i32*

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(1)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0
  ret i32 %2
}

; CHECK-LABEL: define {{.*}} @skipBarrier2(
define i32 @skipBarrier2(i32* %a) {

; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = load i32
  %v = load i32, i32* %a, align 4, !invariant.group !0

  %1 = bitcast i32* %a to i8*
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v2 = load i32
  %v2 = load i32, i32* %a32, align 4, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4

; CHECK: MemoryUse(2) {{.*}} clobbered by liveOnEntry
; CHECK-NEXT: %v3 = load i32
  %v3 = load i32, i32* %a32, align 4, !invariant.group !0
  %add = add nsw i32 %v2, %v3
  %add2 = add nsw i32 %add, %v
  ret i32 %add2
}

; CHECK-LABEL: define {{.*}} @handleInvariantGroups(
define i32 @handleInvariantGroups(i32* %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, i32* %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, i32* @g, align 4
  %1 = bitcast i32* %a to i8*
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a8 = call i8* @llvm.launder.invariant.group.p0i8(i8* %1)
  %a32 = bitcast i8* %a8 to i32*

; CHECK: MemoryUse(2)
; CHECK-NEXT: %2 = load i32
  %2 = load i32, i32* %a32, align 4, !invariant.group !0

; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: store i32 2
  store i32 2, i32* @g, align 4

; CHECK: MemoryUse(4) {{.*}} clobbered by 2
; CHECK-NEXT: %3 = load i32
  %3 = load i32, i32* %a32, align 4, !invariant.group !0
  %add = add nsw i32 %2, %3
  ret i32 %add
}

; CHECK-LABEL: define {{.*}} @loop(
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
; CHECK: MemoryUse(2) {{.*}} clobbered by 1
; CHECK-NEXT: %1 = load i32
  %1 = load i32, i32* %0, !invariant.group !0
  br i1 %a, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(2) {{.*}} clobbered by 1
; CHECK-NEXT: %2 = load
  %2 = load i32, i32* %0, align 4, !invariant.group !0
  br i1 %a, label %Ret, label %Loop.Body

Ret:
  ret i32 %2
}

; CHECK-LABEL: define {{.*}} @loop2(
define i8 @loop2(i8* %p) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, i8* %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(i8* %p)

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %after = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  %after = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  br i1 undef, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: MemoryUse(6)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, i8* %after, !invariant.group !0

; CHECK: MemoryUse(6) {{.*}} clobbered by 1
; CHECK-NEXT: %1 = load i8
  %1 = load i8, i8* %p, !invariant.group !0

; CHECK: 4 = MemoryDef(6)
  store i8 4, i8* %after, !invariant.group !0

  br i1 undef, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(5)
; CHECK-NEXT: %2 = load
  %2 = load i8, i8* %after, align 4, !invariant.group !0

; CHECK: MemoryUse(5) {{.*}} clobbered by 1
; CHECK-NEXT: %3 = load
  %3 = load i8, i8* %p, align 4, !invariant.group !0
  br i1 undef, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}


; CHECK-LABEL: define {{.*}} @loop3(
define i8 @loop3(i8* %p) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, i8* %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(i8* %p)

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %after = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  %after = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  br i1 undef, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: MemoryUse(8)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, i8* %after, !invariant.group !0

; CHECK: 4 = MemoryDef(8)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(i8* %after)

; CHECK: MemoryUse(4) {{.*}} clobbered by 8
; CHECK-NEXT: %1 = load i8
  %1 = load i8, i8* %after, !invariant.group !0

  br i1 undef, label %Loop.next, label %Loop.Body
Loop.next:
; CHECK: 5 = MemoryDef(4)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(i8* %after)

; CHECK: MemoryUse(5) {{.*}} clobbered by 8
; CHECK-NEXT: %2 = load i8
  %2 = load i8, i8* %after, !invariant.group !0

  br i1 undef, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(7)
; CHECK-NEXT: %3 = load
  %3 = load i8, i8* %after, align 4, !invariant.group !0

; CHECK: 6 = MemoryDef(7)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(i8* %after)

; CHECK: MemoryUse(6) {{.*}} clobbered by 7
; CHECK-NEXT: %4 = load
  %4 = load i8, i8* %after, align 4, !invariant.group !0
  br i1 undef, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}

; CHECK-LABEL: define {{.*}} @loop4(
define i8 @loop4(i8* %p) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, i8* %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(i8* %p)
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %after = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  %after = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  br i1 undef, label %Loop.Pre, label %Loop.End

Loop.Pre:
; CHECK: MemoryUse(2)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, i8* %after, !invariant.group !0
  br label %Loop.Body
Loop.Body:
; CHECK: MemoryUse(6)
; CHECK-NEXT: %1 = load i8
  %1 = load i8, i8* %after, !invariant.group !0

; CHECK: MemoryUse(6) {{.*}} clobbered by 1
; CHECK-NEXT: %2 = load i8
  %2 = load i8, i8* %p, !invariant.group !0

; CHECK: 4 = MemoryDef(6)
  store i8 4, i8* %after, !invariant.group !0
  br i1 undef, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(5)
; CHECK-NEXT: %3 = load
  %3 = load i8, i8* %after, align 4, !invariant.group !0

; CHECK: MemoryUse(5) {{.*}} clobbered by 1
; CHECK-NEXT: %4 = load
  %4 = load i8, i8* %p, align 4, !invariant.group !0
  br i1 undef, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}

; In the future we would like to CSE barriers if there is no clobber between.
; CHECK-LABEL: define {{.*}} @optimizable(
define i8 @optimizable() {
entry:
  %ptr = alloca i8
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 42, i8* %ptr, align 1, !invariant.group !0
  store i8 42, i8* %ptr, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call i8* @llvm.launder.invariant.group
  %ptr2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
; FIXME: This one could be CSEd.
; CHECK: 3 = MemoryDef(2)
; CHECK: call i8* @llvm.launder.invariant.group
  %ptr3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: call void @clobber8(i8* %ptr)
  call void @clobber8(i8* %ptr)
; CHECK: 5 = MemoryDef(4)
; CHECK-NEXT: call void @use(i8* %ptr2)
  call void @use(i8* %ptr2)
; CHECK: 6 = MemoryDef(5)
; CHECK-NEXT: call void @use(i8* %ptr3)
  call void @use(i8* %ptr3)
; CHECK: MemoryUse(6)
; CHECK-NEXT: load i8, i8* %ptr3, {{.*}}!invariant.group
  %v = load i8, i8* %ptr3, !invariant.group !0

  ret i8 %v
}

; CHECK-LABEL: define {{.*}} @unoptimizable2()
define i8 @unoptimizable2() {
  %ptr = alloca i8
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 42, i8* %ptr, align 1, !invariant.group !0
  store i8 42, i8* %ptr, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call i8* @llvm.launder.invariant.group
  %ptr2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
; CHECK: 3 = MemoryDef(2)
  store i8 43, i8* %ptr
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: call i8* @llvm.launder.invariant.group
  %ptr3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %ptr)
; CHECK: 5 = MemoryDef(4)
; CHECK-NEXT: call void @clobber8(i8* %ptr)
  call void @clobber8(i8* %ptr)
; CHECK: 6 = MemoryDef(5)
; CHECK-NEXT: call void @use(i8* %ptr2)
  call void @use(i8* %ptr2)
; CHECK: 7 = MemoryDef(6)
; CHECK-NEXT: call void @use(i8* %ptr3)
  call void @use(i8* %ptr3)
; CHECK: MemoryUse(7)
; CHECK-NEXT: %v = load i8, i8* %ptr3, align 1, !invariant.group !0
  %v = load i8, i8* %ptr3, !invariant.group !0
  ret i8 %v
}


declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare void @clobber(i32*)
declare void @clobber8(i8*)
declare void @use(i8* readonly)

!0 = !{!"group1"}
