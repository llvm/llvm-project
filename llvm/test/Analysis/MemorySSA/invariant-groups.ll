; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa-walker>' -verify-memoryssa < %s 2>&1 | FileCheck %s

@g = external global i32

; CHECK-LABEL: define {{.*}} @global(
define i32 @global() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, ptr @g, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(ptr @g)

; FIXME: this could be clobbered by 1 if we walked the instruction list for loads/stores to @g.
; But we can't look at the uses of @g in a function analysis.
; CHECK: MemoryUse(2) {{.*}} clobbered by 2
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr @g, align 4, !invariant.group !0
  ret i32 %1
}

; CHECK-LABEL: define {{.*}} @global2(
define i32 @global2() {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, ptr inttoptr (i64 ptrtoint (ptr @g to i64) to ptr), align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(ptr inttoptr (i64 ptrtoint (ptr @g to i64) to ptr))

; FIXME: this could be clobbered by 1 if we walked the instruction list for loads/stores to @g.
; But we can't look at the uses of @g in a function analysis.
; CHECK: MemoryUse(2) {{.*}} clobbered by 2
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr inttoptr (i64 ptrtoint (ptr @g to i64) to ptr), align 4, !invariant.group !0
  ret i32 %1
}

; CHECK-LABEL: define {{.*}} @foo(
define i32 @foo(ptr %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, ptr %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, ptr @g, align 4

; CHECK:  3 = MemoryDef(2)
; CHECK-NEXT: %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)
  %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)

; This have to be MemoryUse(2), because we can't skip the barrier based on
; invariant.group.
; CHECK: MemoryUse(2)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr %a8, align 4, !invariant.group !0
  ret i32 %1
}

; CHECK-LABEL: define {{.*}} @volatile1(
define void @volatile1(ptr %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, ptr %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(ptr %a)

; CHECK: 3 = MemoryDef(2){{.*}} clobbered by 2
; CHECK-NEXT: load volatile
  %b = load volatile i32, ptr %a, align 4, !invariant.group !0

  ret void
}

; CHECK-LABEL: define {{.*}} @volatile2(
define void @volatile2(ptr %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store volatile i32 0
  store volatile i32 0, ptr %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(ptr %a)

; CHECK: MemoryUse(2){{.*}} clobbered by 2
; CHECK-NEXT: load i32
  %b = load i32, ptr %a, align 4, !invariant.group !0

  ret void
}

; CHECK-LABEL: define {{.*}} @skipBarrier(
define i32 @skipBarrier(ptr %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, ptr %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)
  %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)  

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(1)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr %a8, align 4, !invariant.group !0
  ret i32 %1
}

; CHECK-LABEL: define {{.*}} @skipBarrier2(
define i32 @skipBarrier2(ptr %a) {

; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v = load i32
  %v = load i32, ptr %a, align 4, !invariant.group !0

; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)
  %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)

; We can skip the barrier only if the "skip" is not based on !invariant.group.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: %v2 = load i32
  %v2 = load i32, ptr %a8, align 4, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, ptr @g, align 4

; CHECK: MemoryUse(2) {{.*}} clobbered by liveOnEntry
; CHECK-NEXT: %v3 = load i32
  %v3 = load i32, ptr %a8, align 4, !invariant.group !0
  %add = add nsw i32 %v2, %v3
  %add2 = add nsw i32 %add, %v
  ret i32 %add2
}

; CHECK-LABEL: define {{.*}} @handleInvariantGroups(
define i32 @handleInvariantGroups(ptr %a) {
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0
  store i32 0, ptr %a, align 4, !invariant.group !0

; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: store i32 1
  store i32 1, ptr @g, align 4
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)
  %a8 = call ptr @llvm.launder.invariant.group.p0(ptr %a)

; CHECK: MemoryUse(2)
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr %a8, align 4, !invariant.group !0

; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: store i32 2
  store i32 2, ptr @g, align 4

; CHECK: MemoryUse(4) {{.*}} clobbered by 2
; CHECK-NEXT: %2 = load i32
  %2 = load i32, ptr %a8, align 4, !invariant.group !0
  %add = add nsw i32 %1, %2
  ret i32 %add
}

; CHECK-LABEL: define {{.*}} @loop(
define i32 @loop(i1 %a) {
entry:
  %0 = alloca i32, align 4
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 4
  store i32 4, ptr %0, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber(ptr %0)
  br i1 %a, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: MemoryUse(2) {{.*}} clobbered by 1
; CHECK-NEXT: %1 = load i32
  %1 = load i32, ptr %0, !invariant.group !0
  br i1 %a, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(2) {{.*}} clobbered by 1
; CHECK-NEXT: %2 = load
  %2 = load i32, ptr %0, align 4, !invariant.group !0
  br i1 %a, label %Ret, label %Loop.Body

Ret:
  ret i32 %2
}

; CHECK-LABEL: define {{.*}} @loop2(
define i8 @loop2(ptr %p, i1 %arg) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, ptr %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(ptr %p)

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %after = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  %after = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  br i1 %arg, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: MemoryUse(6)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, ptr %after, !invariant.group !0

; CHECK: MemoryUse(6) {{.*}} clobbered by 1
; CHECK-NEXT: %1 = load i8
  %1 = load i8, ptr %p, !invariant.group !0

; CHECK: 4 = MemoryDef(6)
  store i8 4, ptr %after, !invariant.group !0

  br i1 %arg, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(5)
; CHECK-NEXT: %2 = load
  %2 = load i8, ptr %after, align 4, !invariant.group !0

; CHECK: MemoryUse(5) {{.*}} clobbered by 1
; CHECK-NEXT: %3 = load
  %3 = load i8, ptr %p, align 4, !invariant.group !0
  br i1 %arg, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}


; CHECK-LABEL: define {{.*}} @loop3(
define i8 @loop3(ptr %p, i1 %arg) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, ptr %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(ptr %p)

; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %after = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  %after = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  br i1 %arg, label %Loop.Body, label %Loop.End

Loop.Body:
; CHECK: MemoryUse(8)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, ptr %after, !invariant.group !0

; CHECK: 4 = MemoryDef(8)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(ptr %after)

; CHECK: MemoryUse(4) {{.*}} clobbered by 8
; CHECK-NEXT: %1 = load i8
  %1 = load i8, ptr %after, !invariant.group !0

  br i1 %arg, label %Loop.next, label %Loop.Body
Loop.next:
; CHECK: 5 = MemoryDef(4)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(ptr %after)

; CHECK: MemoryUse(5) {{.*}} clobbered by 8
; CHECK-NEXT: %2 = load i8
  %2 = load i8, ptr %after, !invariant.group !0

  br i1 %arg, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(7)
; CHECK-NEXT: %3 = load
  %3 = load i8, ptr %after, align 4, !invariant.group !0

; CHECK: 6 = MemoryDef(7)
; CHECK-NEXT: call void @clobber8
  call void @clobber8(ptr %after)

; CHECK: MemoryUse(6) {{.*}} clobbered by 7
; CHECK-NEXT: %4 = load
  %4 = load i8, ptr %after, align 4, !invariant.group !0
  br i1 %arg, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}

; CHECK-LABEL: define {{.*}} @loop4(
define i8 @loop4(ptr %p, i1 %arg) {
entry:
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8
  store i8 4, ptr %p, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call void @clobber
  call void @clobber8(ptr %p)
; CHECK: 3 = MemoryDef(2)
; CHECK-NEXT: %after = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  %after = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  br i1 %arg, label %Loop.Pre, label %Loop.End

Loop.Pre:
; CHECK: MemoryUse(2)
; CHECK-NEXT: %0 = load i8
  %0 = load i8, ptr %after, !invariant.group !0
  br label %Loop.Body
Loop.Body:
; CHECK: MemoryUse(6)
; CHECK-NEXT: %1 = load i8
  %1 = load i8, ptr %after, !invariant.group !0

; CHECK: MemoryUse(6) {{.*}} clobbered by 1
; CHECK-NEXT: %2 = load i8
  %2 = load i8, ptr %p, !invariant.group !0

; CHECK: 4 = MemoryDef(6)
  store i8 4, ptr %after, !invariant.group !0
  br i1 %arg, label %Loop.End, label %Loop.Body

Loop.End:
; CHECK: MemoryUse(5)
; CHECK-NEXT: %3 = load
  %3 = load i8, ptr %after, align 4, !invariant.group !0

; CHECK: MemoryUse(5) {{.*}} clobbered by 1
; CHECK-NEXT: %4 = load
  %4 = load i8, ptr %p, align 4, !invariant.group !0
  br i1 %arg, label %Ret, label %Loop.Body

Ret:
  ret i8 %3
}

; In the future we would like to CSE barriers if there is no clobber between.
; CHECK-LABEL: define {{.*}} @optimizable(
define i8 @optimizable() {
entry:
  %ptr = alloca i8
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 42, ptr %ptr, align 1, !invariant.group !0
  store i8 42, ptr %ptr, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call ptr @llvm.launder.invariant.group
  %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; FIXME: This one could be CSEd.
; CHECK: 3 = MemoryDef(2)
; CHECK: call ptr @llvm.launder.invariant.group
  %ptr3 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: call void @clobber8(ptr %ptr)
  call void @clobber8(ptr %ptr)
; CHECK: 5 = MemoryDef(4)
; CHECK-NEXT: call void @use(ptr %ptr2)
  call void @use(ptr %ptr2)
; CHECK: 6 = MemoryDef(5)
; CHECK-NEXT: call void @use(ptr %ptr3)
  call void @use(ptr %ptr3)
; CHECK: MemoryUse(6)
; CHECK-NEXT: load i8, ptr %ptr3, {{.*}}!invariant.group
  %v = load i8, ptr %ptr3, !invariant.group !0

  ret i8 %v
}

; CHECK-LABEL: define {{.*}} @unoptimizable2()
define i8 @unoptimizable2() {
  %ptr = alloca i8
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 42, ptr %ptr, align 1, !invariant.group !0
  store i8 42, ptr %ptr, !invariant.group !0
; CHECK: 2 = MemoryDef(1)
; CHECK-NEXT: call ptr @llvm.launder.invariant.group
  %ptr2 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: 3 = MemoryDef(2)
  store i8 43, ptr %ptr
; CHECK: 4 = MemoryDef(3)
; CHECK-NEXT: call ptr @llvm.launder.invariant.group
  %ptr3 = call ptr @llvm.launder.invariant.group.p0(ptr %ptr)
; CHECK: 5 = MemoryDef(4)
; CHECK-NEXT: call void @clobber8(ptr %ptr)
  call void @clobber8(ptr %ptr)
; CHECK: 6 = MemoryDef(5)
; CHECK-NEXT: call void @use(ptr %ptr2)
  call void @use(ptr %ptr2)
; CHECK: 7 = MemoryDef(6)
; CHECK-NEXT: call void @use(ptr %ptr3)
  call void @use(ptr %ptr3)
; CHECK: MemoryUse(7)
; CHECK-NEXT: %v = load i8, ptr %ptr3, align 1, !invariant.group !0
  %v = load i8, ptr %ptr3, !invariant.group !0
  ret i8 %v
}


declare ptr @llvm.launder.invariant.group.p0(ptr)
declare void @clobber(ptr)
declare void @clobber8(ptr)
declare void @use(ptr readonly)

!0 = !{!"group1"}
