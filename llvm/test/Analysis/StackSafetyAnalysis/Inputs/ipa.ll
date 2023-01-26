target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

define dso_local void @Write1(ptr %p) #0 {
entry:
  store i8 0, ptr %p, align 1
  ret void
}

define dso_local void @Write4(ptr %p) #0 {
entry:
  store i32 0, ptr %p, align 1
  ret void
}

define dso_local void @Write4_2(ptr %p, ptr %q) #0 {
entry:
  store i32 0, ptr %p, align 1
  store i32 0, ptr %q, align 1
  ret void
}

define dso_local void @Write8(ptr %p) #0 {
entry:
  store i64 0, ptr %p, align 1
  ret void
}

define dso_local ptr @WriteAndReturn8(ptr %p) #0 {
entry:
  store i8 0, ptr %p, align 1
  ret ptr %p
}

declare dso_local void @ExternalCall(ptr %p)

define dso_preemptable void @PreemptableWrite1(ptr %p) #0 {
entry:
  store i8 0, ptr %p, align 1
  ret void
}

define linkonce dso_local void @InterposableWrite1(ptr %p) #0 {
entry:
  store i8 0, ptr %p, align 1
  ret void
}

define dso_local ptr @ReturnDependent(ptr %p) #0 {
entry:
  %p2 = getelementptr i8, ptr %p, i64 2
  ret ptr %p2
}

; access range [2, 6)
define dso_local void @Rec0(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 2
  call void @Write4(ptr %p1)
  ret void
}

; access range [3, 7)
define dso_local void @Rec1(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 1
  call void @Rec0(ptr %p1)
  ret void
}

; access range [-2, 2)
define dso_local void @Rec2(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 -5
  call void @Rec1(ptr %p1)
  ret void
}

; Recursive function that passes %acc unchanged => access range [0, 4).
define dso_local void @RecursiveNoOffset(ptr %p, i32 %size, ptr %acc) {
entry:
  %cmp = icmp eq i32 %size, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  %load0 = load i32, ptr %p, align 4
  %load1 = load i32, ptr %acc, align 4
  %add = add nsw i32 %load1, %load0
  store i32 %add, ptr %acc, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 1
  %sub = add nsw i32 %size, -1
  tail call void @RecursiveNoOffset(ptr %add.ptr, i32 %sub, ptr %acc)
  ret void

return:
  ret void
}

; Recursive function that advances %acc on each iteration => access range unlimited.
define dso_local void @RecursiveWithOffset(i32 %size, ptr %acc) {
entry:
  %cmp = icmp eq i32 %size, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  store i32 0, ptr %acc, align 4
  %acc2 = getelementptr inbounds i32, ptr %acc, i64 1
  %sub = add nsw i32 %size, -1
  tail call void @RecursiveWithOffset(i32 %sub, ptr %acc2)
  ret void

return:
  ret void
}

define dso_local ptr @ReturnAlloca() {
entry:
  %x = alloca i64, align 4
  ret ptr %x
}

define dso_local void @Write1Private(ptr %p) #0 {
entry:
  call void @Private(ptr %p)
  ret void
}

define dso_local void @Write1SameModule(ptr %p) #0 {
entry:
  call void @Write1(ptr %p)
  ret void
}

declare void @Write1Module0(ptr %p)

define dso_local void @Write1DiffModule(ptr %p) #0 {
entry:
  call void @Write1Module0(ptr %p)
  ret void
}

define private dso_local void @Private(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 -1
  store i8 0, ptr %p1, align 1
  ret void
}

define dso_local void @Write1Weak(ptr %p) #0 {
entry:
  call void @Weak(ptr %p)
  ret void
}

define weak dso_local void @Weak(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 -1
  store i8 0, ptr %p1, align 1
  ret void
}

