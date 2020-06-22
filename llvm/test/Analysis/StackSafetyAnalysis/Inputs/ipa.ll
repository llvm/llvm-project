target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

define dso_local void @Write1(i8* %p) #0 {
entry:
  store i8 0, i8* %p, align 1
  ret void
}

define dso_local void @Write4(i8* %p) #0 {
entry:
  %0 = bitcast i8* %p to i32*
  store i32 0, i32* %0, align 1
  ret void
}

define dso_local void @Write4_2(i8* %p, i8* %q) #0 {
entry:
  %0 = bitcast i8* %p to i32*
  store i32 0, i32* %0, align 1
  %1 = bitcast i8* %q to i32*
  store i32 0, i32* %1, align 1
  ret void
}

define dso_local void @Write8(i8* %p) #0 {
entry:
  %0 = bitcast i8* %p to i64*
  store i64 0, i64* %0, align 1
  ret void
}

define dso_local i8* @WriteAndReturn8(i8* %p) #0 {
entry:
  store i8 0, i8* %p, align 1
  ret i8* %p
}

declare dso_local void @ExternalCall(i8* %p)

define dso_preemptable void @PreemptableWrite1(i8* %p) #0 {
entry:
  store i8 0, i8* %p, align 1
  ret void
}

define linkonce dso_local void @InterposableWrite1(i8* %p) #0 {
entry:
  store i8 0, i8* %p, align 1
  ret void
}

define dso_local i8* @ReturnDependent(i8* %p) #0 {
entry:
  %p2 = getelementptr i8, i8* %p, i64 2
  ret i8* %p2
}

; access range [2, 6)
define dso_local void @Rec0(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 2
  call void @Write4(i8* %p1)
  ret void
}

; access range [3, 7)
define dso_local void @Rec1(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 1
  call void @Rec0(i8* %p1)
  ret void
}

; access range [-2, 2)
define dso_local void @Rec2(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 -5
  call void @Rec1(i8* %p1)
  ret void
}

; Recursive function that passes %acc unchanged => access range [0, 4).
define dso_local void @RecursiveNoOffset(i32* %p, i32 %size, i32* %acc) {
entry:
  %cmp = icmp eq i32 %size, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  %0 = load i32, i32* %p, align 4
  %1 = load i32, i32* %acc, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %acc, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %sub = add nsw i32 %size, -1
  tail call void @RecursiveNoOffset(i32* %add.ptr, i32 %sub, i32* %acc)
  ret void

return:
  ret void
}

; Recursive function that advances %acc on each iteration => access range unlimited.
define dso_local void @RecursiveWithOffset(i32 %size, i32* %acc) {
entry:
  %cmp = icmp eq i32 %size, 0
  br i1 %cmp, label %return, label %if.end

if.end:
  store i32 0, i32* %acc, align 4
  %acc2 = getelementptr inbounds i32, i32* %acc, i64 1
  %sub = add nsw i32 %size, -1
  tail call void @RecursiveWithOffset(i32 %sub, i32* %acc2)
  ret void

return:
  ret void
}

define dso_local i64* @ReturnAlloca() {
entry:
  %x = alloca i64, align 4
  ret i64* %x
}
