; This test creates a monster SCC with a very pernicious call graph. It builds
; a cycle of cross-connected pairs of functions with interesting inlining
; decisions throughout, but ultimately trivial code complexity.
;
; Typically, a greedy approach to inlining works well for bottom-up inliners
; such as LLVM's. However, there is no way to be bottom-up over an SCC: it's
; a cycle! Greedily inlining as much as possible into each function of this
; *SCC* will have the disasterous effect of inlining all N-1 functions into the
; first one visited, N-2 functions into the second one visited, N-3 into the
; third, and so on. This is because until inlining occurs, each function in
; isolation appears to be an excellent inline candidate.
;
; Note that the exact number of calls in each function doesn't really matter.
; It is mostly a function of cost thresholds and visit order. Because this is an
; SCC there is no "right" or "wrong" answer here as long as no function blows up
; to be *huge*. The specific concerning pattern is if one or more functions get
; more than 16 calls in them.
;
; This test is extracted from the following C++ program compiled with Clang.
; The IR is simplified with SROA, instcombine, and simplifycfg. Then C++
; linkage stuff, attributes, target specific things, metadata and comments were
; removed. The order of the fuctions is also made more predictable than Clang's
; output order.
;
;   void g(int);
;
;   template <bool K, int N> void f(bool *B, bool *E) {
;     if (K)
;       g(N);
;     if (B == E)
;       return;
;     if (*B)
;       f<true, N + 1>(B + 1, E);
;     else
;       f<false, N + 1>(B + 1, E);
;   }
;   template <> void f<false, MAX>(bool *B, bool *E) { return f<false, 0>(B, E); }
;   template <> void f<true, MAX>(bool *B, bool *E) { return f<true, 0>(B, E); }
;
;   void test(bool *B, bool *E) { f<false, 0>(B, E); }
;
; RUN: opt -S < %s -inline -inline-threshold=150 -enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK,OLD
; RUN: opt -S < %s -passes=inline -inline-threshold=150 | FileCheck %s --check-prefixes=CHECK,NEW
; RUN: opt -S < %s -passes=inliner-wrapper -inline-threshold=150 | FileCheck %s --check-prefixes=CHECK,NEW

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @_Z1gi(i32)

; CHECK-LABEL: define void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi1EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi2EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi2EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi2EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi2EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb0ELi0EEvPbS0_(i8* %B, i8* %E) {
entry:
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi1EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi1EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi1EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi1EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi2EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi2EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb1ELi0EEvPbS0_(i8* %B, i8* %E) {
entry:
  call void @_Z1gi(i32 0)
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi1EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi1EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb0ELi1EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi2EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi2EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi3EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi3EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb0ELi1EEvPbS0_(i8* %B, i8* %E) {
entry:
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi2EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi2EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb1ELi1EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi2EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi3EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi3EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi3EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi3EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb1ELi1EEvPbS0_(i8* %B, i8* %E) {
entry:
  call void @_Z1gi(i32 1)
  %cmp = icmp eq i8* %B, %E
; CHECK-NOT: call
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi2EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi2EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb0ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi0EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi0EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi4EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi4EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb0ELi2EEvPbS0_(i8* %B, i8* %E) {
entry:
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi3EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi3EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb1ELi2EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1gi(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi4EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi4EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi4EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi4EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb1ELi2EEvPbS0_(i8* %B, i8* %E) {
entry:
  call void @_Z1gi(i32 2)
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi3EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi3EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb0ELi3EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb0ELi0EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi1EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi1EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi0EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb0ELi3EEvPbS0_(i8* %B, i8* %E) {
entry:
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi4EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi4EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb1ELi3EEvPbS0_(
; CHECK-NOT: call
; CHECK: call void @_Z1gi(
; CHECK-NOT: call
; CHECK: call void @_Z1fILb1ELi0EEvPbS0_(
; CHECK-NOT: call
; CHECK: call void @_Z1fILb0ELi0EEvPbS0_(
; CHECK-NOT: call
define void @_Z1fILb1ELi3EEvPbS0_(i8* %B, i8* %E) {
entry:
  call void @_Z1gi(i32 3)
  %cmp = icmp eq i8* %B, %E
  br i1 %cmp, label %if.end3, label %if.end

if.end:
  %0 = load i8, i8* %B, align 1
  %tobool = icmp eq i8 %0, 0
  %add.ptr2 = getelementptr inbounds i8, i8* %B, i64 1
  br i1 %tobool, label %if.else, label %if.then1

if.then1:
  call void @_Z1fILb1ELi4EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.else:
  call void @_Z1fILb0ELi4EEvPbS0_(i8* %add.ptr2, i8* %E)
  br label %if.end3

if.end3:
  ret void
}

; CHECK-LABEL: define void @_Z1fILb0ELi4EEvPbS0_(
; CHECK-NOT: call
; CHECK: call void @_Z1fILb0ELi0EEvPbS0_(
; CHECK-NOT: call
define void @_Z1fILb0ELi4EEvPbS0_(i8* %B, i8* %E) {
entry:
  call void @_Z1fILb0ELi0EEvPbS0_(i8* %B, i8* %E)
  ret void
}

; CHECK-LABEL: define void @_Z1fILb1ELi4EEvPbS0_(
; OLD-NOT: call
; OLD: call void @_Z1fILb1ELi0EEvPbS0_(
; OLD-NOT: call
; NEW-NOT: call
; NEW: call void @_Z1gi(
; NEW-NOT: call
; NEW: call void @_Z1fILb1ELi1EEvPbS0_(
; NEW-NOT: call
; NEW: call void @_Z1fILb0ELi1EEvPbS0_(
; NEW-NOT: call
define void @_Z1fILb1ELi4EEvPbS0_(i8* %B, i8* %E) {
entry:
  call void @_Z1fILb1ELi0EEvPbS0_(i8* %B, i8* %E)
  ret void
}

; CHECK-LABEL: define void @_Z4testPbS_(
; CHECK: call
; CHECK-NOT: call
define void @_Z4testPbS_(i8* %B, i8* %E) {
entry:
  call void @_Z1fILb0ELi0EEvPbS0_(i8* %B, i8* %E)
  ret void
}

