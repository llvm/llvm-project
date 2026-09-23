; Constant arrays that differ only where one of them holds poison describe the
; same value, so they can be merged into the more defined of the two. Lookup
; tables built for copies of an inlined switch differ exactly this way.
; RUN: opt -passes=constmerge -S < %s | FileCheck %s
; Comparing candidates against each other is bounded. With no room for it,
; nothing is merged and nothing is wrong.
; RUN: opt -passes=constmerge -constmerge-max-poison-candidates=1 -S < %s \
; RUN:   | FileCheck %s --check-prefix=NOBUDGET

; NOBUDGET: @a = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr poison, ptr @f2, ptr @f3]
; NOBUDGET: @b = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr @f1, ptr poison, ptr @f3]
; NOBUDGET: @c = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr @f1, ptr @f2, ptr @f3]

; The merge target is the constant needing fewest elements filled in, not
; whichever comes first, so reordering the globals does not change the result.

; CHECK: @c = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr @f1, ptr @f2, ptr @f3]
; CHECK-NOT: @a =
; CHECK-NOT: @b =
; CHECK: @different = private unnamed_addr constant [4 x ptr] [ptr @f3, ptr @f1, ptr @f2, ptr @f3]

@a = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr poison, ptr @f2, ptr @f3]
@b = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr @f1, ptr poison, ptr @f3]
@c = private unnamed_addr constant [4 x ptr] [ptr @f0, ptr @f1, ptr @f2, ptr @f3]
@different = private unnamed_addr constant [4 x ptr] [ptr @f3, ptr @f1, ptr @f2, ptr @f3]

declare void @f0()
declare void @f1()
declare void @f2()
declare void @f3()

define ptr @ua(i64 %i) {
; CHECK-LABEL: @ua(
; CHECK: getelementptr [4 x ptr], ptr @c,
  %p = getelementptr [4 x ptr], ptr @a, i64 0, i64 %i
  %v = load ptr, ptr %p
  ret ptr %v
}

define ptr @ub(i64 %i) {
; CHECK-LABEL: @ub(
; CHECK: getelementptr [4 x ptr], ptr @c,
  %p = getelementptr [4 x ptr], ptr @b, i64 0, i64 %i
  %v = load ptr, ptr %p
  ret ptr %v
}

define ptr @uc(i64 %i) {
; CHECK-LABEL: @uc(
; CHECK: getelementptr [4 x ptr], ptr @c,
  %p = getelementptr [4 x ptr], ptr @c, i64 0, i64 %i
  %v = load ptr, ptr %p
  ret ptr %v
}

define ptr @ud(i64 %i) {
; CHECK-LABEL: @ud(
; CHECK: getelementptr [4 x ptr], ptr @different,
  %p = getelementptr [4 x ptr], ptr @different, i64 0, i64 %i
  %v = load ptr, ptr %p
  ret ptr %v
}
