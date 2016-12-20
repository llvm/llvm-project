; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @foo1(i32* %a) #0 {
entry:
  %0 = load i32, i32* %a, align 4

; Check that the alignment has been upgraded and that the assume has not
; been removed:
; CHECK-LABEL: @foo1
; CHECK-DAG: load i32, i32* %a, align 32
; CHECK-DAG: call void @llvm.assume
; CHECK: ret i32

  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)

  ret i32 %0
}

; Function Attrs: nounwind uwtable
define i32 @foo2(i32* %a) #0 {
entry:
; Same check as in @foo1, but make sure it works if the assume is first too.
; CHECK-LABEL: @foo2
; CHECK-DAG: load i32, i32* %a, align 32
; CHECK-DAG: call void @llvm.assume
; CHECK: ret i32

  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)

  %0 = load i32, i32* %a, align 4
  ret i32 %0
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

define i32 @simple(i32 %a) #1 {
entry:

; CHECK-LABEL: @simple
; CHECK: call void @llvm.assume
; CHECK: ret i32 4

  %cmp = icmp eq i32 %a, 4
  tail call void @llvm.assume(i1 %cmp)
  ret i32 %a
}

; Function Attrs: nounwind uwtable
define i32 @can1(i1 %a, i1 %b, i1 %c) {
entry:
  %and1 = and i1 %a, %b
  %and  = and i1 %and1, %c
  tail call void @llvm.assume(i1 %and)

; CHECK-LABEL: @can1
; CHECK: call void @llvm.assume(i1 %a)
; CHECK: call void @llvm.assume(i1 %b)
; CHECK: call void @llvm.assume(i1 %c)
; CHECK: ret i32

  ret i32 5
}

; Function Attrs: nounwind uwtable
define i32 @can2(i1 %a, i1 %b, i1 %c) {
entry:
  %v = or i1 %a, %b
  %w = xor i1 %v, 1
  tail call void @llvm.assume(i1 %w)

; CHECK-LABEL: @can2
; CHECK: %[[V1:[^ ]+]] = xor i1 %a, true
; CHECK: call void @llvm.assume(i1 %[[V1]])
; CHECK: %[[V2:[^ ]+]] = xor i1 %b, true
; CHECK: call void @llvm.assume(i1 %[[V2]])
; CHECK: ret i32

  ret i32 5
}

define i32 @bar1(i32 %a) #0 {
entry:
  %and1 = and i32 %a, 3

; CHECK-LABEL: @bar1
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  ret i32 %and1
}

; Function Attrs: nounwind uwtable
define i32 @bar2(i32 %a) #0 {
entry:
; CHECK-LABEL: @bar2
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  %and1 = and i32 %a, 3
  ret i32 %and1
}

; Function Attrs: nounwind uwtable
define i32 @bar3(i32 %a, i1 %x, i1 %y) #0 {
entry:
  %and1 = and i32 %a, 3

; Don't be fooled by other assumes around.
; CHECK-LABEL: @bar3
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  tail call void @llvm.assume(i1 %x)

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  tail call void @llvm.assume(i1 %y)

  ret i32 %and1
}

; Function Attrs: nounwind uwtable
define i32 @bar4(i32 %a, i32 %b) {
entry:
  %and1 = and i32 %b, 3

; CHECK-LABEL: @bar4
; CHECK: call void @llvm.assume
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

  %and = and i32 %a, 7
  %cmp = icmp eq i32 %and, 1
  tail call void @llvm.assume(i1 %cmp)

  %cmp2 = icmp eq i32 %a, %b
  tail call void @llvm.assume(i1 %cmp2)

  ret i32 %and1
}

define i32 @icmp1(i32 %a) #0 {
entry:
  %cmp = icmp sgt i32 %a, 5
  tail call void @llvm.assume(i1 %cmp)
  %conv = zext i1 %cmp to i32
  ret i32 %conv

; CHECK-LABEL: @icmp1
; CHECK: call void @llvm.assume
; CHECK: ret i32 1

}

; Function Attrs: nounwind uwtable
define i32 @icmp2(i32 %a) #0 {
entry:
  %cmp = icmp sgt i32 %a, 5
  tail call void @llvm.assume(i1 %cmp)
  %0 = zext i1 %cmp to i32
  %lnot.ext = xor i32 %0, 1
  ret i32 %lnot.ext

; CHECK-LABEL: @icmp2
; CHECK: call void @llvm.assume
; CHECK: ret i32 0
}

declare void @escape(i32* %a)

; Do we canonicalize a nonnull assumption on a load into
; metadata form?
define i1 @nonnull1(i32** %a) {
entry:
  %load = load i32*, i32** %a
  %cmp = icmp ne i32* %load, null
  tail call void @llvm.assume(i1 %cmp)
  tail call void @escape(i32* %load)
  %rval = icmp eq i32* %load, null
  ret i1 %rval

; CHECK-LABEL: @nonnull1
; CHECK: !nonnull
; CHECK-NOT: call void @llvm.assume
; CHECK: ret i1 false
}

; Make sure the above canonicalization applies only
; to pointer types.  Doing otherwise would be illegal.
define i1 @nonnull2(i32* %a) {
entry:
  %load = load i32, i32* %a
  %cmp = icmp ne i32 %load, 0
  tail call void @llvm.assume(i1 %cmp)
  %rval = icmp eq i32 %load, 0
  ret i1 %rval

; CHECK-LABEL: @nonnull2
; CHECK-NOT: !nonnull
; CHECK: call void @llvm.assume
}

; Make sure the above canonicalization does not trigger
; if the assume is control dependent on something else
define i1 @nonnull3(i32** %a, i1 %control) {
entry:
  %load = load i32*, i32** %a
  %cmp = icmp ne i32* %load, null
  br i1 %control, label %taken, label %not_taken
taken:
  tail call void @llvm.assume(i1 %cmp)
  %rval = icmp eq i32* %load, null
  ret i1 %rval
not_taken:
  ret i1 true

; CHECK-LABEL: @nonnull3
; CHECK-NOT: !nonnull
; CHECK: call void @llvm.assume
}

; Make sure the above canonicalization does not trigger
; if the path from the load to the assume is potentially 
; interrupted by an exception being thrown
define i1 @nonnull4(i32** %a) {
entry:
  %load = load i32*, i32** %a
  ;; This call may throw!
  tail call void @escape(i32* %load)
  %cmp = icmp ne i32* %load, null
  tail call void @llvm.assume(i1 %cmp)
  %rval = icmp eq i32* %load, null
  ret i1 %rval

; CHECK-LABEL: @nonnull4
; CHECK-NOT: !nonnull
; CHECK: call void @llvm.assume
}




attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

