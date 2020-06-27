; RUN: opt -S -basic-aa -gvn < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

@sortlist = external global [5001 x i32], align 4

define void @Bubble() nounwind noinline {
; CHECK: entry:
; CHECK-NEXT: %tmp7.pre = load i32
entry:
  br label %while.body5

; CHECK: while.body5:
; CHECK: %tmp7 = phi i32
; CHECK-NOT: %tmp7 = load i32
while.body5:
  %indvar = phi i32 [ 0, %entry ], [ %tmp6, %if.end ]
  %tmp5 = add i32 %indvar, 2
  %arrayidx9 = getelementptr [5001 x i32], [5001 x i32]* @sortlist, i32 0, i32 %tmp5
  %tmp6 = add i32 %indvar, 1
  %arrayidx = getelementptr [5001 x i32], [5001 x i32]* @sortlist, i32 0, i32 %tmp6
  %tmp7 = load i32, i32* %arrayidx, align 4
  %tmp10 = load i32, i32* %arrayidx9, align 4
  %cmp11 = icmp sgt i32 %tmp7, %tmp10
  br i1 %cmp11, label %if.then, label %if.end

; CHECK: if.then:
if.then:
  store i32 %tmp10, i32* %arrayidx, align 4
  store i32 %tmp7, i32* %arrayidx9, align 4
  br label %if.end

if.end:
  %exitcond = icmp eq i32 %tmp6, 100
  br i1 %exitcond, label %while.end.loopexit, label %while.body5

while.end.loopexit:
  ret void
}

declare void @hold(i32) readonly
declare void @clobber()

; This is a classic LICM case
define i32 @test1(i1 %cnd, i32* %p) {
; CHECK-LABEL: @test1
entry: 
; CHECK-LABEL: entry
; CHECK-NEXT: %v1.pre = load i32, i32* %p
  br label %header

header:
; CHECK-LABEL: header
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br label %header
}


; Slightly more complicated case to highlight that MemoryDependenceAnalysis
; can compute availability for internal control flow.  In this case, because
; the value is fully available across the backedge, we only need to establish
; anticipation for the preheader block (which is trivial in this case.)
define i32 @test2(i1 %cnd, i32* %p) {
; CHECK-LABEL: @test2
entry: 
; CHECK-LABEL: entry
; CHECK-NEXT: %v1.pre = load i32, i32* %p
  br label %header

header:
; CHECK-LABEL: header
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br i1 %cnd, label %bb1, label %bb2

bb1:
  br label %merge

bb2:
  br label %merge

merge:
  br label %header
}


; TODO: at the moment, our anticipation check does not handle anything
; other than straight-line unconditional fallthrough.  This particular
; case could be solved through either a backwards anticipation walk or
; use of the "safe to speculate" status (if we annotate the param)
define i32 @test3(i1 %cnd, i32* %p) {
entry: 
; CHECK-LABEL: @test3
; CHECK-LABEL: entry
  br label %header

header:
  br i1 %cnd, label %bb1, label %bb2

bb1:
  br label %merge

bb2:
  br label %merge

merge:
; CHECK-LABEL: merge
; CHECK: load i32, i32* %p
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br label %header
}

; Highlight that we can PRE into a latch block when there are multiple
; latches only one of which clobbers an otherwise invariant value.
define i32 @test4(i1 %cnd, i32* %p) {
; CHECK-LABEL: @test4
entry: 
; CHECK-LABEL: entry
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br label %header

header:
; CHECK-LABEL: header
  %v2 = load i32, i32* %p
  call void @hold(i32 %v2)
  br i1 %cnd, label %bb1, label %bb2

bb1:
  br label %header

bb2:
; CHECK-LABEL: bb2
; CHECK:       call void @clobber()
; CHECK-NEXT:  %v2.pre = load i32, i32* %p
; CHECK-NEXT:  br label %header

  call void @clobber()
  br label %header
}

; Highlight the fact that we can PRE into a single clobbering latch block
; even in loop simplify form (though multiple applications of the same
; transformation).
define i32 @test5(i1 %cnd, i32* %p) {
; CHECK-LABEL: @test5
entry: 
; CHECK-LABEL: entry
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br label %header

header:
; CHECK-LABEL: header
  %v2 = load i32, i32* %p
  call void @hold(i32 %v2)
  br i1 %cnd, label %bb1, label %bb2

bb1:
  br label %merge

bb2:
; CHECK-LABEL: bb2
; CHECK:       call void @clobber()
; CHECK-NEXT:  %v2.pre.pre = load i32, i32* %p
; CHECK-NEXT:  br label %merge

  call void @clobber()
  br label %merge

merge:
  br label %header
}

declare void @llvm.experimental.guard(i1 %cnd, ...)

; These two tests highlight speculation safety when we can not establish
; anticipation (since the original load might actually not execcute)
define i32 @test6a(i1 %cnd, i32* %p) {
entry: 
; CHECK-LABEL: @test6a
  br label %header

header:
; CHECK-LABEL: header
; CHECK: load i32, i32* %p
  call void (i1, ...) @llvm.experimental.guard(i1 %cnd) ["deopt"()]
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br label %header
}

define i32 @test6b(i1 %cnd, i32* dereferenceable(8) align 4 %p) {
entry: 
; CHECK-LABEL: @test6b
; CHECK: load i32, i32* %p
  br label %header

header:
; CHECK-LABEL: header
  call void (i1, ...) @llvm.experimental.guard(i1 %cnd) ["deopt"()]
  %v1 = load i32, i32* %p
  call void @hold(i32 %v1)
  br label %header
}
