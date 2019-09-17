; RUN: opt -attributor --attributor-disable=false -S < %s | FileCheck %s
; TODO: Add max-iteration check
; ModuleID = 'value-simplify.ll'
source_filename = "value-simplify.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
declare void @f(i32)

; Test1: Replace argument with constant
define internal void @test1(i32 %a) {
; CHECK: tail call void @f(i32 1)
  tail call void @f(i32 %a)
  ret void
}

define void @test1_helper() {
  tail call void @test1(i32 1)
  ret void
}

; TEST 2 : Simplify return value
define i32 @return0() {
  ret i32 0
}

define i32 @return1() {
  ret i32 1
}

; CHECK: define i32 @test2_1(i1 %c)
define i32 @test2_1(i1 %c) {
  br i1 %c, label %if.true, label %if.false
if.true:
  %call = tail call i32 @return0()

; FIXME: %ret0 should be replaced with i32 1.
; CHECK: %ret0 = add i32 0, 1
  %ret0 = add i32 %call, 1
  br label %end
if.false:
  %ret1 = tail call i32 @return1()
  br label %end
end:

; FIXME: %ret should be replaced with i32 1.
; CHECK: %ret = phi i32 [ %ret0, %if.true ], [ 1, %if.false ]
  %ret = phi i32 [ %ret0, %if.true ], [ %ret1, %if.false ]

; FIXME: ret i32 1
; CHECK: ret i32 %ret
  ret i32 %ret
}



; CHECK: define i32 @test2_2(i1 %c)
define i32 @test2_2(i1 %c) {
; FIXME: %ret should be replaced with i32 1.
  %ret = tail call i32 @test2_1(i1 %c)
; FIXME: ret i32 1
; CHECK: ret i32 %ret
  ret i32 %ret
}

declare void @use(i32)
; CHECK: define void @test3(i1 %c)
define void @test3(i1 %c) {
  br i1 %c, label %if.true, label %if.false
if.true:
  br label %end
if.false:
  %ret1 = tail call i32 @return1()
  br label %end
end:

; CHECK: %r = phi i32 [ 1, %if.true ], [ 1, %if.false ]
  %r = phi i32 [ 1, %if.true ], [ %ret1, %if.false ]

; CHECK: tail call void @use(i32 1)
  tail call void @use(i32 %r)
  ret void
}

define void @test-select-phi(i1 %c) {
  %select-same = select i1 %c, i32 1, i32 1
  ; CHECK: tail call void @use(i32 1)
  tail call void @use(i32 %select-same)

  %select-not-same = select i1 %c, i32 1, i32 0
  ; CHECK: tail call void @use(i32 %select-not-same)
  tail call void @use(i32 %select-not-same)
  br i1 %c, label %if-true, label %if-false
if-true:
  br label %end
if-false:
  br label %end
end:
  %phi-same = phi i32 [ 1, %if-true ], [ 1, %if-false ]
  %phi-not-same = phi i32 [ 0, %if-true ], [ 1, %if-false ]
  %phi-same-prop = phi i32 [ 1, %if-true ], [ %select-same, %if-false ]
  %phi-same-undef = phi i32 [ 1, %if-true ], [ undef, %if-false ]
  %select-not-same-undef = select i1 %c, i32 %phi-not-same, i32 undef


  ; CHECK: tail call void @use(i32 1)
  tail call void @use(i32 %phi-same)

  ; CHECK: tail call void @use(i32 %phi-not-same)
  tail call void @use(i32 %phi-not-same)

  ; CHECK: tail call void @use(i32 1)
  tail call void @use(i32 %phi-same-prop)

  ; CHECK: tail call void @use(i32 1)
  tail call void @use(i32 %phi-same-undef)

  ; CHECK: tail call void @use(i32 %select-not-same-undef)
  tail call void @use(i32 %select-not-same-undef)

  ret void

}
