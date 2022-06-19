; RUN: opt < %s -passes=tailcallelim -verify-dom-info -S | FileCheck %s
; PR7328
; PR7506
define i32 @test1_constants(i32 %x) {
entry:
  %cond = icmp ugt i32 %x, 0                      ; <i1> [#uses=1]
  br i1 %cond, label %return, label %body

body:                                             ; preds = %entry
  %y = add i32 %x, 1                              ; <i32> [#uses=1]
  %recurse = call i32 @test1_constants(i32 %y)        ; <i32> [#uses=0]
  ret i32 0

return:                                           ; preds = %entry
  ret i32 1
}

; CHECK-LABEL: define i32 @test1_constants(
; CHECK: tailrecurse:
; CHECK: %ret.tr = phi i32 [ undef, %entry ], [ %current.ret.tr, %body ]
; CHECK: %ret.known.tr = phi i1 [ false, %entry ], [ true, %body ]
; CHECK: body:
; CHECK-NOT: %recurse
; CHECK: %current.ret.tr = select i1 %ret.known.tr, i32 %ret.tr, i32 0
; CHECK-NOT: ret
; CHECK: return:
; CHECK: %current.ret.tr1 = select i1 %ret.known.tr, i32 %ret.tr, i32 1
; CHECK: ret i32 %current.ret.tr1

define i32 @test2_non_constants(i32 %x) {
entry:
  %cond = icmp ugt i32 %x, 0
  br i1 %cond, label %return, label %body

body:
  %y = add i32 %x, 1
  %helper1 = call i32 @test2_helper()
  %recurse = call i32 @test2_non_constants(i32 %y)
  ret i32 %helper1

return:
  %helper2 = call i32 @test2_helper()
  ret i32 %helper2
}

declare i32 @test2_helper()

; CHECK-LABEL: define i32 @test2_non_constants(
; CHECK: tailrecurse:
; CHECK: %ret.tr = phi i32 [ undef, %entry ], [ %current.ret.tr, %body ]
; CHECK: %ret.known.tr = phi i1 [ false, %entry ], [ true, %body ]
; CHECK: body:
; CHECK-NOT: %recurse
; CHECK: %current.ret.tr = select i1 %ret.known.tr, i32 %ret.tr, i32 %helper1
; CHECK-NOT: ret
; CHECK: return:
; CHECK: %current.ret.tr1 = select i1 %ret.known.tr, i32 %ret.tr, i32 %helper2
; CHECK: ret i32 %current.ret.tr1

define i32 @test3_mixed(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 0, label %case0
    i32 1, label %case1
    i32 2, label %case2
  ]

case0:
  %helper1 = call i32 @test3_helper()
  br label %return

case1:
  %y1 = add i32 %x, -1
  %recurse1 = call i32 @test3_mixed(i32 %y1)
  br label %return

case2:
  %y2 = add i32 %x, -1
  %helper2 = call i32 @test3_helper()
  %recurse2 = call i32 @test3_mixed(i32 %y2)
  br label %return

default:
  %y3 = urem i32 %x, 3
  %recurse3 = call i32 @test3_mixed(i32 %y3)
  br label %return

return:
  %retval = phi i32 [ %recurse3, %default ], [ %helper2, %case2 ], [ 9, %case1 ], [ %helper1, %case0 ]
  ret i32 %retval
}

declare i32 @test3_helper()

; CHECK-LABEL: define i32 @test3_mixed(
; CHECK: tailrecurse:
; CHECK: %ret.tr = phi i32 [ undef, %entry ], [ %current.ret.tr, %case1 ], [ %current.ret.tr1, %case2 ], [ %ret.tr, %default ]
; CHECK: %ret.known.tr = phi i1 [ false, %entry ], [ true, %case1 ], [ true, %case2 ], [ %ret.known.tr, %default ]
; CHECK: case1:
; CHECK-NOT: %recurse
; CHECK: %current.ret.tr = select i1 %ret.known.tr, i32 %ret.tr, i32 9
; CHECK: br label %tailrecurse
; CHECK: case2:
; CHECK-NOT: %recurse
; CHECK: %current.ret.tr1 = select i1 %ret.known.tr, i32 %ret.tr, i32 %helper2
; CHECK: br label %tailrecurse
; CHECK: default:
; CHECK-NOT: %recurse
; CHECK: br label %tailrecurse
; CHECK: return:
; CHECK: %current.ret.tr2 = select i1 %ret.known.tr, i32 %ret.tr, i32 %helper1
; CHECK: ret i32 %current.ret.tr2
