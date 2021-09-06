; REQUIRES: assert
; RUN: opt -passes=print-predicateinfo -debug < %s 2>&1 | FileCheck %s

declare void @use(i32)

; Make sure we are visiting the values to build predicate infos for in a
; deterministic order.
define i32 @test12(i32 %x, i32 %y) {
; CHECK: Visiting i32 %x
; CHECK: Visiting i32 %y
; CHECK: Visiting   %lcmp = icmp eq i32 %x, 0
; CHECK: Visiting   %lcmp2 = icmp slt i32 %y, 1000
; CHECK: Visiting   %lcmp3 = icmp slt i32 %y.0, 900
; CHECK: Visiting   %lcmp4 = icmp slt i32 %y.0.1, 700
; CHECK: Visiting   %lcmp5 = icmp slt i32 %y.0.1.2, 700
; CHECK: Visiting   %lcmp6 = icmp slt i32 %y.0.1.2.3, 700
; CHECK: Visiting   %lcmp7 = icmp slt i32 %y.0.1.2.3.4, 700
; CHECK: Visiting   %rcmp = icmp eq i32 %x, 0
entry:
  br i1 undef, label %left, label %right

left:
  %lcmp = icmp eq i32 %x, 0
  br i1 %lcmp, label %left_cond_true, label %left_cond_false

left_cond_true:
  %lcmp2 = icmp slt i32 %y, 1000
  br i1 %lcmp2, label %left_cond_true2, label %left_ret

left_cond_true2:
  call void @use(i32 %y)
  %lcmp3 = icmp slt i32 %y, 900
  br i1 %lcmp3, label %left_cond_true3, label %left_ret

left_cond_true3:
  call void @use(i32 %y)
  %lcmp4 = icmp slt i32 %y, 700
  br i1 %lcmp4, label %left_cond_true4, label %left_ret

left_cond_true4:
  call void @use(i32 %y)
  %lcmp5 = icmp slt i32 %y, 700
  br i1 %lcmp5, label %left_cond_true5, label %left_ret

left_cond_true5:
  call void @use(i32 %y)
  %lcmp6 = icmp slt i32 %y, 700
  br i1 %lcmp6, label %left_cond_true6, label %left_ret

left_cond_true6:
  call void @use(i32 %y)
  %lcmp7 = icmp slt i32 %y, 700
  br i1 %lcmp7, label %left_cond_true7, label %left_ret

left_cond_true7:
  ret i32 %y

left_cond_false:
  br label %left_ret

left_ret:
  %lres = phi i32 [ %x, %left_cond_true ], [ %x, %left_cond_false ], [ %x, %left_cond_true2 ], [ %x, %left_cond_true3 ], [ %x, %left_cond_true4 ], [ %x, %left_cond_true5 ], [ %x, %left_cond_true6 ]

  ret i32 %lres

right:
  %rcmp = icmp eq i32 %x, 0
  br i1 %rcmp, label %right_cond_true, label %right_cond_false

right_cond_true:
  br label %right_ret

right_cond_false:
  br label %right_ret

right_ret:
  %rres = phi i32 [ %x, %right_cond_true ], [ %x, %right_cond_false ]
  ret i32 %rres
}
