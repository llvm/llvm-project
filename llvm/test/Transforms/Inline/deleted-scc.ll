; RUN: opt -passes=inline < %s -S | FileCheck %s

; CHECK-NOT: define
; CHECK: define void @e()
; CHECK-NOT: define

@b = external local_unnamed_addr global i32, align 4

define void @e() local_unnamed_addr {
entry:
  call fastcc void @d()
  ret void
}

define internal fastcc void @f() unnamed_addr {
entry:
  call fastcc void @d()
  ret void
}

define internal fastcc void @d() unnamed_addr {
entry:
  br label %L

L:                                                ; preds = %cleanup9, %entry
  %cleanup.dest.slot.0 = phi i32 [ undef, %entry ], [ %cleanup.dest.slot.3, %cleanup9 ]
  store i32 0, ptr @b, align 4
  %tobool.not = icmp eq i32 0, 0
  br i1 %tobool.not, label %if.then, label %while.cond

while.cond:                                       ; preds = %cleanup9, %L
  %cleanup.dest.slot.2 = phi i32 [ %cleanup.dest.slot.0, %L ], [ 0, %cleanup9 ]
  %0 = load i32, ptr @b, align 4
  %tobool3.not = icmp eq i32 %0, 0
  br i1 %tobool3.not, label %cleanup9, label %while.body4

while.body4:                                      ; preds = %while.cond
  call fastcc void @f()
  br label %cleanup9

cleanup9:                                         ; preds = %while.cond, %while.body4
  %cleanup.dest.slot.3 = phi i32 [ %cleanup.dest.slot.2, %while.body4 ], [ 0, %while.cond ]
  switch i32 %cleanup.dest.slot.3, label %common.ret [
    i32 0, label %while.cond
    i32 2, label %L
  ]

common.ret:                                       ; preds = %cleanup9, %if.then
  ret void

if.then:                                          ; preds = %L
  call void @e()
  br label %common.ret
}
