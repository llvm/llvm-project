; RUN: opt -passes=newgvn -S < %s | FileCheck %s
@y = external global i32
@z = external global i32

; Function Attrs: nounwind ssp uwtable
define void @foo(i32 %x) {
; CHECK: @foo(i32 %x)
; CHECK: %.pre = load i32, ptr @y
; CHECK: call void @bar(i32 %.pre)

  %t = sub i32 %x, %x
  %.pre = load i32, ptr @y, align 4
  %cmp = icmp sgt i32 %t, 2
  br i1 %cmp, label %if.then, label %entry.if.end_crit_edge

entry.if.end_crit_edge:                           ; preds = %entry
  br label %if.end

if.then:                                          ; preds = %entry
  %add = add nsw i32 %x, 3
  store i32 %add, ptr @y, align 4
  br label %if.end

if.end:                                           ; preds = %entry.if.end_crit_edge, %if.then
  %1 = phi i32 [ %.pre, %entry.if.end_crit_edge ], [ %add, %if.then ]
  tail call void @bar(i32 %1)
  ret void
}

define void @foo2(i32 %x) {
; CHECK: @foo2(i32 %x)
; CHECK: %.pre = load i32, ptr @y
; CHECK: tail call void @bar(i32 %.pre)
entry:
  %t = sub i32 %x, %x
  %.pre = load i32, ptr @y, align 4
  %cmp = icmp sgt i32 %t, 2
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %add = add nsw i32 %x, 3
  store i32 %add, ptr @y, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 1, ptr @z, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %0 = phi i32 [ %.pre, %if.else ], [ %add, %if.then ]
  tail call void @bar(i32 %0)
  ret void
}

declare void @bar(i32)
