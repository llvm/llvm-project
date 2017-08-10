; RUN: opt -S -newgvn %s | FileCheck %s

; CHECK: define i32 @main() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp = load i32, i32* @d, align 4
; CHECK-NEXT:   %tmp1 = load i32, i32* @c, align 4
; CHECK-NEXT:   %tobool = icmp eq i32 %tmp1, -1
; CHECK-NEXT:   br i1 %tobool, label %if.end, label %if.then
; CHECK: if.then:
; CHECK-NEXT:   br label %L
; CHECK: L:
; CHECK-NEXT:   %e.0 = phi i32 [ 0, %if.then ], [ %e.1, %if.then4 ]
; CHECK-NEXT:   br label %if.end
; CHECK: if.end:
; CHECK-NEXT:   %e.1 = phi i32 [ %e.0, %L ], [ %tmp, %entry ]
; CHECK-NEXT:   store i32 %e.1, i32* @a, align 4
; CHECK-NEXT:   %tmp2 = load i32, i32* @b, align 4
; CHECK-NEXT:   store i32 0, i32* @b, align 4
; CHECK-NEXT:   %sext = shl i32 %tmp2, 16
; CHECK-NEXT:   %conv1 = ashr exact i32 %sext, 16
; CHECK-NEXT:   %add = add nsw i32 %conv1, %tmp1
; CHECK-NEXT:   %add2 = add nsw i32 %add, %e.1
; CHECK-NEXT:   store i32 %add2, i32* @a, align 4
; CHECK-NEXT:   %tobool3 = icmp eq i32 %add2, 0
; CHECK-NEXT:   br i1 %tobool3, label %if.end5, label %if.then4
; CHECK: if.then4:
; CHECK-NEXT:   br label %L
; CHECK: if.end5:
; CHECK-NEXT:   ret i32 0
; CHECK-NEXT: }

@d = global i32 1, align 4
@c = common global i32 0, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4

define i32 @main() {
entry:
  %tmp = load i32, i32* @d, align 4
  %tmp1 = load i32, i32* @c, align 4
  %tobool = icmp eq i32 %tmp1, -1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %L

L:                                                ; preds = %if.then4, %if.then
  %e.0 = phi i32 [ 0, %if.then ], [ %e.1, %if.then4 ]
  br label %if.end

if.end:                                           ; preds = %L, %entry
  %e.1 = phi i32 [ %e.0, %L ], [ %tmp, %entry ]
  store i32 %e.1, i32* @a, align 4
  %tmp2 = load i32, i32* @b, align 4
  store i32 0, i32* @b, align 4
  %sext = shl i32 %tmp2, 16
  %conv1 = ashr exact i32 %sext, 16
  %tmp3 = load i32, i32* @c, align 4
  %add = add nsw i32 %conv1, %tmp3
  %tmp4 = load i32, i32* @a, align 4
  %and = and i32 %tmp4, %e.1
  %add2 = add nsw i32 %add, %and
  store i32 %add2, i32* @a, align 4
  %tobool3 = icmp eq i32 %add2, 0
  br i1 %tobool3, label %if.end5, label %if.then4

if.then4:                                         ; preds = %if.end
  br label %L

if.end5:                                          ; preds = %if.end
  ret i32 0
}
