; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; Unused pointer argument should be removed from the function's signature
; while the used arguments should be promoted if they are pointers.
; The pass should not touch any unused non-pointer arguments.
define internal i32 @callee(i1 %c, i1 %d, i32* %used, i32* %unused) nounwind {
; CHECK-LABEL: define {{[^@]+}}@callee
; CHECK-SAME: (i1 [[C:%.*]], i1 [[D:%.*]], i32 [[USED_VAL:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C]], label %if, label %else
; CHECK:       if:
; CHECK-NEXT:    ret i32 [[USED_VAL]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 -1
;
entry:
  %x = load i32, i32* %used, align 4
  br i1 %c, label %if, label %else

if:
  ret i32 %x

else:
  ret i32 -1
}

; Unused byval argument should be removed from the function's signature
; while the used arguments should be promoted if they are pointers.
; The pass should not touch any unused non-pointer arguments.
define internal i32 @callee_byval(i1 %c, i1 %d, i32* byval(i32) align 4 %used, i32* byval(i32) align 4 %unused) nounwind {
; CHECK-LABEL: define {{[^@]+}}@callee_byval
; CHECK-SAME: (i1 [[C:%.*]], i1 [[D:%.*]], i32 [[USED_VAL:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[C]], label %if, label %else
; CHECK:       if:
; CHECK-NEXT:    ret i32 [[USED_VAL]]
; CHECK:       else:
; CHECK-NEXT:    ret i32 -1
;
entry:
  %x = load i32, i32* %used, align 4
  br i1 %c, label %if, label %else

if:
  ret i32 %x

else:
  ret i32 -1
}

define i32 @caller(i1 %c, i1 %d, i32* %arg) nounwind {
; CHECK-LABEL: define {{[^@]+}}@caller
; CHECK-SAME: (i1 [[C:%.*]], i1 [[D:%.*]], i32* [[ARG:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARG_VAL_0:%.*]] = load i32, i32* [[ARG]], align 4
; CHECK-NEXT:    [[RES_0:%.*]] = call i32 @callee_byval(i1 [[C]], i1 [[D]], i32 [[ARG_VAL_0]]) #[[ATTR0]]
; CHECK-NEXT:    [[ARG_VAL_1:%.*]] = load i32, i32* [[ARG]], align 4
; CHECK-NEXT:    [[RES_1:%.*]] = call i32 @callee(i1 [[C]], i1 [[D]], i32 [[ARG_VAL_1]]) #[[ATTR0]]
; CHECK-NEXT:  ret i32 1
;
entry:
  call i32 @callee_byval(i1 %c, i1 %d, i32* byval(i32) align 4 %arg, i32* byval(i32) align 4 %arg) nounwind
  call i32 @callee(i1 %c, i1 %d, i32* %arg, i32* %arg) nounwind
  ret i32 1
}
