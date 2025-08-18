; RUN: opt -passes=inline -inline-threshold=0 -inline-all-viable-calls -S < %s | FileCheck %s

; Check that viable calls that are beyond the cost threshold are still inlined.
define i32 @callee_simple(i32 %x) {
  %1 = add i32 %x, 1
  %2 = mul i32 %1, 2
  %3 = sub i32 %2, 1
  %4 = add i32 %3, 3
  %5 = mul i32 %4, 2
  %6 = sub i32 %5, 2
  %7 = add i32 %6, 1
  ret i32 %7
}

; Check that user decisions are respected.
define i32 @callee_alwaysinline(i32 %x) alwaysinline {
  %sub = sub i32 %x, 3
  ret i32 %sub
}

define i32 @callee_noinline(i32 %x) noinline {
  %div = sdiv i32 %x, 2
  ret i32 %div
}

define i32 @callee_optnone(i32 %x) optnone noinline {
  %rem = srem i32 %x, 2
  ret i32 %rem
}

define i32 @caller(i32 %a) {
; CHECK-LABEL: define i32 @caller(
; CHECK-SAME: i32 [[A:%.*]]) {
; CHECK-NEXT:    [[TMP7:%.*]] = add i32 [[A]], 1
; CHECK-NEXT:    [[TMP8:%.*]] = mul i32 [[TMP7]], 2
; CHECK-NEXT:    [[TMP3:%.*]] = sub i32 [[TMP8]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], 3
; CHECK-NEXT:    [[TMP5:%.*]] = mul i32 [[TMP4]], 2
; CHECK-NEXT:    [[TMP6:%.*]] = sub i32 [[TMP5]], 2
; CHECK-NEXT:    [[ADD_I:%.*]] = add i32 [[TMP6]], 1
; CHECK-NEXT:    [[SUB_I:%.*]] = sub i32 [[ADD_I]], 3
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @callee_noinline(i32 [[SUB_I]])
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @callee_optnone(i32 [[TMP1]])
; CHECK-NEXT:    [[SUM:%.*]] = add i32 [[TMP2]], [[TMP1]]
; CHECK-NEXT:    ret i32 [[SUM]]
;
  %1 = call i32 @callee_simple(i32 %a)
  %2 = call i32 @callee_alwaysinline(i32 %1)
  %3 = call i32 @callee_noinline(i32 %2)
  %4 = call i32 @callee_optnone(i32 %3)
  %sum = add i32 %4, %3
  ret i32 %sum
}

; Check that non-viable calls are not inlined

; Test recursive function is not inlined
define i32 @recursive(i32 %n) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret i32 0

recurse:
  %dec = sub i32 %n, 1
  %rec = call i32 @recursive(i32 %dec)
  %add = add i32 %rec, 1
  ret i32 %add
}

define i32 @call_recursive(i32 %x) {
; CHECK-LABEL: define i32 @call_recursive(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:    [[R:%.*]] = call i32 @recursive(i32 [[X]])
; CHECK-NEXT:    ret i32 [[R]]
;
  %r = call i32 @recursive(i32 %x)
  ret i32 %r
}

; Test indirectbr prevents inlining
define void @has_indirectbr(ptr %ptr, i32 %cond) {
entry:
  switch i32 %cond, label %default [
  i32 0, label %target0
  i32 1, label %target1
  ]

target0:
  br label %end

target1:
  br label %end

default:
  br label %end

end:
  indirectbr ptr %ptr, [label %target0, label %target1]
  ret void
}

define void @call_indirectbr(ptr %p, i32 %c) {
; CHECK-LABEL: define void @call_indirectbr(
; CHECK-SAME: ptr [[P:%.*]], i32 [[C:%.*]]) {
; CHECK-NEXT:    call void @has_indirectbr(ptr [[P]], i32 [[C]])
; CHECK-NEXT:    ret void
;
  call void @has_indirectbr(ptr %p, i32 %c)
  ret void
}

