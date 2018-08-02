; REQUIRES: asserts
; RUN: opt < %s -div-rem-pairs -debug-counter=div-rem-pairs-transform-skip=1,div-rem-pairs-transform-count=1 \
; RUN: -S -mtriple=x86_64-unknown-unknown    | FileCheck %s
;; Test that, with debug counters on, we only skip the first div-rem-pairs opportunity, optimize one after it,
;; and then ignore all the others. There is 1 optimization opportunity in f1, 2 in f2, and another 1 in f3,
;; only the first one in f2 will be performed.

define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: @f1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = urem i64 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[DIV:%.*]] = udiv i64 %a, %b
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i64 [ [[DIV]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i64 [[RET]]
;
entry:
  %rem = urem i64 %a, %b
  %cmp = icmp eq i64 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %div = udiv i64 %a, %b
  br label %end

end:
  %ret = phi i64 [ %div, %if ], [ 3, %entry ]
  ret i64 %ret
}

define i16 @f2(i16 %a, i16 %b) {
; CHECK-LABEL: @f2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DIV1:%.*]] = sdiv i16 %a, %b
; CHECK-NEXT:    [[REM1:%.*]] = srem i16 %a, %b
; CHECK-NEXT:    [[DIV2:%.*]] = udiv i16 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[DIV1]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[REM2:%.*]] = urem i16 %a, %b
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i16 [ [[REM1]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i16 [[RET]]
;
entry:
  %div1 = sdiv i16 %a, %b
  %div2 = udiv i16 %a, %b
  %cmp = icmp eq i16 %div1, 42
  br i1 %cmp, label %if, label %end

if:
  %rem1 = srem i16 %a, %b
  %rem2 = urem i16 %a, %b
  br label %end

end:
  %ret = phi i16 [ %rem1, %if ], [ 3, %entry ]
  ret i16 %ret
}

define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: @f3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i32 [[RET]]
;
entry:
  %rem = srem i32 %a, %b
  %cmp = icmp eq i32 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %div = sdiv i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ 3, %entry ]
  ret i32 %ret
}