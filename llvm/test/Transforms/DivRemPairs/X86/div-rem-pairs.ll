; RUN: opt < %s -div-rem-pairs -S -mtriple=x86_64-unknown-unknown    | FileCheck %s

declare void @foo(i32, i32)

define void @decompose_illegal_srem_same_block(i32 %a, i32 %b) {
; CHECK-LABEL: @decompose_illegal_srem_same_block(
; CHECK-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; CHECK-NEXT:    call void @foo(i32 [[REM]], i32 [[DIV]])
; CHECK-NEXT:    ret void
;
  %rem = srem i32 %a, %b
  %div = sdiv i32 %a, %b
  call void @foo(i32 %rem, i32 %div)
  ret void
}

define void @decompose_illegal_urem_same_block(i32 %a, i32 %b) {
; CHECK-LABEL: @decompose_illegal_urem_same_block(
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 %a, %b
; CHECK-NEXT:    [[REM:%.*]] = urem i32 %a, %b
; CHECK-NEXT:    call void @foo(i32 [[REM]], i32 [[DIV]])
; CHECK-NEXT:    ret void
;
  %div = udiv i32 %a, %b
  %rem = urem i32 %a, %b
  call void @foo(i32 %rem, i32 %div)
  ret void
}

; Hoist and optionally decompose the sdiv because it's safe and free.
; PR31028 - https://bugs.llvm.org/show_bug.cgi?id=31028

define i32 @hoist_sdiv(i32 %a, i32 %b) {
; CHECK-LABEL: @hoist_sdiv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
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

; Hoist and optionally decompose the udiv because it's safe and free.

define i64 @hoist_udiv(i64 %a, i64 %b) {
; CHECK-LABEL: @hoist_udiv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = urem i64 %a, %b
; CHECK-NEXT:    [[DIV:%.*]] = udiv i64 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
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

; Hoist the srem if it's safe and free, otherwise decompose it.

define i16 @hoist_srem(i16 %a, i16 %b) {
; CHECK-LABEL: @hoist_srem(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i16 %a, %b
; CHECK-NEXT:    [[REM:%.*]] = srem i16 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[DIV]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i16 [ [[REM]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i16 [[RET]]
;
entry:
  %div = sdiv i16 %a, %b
  %cmp = icmp eq i16 %div, 42
  br i1 %cmp, label %if, label %end

if:
  %rem = srem i16 %a, %b
  br label %end

end:
  %ret = phi i16 [ %rem, %if ], [ 3, %entry ]
  ret i16 %ret
}

; Hoist the urem if it's safe and free, otherwise decompose it.

define i8 @hoist_urem(i8 %a, i8 %b) {
; CHECK-LABEL: @hoist_urem(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DIV:%.*]] = udiv i8 %a, %b
; CHECK-NEXT:    [[REM:%.*]] = urem i8 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[DIV]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i8 [ [[REM]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i8 [[RET]]
;
entry:
  %div = udiv i8 %a, %b
  %cmp = icmp eq i8 %div, 42
  br i1 %cmp, label %if, label %end

if:
  %rem = urem i8 %a, %b
  br label %end

end:
  %ret = phi i8 [ %rem, %if ], [ 3, %entry ]
  ret i8 %ret
}

; If the ops don't match, don't do anything: signedness.

define i32 @dont_hoist_udiv(i32 %a, i32 %b) {
; CHECK-LABEL: @dont_hoist_udiv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[DIV:%.*]] = udiv i32 %a, %b
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
  %div = udiv i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ 3, %entry ]
  ret i32 %ret
}

; If the ops don't match, don't do anything: operation.

define i32 @dont_hoist_srem(i32 %a, i32 %b) {
; CHECK-LABEL: @dont_hoist_srem(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = urem i32 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[REM2:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i32 [ [[REM2]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i32 [[RET]]
;
entry:
  %rem = urem i32 %a, %b
  %cmp = icmp eq i32 %rem, 42
  br i1 %cmp, label %if, label %end

if:
  %rem2 = srem i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %rem2, %if ], [ 3, %entry ]
  ret i32 %ret
}

; If the ops don't match, don't do anything: operands.

define i32 @dont_hoist_sdiv(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @dont_hoist_sdiv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[REM]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %c
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
  %div = sdiv i32 %a, %c
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ 3, %entry ]
  ret i32 %ret
}

; If the target doesn't have a unified div/rem op for the type, decompose rem in-place to mul+sub.

define i128 @dont_hoist_urem(i128 %a, i128 %b) {
; CHECK-LABEL: @dont_hoist_urem(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DIV:%.*]] = udiv i128 %a, %b
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i128 [[DIV]], 42
; CHECK-NEXT:    br i1 [[CMP]], label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[TMP0:%.*]] = mul i128 [[DIV]], %b
; CHECK-NEXT:    [[TMP1:%.*]] = sub i128 %a, [[TMP0]]
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i128 [ [[TMP1]], %if ], [ 3, %entry ]
; CHECK-NEXT:    ret i128 [[RET]]
;
entry:
  %div = udiv i128 %a, %b
  %cmp = icmp eq i128 %div, 42
  br i1 %cmp, label %if, label %end

if:
  %rem = urem i128 %a, %b
  br label %end

end:
  %ret = phi i128 [ %rem, %if ], [ 3, %entry ]
  ret i128 %ret
}

; We don't hoist if one op does not dominate the other,
; but we could hoist both ops to the common predecessor block?

define i32 @no_domination(i1 %cmp, i32 %a, i32 %b) {
; CHECK-LABEL: @no_domination(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cmp, label %if, label %else
; CHECK:       if:
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i32 %a, %b
; CHECK-NEXT:    br label %end
; CHECK:       else:
; CHECK-NEXT:    [[REM:%.*]] = srem i32 %a, %b
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[RET:%.*]] = phi i32 [ [[DIV]], %if ], [ [[REM]], %else ]
; CHECK-NEXT:    ret i32 [[RET]]
;
entry:
  br i1 %cmp, label %if, label %else

if:
  %div = sdiv i32 %a, %b
  br label %end

else:
  %rem = srem i32 %a, %b
  br label %end

end:
  %ret = phi i32 [ %div, %if ], [ %rem, %else ]
  ret i32 %ret
}

