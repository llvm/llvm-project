; RUN: opt < %s -slsr -gvn -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

define void @shl(i32 %b, i32 %s) {
; CHECK-LABEL: @shl(
  %1 = add i32 %b, %s
; [[BASIS:%[a-zA-Z0-9]+]] = add i32 %b, %s
  call void @foo(i32 %1)
  %s2 = shl i32 %s, 1
  %2 = add i32 %b, %s2
; add i32 [[BASIS]], %s
  call void @foo(i32 %2)
  ret void
}

define void @stride_is_2s(i32 %b, i32 %s) {
; CHECK-LABEL: @stride_is_2s(
  %s2 = shl i32 %s, 1
; CHECK: %s2 = shl i32 %s, 1
  %1 = add i32 %b, %s2
; CHECK: [[t1:%[a-zA-Z0-9]+]] = add i32 %b, %s2
  call void @foo(i32 %1)
  %s4 = shl i32 %s, 2
  %2 = add i32 %b, %s4
; CHECK: [[t2:%[a-zA-Z0-9]+]] = add i32 [[t1]], %s2
  call void @foo(i32 %2)
  %s6 = mul i32 %s, 6
  %3 = add i32 %b, %s6
; CHECK: add i32 [[t2]], %s2
  call void @foo(i32 %3)
  ret void
}

define void @stride_is_3s(i32 %b, i32 %s) {
; CHECK-LABEL: @stride_is_3s(
  %1 = add i32 %s, %b
; CHECK: [[t1:%[a-zA-Z0-9]+]] = add i32 %s, %b
  call void @foo(i32 %1)
  %s4 = shl i32 %s, 2
  %2 = add i32 %s4, %b
; CHECK: [[bump:%[a-zA-Z0-9]+]] = mul i32 %s, 3
; CHECK: [[t2:%[a-zA-Z0-9]+]] = add i32 [[t1]], [[bump]]
  call void @foo(i32 %2)
  %s7 = mul i32 %s, 7
  %3 = add i32 %s7, %b
; CHECK: add i32 [[t2]], [[bump]]
  call void @foo(i32 %3)
  ret void
}

; foo(b + 6 * s);
; foo(b + 4 * s);
; foo(b + 2 * s);
;   =>
; t1 = b + 6 * s;
; foo(t1);
; s2 = 2 * s;
; t2 = t1 - s2;
; foo(t2);
; t3 = t2 - s2;
; foo(t3);
define void @stride_is_minus_2s(i32 %b, i32 %s) {
; CHECK-LABEL: @stride_is_minus_2s(
  %s6 = mul i32 %s, 6
  %1 = add i32 %b, %s6
; CHECK: [[t1:%[a-zA-Z0-9]+]] = add i32 %b, %s6
; CHECK: call void @foo(i32 [[t1]])
  call void @foo(i32 %1)
  %s4 = shl i32 %s, 2
  %2 = add i32 %b, %s4
; CHECK: [[bump:%[a-zA-Z0-9]+]] = shl i32 %s, 1
; CHECK: [[t2:%[a-zA-Z0-9]+]] = sub i32 [[t1]], [[bump]]
  call void @foo(i32 %2)
; CHECK: call void @foo(i32 [[t2]])
  %s2 = shl i32 %s, 1
  %3 = add i32 %b, %s2
; CHECK: [[t3:%[a-zA-Z0-9]+]] = sub i32 [[t2]], [[bump]]
  call void @foo(i32 %3)
; CHECK: call void @foo(i32 [[t3]])
  ret void
}

; t = b + (s << 3);
; foo(t);
; foo(b + s);
;
; do not rewrite b + s to t - 7 * s because the latter is more complicated.
define void @simple_enough(i32 %b, i32 %s) {
; CHECK-LABEL: @simple_enough(
  %s8 = shl i32 %s, 3
  %1 = add i32 %b, %s8
  call void @foo(i32 %1)
  %2 = add i32 %b, %s
; CHECK: [[t:%[a-zA-Z0-9]+]] = add i32 %b, %s{{$}}
  call void @foo(i32 %2)
; CHECK: call void @foo(i32 [[t]])
  ret void
}

define void @slsr_strided_add_128bit(i128 %b, i128 %s) {
; CHECK-LABEL: @slsr_strided_add_128bit(
  %s125 = shl i128 %s, 125
  %s126 = shl i128 %s, 126
  %1 = add i128 %b, %s125
; CHECK: [[t1:%[a-zA-Z0-9]+]] = add i128 %b, %s125
  call void @bar(i128 %1)
  %2 = add i128 %b, %s126
; CHECK: [[t2:%[a-zA-Z0-9]+]] = add i128 [[t1]], %s125
  call void @bar(i128 %2)
; CHECK: call void @bar(i128 [[t2]])
  ret void
}

declare void @foo(i32)
declare void @bar(i128)
