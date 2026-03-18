; RUN: opt < %s -passes="slsr" -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

%struct.B = type { i16 }
%struct.A = type { %struct.B, %struct.B }

define i32 @pick(i32 %0, ptr %addr) {
; `d` can be optimized by 2 approaches
; 1. a = 1 + 1 * %0
;    d = 1 + 8 * %0
;      = a + 7 * %0
; 2. c = (8 * %0) + 3
;    d = (8 * %0) + 1
;      = c - 2
; Pick candidate (2) as it can save 1 instruction from (7 * %0)
;
; CHECK-LABEL: pick
; CHECK: [[A:%.*]] = add i32 %0, 1
; CHECK: [[B:%.*]] = shl i32 %0, 3
; CHECK: [[C:%.*]] = add i32 [[B]], 3
; CHECK: store i32 [[C]], ptr %addr
; CHECK: [[D:%.*]] = add i32 [[C]], -2
; CHECK: ret i32 %d

  %a = add i32 %0, 1
  %b = shl i32 %0, 3
  %c = add i32 %b, 3
  store i32 %c, ptr %addr
  %d = add i32 %b, 1
  ret i32 %d
}
