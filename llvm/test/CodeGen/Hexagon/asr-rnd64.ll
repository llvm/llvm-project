; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Check if we generate rounding-asr instruction.  It is equivalent to
; Rd = ((Rs >> #u) +1) >> 1.

target triple = "hexagon"

define i32 @f0(i32 %a0) {
b0:
; CHECK: asr{{.*}}:rnd
  %v0 = alloca i32, align 4
  store i32 %a0, ptr %v0, align 4
  %v1 = load i32, ptr %v0, align 4
  %v2 = ashr i32 %v1, 10
  %v3 = add nsw i32 %v2, 1
  %v4 = ashr i32 %v3, 1
  ret i32 %v4
}

define i64 @f1(i64 %a0) {
b0:
; CHECK: asr{{.*}}:rnd
  %v0 = alloca i64, align 8
  store i64 %a0, ptr %v0, align 8
  %v1 = load i64, ptr %v0, align 8
  %v2 = ashr i64 %v1, 17
  %v3 = add nsw i64 %v2, 1
  %v4 = ashr i64 %v3, 1
  ret i64 %v4
}
