; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; The operand-swapping code in HexagonPeephole was not handling subregisters
; correctly, resulting in a crash on this code.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@float_rounding_mode = external global i8, align 1
@float_exception_flags = external global i8, align 1

; Function Attrs: nounwind
define i64 @fred(i32 %a) #0 {
entry:
  br i1 undef, label %cleanup, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp3 = icmp eq i32 undef, 255
  %tobool4 = icmp ne i32 undef, 0
  %or.cond = and i1 %tobool4, %cmp3
  %. = select i1 %or.cond, i64 9223372036854775807, i64 -9223372036854775808
  br label %cleanup

cleanup:                                          ; preds = %lor.lhs.false, %entry
  %retval.0 = phi i64 [ 9223372036854775807, %entry ], [ %., %lor.lhs.false ]
  ret i64 %retval.0
}

attributes #0 = { nounwind }
