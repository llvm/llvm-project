; RUN: opt -S -passes=pgo-icall-prom -icp-total-percent-threshold=0 < %s 2>&1 | FileCheck %s

; Test that CallPromotionUtils will promote calls which require pointer cast
; safely, i.e. drop incompatible attributes.

@foo = common global ptr null, align 8

; correct type, preserve attributes
define double @func_double(double %a) {
  ret double poison
}

; drop nofpclass attributes
define i64 @func_i64(i64 %a) {
  ret i64 poison
}

define double @cast_scalar_fp(double %arg) {
  %tmp = load ptr, ptr @foo, align 8

; Make sure callsite attributes are dropped on arguments and retval.
; CHECK: [[ARG:%[0-9]+]] = bitcast double %arg to i64
; CHECK-NEXT: call i64 @func_i64(i64 [[ARG]])

; Make sure callsite attributes are preserved on arguments and retval.
; CHECK: call nofpclass(inf) double @func_double(double nofpclass(nan)

; CHECK: call nofpclass(inf) double %tmp(double nofpclass(nan) %arg)
  %call = call nofpclass(inf) double %tmp(double nofpclass(nan) %arg), !prof !0
  ret double %call
}

; ; correct type, preserve attributes
define [2 x [2 x <2 x double>]] @func_array_vector_f64([2 x [2 x <2 x double>]] %a) {
  ret [2 x [2 x <2 x double>]] poison
}

; drop nofpclass attributes
define [2 x [2 x <2 x i64>]] @func_array_vector_i64([2 x [2 x <2 x i64>]] %a) {
  ret [2 x [2 x <2 x i64>]] poison
}

; FIXME: This is not promoted
; CHECK: %call = call nofpclass(inf) [2 x [2 x <2 x double>]] %tmp([2 x [2 x <2 x double>]] nofpclass(nan) %arg)
define [2 x [2 x <2 x double>]] @cast_array_vector([2 x [2 x <2 x double>]] %arg) {
  %tmp = load ptr, ptr @foo, align 8
  %call = call nofpclass(inf) [2 x [2 x <2 x double>]] %tmp([2 x [2 x <2 x double>]] nofpclass(nan) %arg), !prof !1
  ret [2 x [2 x <2 x double>]] %call
}

!0 = !{!"VP", i32 0, i64 1440, i64 15573779287943805696, i64 1030, i64 16900752280434761561, i64 410}
!1 = !{!"VP", i32 0, i64 1440, i64 1124945363680759394, i64 1030, i64 16341336592352938424, i64 410}
