; RUN: true
; Constants for edge case float testing.

; Generated with:
; clang -S -emit-llvm floating-point-constants.c
;
; // floating-point-constants.c
; #include <math.h>
; #include <float.h>
;
; const float flt_nan = NAN;
; const float flt_pos_min_subnormal = FLT_TRUE_MIN;
; const float flt_pos_min_normal = FLT_MIN;
; const float flt_pos_max = FLT_MAX;
; const float flt_pos_infinity = INFINITY;
; const float flt_neg_min_subnormal = -FLT_TRUE_MIN;
; const float flt_neg_min_normal = -FLT_MIN;
; const float flt_neg_max = -FLT_MAX;
; const float flt_neg_infinity = -INFINITY;
;
; const double dbl_nan = NAN;
; const double dbl_pos_min_subnormal = DBL_TRUE_MIN;
; const double dbl_pos_min_normal = DBL_MIN;
; const double dbl_pos_max = DBL_MAX;
; const double dbl_pos_infinity = INFINITY;
; const double dbl_neg_min_subnormal = -DBL_TRUE_MIN;
; const double dbl_neg_min_normal = -DBL_MIN;
; const double dbl_neg_max = -DBL_MAX;
; const double dbl_neg_infinity = -INFINITY;

@flt_nan = dso_local constant float 0x7FF8000000000000, align 4
@flt_pos_min_subnormal = dso_local constant float 0x36A0000000000000, align 4
@flt_pos_min_normal = dso_local constant float 0x3810000000000000, align 4
@flt_pos_max = dso_local constant float 0x47EFFFFFE0000000, align 4
@flt_pos_infinity = dso_local constant float 0x7FF0000000000000, align 4
@flt_neg_min_subnormal = dso_local constant float 0xB6A0000000000000, align 4
@flt_neg_min_normal = dso_local constant float 0xB810000000000000, align 4
@flt_neg_max = dso_local constant float 0xC7EFFFFFE0000000, align 4
@flt_neg_infinity = dso_local constant float 0xFFF0000000000000, align 4
@dbl_nan = dso_local constant double 0x7FF8000000000000, align 8
@dbl_pos_min_subnormal = dso_local constant double 4.940660e-324, align 8
@dbl_pos_min_normal = dso_local constant double 0x10000000000000, align 8
@dbl_pos_max = dso_local constant double 0x7FEFFFFFFFFFFFFF, align 8
@dbl_pos_infinity = dso_local constant double 0x7FF0000000000000, align 8
@dbl_neg_min_subnormal = dso_local constant double -4.940660e-324, align 8
@dbl_neg_min_normal = dso_local constant double 0x8010000000000000, align 8
@dbl_neg_max = dso_local constant double 0xFFEFFFFFFFFFFFFF, align 8
@dbl_neg_infinity = dso_local constant double 0xFFF0000000000000, align 8
