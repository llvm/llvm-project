; RUN: opt -passes=nsan -mtriple=s390x-linux-gnu -S %s -o - | FileCheck %s
; REQUIRES: systemz-registered-target
;
; Check that NSan emits signext/zeroext on the narrow integer arguments and
; return values of its runtime calls.

; CHECK: declare signext i32 @__nsan_internal_check_float_d(float, double, i32 signext, i64)
; CHECK: declare void @__nsan_fcmp_fail_float_d(float, float, double, double, i32 signext, i1 zeroext, i1 zeroext)

; CHECK: declare signext i32 @__nsan_internal_check_double_q(double, fp128, i32 signext, i64)
; CHECK: declare void @__nsan_fcmp_fail_double_q(double, double, fp128, fp128, i32 signext, i1 zeroext, i1 zeroext)
