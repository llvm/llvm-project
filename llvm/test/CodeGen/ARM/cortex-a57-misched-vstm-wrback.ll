; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -mattr=use-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have VSTM instruction combined from single-stores
; CHECK:       ********** MI Scheduling **********
; CHECK:       schedule starting
; CHECK:       VSTMDIA_UPD
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 4
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=1

@a = global double 0.0, align 4
@b = global double 0.0, align 4
@c = global double 0.0, align 4

define i32 @bar(double* %vptr, i32 %iv1, i32* %iptr) minsize {
  
  %vp2 = getelementptr double, double* %vptr, i32 1
  %vp3 = getelementptr double, double* %vptr, i32 2

  %v1 = load double, double* %vptr, align 8
  %v2 = load double, double* %vp2, align 8
  %v3 = load double, double* %vp3, align 8

  store double %v1, double* @a, align 8
  store double %v2, double* @b, align 8
  store double %v3, double* @c, align 8

  %ptr_after = getelementptr double, double* @a, i32 3

  %ptr_new_ival = ptrtoint double* %ptr_after to i32
  %ptr_new = inttoptr i32 %ptr_new_ival to i32*

  store i32 %ptr_new_ival, i32* %iptr, align 8

  %mul1 = mul i32 %ptr_new_ival, %iv1

  ret i32 %mul1
}

