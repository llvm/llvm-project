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

@a = dso_local global double 0.0, align 4
@b = dso_local global double 0.0, align 4
@c = dso_local global double 0.0, align 4

define dso_local i32 @bar(ptr %vptr, i32 %iv1, ptr %iptr) minsize {
  
  %vp2 = getelementptr double, ptr %vptr, i32 1
  %vp3 = getelementptr double, ptr %vptr, i32 2

  %v1 = load double, ptr %vptr, align 8
  %v2 = load double, ptr %vp2, align 8
  %v3 = load double, ptr %vp3, align 8

  store double %v1, ptr @a, align 8
  store double %v2, ptr @b, align 8
  store double %v3, ptr @c, align 8

  %ptr_after = getelementptr double, ptr @a, i32 3

  %ptr_new_ival = ptrtoint ptr %ptr_after to i32
  %ptr_new = inttoptr i32 %ptr_new_ival to ptr

  store i32 %ptr_new_ival, ptr %iptr, align 8

  %mul1 = mul i32 %ptr_new_ival, %iv1

  ret i32 %mul1
}

