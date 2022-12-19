; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -mattr=use-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; 

@a = dso_local global double 0.0, align 4
@b = dso_local global double 0.0, align 4
@c = dso_local global double 0.0, align 4

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have VLDM instruction combined from single-loads
; CHECK:       ********** MI Scheduling **********
; CHECK:       VLDMDIA_UPD
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 6
; CHECK:       Successors:
; CHECK:       Data
; CHECK-SAME:  Latency=1
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=1
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=5
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=0
; CHECK-NEXT:  Data
; CHECK-SAME:  Latency=0
define dso_local i32 @bar(ptr %iptr) minsize optsize {
  %1 = load double, ptr @a, align 8
  %2 = load double, ptr @b, align 8
  %3 = load double, ptr @c, align 8

  %ptr_after = getelementptr double, ptr @a, i32 3

  %ptr_new_ival = ptrtoint ptr %ptr_after to i32
  %ptr_new = inttoptr i32 %ptr_new_ival to ptr

  store i32 %ptr_new_ival, ptr %iptr, align 8
  
  %v1 = fptoui double %1 to i32

  %mul1 = mul i32 %ptr_new_ival, %v1

  %v2 = fptoui double %2 to i32
  %v3 = fptoui double %3 to i32
  
  %mul2 = mul i32 %mul1, %v2
  %mul3 = mul i32 %mul2, %v3
  
  ret i32 %mul3
}

