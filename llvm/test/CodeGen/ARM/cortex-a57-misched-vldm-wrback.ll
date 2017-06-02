; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -misched-postra -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; 

@a = global double 0.0, align 4
@b = global double 0.0, align 4
@c = global double 0.0, align 4

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have VLDM instruction combined from single-loads
; CHECK:       ********** MI Scheduling **********
; CHECK:       VLDMDIA_UPD
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 6
; CHECK:       Successors:
; CHECK:       data
; CHECK-SAME:  Latency=1
; CHECK-NEXT:  data
; CHECK-SAME:  Latency=1
; CHECK-NEXT:  data
; CHECK-SAME:  Latency=5
; CHECK-NEXT:  data 
; CHECK-SAME:  Latency=5
; CHECK-NEXT:  data 
; CHECK-SAME:  Latency=6
define i32 @bar(i32* %iptr) minsize optsize {
  %1 = load double, double* @a, align 8
  %2 = load double, double* @b, align 8
  %3 = load double, double* @c, align 8

  %ptr_after = getelementptr double, double* @a, i32 3

  %ptr_new_ival = ptrtoint double* %ptr_after to i32
  %ptr_new = inttoptr i32 %ptr_new_ival to i32*

  store i32 %ptr_new_ival, i32* %iptr, align 8
  
  %v1 = fptoui double %1 to i32

  %mul1 = mul i32 %ptr_new_ival, %v1

  %v2 = fptoui double %2 to i32
  %v3 = fptoui double %3 to i32
  
  %mul2 = mul i32 %mul1, %v2
  %mul3 = mul i32 %mul2, %v3
  
  ret i32 %mul3
}

