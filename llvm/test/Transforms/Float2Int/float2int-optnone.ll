; RUN: opt < %s -passes=float2int -S | FileCheck %s
;
; Verify that pass float2int is not run on optnone functions.

define i16 @simple1(i8 %a) #0 {
; CHECK-LABEL: @simple1
; CHECK:  %1 = uitofp i8 %a to float
; CHECK-NEXT:  %2 = fadd float %1, 1.0
; CHECK-NEXT:  %3 = fptoui float %2 to i16
; CHECK-NEXT:  ret i16 %3
  %1 = uitofp i8 %a to float
  %2 = fadd float %1, 1.0
  %3 = fptoui float %2 to i16
  ret i16 %3
}

attributes #0 = { noinline optnone }
