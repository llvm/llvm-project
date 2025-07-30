; RUN: llc -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mtriple=powerpc-ibm-aix -mcpu=pwr7 < %s | FileCheck %s

define i64 @foo(i32 noundef %argc) #0 {
entry:
  %argc.addr = alloca i32, align 4
  %num = alloca i64, align 8
  store i32 %argc, ptr %argc.addr, align 4
  %0 = load i32, ptr %argc.addr, align 4
  %sub = sub nsw i32 %0, 2
  %conv = sext i32 %sub to i64
  store i64 %conv, ptr %num, align 8
  %1 = load i64, ptr %num, align 8
  %sub1 = sub nsw i64 0, %1
  ret i64 %sub1
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all"  "stack-protector-buffer-size"="8" }

; CHECK:      .foo:
; CHECK-NEXT: # %bb.0:                                # %entry
; CHECK-NEXT:   stw r3, -8(r1)
; CHECK-NEXT:   lwz r3, -8(r1)
; CHECK-NEXT:   addi r3, r3, -2
; CHECK-NEXT:   srawi r4, r3, 31
; CHECK-NEXT:   stw r3, -12(r1)
; CHECK-NEXT:   stw r4, -16(r1)
; CHECK-NEXT:   lwz r3, -16(r1)
; CHECK-NEXT:   lwz r4, -12(r1)
; CHECK-NEXT:   subfic r4, r4, 0
; CHECK-NEXT:   subfze r3, r3
; CHECK-NEXT:   blr
