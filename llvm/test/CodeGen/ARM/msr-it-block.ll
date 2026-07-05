; RUN: llc < %s -mtriple=thumbv6m-none-eabi | FileCheck %s --check-prefix=V6M --check-prefix=CHECK
; RUN: llc < %s -mtriple=thumbv7m-none-eabi | FileCheck %s --check-prefix=V7M --check-prefix=CHECK
; RUN: llc < %s -mtriple=thumbv7a-none-eabi | FileCheck %s --check-prefix=V7A --check-prefix=CHECK
; RUN: llc < %s -mtriple=armv7a-none-eabi   | FileCheck %s --check-prefix=V7A --check-prefix=CHECK


target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7a-arm-none-eabi"

define void @test_const(i32 %val) {
; CHECK-LABEL: test_const:
entry:
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %write_reg, label %exit

write_reg:
  tail call void @llvm.write_register.i32(metadata !0, i32 0)
  tail call void @llvm.write_register.i32(metadata !0, i32 0)
; V6M: msr     apsr, {{r[0-9]+}}
; V6M: msr     apsr, {{r[0-9]+}}
; V7M: msr     apsr_nzcvq, {{r[0-9]+}}
; V7M: msr     apsr_nzcvq, {{r[0-9]+}}
; V7A: msr     APSR_nzcvq, {{r[0-9]+}}
; V7A: msr     APSR_nzcvq, {{r[0-9]+}}
  br label %exit

exit:
  ret void
}

define void @test_var(i32 %val, i32 %apsr) {
; CHECK-LABEL: test_var:
entry:
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %write_reg, label %exit

write_reg:
  tail call void @llvm.write_register.i32(metadata !0, i32 %apsr)
  tail call void @llvm.write_register.i32(metadata !0, i32 %apsr)
; V6M: msr     apsr, {{r[0-9]+}}
; V6M: msr     apsr, {{r[0-9]+}}
; V7M: msr     apsr_nzcvq, {{r[0-9]+}}
; V7M: msr     apsr_nzcvq, {{r[0-9]+}}
; V7A: msr     APSR_nzcvq, {{r[0-9]+}}
; V7A: msr     APSR_nzcvq, {{r[0-9]+}}
  br label %exit

exit:
  ret void
}


declare void @llvm.write_register.i32(metadata, i32)

!0 = !{!"apsr"}
