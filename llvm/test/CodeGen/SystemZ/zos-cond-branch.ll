; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z13 | FileCheck %s

; Test that SystemZHLASMInstPrinter prints BC via BCAsm with numerical mask
; when condition code 3 is possible (CCValid & 1 != 0) and the condition mask
; has no valid HLASM extended mnemonic (e.g. mask 9 from fcmp one).
define void @test_bc_numeric_mask(float %a, float %b, ptr %p) {
; CHECK-LABEL: test_bc_numeric_mask DS 0H
; CHECK:       cebr  0,2
; CHECK-NEXT:  bc    9,2(7)
; CHECK:       mvhi  0(3),1
; CHECK-NEXT:  b     2(7)
entry:
  %cmp = fcmp one float %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 1, ptr %p
  br label %if.end

if.end:
  ret void
}

; Test that SystemZHLASMInstPrinter prints BRC via BRCAsm with numerical mask
; when condition code 3 is possible (CCValid & 1 != 0) and the condition mask
; has no valid HLASM extended mnemonic (e.g. mask 9 from fcmp one).
define void @test_brc_numeric_mask(float %a, float %b) {
; CHECK-LABEL: test_brc_numeric_mask DS 0H
; CHECK:       cebr  0,2
; CHECK-NEXT:  brc   9,[[LABEL:L#BB[0-9_]+]]
; CHECK:       basr  7,6
; CHECK:       [[LABEL]] DS 0H
entry:
  %cmp = fcmp one float %a, %b
  br i1 %cmp, label %bb1, label %bb2

bb1:
  call void @foo()
  br label %bb2

bb2:
  ret void
}

declare void @foo()
