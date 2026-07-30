; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs < %s | FileCheck %s

define i32 @relax_bcc(i1 %b) {
; CHECK-LABEL: relax_bcc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    andi. 3, 3, 1
; CHECK-NEXT:    #APP
; CHECK-NEXT:  label:
; CHECK-NEXT:    add 3, 3, 5
; CHECK-NEXT:    cmpd    4, 3
; CHECK-NEXT:    bne     0, label
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    bc 12, 1, .+8
; CHECK-NEXT:    b .LBB0_4
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    li 3, 101
; CHECK-NEXT:    mtctr 3
; CHECK-NEXT:    .p2align        4
; CHECK-NEXT:  .LBB0_2:
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    bdnz .LBB0_2
; CHECK-NEXT:  # %bb.3:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    .space 32748
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB0_4: # %tail
; CHECK-NEXT:    li 3, 1
; CHECK-NEXT:    blr
entry:
  call void asm sideeffect "label:\0A\09add 3,3,5\0A\09cmpd 4,3\0A\09bne label", ""()
  br i1 %b, label %for.body, label %tail

for.body:                                         ; preds = %for.body, %entry
   %0 = phi i32 [0, %entry], [%1, %for.body]
   %1 = add i32 %0, 1
   %2 = icmp sgt i32 %1, 100
   br i1 %2, label %exit, label %for.body

exit:
  call void asm sideeffect ".space 32748", ""()
  br label %tail

tail:
  ret i32 1
}
