; RUN: llc < %s -mtriple=thumb-eabi   | FileCheck %s --check-prefixes CHECK,CHECK-LITTLE
; RUN: llc < %s -mtriple=thumbeb-eabi | FileCheck %s --check-prefixes CHECK,CHECK-BIG

define i1 @umulo32(i32 %l, i32 %r) unnamed_addr #0 {
; CHECK-LABEL: umulo32:
; CHECK:                @ %bb.0: @ %start
; CHECK-NEXT:           .save {r7, lr}
; CHECK-NEXT:           push {r7, lr}
; CHECK-LITTLE-NEXT:    movs r2, r1
; CHECK-LITTLE-NEXT:    movs r1, #0
; CHECK-NEXT:           movs r3, r1
; CHECK-BIG-NEXT:       movs r1, r0
; CHECK-BIG-NEXT:       movs r0, #0
; CHECK-BIG-NEXT:       movs r2, r0
; CHECK-NEXT:           bl __aeabi_lmul
; CHECK-LITTLE-NEXT:    cmp  r1, #0
; CHECK-LITTLE-NEXT:    bne  .LBB0_2
; CHECK-LITTLE-NEXT:    @ %bb.1:
; CHECK-LITTLE-NEXT:    movs r0, r1
; CHECK-LITTLE-NEXT:    b    .LBB0_3
; CHECK-LITTLE-NEXT:    .LBB0_2:
; CHECK-LITTLE-NEXT:    movs r0, #1
; CHECK-LITTLE-NEXT:    .LBB0_3:
; CHECK-BIG-NEXT:       cmp  r0, #0
; CHECK-BIG-NEXT:       beq  .LBB0_2
; CHECK-BIG-NEXT:       @ %bb.1:
; CHECK-BIG-NEXT:       movs r0, #1
; CHECK-BIG-NEXT:       .LBB0_2:
; CHECK-NEXT:           pop {r7}
; CHECK-NEXT:           pop {r1}
; CHECK-NEXT:           bx r1
start:
  %0 = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %l, i32 %r) #2
  %1 = extractvalue { i32, i1 } %0, 1
  ret i1 %1
}

; Function Attrs: nounwind readnone speculatable
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #1

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }
