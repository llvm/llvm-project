; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s --check-prefixes=PPC64
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s --check-prefixes=PPC32

define { i128, i8 } @muloti_test(i128 %l, i128 %r) unnamed_addr #0 {

; PPC64-LABEL muloti_test:
; PPC64: mulld 8, 5, 4
; PPC64-NEXT: cmpdi 5, 3, 0
; PPC64-NEXT: mulhdu. 9, 3, 6
; PPC64-NEXT: mulld 3, 3, 6
; PPC64-NEXT: mcrf 1, 0
; PPC64-NEXT: add 3, 3, 8
; PPC64-NEXT: cmpdi   5, 0
; PPC64-NEXT: crnor 20, 2, 22
; PPC64-NEXT: cmpldi  3, 0
; PPC64-NEXT: mulhdu 8, 4, 6
; PPC64-NEXT: add 3, 8, 3
; PPC64-NEXT: cmpld 6, 3, 8
; PPC64-NEXT: crandc 21, 24, 2
; PPC64-NEXT: crorc 20, 20, 6
; PPC64-NEXT: li 7, 1
; PPC64-NEXT: mulhdu. 5, 5, 4
; PPC64-NEXT: crorc 20, 20, 2
; PPC64-NEXT: crnor 20, 20, 21
; PPC64-NEXT: mulld 4, 4, 6
; PPC64-NEXT: bc 12, 20, .LBB0_2
; PPC64: ori 5, 7, 0
; PPC64-NEXT: blr
; PPC64-NEXT: .LBB0_2:
; PPC64-NEXT: addi 5, 0, 0
; PPC64-NEXT: blr
;
; PPC32-LABEL muloti_test:
; PPC32: mflr 0
; PPC32-NEXT: stw 0, 4(1)
; PPC32-NEXT: stwu 1, -80(1)
; PPC32-NEXT: .cfi_def_cfa_offset 80
; PPC32-NEXT: .cfi_offset lr, 4
; PPC32-NEXT: .cfi_offset r20, -48
; PPC32-NEXT: .cfi_offset r21, -44
; PPC32-NEXT: .cfi_offset r22, -40
; PPC32-NEXT: .cfi_offset r23, -36
; PPC32-NEXT: .cfi_offset r24, -32
; PPC32-NEXT: .cfi_offset r25, -28
; PPC32-NEXT: .cfi_offset r26, -24
; PPC32-NEXT: .cfi_offset r27, -20
; PPC32-NEXT: .cfi_offset r28, -16
; PPC32-NEXT: .cfi_offset r29, -12
; PPC32-NEXT: .cfi_offset r30, -8
; PPC32-NEXT: stw 26, 56(1)
; PPC32-NEXT: stw 27, 60(1)
; PPC32-NEXT: stw 29, 68(1)
; PPC32-NEXT: stw 30, 72(1)
; PPC32-NEXT: mfcr 12
; PPC32-NEXT: mr 30, 8
; PPC32-NEXT: mr 29, 7
; PPC32-NEXT: mr 27, 4
; PPC32-NEXT: mr 26, 3
; PPC32-NEXT: li 3, 0
; PPC32-NEXT: li 4, 0
; PPC32-NEXT: li 7, 0
; PPC32-NEXT: li 8, 0
; PPC32-NEXT: stw 20, 32(1)
; PPC32-NEXT: stw 21, 36(1)
; PPC32-NEXT: stw 22, 40(1)
; PPC32-NEXT: stw 23, 44(1)
; PPC32-NEXT: stw 24, 48(1)
; PPC32-NEXT: stw 25, 52(1)
; PPC32-NEXT: stw 28, 64(1)
; PPC32-NEXT: mr 25, 10
; PPC32-NEXT: stw 12, 28(1)
; PPC32-NEXT: mr 28, 9
; PPC32-NEXT: mr 23, 6
; PPC32-NEXT: mr 24, 5
; PPC32-NEXT: bl __multi3@PLT
; PPC32-NEXT: mr 7, 4
; PPC32-NEXT: mullw 4, 24, 30
; PPC32-NEXT: mullw 8, 29, 23
; PPC32-NEXT: mullw 10, 28, 27
; PPC32-NEXT: mullw 11, 26, 25
; PPC32-NEXT: mulhwu 9, 30, 23
; PPC32-NEXT: mulhwu 12, 27, 25
; PPC32-NEXT: mullw 0, 30, 23
; PPC32-NEXT: mullw 22, 27, 25
; PPC32-NEXT: add 21, 8, 4
; PPC32-NEXT: add 10, 11, 10
; PPC32-NEXT: addc 4, 22, 0
; PPC32-NEXT: add 11, 9, 21
; PPC32-NEXT: add 0, 12, 10
; PPC32-NEXT: adde 8, 0, 11
; PPC32-NEXT: addc 4, 7, 4
; PPC32-NEXT: adde 8, 3, 8
; PPC32-NEXT: xor 22, 4, 7
; PPC32-NEXT: xor 20, 8, 3
; PPC32-NEXT: or. 22, 22, 20
; PPC32-NEXT: mcrf 1, 0
; PPC32-NEXT: cmpwi   29, 0
; PPC32-NEXT: cmpwi 5, 24, 0
; PPC32-NEXT: cmpwi 6, 26, 0
; PPC32-NEXT: cmpwi 7, 28, 0
; PPC32-NEXT: crnor 8, 22, 2
; PPC32-NEXT: mulhwu. 23, 29, 23
; PPC32-NEXT: crnor 9, 30, 26
; PPC32-NEXT: mcrf 5, 0
; PPC32-NEXT: cmplwi  21, 0
; PPC32-NEXT: cmplw 6, 11, 9
; PPC32-NEXT: cmplwi 7, 10, 0
; PPC32-NEXT: crandc 10, 24, 2
; PPC32-NEXT: cmplw 3, 0, 12
; PPC32-NEXT: mulhwu. 9, 24, 30
; PPC32-NEXT: mcrf 6, 0
; PPC32-NEXT: crandc 11, 12, 30
; PPC32-NEXT: cmplw   4, 7
; PPC32-NEXT: cmplw 7, 8, 3
; PPC32-NEXT: crand 12, 30, 0
; PPC32-NEXT: crandc 13, 28, 30
; PPC32-NEXT: mulhwu. 3, 26, 25
; PPC32-NEXT: mcrf 7, 0
; PPC32-NEXT: cror 0, 12, 13
; PPC32-NEXT: crandc 12, 0, 6
; PPC32-NEXT: crorc 20, 8, 22
; PPC32-NEXT: crorc 20, 20, 26
; PPC32-NEXT: mulhwu. 3, 28, 27
; PPC32-NEXT: mcrf 1, 0
; PPC32-NEXT: crorc 25, 9, 30
; PPC32-NEXT: or. 3, 27, 26
; PPC32-NEXT: cror 24, 20, 10
; PPC32-NEXT: mcrf 5, 0
; PPC32-NEXT: crorc 25, 25, 6
; PPC32-NEXT: or. 3, 30, 29
; PPC32-NEXT: cror 25, 25, 11
; PPC32-NEXT: crnor 20, 2, 22
; PPC32-NEXT: lwz 12, 28(1)
; PPC32-NEXT: cror 20, 20, 25
; PPC32-NEXT: cror 20, 20, 24
; PPC32-NEXT: crnor 20, 20, 12
; PPC32-NEXT: li 3, 1
; PPC32-NEXT: bc 12, 20, .LBB0_2
; PPC32: ori 7, 3, 0
; PPC32-NEXT: b .LBB0_3
; PPC32-NEXT:.LBB0_2:
; PPC32-NEXT: addi 7, 0, 0
; PPC32-NEXT:.LBB0_3:
; PPC32-NEXT: mr 3, 8
; PPC32-NEXT: mtcrf 32, 12
; PPC32-NEXT: mtcrf 16, 12
; PPC32-NEXT: lwz 30, 72(1)
; PPC32-NEXT: lwz 29, 68(1)
; PPC32-NEXT: lwz 28, 64(1)
; PPC32-NEXT: lwz 27, 60(1)
; PPC32-NEXT: lwz 26, 56(1)
; PPC32-NEXT: lwz 25, 52(1)
; PPC32-NEXT: lwz 24, 48(1)
; PPC32-NEXT: lwz 23, 44(1)
; PPC32-NEXT: lwz 22, 40(1)
; PPC32-NEXT: lwz 21, 36(1)
; PPC32-NEXT: lwz 20, 32(1)
; PPC32-NEXT: lwz 0, 84(1)
; PPC32-NEXT: addi 1, 1, 80
; PPC32-NEXT: mtlr 0
; PPC32-NEXT: blr
start:
  %0 = tail call { i128, i1 } @llvm.umul.with.overflow.i128(i128 %l, i128 %r) #2
  %1 = extractvalue { i128, i1 } %0, 0
  %2 = extractvalue { i128, i1 } %0, 1
  %3 = zext i1 %2 to i8
  %4 = insertvalue { i128, i8 } undef, i128 %1, 0
  %5 = insertvalue { i128, i8 } %4, i8 %3, 1
  ret { i128, i8 } %5
}

; Function Attrs: nounwind readnone speculatable
declare { i128, i1 } @llvm.umul.with.overflow.i128(i128, i128) #1

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }
