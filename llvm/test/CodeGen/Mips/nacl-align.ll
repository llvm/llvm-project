; RUN: llc -filetype=asm -mtriple=mipsel-none-nacl -relocation-model=static \
; RUN:     -O3 < %s | FileCheck %s


; This test tests that NaCl functions are bundle-aligned.

define void @test0() {
  ret void

; CHECK:          .p2align  4
; CHECK-NOT:      .p2align
; CHECK-LABEL:    test0:

}


; This test tests that blocks that are jumped to through jump table are
; bundle-aligned.

define i32 @test1(i32 %i) {
entry:
  switch i32 %i, label %default [
    i32 0, label %bb1
    i32 1, label %bb2
    i32 2, label %bb3
    i32 3, label %bb4
  ]

bb1:
  ret i32 111
bb2:
  ret i32 222
bb3:
  ret i32 333
bb4:
  ret i32 444
default:
  ret i32 555


; CHECK-LABEL:       test1:

; CHECK:             .p2align  4
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 111
; CHECK-NEXT:        .p2align  4
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 222
; CHECK-NEXT:        .p2align  4
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 333
; CHECK-NEXT:        .p2align  4
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 444
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 555

}


; This test tests that a block whose address is taken is bundle-aligned in NaCl.

@bb_array = constant [2 x ptr] [ptr blockaddress(@test2, %bb1),
                                ptr blockaddress(@test2, %bb2)], align 4

define i32 @test2(i32 %i) {
entry:
  %elementptr = getelementptr inbounds [2 x ptr], ptr @bb_array, i32 0, i32 %i
  %0 = load ptr, ptr %elementptr, align 4
  indirectbr ptr %0, [label %bb1, label %bb2]

bb1:
  ret i32 111
bb2:
  ret i32 222


; CHECK-LABEL:       test2:

; Note that there are two consecutive labels - one temporary and one for
; basic block.

; CHECK:             .p2align  4
; CHECK-NEXT:    ${{[a-zA-Z0-9]+}}:
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 111
; CHECK-NEXT:        .p2align  4
; CHECK-NEXT:    ${{[a-zA-Z0-9]+}}:
; CHECK-NEXT:    ${{BB[0-9]+_[0-9]+}}:
; CHECK-NEXT:        jr      $ra
; CHECK-NEXT:        addiu   $2, $zero, 222

}
