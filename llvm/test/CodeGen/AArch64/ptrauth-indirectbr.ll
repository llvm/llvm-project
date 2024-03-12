; RUN: llc -mtriple arm64e-apple-darwin \
; RUN:   -asm-verbose=false -aarch64-enable-collect-loh=false \
; RUN:   -o - %s | FileCheck %s

; RUN: llc -mtriple arm64e-apple-darwin \
; RUN:   -global-isel -global-isel-abort=1 -verify-machineinstrs \
; RUN:   -asm-verbose=false -aarch64-enable-collect-loh=false \
; RUN:   -o - %s | FileCheck %s

; The discriminator is the same for all blockaddresses in the function.
; ptrauth_string_discriminator("test_blockaddress blockaddress") == 52152

; CHECK-LABEL: _test_blockaddress:
; CHECK:         adrp x16, [[F1BB1ADDR:Ltmp[0-9]+]]@PAGE
; CHECK-NEXT:    add x16, x16, [[F1BB1ADDR]]@PAGEOFF
; CHECK-NEXT:    mov x17, #[[F1DISCVAL:52152]]
; CHECK-NEXT:    pacia x16, x17
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    adrp x16, [[F1BB2ADDR:Ltmp[0-9]+]]@PAGE
; CHECK-NEXT:    add x16, x16, [[F1BB2ADDR]]@PAGEOFF
; CHECK-NEXT:    mov x17, #[[F1DISCVAL]]
; CHECK-NEXT:    pacia x16, x17
; CHECK-NEXT:    mov x1, x16
; CHECK-NEXT:    bl _dummy_choose
; CHECK-NEXT:    mov x17, #[[F1DISCVAL]]
; CHECK-NEXT:    braa x0, x17
; CHECK:        [[F1BB1ADDR]]:
; CHECK-NEXT:   [[F1BB1:LBB[0-9_]+]]:
; CHECK-NEXT:    mov w0, #1
; CHECK:        [[F1BB2ADDR]]:
; CHECK-NEXT:   [[F1BB2:LBB[0-9_]+]]:
; CHECK-NEXT:    mov w0, #2
define i32 @test_blockaddress() #0 {
entry:
  %tmp0 = call ptr @dummy_choose(ptr blockaddress(@test_blockaddress, %bb1), ptr blockaddress(@test_blockaddress, %bb2))
  indirectbr ptr %tmp0, [label %bb1, label %bb2]

bb1:
  ret i32 1

bb2:
  ret i32 2
}

; Test another function to compare the discriminator.
; ptrauth_string_discriminator("test_blockaddress_2 blockaddress") == 22012

; CHECK-LABEL: _test_blockaddress_2:
; CHECK:         adrp x16, [[F2BB1ADDR:Ltmp[0-9]+]]@PAGE
; CHECK-NEXT:    add x16, x16, [[F2BB1ADDR]]@PAGEOFF
; CHECK-NEXT:    mov x17, #[[F2DISCVAL:22012]]
; CHECK-NEXT:    pacia x16, x17
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    adrp x16, [[F2BB2ADDR:Ltmp[0-9]+]]@PAGE
; CHECK-NEXT:    add x16, x16, [[F2BB2ADDR]]@PAGEOFF
; CHECK-NEXT:    mov x17, #[[F2DISCVAL]]
; CHECK-NEXT:    pacia x16, x17
; CHECK-NEXT:    mov x1, x16
; CHECK-NEXT:    bl _dummy_choose
; CHECK-NEXT:    mov x17, #[[F2DISCVAL]]
; CHECK-NEXT:    braa x0, x17
; CHECK:        [[F2BB1ADDR]]:
; CHECK-NEXT:   [[F2BB1:LBB[0-9_]+]]:
; CHECK-NEXT:    mov w0, #1
; CHECK:        [[F2BB2ADDR]]:
; CHECK-NEXT:   [[F2BB2:LBB[0-9_]+]]:
; CHECK-NEXT:    mov w0, #2
define i32 @test_blockaddress_2() #0 {
entry:
  %tmp0 = call ptr @dummy_choose(ptr blockaddress(@test_blockaddress_2, %bb1), ptr blockaddress(@test_blockaddress_2, %bb2))
  indirectbr ptr %tmp0, [label %bb1, label %bb2]

bb1:
  ret i32 1

bb2:
  ret i32 2
}

; CHECK-LABEL: _test_blockaddress_other_function:
; CHECK:         adrp x16, [[F1BB1ADDR]]@PAGE
; CHECK-NEXT:    add x16, x16, [[F1BB1ADDR]]@PAGEOFF
; CHECK-NEXT:    mov x17, #[[F1DISCVAL]]
; CHECK-NEXT:    pacia x16, x17
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    ret
define ptr @test_blockaddress_other_function() #0 {
  ret ptr blockaddress(@test_blockaddress, %bb1)
}

; CHECK-LABEL: .section __DATA,__const
; CHECK-NEXT:  .globl _test_blockaddress_array
; CHECK-NEXT:  .p2align 4
; CHECK-NEXT:  _test_blockaddress_array:
; CHECK-NEXT:   .quad [[F1BB1ADDR]]@AUTH(ia,[[F1DISCVAL]]
; CHECK-NEXT:   .quad [[F1BB2ADDR]]@AUTH(ia,[[F1DISCVAL]]
; CHECK-NEXT:   .quad [[F2BB1ADDR]]@AUTH(ia,[[F2DISCVAL]]
; CHECK-NEXT:   .quad [[F2BB2ADDR]]@AUTH(ia,[[F2DISCVAL]]
@test_blockaddress_array = constant [4 x ptr] [
  ptr blockaddress(@test_blockaddress, %bb1), ptr blockaddress(@test_blockaddress, %bb2),
  ptr blockaddress(@test_blockaddress_2, %bb1), ptr blockaddress(@test_blockaddress_2, %bb2)
]

declare ptr @dummy_choose(ptr, ptr)

attributes #0 = { "ptrauth-indirect-gotos" nounwind }
