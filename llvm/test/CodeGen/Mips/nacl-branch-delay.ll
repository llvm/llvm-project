; RUN: llc -filetype=asm -mtriple=mipsel-none-linux -relocation-model=static \
; RUN:     -O3 < %s | FileCheck %s

; RUN: llc -filetype=asm -mtriple=mipsel-none-nacl -relocation-model=static \
; RUN:     -O3 < %s | FileCheck %s -check-prefix=CHECK-NACL

@x = global i32 0, align 4
declare void @f1(i32)
declare void @f2()


define void @test1() {
  %1 = load i32, i32* @x, align 4
  call void @f1(i32 %1)
  ret void


; CHECK-LABEL:       test1

; We first make sure that for non-NaCl targets branch-delay slot contains
; dangerous instructions.

; Check that branch-delay slot is used to load argument from x before function
; call.

; CHECK:             jal
; CHECK-NEXT:        lw      $4, %lo(x)(${{[0-9]+}})

; Check that branch-delay slot is used for adjusting sp before return.

; CHECK:             jr      $ra
; CHECK-NEXT:        addiu   $sp, $sp, {{[0-9]+}}


; For NaCl, check that branch-delay slot doesn't contain dangerous instructions.

; CHECK-NACL:             jal
; CHECK-NACL-NEXT:        nop

; CHECK-NACL:             jr      $ra
; CHECK-NACL-NEXT:        nop
}


define void @test2() {
  store i32 1, i32* @x, align 4
  call void @f2()
  ret void


; CHECK-LABEL:       test2

; Check that branch-delay slot is used for storing to x before function call.

; CHECK:             jal
; CHECK-NEXT:        sw      ${{[0-9]+}}, %lo(x)(${{[0-9]+}})

; Check that branch-delay slot is used for adjusting sp before return.

; CHECK:             jr      $ra
; CHECK-NEXT:        addiu   $sp, $sp, {{[0-9]+}}


; For NaCl, check that branch-delay slot doesn't contain dangerous instructions.

; CHECK-NACL:             jal
; CHECK-NACL-NEXT:        nop

; CHECK-NACL:             jr      $ra
; CHECK-NACL-NEXT:        nop
}
