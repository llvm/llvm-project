; RUN: llc -mtriple=mipsel-elf -mcpu=mips32   -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,NOT-R6
; RUN: llc -mtriple=mipsel-elf -mcpu=mips32r2 -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,NOT-R6
; RUN: llc -mtriple=mipsel-elf -mcpu=mips32r6 -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,R6

declare void @llvm.eh.return.i32(i32, ptr)
declare void @foo(...)

define ptr @f1(i32 %offset, ptr %handler) {
entry:
  call void (...) @foo()
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable

; CHECK:    f1:
; CHECK:        addiu   $sp, $sp, -[[spoffset:[0-9]+]]

; check that $a0-$a3 are saved on stack.
; CHECK:        sw      $4, [[offset0:[0-9]+]]($sp)
; CHECK:        sw      $5, [[offset1:[0-9]+]]($sp)
; CHECK:        sw      $6, [[offset2:[0-9]+]]($sp)
; CHECK:        sw      $7, [[offset3:[0-9]+]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a3.
; CHECK:        .cfi_offset 4,
; CHECK:        .cfi_offset 5,
; CHECK:        .cfi_offset 6,
; CHECK:        .cfi_offset 7,

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        move    $[[R0:[a-z0-9]+]], $5
; CHECK:        move    $[[R1:[a-z0-9]+]], $4
; CHECK:        move    $3, $[[R1]]
; CHECK:        move    $2, $[[R0]]

; check that $a0-$a3 are restored from stack.
; CHECK:        lw      $4, [[offset0]]($sp)
; CHECK:        lw      $5, [[offset1]]($sp)
; CHECK:        lw      $6, [[offset2]]($sp)
; CHECK:        lw      $7, [[offset3]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; also check that $25 contains handler value
; CHECK:        addiu   $sp, $sp, [[spoffset]]
; CHECK:        move    $25, $2
; CHECK:        move    $ra, $2
; NOT-R6:       jr      $ra # <MCInst #{{[0-9]+}} JR
; R6:           jr      $ra # <MCInst #{{[0-9]+}} JALR
; CHECK:        addu    $sp, $sp, $3
}

define ptr @f2(i32 %offset, ptr %handler) {
entry:
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable

; CHECK:    f2:
; CHECK:        addiu   $sp, $sp, -[[spoffset:[0-9]+]]

; check that $a0-$a3 are saved on stack.
; CHECK:        sw      $4, [[offset0:[0-9]+]]($sp)
; CHECK:        sw      $5, [[offset1:[0-9]+]]($sp)
; CHECK:        sw      $6, [[offset2:[0-9]+]]($sp)
; CHECK:        sw      $7, [[offset3:[0-9]+]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a3.
; CHECK:        .cfi_offset 4,
; CHECK:        .cfi_offset 5,
; CHECK:        .cfi_offset 6,
; CHECK:        .cfi_offset 7,

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        move    $3, $4
; CHECK:        move    $2, $5

; check that $a0-$a3 are restored from stack.
; CHECK:        lw      $4, [[offset0]]($sp)
; CHECK:        lw      $5, [[offset1]]($sp)
; CHECK:        lw      $6, [[offset2]]($sp)
; CHECK:        lw      $7, [[offset3]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; also check that $25 contains handler value
; CHECK:        addiu   $sp, $sp, [[spoffset]]
; CHECK:        move    $25, $2
; CHECK:        move    $ra, $2
; NOT-R6:       jr      $ra # <MCInst #{{[0-9]+}} JR
; R6:           jr      $ra # <MCInst #{{[0-9]+}} JALR
; CHECK:        addu    $sp, $sp, $3
}
