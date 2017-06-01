; RUN: llc -march=mips64el -mcpu=mips4    -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,NOT-R6
; RUN: llc -march=mips64el -mcpu=mips64   -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,NOT-R6
; RUN: llc -march=mips64el -mcpu=mips64r2 -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,NOT-R6
; RUN: llc -march=mips64el -mcpu=mips64r6 -asm-show-inst -relocation-model=pic < %s | FileCheck %s -check-prefixes=CHECK,R6

declare void @llvm.eh.return.i64(i64, i8*)
declare void @foo(...)

define void @f1(i64 %offset, i8* %handler) {
entry:
  call void (...) @foo()
  call void @llvm.eh.return.i64(i64 %offset, i8* %handler)
  unreachable

; CHECK:    f1:
; CHECK:        daddiu  $sp, $sp, -[[spoffset:[0-9]+]]

; check that $a0-$a3 are saved on stack.
; CHECK:        sd      $4, [[offset0:[0-9]+]]($sp)
; CHECK:        sd      $5, [[offset1:[0-9]+]]($sp)
; CHECK:        sd      $6, [[offset2:[0-9]+]]($sp)
; CHECK:        sd      $7, [[offset3:[0-9]+]]($sp)

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
; CHECK:        ld      $4, [[offset0]]($sp)
; CHECK:        ld      $5, [[offset1]]($sp)
; CHECK:        ld      $6, [[offset2]]($sp)
; CHECK:        ld      $7, [[offset3]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; also check that $25 contains handler value
; CHECK:        daddiu  $sp, $sp, [[spoffset]]
; CHECK:        move    $25, $2
; CHECK:        move    $ra, $2
; NOT-R6:       jr      $ra # <MCInst #{{[0-9]+}} JR
; R6:           jr      $ra # <MCInst #{{[0-9]+}} JALR
; CHECK:        daddu   $sp, $sp, $3
}

define void @f2(i64 %offset, i8* %handler) {
entry:
  call void @llvm.eh.return.i64(i64 %offset, i8* %handler)
  unreachable

; CHECK:    f2:
; CHECK:        .cfi_startproc
; CHECK:        daddiu  $sp, $sp, -[[spoffset:[0-9]+]]
; CHECK:        .cfi_def_cfa_offset [[spoffset]]

; check that $a0-$a3 are saved on stack.
; CHECK:        sd      $4, [[offset0:[0-9]+]]($sp)
; CHECK:        sd      $5, [[offset1:[0-9]+]]($sp)
; CHECK:        sd      $6, [[offset2:[0-9]+]]($sp)
; CHECK:        sd      $7, [[offset3:[0-9]+]]($sp)

; check that .cfi_offset directives are emitted for $a0-$a3.
; CHECK:        .cfi_offset 4, -8
; CHECK:        .cfi_offset 5, -16
; CHECK:        .cfi_offset 6, -24
; CHECK:        .cfi_offset 7, -32

; check that stack adjustment and handler are put in $v1 and $v0.
; CHECK:        move    $3, $4
; CHECK:        move    $2, $5

; check that $a0-$a3 are restored from stack.
; CHECK:        ld      $4, [[offset0]]($sp)
; CHECK:        ld      $5, [[offset1]]($sp)
; CHECK:        ld      $6, [[offset2]]($sp)
; CHECK:        ld      $7, [[offset3]]($sp)

; check that stack is adjusted by $v1 and that code returns to address in $v0
; also check that $25 contains handler value
; CHECK:        daddiu  $sp, $sp, [[spoffset]]
; CHECK:        move    $25, $2
; CHECK:        move    $ra, $2
; NOT-R6:       jr      $ra # <MCInst #{{[0-9]+}} JR
; R6:           jr      $ra # <MCInst #{{[0-9]+}} JALR
; CHECK:        daddu   $sp, $sp, $3
; CHECK:        .cfi_endproc
}
