; RUN: llc -mtriple=mips-elf < %s | FileCheck %s -check-prefix=O32
; RUN: llc -mtriple=mips64-elf -target-abi=n32 < %s | FileCheck %s -check-prefix=N32
; RUN: llc -mtriple=mips64-elf < %s | FileCheck %s -check-prefix=N64
; RUN: llc -mtriple=mipsel-windows-gnu < %s | FileCheck %s -check-prefix=MIPSEL

; We only use the '$' prefix on O32. The others use the ELF convention.
; O32: $JTI0_0
; N32: .LJTI0_0
; N64: .LJTI0_0

; Check basic block labels while we're at it.
; O32: $BB0_2:
; N32: .LBB0_2:
; N64: .LBB0_2:

; MIPSEL-LABEL: _Z3fooi:
; MIPSEL:       # %bb.0: # %entry
; MIPSEL-NEXT:    addiu $sp, $sp, -16
; MIPSEL-NEXT:    sltiu	$1, $4, 7
; MIPSEL-NEXT:    beqz $1, .LBB0_6
; MIPSEL-NEXT:    sw $4, 4($sp)
; MIPSEL-NEXT:  # %bb.1: # %entry
; MIPSEL-NEXT:    sll $1, $4, 2
; MIPSEL-NEXT:    lui $2, %hi($JTI0_0)
; MIPSEL-NEXT:    addu $1, $1, $2
; MIPSEL-NEXT:    lw $1, %lo($JTI0_0)($1)
; MIPSEL-NEXT:    jr $1
; MIPSEL-NEXT:    nop
; MIPSEL-NEXT:  .LBB0_2: # %sw.bb
; MIPSEL-NEXT:    lui $1, %hi($.str)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_3: # %sw.bb4
; MIPSEL-NEXT:    lui $1, %hi($.str.4)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.4)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_4: # %sw.bb2
; MIPSEL-NEXT:    lui $1, %hi($.str.2)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.2)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_5: # %sw.bb3
; MIPSEL-NEXT:    lui $1, %hi($.str.3)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.3)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_6: # %sw.epilog
; MIPSEL-NEXT:    lui $1, %hi($.str.7)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.7)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_7: # %sw.bb1
; MIPSEL-NEXT:    lui $1, %hi($.str.1)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.1)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_8: # %sw.bb5
; MIPSEL-NEXT:    lui $1, %hi($.str.5)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.5)
; MIPSEL-NEXT:    j .LBB0_10
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_9: # %sw.bb6
; MIPSEL-NEXT:    lui $1, %hi($.str.6)
; MIPSEL-NEXT:    addiu	$1, $1, %lo($.str.6)
; MIPSEL-NEXT:    sw $1, 8($sp)
; MIPSEL-NEXT:  .LBB0_10: # %return
; MIPSEL-NEXT:    lw $2, 8($sp)
; MIPSEL-NEXT:    jr $ra
; MIPSEL-NEXT:    addiu	$sp, $sp, 16

@.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"B\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"C\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"D\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"E\00", align 1
@.str.5 = private unnamed_addr constant [2 x i8] c"F\00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"G\00", align 1
@.str.7 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define ptr @_Z3fooi(i32 signext %Letter) {
entry:
  %retval = alloca ptr, align 8
  %Letter.addr = alloca i32, align 4
  store i32 %Letter, ptr %Letter.addr, align 4
  %0 = load i32, ptr %Letter.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
    i32 5, label %sw.bb5
    i32 6, label %sw.bb6
  ]

sw.bb:
  store ptr @.str, ptr %retval, align 8
  br label %return

sw.bb1:
  store ptr @.str.1, ptr %retval, align 8
  br label %return

sw.bb2:
  store ptr @.str.2, ptr %retval, align 8
  br label %return

sw.bb3:
  store ptr @.str.3, ptr %retval, align 8
  br label %return

sw.bb4:
  store ptr @.str.4, ptr %retval, align 8
  br label %return

sw.bb5:
  store ptr @.str.5, ptr %retval, align 8
  br label %return

sw.bb6:
  store ptr @.str.6, ptr %retval, align 8
  br label %return

sw.epilog:
  store ptr @.str.7, ptr %retval, align 8
  br label %return

return:
  %1 = load ptr, ptr %retval, align 8
  ret ptr %1
}
