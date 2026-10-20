; Test all important variants of the 'ret' instruction.
;
; For non-void returns it is necessary to have something to return so we also
; test constant generation here.
;
; We'll test pointer returns in a separate file since the relocation model
; affects it and it's undesirable to repeat the non-pointer returns for each
; relocation model.

; RUN: llc -mtriple=mips   -mcpu=mips32   -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR32,NO-MTHC1,NOT-R6
; RUN: llc -mtriple=mips   -mcpu=mips32r2 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR32,MTHC1,NOT-R6
; RUN: llc -mtriple=mips   -mcpu=mips32r3 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR32,MTHC1,NOT-R6
; RUN: llc -mtriple=mips   -mcpu=mips32r5 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR32,MTHC1,NOT-R6
; RUN: llc -mtriple=mips   -mcpu=mips32r6 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR32,MTHC1,R6C
; RUN: llc -mtriple=mips64 -mcpu=mips4    -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR64,DMTC1,NOT-R6
; RUN: llc -mtriple=mips64 -mcpu=mips64   -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR64,DMTC1,NOT-R6
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR64,DMTC1,NOT-R6
; RUN: llc -mtriple=mips64 -mcpu=mips64r3 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR64,DMTC1,NOT-R6
; RUN: llc -mtriple=mips64 -mcpu=mips64r5 -asm-show-inst < %s | FileCheck %s -check-prefixes=ALL,GPR64,DMTC1,NOT-R6

; FIXME: for the test ret_double_0x0, the delay slot of jr cannot be filled
;        as mthc1 has unmodeled side effects. This is an artifact of our backend.
;        Force the delay slot filler off to check that the sequence jr $ra; nop is
;        turned into jic 0, $ra.

; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -asm-show-inst -disable-mips-delay-filler < %s | FileCheck %s -check-prefixes=ALL,GPR64,DMTC1,R6C

define void @ret_void() {
; ALL-LABEL: ret_void:

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR
; R6C-DAG:       jrc $ra # <MCInst #{{[0-9]+}} JIC

  ret void
}

define i8 @ret_i8() {
; ALL-LABEL: ret_i8:
; ALL-DAG:       addiu $2, $zero, 3

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i8 3
}

define i16 @ret_i16_3() {
; ALL-LABEL: ret_i16_3:
; ALL-DAG:       addiu $2, $zero, 3

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i16 3
}

define i16 @ret_i16_256() {
; ALL-LABEL: ret_i16_256:
; ALL-DAG:       addiu $2, $zero, 256

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i16 256
}

define i16 @ret_i16_257() {
; ALL-LABEL: ret_i16_257:
; ALL-DAG:       addiu $2, $zero, 257

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i16 257
}

define i32 @ret_i32_257() {
; ALL-LABEL: ret_i32_257:
; ALL-DAG:       addiu $2, $zero, 257

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i32 257
}

define i32 @ret_i32_65536() {
; ALL-LABEL: ret_i32_65536:
; ALL-DAG:       lui $2, 1

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i32 65536
}

define i32 @ret_i32_65537() {
; ALL-LABEL: ret_i32_65537:
; ALL:           lui $[[T0:[0-9]+]], 1
; ALL-DAG:       ori $2, $[[T0]], 1

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i32 65537
}

define i64 @ret_i64_65537() {
; ALL-LABEL: ret_i64_65537:
; ALL:           lui $[[T0:[0-9]+]], 1

; GPR32-DAG:     ori $3, $[[T0]], 1
; GPR32-DAG:     addiu $2, $zero, 0

; GPR64-DAG:     ori $2, $[[T0]], 1

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i64 65537
}

define i64 @ret_i64_281479271677952() {
; ALL-LABEL: ret_i64_281479271677952:
; ALL-DAG:       lui $[[T0:[0-9]+]], 1

; GPR32-DAG:     ori $2, $[[T0]], 1
; GPR32-DAG:     addiu $3, $zero, 0

; GPR64-DAG:     daddiu $[[T1:[0-9]+]], $[[T0]], 1
; GPR64-DAG:     dsll $2, $[[T1]], 32

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i64 281479271677952
}

define i64 @ret_i64_281479271809026() {
; ALL-LABEL: ret_i64_281479271809026:
; GPR32-DAG:     lui $[[T0:[0-9]+]], 1
; GPR32-DAG:     lui $[[T1:[0-9]+]], 2
; GPR32-DAG:     ori $2, $[[T0]], 1
; GPR32-DAG:     ori $3, $[[T1]], 2

; GPR64-DAG:     ori  $[[T0:[0-9]+]], $zero, 32769
; GPR64-DAG:     dsll $[[T1:[0-9]+]], $[[T0]], 16
; GPR64-DAG:     daddiu $[[T0:[0-9]+]], $[[T0]], -32767
; GPR64-DAG:     dsll $[[T1:[0-9]+]], $[[T0]], 17
; GPR64-DAG:     daddiu $2, $[[T1]], 2

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret i64 281479271809026
}

define float @ret_float_0x0() {
; ALL-LABEL: ret_float_0x0:

; NO-MTHC1-DAG:  mtc1 $zero, $f0

; MTHC1-DAG:     mtc1 $zero, $f0

; DMTC1-DAG:      mtc1 $zero, $f0

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR

  ret float 0x0000000000000000
}

define float @ret_float_0x3() {
; ALL-LABEL: ret_float_0x3:

; Use a constant pool
; O32-DAG:       lwc1 $f0, %lo($CPI
; N64-DAG:       lwc1 $f0, %got_ofst($CPI

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR
; R6C-DAG:       jrc $ra # <MCInst #{{[0-9]+}} JIC

; float constants are written as double constants
  ret float 0x36b8000000000000
}

define double @ret_double_0x0() {
; ALL-LABEL: ret_double_0x0:

; NO-MTHC1-DAG:  mtc1 $zero, $f0
; NO-MTHC1-DAG:  mtc1 $zero, $f1

; MTHC1-DAG:     mtc1 $zero, $f0
; MTHC1-DAG:     mthc1 $zero, $f0

; DMTC1-DAG:      dmtc1 $zero, $f0

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR
; R6C-DAG:       jrc $ra # <MCInst #{{[0-9]+}} JIC

  ret double 0x0000000000000000
}

define double @ret_double_0x3() {
; ALL-LABEL: ret_double_0x3:

; Use a constant pool
; O32-DAG:       ldc1 $f0, %lo($CPI
; N64-DAG:       ldc1 $f0, %got_ofst($CPI

; NOT-R6-DAG:    jr $ra # <MCInst #{{[0-9]+}} JR
; R6-DAG:        jr $ra # <MCInst #{{[0-9]+}} JALR
; R6C-DAG:       jrc $ra # <MCInst #{{[0-9]+}} JIC

  ret double 0x0000000000000003
}
