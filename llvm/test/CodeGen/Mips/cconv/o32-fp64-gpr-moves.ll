; RUN: llc -mtriple=mips64 -mcpu=mips3 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips3 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips4 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips4 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -target-abi=o32 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64 -target-abi=o32 -mattr=+nooddspreg -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64 -target-abi=o32 -mattr=+nooddspreg -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=o32 -mattr=+nooddspreg -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi=o32 -mattr=+nooddspreg -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=SPILL
; RUN: llc -mtriple=mips64el -mcpu=mips64 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=SPILL
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=DIRECT-BE
; RUN: llc -mtriple=mips64el -mcpu=mips64r2 -target-abi=o32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=DIRECT-LE

; An integer argument before a double makes O32 pass that double in two
; 32-bit GPRs. Check both directions of the transfer with 64-bit FPRs.
; Before revision 2, MTHC1/MFHC1 are unavailable: the upper half of an FPR
; cannot be accessed by applying MTC1/MFC1 to its sub_hi register.
define double @from_gprs(i32 %pad, double %value) {
; SPILL-LABEL: from_gprs:
; SPILL:       addiu $sp, $sp, -8
; SPILL:       sw $6, 0($sp)
; SPILL-NEXT:  sw $7, 4($sp)
; SPILL-NEXT:  ldc1 $f0, 0($sp)
; SPILL:       addiu $sp, $sp, 8
;
; DIRECT-BE-LABEL: from_gprs:
; DIRECT-BE:       mtc1 $7, $f0
; DIRECT-BE-NEXT:  mthc1 $6, $f0
;
; DIRECT-LE-LABEL: from_gprs:
; DIRECT-LE:       mtc1 $6, $f0
; DIRECT-LE-NEXT:  mthc1 $7, $f0
  ret double %value
}

declare void @consume(i32, double)

define void @to_gprs(double %value) {
; SPILL-LABEL: to_gprs:
; SPILL:       sdc1 $f12, 16($sp)
; SPILL-NEXT:  lw $6, 16($sp)
; SPILL-NEXT:  sdc1 $f12, 16($sp)
; SPILL-NEXT:  lw $7, 20($sp)
; SPILL:       jal consume
;
; DIRECT-BE-LABEL: to_gprs:
; DIRECT-BE-DAG:   mfhc1 $6, $f12
; DIRECT-BE-DAG:   mfc1 $7, $f12
; DIRECT-BE:       jal consume
;
; DIRECT-LE-LABEL: to_gprs:
; DIRECT-LE-DAG:   mfc1 $6, $f12
; DIRECT-LE-DAG:   mfhc1 $7, $f12
; DIRECT-LE:       jal consume
  call void @consume(i32 1, double %value)
  ret void
}
