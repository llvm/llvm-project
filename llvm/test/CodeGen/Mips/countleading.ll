; RUN: llc -march=mipsel -mcpu=mips32   < %s | FileCheck -check-prefixes=ALL,MIPS32-R1-R2,MIPS32-GT-R1 %s
; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck -check-prefixes=ALL,MIPS32-R1-R2,MIPS32-GT-R1 %s
; RUN: llc -march=mipsel -mcpu=mips32r6 < %s | FileCheck -check-prefixes=ALL,MIPS32-R6,MIPS32-GT-R1 %s
; RUN: llc -march=mips64el -mcpu=mips4    < %s | FileCheck -check-prefixes=ALL,MIPS4 %s
; RUN: llc -march=mips64el -mcpu=mips64   < %s | FileCheck -check-prefixes=ALL,MIPS64-GT-R1 %s
; RUN: llc -march=mips64el -mcpu=mips64r2 < %s | FileCheck -check-prefixes=ALL,MIPS64-GT-R1 %s
; RUN: llc -march=mips64el -mcpu=mips64r6 < %s | FileCheck -check-prefixes=ALL,MIPS64-GT-R1 %s
; RUN: llc -march=mips64el -mcpu=mips64r6 -mattr=micromips < %s | FileCheck -check-prefixes=ALL,MICROMIPS64 %s

; Prefixes:
;   ALL      - All
;   MIPS32-GT-R1 - MIPS64r1 and above (does not include MIPS64's)
;   MIPS64-GT-R1 - MIPS64r1 and above

define i32 @ctlz_i32(i32 signext %X) nounwind readnone {
entry:
; ALL-LABEL: ctlz_i32:

; MIPS4-NOT:     clz

; MIPS32-GT-R1:  clz $2, $4

; MIPS64-GT-R1:  clz $2, $4

; MICROMIPS64:   clz $2, $4

  %tmp1 = tail call i32 @llvm.ctlz.i32(i32 %X, i1 true)
  ret i32 %tmp1
}

declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone

define i32 @ctlo_i32(i32 signext %X) nounwind readnone {
entry:
; ALL-LABEL: ctlo_i32:

; MIPS4-NOT:     clo

; MIPS32-GT-R1:  clo $2, $4

; MIPS64-GT-R1:  clo $2, $4

; MICROMIPS64:   clo $2, $4

  %neg = xor i32 %X, -1
  %tmp1 = tail call i32 @llvm.ctlz.i32(i32 %neg, i1 true)
  ret i32 %tmp1
}

define i64 @ctlz_i64(i64 %X) nounwind readnone {
entry:
; ALL-LABEL: ctlz_i64:

; MIPS4-NOT:     dclz

; MIPS32-GT-R1-DAG: clz $[[R0:[0-9]+]], $4
; MIPS32-GT-R1-DAG: clz $[[R1:[0-9]+]], $5
; MIPS32-GT-R1-DAG: addiu $[[R2:2+]], $[[R0]], 32
; MIPS32-R1-R2-DAG: movn $[[R2]], $[[R1]], $5
; MIPS32-R6-DAG:    seleqz $[[R5:[0-9]+]], $[[R2]], $5
; MIPS32-R6-DAG:    selnez $[[R6:[0-9]+]], $[[R1]], $5
; MIPS32-R6-DAG:    or $2, $[[R6]], $[[R5]]
; MIPS32-GT-R1-DAG: addiu $3, $zero, 0

; MIPS64-GT-R1:  dclz $2, $4
; MICROMIPS64:   dclz $2, $4

  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %X, i1 true)
  ret i64 %tmp1
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i64 @ctlo_i64(i64 %X) nounwind readnone {
entry:
; ALL-LABEL: ctlo_i64:

; MIPS4-NOT:     dclo

; MIPS32-GT-R1-DAG: clo $[[R0:[0-9]+]], $4
; MIPS32-GT-R1-DAG: clo $[[R1:[0-9]+]], $5
; MIPS32-GT-R1-DAG: addiu $[[R2:2+]], $[[R0]], 32
; MIPS32-GT-R1-DAG: addiu $[[R3:[0-9]+]], $zero, -1
; MIPS32-GT-R1-DAG: xor $[[R4:[0-9]+]], $5, $[[R3]]
; MIPS32-R1-R2-DAG: movn $[[R2]], $[[R1]], $[[R4]]
; MIPS32-R6-DAG:    selnez $[[R5:[0-9]+]], $[[R1]], $[[R4]]
; MIPS32-R6-DAG:    seleqz $[[R6:[0-9]+]], $[[R2]], $[[R4]]
; MIPS32-R6-DAG:    or $2, $[[R5]], $[[R6]]
; MIPS32-GT-R1-DAG: addiu $3, $zero, 0

; MIPS64-GT-R1:  dclo $2, $4
; MICROMIPS64:   dclo $2, $4

  %neg = xor i64 %X, -1
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %neg, i1 true)
  ret i64 %tmp1
}
