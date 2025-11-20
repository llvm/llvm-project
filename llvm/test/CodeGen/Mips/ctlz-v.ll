; RUN: llc  < %s -mtriple=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=MIPS32
; RUN: llc  < %s -mtriple=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefix=MIPS64

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1)

define <2 x i32> @ctlzv2i32(<2 x i32> %x) {
entry:
; MIPS32: clz     $2, $4
; MIPS32: clz     $3, $5

; MIPS64-DAG: dsrl $[[A0:[0-9]+]], $4, 32
; MIPS64-DAG: sll $[[A1:[0-9]+]], $[[A0]], 0
; MIPS64-DAG: clz $[[R0:[0-9]+]], $[[A1]]
; MIPS64-DAG: dsll $[[R1:[0-9]+]], $[[R0]], 32
; MIPS64-DAG: sll $[[A2:[0-9]+]], $4, 0
; MIPS64-DAG: clz $[[R2:[0-9]+]], $[[A2]]
; MIPS64-DAG: dext $[[R3:[0-9]+]], $[[R2]], 0, 32
; MIPS64-DAG: or $2, $[[R3]], $[[R1]]

  %ret = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %x, i1 true)
  ret <2 x i32> %ret
}

