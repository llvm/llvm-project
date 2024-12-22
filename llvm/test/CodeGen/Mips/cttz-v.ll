; RUN: llc  < %s -march=mipsel -mcpu=mips32r2 | FileCheck %s -check-prefix=MIPS32
; RUN: llc  < %s -march=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefix=MIPS64

declare <2 x i32> @llvm.cttz.v2i32(<2 x i32>, i1)

define <2 x i32> @cttzv2i32(<2 x i32> %x) {
entry:
; MIPS32-DAG: addiu   $[[R0:[0-9]+]], $4, -1
; MIPS32-DAG: not     $[[R1:[0-9]+]], $4
; MIPS32-DAG: and     $[[R2:[0-9]+]], $[[R1]], $[[R0]]
; MIPS32-DAG: clz     $[[R3:[0-9]+]], $[[R2]]
; MIPS32-DAG: addiu   $[[R4:[0-9]+]], $zero, 32
; MIPS32-DAG: subu    $2, $[[R4]], $[[R3]]
; MIPS32-DAG: addiu   $[[R5:[0-9]+]], $5, -1
; MIPS32-DAG: not     $[[R6:[0-9]+]], $5
; MIPS32-DAG: and     $[[R7:[0-9]+]], $[[R6]], $[[R5]]
; MIPS32-DAG: clz     $[[R8:[0-9]+]], $[[R7]]
; MIPS32-DAG: jr      $ra
; MIPS32-DAG: subu    $3, $[[R4]], $[[R8]]

; MIPS64-DAG: sll     $[[A0:[0-9]+]], $4, 0
; MIPS64-DAG: addiu   $[[R0:[0-9]+]], $[[A0]], -1
; MIPS64-DAG: not     $[[R1:[0-9]+]], $[[A0]]
; MIPS64-DAG: and     $[[R2:[0-9]+]], $[[R1]], $[[R0]]
; MIPS64-DAG: clz     $[[R3:[0-9]+]], $[[R2]]
; MIPS64-DAG: addiu   $[[R4:[0-9]+]], $zero, 32
; MIPS64-DAG: subu    $[[R5:[0-9]+]], $[[R4]], $[[R3]]
; MIPS64-DAG: dsrl    $[[R6:[0-9]+]], $4, 32
; MIPS64-DAG: sll     $[[R7:[0-9]+]], $[[R6]], 0
; MIPS64-DAG: dext    $[[R8:[0-9]+]], $[[R5]], 0, 32
; MIPS64-DAG: addiu   $[[R9:[0-9]+]], $[[R7]], -1
; MIPS64-DAG: not     $[[R10:[0-9]+]], $[[R7]]
; MIPS64-DAG: and     $[[R11:[0-9]+]], $[[R10]], $[[R9]]
; MIPS64-DAG: clz     $[[R12:[0-9]+]], $[[R11]]
; MIPS64-DAG: subu    $[[R13:[0-9]+]], $[[R4]], $[[R12]]
; MIPS64-DAG: dsll    $[[R14:[0-9]+]], $[[R13]], 32
; MIPS64-DAG: or      $2, $[[R8]], $[[R14]]

  %ret = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %x, i1 true)
  ret <2 x i32> %ret
}

