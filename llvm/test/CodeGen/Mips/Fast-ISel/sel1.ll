; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -O2 -relocation-model=pic \
; RUN:          -fast-isel -fast-isel-abort=1 | FileCheck %s

define i1 @sel_i1(i1 %j, i1 %k, i1 %l) {
entry:
  ; CHECK-LABEL:  sel_i1:

  ; FIXME: The following instruction is redundant.
  ; CHECK:            xor     $[[T0:[0-9]+]], $4, $zero
  ; CHECK-NEXT:       sltu    $[[T1:[0-9]+]], $zero, $[[T0]]
  ; CHECK-NEXT:       andi    $[[T2:[0-9]+]], $[[T1]], 1
  ; CHECK-NEXT:       movn    $6, $5, $[[T2]]
  ; CHECK:            move    $2, $6
  %cond = icmp ne i1 %j, 0
  %res = select i1 %cond, i1 %k, i1 %l
  ret i1 %res
}

define i8 @sel_i8(i8 %j, i8 %k, i8 %l) {
entry:
  ; CHECK-LABEL:  sel_i8:

  ; CHECK-DAG:        seb     $[[T0:[0-9]+]], $4
  ; FIXME: The following 2 instructions are redundant.
  ; CHECK-DAG:        seb     $[[T1:[0-9]+]], $zero
  ; CHECK:            xor     $[[T2:[0-9]+]], $[[T0]], $[[T1]]
  ; CHECK-NEXT:       sltu    $[[T3:[0-9]+]], $zero, $[[T2]]
  ; CHECK-NEXT:       andi    $[[T4:[0-9]+]], $[[T3]], 1
  ; CHECK-NEXT:       movn    $6, $5, $[[T4]]
  ; CHECK:            move    $2, $6
  %cond = icmp ne i8 %j, 0
  %res = select i1 %cond, i8 %k, i8 %l
  ret i8 %res
}

define i16 @sel_i16(i16 %j, i16 %k, i16 %l) {
entry:
  ; CHECK-LABEL:  sel_i16:

  ; CHECK-DAG:        seh     $[[T0:[0-9]+]], $4
  ; FIXME: The following 2 instructions are redundant.
  ; CHECK-DAG:        seh     $[[T1:[0-9]+]], $zero
  ; CHECK:            xor     $[[T2:[0-9]+]], $[[T0]], $[[T1]]
  ; CHECK-NEXT:       sltu    $[[T3:[0-9]+]], $zero, $[[T2]]
  ; CHECK-NEXT:       andi    $[[T4:[0-9]+]], $[[T3]], 1
  ; CHECK-NEXT:       movn    $6, $5, $[[T4]]
  ; CHECK:            move    $2, $6
  %cond = icmp ne i16 %j, 0
  %res = select i1 %cond, i16 %k, i16 %l
  ret i16 %res
}

define i32 @sel_i32(i32 %j, i32 %k, i32 %l) {
entry:
  ; CHECK-LABEL:  sel_i32:

  ; FIXME: The following instruction is redundant.
  ; CHECK:            xor     $[[T0:[0-9]+]], $4, $zero
  ; CHECK-NEXT:       sltu    $[[T1:[0-9]+]], $zero, $[[T0]]
  ; CHECK-NEXT:       andi    $[[T2:[0-9]+]], $[[T1]], 1
  ; CHECK-NEXT:       movn    $6, $5, $[[T2]]
  ; CHECK:            move    $2, $6
  %cond = icmp ne i32 %j, 0
  %res = select i1 %cond, i32 %k, i32 %l
  ret i32 %res
}

define float @sel_float(i32 %j, float %k, float %l) {
entry:
  ; CHECK-LABEL:  sel_float:

  ; CHECK-DAG:        mtc1    $6, $f0
  ; CHECK-DAG:        mtc1    $5, $f1
  ; CHECK-DAG:        xor     $[[T0:[0-9]+]], $4, $zero
  ; CHECK:            sltu    $[[T1:[0-9]+]], $zero, $[[T0]]
  ; CHECK-NEXT:       andi    $[[T2:[0-9]+]], $[[T1]], 1
  ; CHECK:            movn.s  $f0, $f1, $[[T2]]
  %cond = icmp ne i32 %j, 0
  %res = select i1 %cond, float %k, float %l
  ret float %res
}

define float @sel_float2(float %k, float %l, i32 %j) {
entry:
  ; CHECK-LABEL:  sel_float2:

  ; CHECK-DAG:        xor     $[[T0:[0-9]+]], $6, $zero
  ; CHECK:            sltu    $[[T1:[0-9]+]], $zero, $[[T0]]
  ; CHECK-NEXT:       andi    $[[T2:[0-9]+]], $[[T1]], 1
  ; CHECK:            movn.s  $f14, $f12, $[[T2]]
  ; CHECK:            mov.s   $f0, $f14
  %cond = icmp ne i32 %j, 0
  %res = select i1 %cond, float %k, float %l
  ret float %res
}

define double @sel_double(i32 %j, double %k, double %l) {
entry:
  ; CHECK-LABEL:  sel_double:

  ; CHECK-DAG:        mtc1    $6, $f2
  ; CHECK-DAG:        mthc1   $7, $f2
  ; CHECK-DAG:        ldc1    $f0, 16($sp)
  ; CHECK-DAG:        xor     $[[T0:[0-9]+]], $4, $zero
  ; CHECK:            sltu    $[[T1:[0-9]+]], $zero, $[[T0]]
  ; CHECK-NEXT:       andi    $[[T2:[0-9]+]], $[[T1]], 1
  ; CHECK:            movn.d  $f0, $f2, $[[T2]]
  %cond = icmp ne i32 %j, 0
  %res = select i1 %cond, double %k, double %l
  ret double %res
}

define double @sel_double2(double %k, double %l, i32 %j) {
entry:
  ; CHECK-LABEL:  sel_double2:

  ; CHECK-DAG:        lw      $[[SEL:[0-9]+]], 16($sp)
  ; CHECK-DAG:        xor     $[[T0:[0-9]+]], $[[SEL]], $zero
  ; CHECK:            sltu    $[[T1:[0-9]+]], $zero, $[[T0]]
  ; CHECK-NEXT:       andi    $[[T2:[0-9]+]], $[[T1]], 1
  ; CHECK:            movn.d  $f14, $f12, $[[T2]]
  ; CHECK:            mov.d   $f0, $f14
  %cond = icmp ne i32 %j, 0
  %res = select i1 %cond, double %k, double %l
  ret double %res
}
