; RUN: llc < %s -mtriple=mipsel -mcpu=mips32 -O0 -relocation-model=pic \
; RUN:      -fast-isel-abort=3 | FileCheck %s
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r2 -O0 -relocation-model=pic \
; RUN:      -fast-isel-abort=3 | FileCheck %s

@sj = global i32 200000, align 4
@sk = global i32 -47, align 4
@uj = global i32 200000, align 4
@uk = global i32 43, align 4
@si = common global i32 0, align 4
@ui = common global i32 0, align 4

define void @divs() {
  ; CHECK-LABEL:  divs:

  ; CHECK:        lui     $[[GOT1:[0-9]+]], %hi(_gp_disp)
  ; CHECK:        addiu   $[[GOT2:[0-9]+]], $[[GOT1]], %lo(_gp_disp)
  ; CHECK:        addu    $[[GOT:[0-9]+]], $[[GOT2:[0-9]+]], $25
  ; CHECK-DAG:    lw      $[[I_ADDR:[0-9]+]], %got(si)($[[GOT]])
  ; CHECK-DAG:    lw      $[[K_ADDR:[0-9]+]], %got(sk)($[[GOT]])
  ; CHECK-DAG:    lw      $[[J_ADDR:[0-9]+]], %got(sj)($[[GOT]])
  ; CHECK-DAG:    lw      $[[J:[0-9]+]], 0($[[J_ADDR]])
  ; CHECK-DAG:    lw      $[[K:[0-9]+]], 0($[[K_ADDR]])
  ; CHECK-DAG:    div     $zero, $[[J]], $[[K]]
  ; CHECK-DAG:    teq     $[[K]], $zero, 7
  ; CHECK-DAG:    mflo    $[[RESULT:[0-9]+]]
  ; CHECK:        sw      $[[RESULT]], 0($[[I_ADDR]])
  %1 = load i32, ptr @sj, align 4
  %2 = load i32, ptr @sk, align 4
  %div = sdiv i32 %1, %2
  store i32 %div, ptr @si, align 4
  ret void
}

define void @divu() {
  ; CHECK-LABEL:  divu:

  ; CHECK:            lui     $[[GOT1:[0-9]+]], %hi(_gp_disp)
  ; CHECK:            addiu   $[[GOT2:[0-9]+]], $[[GOT1]], %lo(_gp_disp)
  ; CHECK:            addu    $[[GOT:[0-9]+]], $[[GOT2:[0-9]+]], $25
  ; CHECK-DAG:        lw      $[[I_ADDR:[0-9]+]], %got(ui)($[[GOT]])
  ; CHECK-DAG:        lw      $[[K_ADDR:[0-9]+]], %got(uk)($[[GOT]])
  ; CHECK-DAG:        lw      $[[J_ADDR:[0-9]+]], %got(uj)($[[GOT]])
  ; CHECK-DAG:        lw      $[[J:[0-9]+]], 0($[[J_ADDR]])
  ; CHECK-DAG:        lw      $[[K:[0-9]+]], 0($[[K_ADDR]])
  ; CHECK-DAG:        divu    $zero, $[[J]], $[[K]]
  ; CHECK-DAG:        teq     $[[K]], $zero, 7
  ; CHECK-DAG:        mflo    $[[RESULT:[0-9]+]]
  ; CHECK:            sw      $[[RESULT]], 0($[[I_ADDR]])
  %1 = load i32, ptr @uj, align 4
  %2 = load i32, ptr @uk, align 4
  %div = udiv i32 %1, %2
  store i32 %div, ptr @ui, align 4
  ret void
}
