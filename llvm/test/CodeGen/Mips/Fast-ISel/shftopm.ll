; RUN: llc -march=mipsel -relocation-model=pic -O0 \
; RUN:     -fast-isel-abort=3 -mcpu=mips32r2  < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 \
; RUN:     -fast-isel-abort=3 -mcpu=mips32 < %s | FileCheck %s

@s1 = global i16 -89, align 2
@s2 = global i16 4, align 2
@us1 = global i16 -503, align 2
@us2 = global i16 5, align 2
@s3 = common global i16 0, align 2
@us3 = common global i16 0, align 2

define void @sll() {
entry:
  %0 = load i16, ptr @s1, align 2
  %1 = load i16, ptr @s2, align 2
  %shl = shl i16 %0, %1
  store i16 %shl, ptr @s3, align 2
; CHECK-LABEL:  sll:
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK-DAG:    addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[S3_ADDR:[0-9]+]], %got(s3)($[[REG_GP]])
; CHECK-DAG:    lw      $[[S2_ADDR:[0-9]+]], %got(s2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[S1:[0-9]+]], 0($[[S1_ADDR]])
; CHECK-DAG:    lhu     $[[S2:[0-9]+]], 0($[[S2_ADDR]])
; CHECK-DAG:    sllv    $[[RES:[0-9]+]], $[[S1]], $[[S2]]
; CHECK:        sh      $[[RES]], 0($[[S3_ADDR]])
  ret void
}

define void @slli() {
entry:
  %0 = load i16, ptr @s1, align 2
  %shl = shl i16 %0, 5
  store i16 %shl, ptr @s3, align 2
; CHECK-LABEL:  slli:
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK-DAG:    addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[S1:[0-9]+]], 0($[[S1_ADDR]])
; CHECK:        sll     $[[RES:[0-9]+]], $[[S1]],  5
; CHECK:        lw      $[[S3_ADDR:[0-9]+]], %got(s3)($[[REG_GP]])
; CHECK:        sh      $[[RES]], 0($[[S3_ADDR]])
  ret void
}

define void @srl() {
entry:
  %0 = load i16, ptr @us1, align 2
  %1 = load i16, ptr @us2, align 2
  %shr = lshr i16 %0, %1
  store i16 %shr, ptr @us3, align 2
  ret void
; CHECK-LABEL:  srl:
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK-DAG:    addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US2_ADDR:[0-9]+]], %got(us2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    lhu     $[[US2:[0-9]+]], 0($[[US2_ADDR]])
; CHECK:        srlv    $[[RES:[0-9]+]], $[[US1]], $[[US2]]
; CHECK:        lw      $[[US3_ADDR:[0-9]+]], %got(us3)($[[REG_GP]])
; CHECK:        sh      $[[RES]], 0($[[S3_ADDR]])
}

define void @srli() {
entry:
  %0 = load i16, ptr @us1, align 2
  %shr = lshr i16 %0, 4
  store i16 %shr, ptr @us3, align 2
; CHECK-LABEL:  srli:
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK-DAG:    addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK:        srl     $[[RES:[0-9]+]], $[[US1]], 4
; CHECK:        lw      $[[US3_ADDR:[0-9]+]], %got(us3)($[[REG_GP]])
; CHECK:        sh      $[[RES]], 0($[[S3_ADDR]])
  ret void
}

define void @sra() {
entry:
  %0 = load i16, ptr @s1, align 2
  %1 = load i16, ptr @s2, align 2
  %shr = ashr i16 %0, %1
  store i16 %shr, ptr @s3, align 2
; CHECK-LABEL:  sra:
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK-DAG:    addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[S2_ADDR:[0-9]+]], %got(s2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[S1:[0-9]+]], 0($[[S1_ADDR]])
; CHECK-DAG:    lhu     $[[S2:[0-9]+]], 0($[[S2_ADDR]])
; CHECK:        srav    $[[RES:[0-9]+]], $[[S1]], $[[S2]]
; CHECK:        lw      $[[S3_ADDR:[0-9]+]], %got(s3)($[[REG_GP]])
; CHECK:        sh      $[[RES]], 0($[[S3_ADDR]])
  ret void
}

define void @srai() {
entry:
  %0 = load i16, ptr @s1, align 2
  %shr = ashr i16 %0, 2
  store i16 %shr, ptr @s3, align 2
; CHECK-LABEL:  srai:
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK-DAG:    addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK-DAG:    addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[S1_ADDR:[0-9]+]], %got(s1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[S1:[0-9]+]], 0($[[S1_ADDR]])
; CHECK:        sra     $[[RES:[0-9]+]], $[[S1]], 2
; CHECK:        lw      $[[S3_ADDR:[0-9]+]], %got(s3)($[[REG_GP]])
; CHECK:        sh      $[[RES]], 0($[[S3_ADDR]])
  ret void
}
