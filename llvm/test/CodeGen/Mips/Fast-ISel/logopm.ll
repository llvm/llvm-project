; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 < %s | FileCheck %s

@ub1 = common global i8 0, align 1
@ub2 = common global i8 0, align 1
@ub3 = common global i8 0, align 1
@uc1 = common global i8 0, align 1
@uc2 = common global i8 0, align 1
@uc3 = common global i8 0, align 1
@us1 = common global i16 0, align 2
@us2 = common global i16 0, align 2
@us3 = common global i16 0, align 2
@ub = common global i8 0, align 1
@uc = common global i8 0, align 1
@us = common global i16 0, align 2
@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1
@ui = common global i32 0, align 4
@ui1 = common global i32 0, align 4
@ui2 = common global i32 0, align 4
@ui3 = common global i32 0, align 4

; Function Attrs: noinline nounwind
define void @andUb() #0 {
entry:
  %0 = load i8, i8* @ub1, align 1
  %1 = load i8, i8* @ub2, align 1
  %conv0 = trunc i8 %0 to i1
  %conv1 = trunc i8 %1 to i1
  %and0 = and i1 %conv1, %conv0
  %conv3 = zext i1 %and0 to i8
  store i8 %conv3, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    andUb
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB2_ADDR:[0-9]+]], %got(ub2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    lbu     $[[UB2:[0-9]+]], 0($[[UB2_ADDR]])
; CHECK-DAG:    and     $[[RES1:[0-9]+]], $[[UB2]], $[[UB1]]
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
  ret void
}

; Function Attrs: noinline nounwind
define void @andUb0() #0 {
entry:
  %0 = load i8, i8* @ub1, align 1, !tbaa !2
  %conv = trunc i8 %0 to i1
  %and = and i1 %conv, 0
  %conv1 = zext i1 %and to i8
  store i8 %conv1, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    andUb0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    and     $[[RES1:[0-9]+]], $[[UB1]], $zero
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
; CHECK:        .end    andUb0
  ret void
}

; Function Attrs: noinline nounwind
define void @andUb1() #0 {
; clang uses i8 constants for booleans, so we test with an i8 1.
entry:
  %x = load i8, i8* @ub1, align 1, !tbaa !2
  %and = and i8 %x, 1
  %conv = trunc i8 %and to i1
  %conv1 = zext i1 %conv to i8
  store i8 %conv1, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    andUb1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    addiu   $[[CONST:[0-9]+]], $zero, 1
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    and     $[[RES1:[0-9]+]], $[[UB1]], $[[CONST]]
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
; CHECK:        .end    andUb1
  ret void
}

; Function Attrs: noinline nounwind
define void @orUb() #0 {
entry:
  %0 = load i8, i8* @ub1, align 1
  %1 = load i8, i8* @ub2, align 1
  %conv0 = trunc i8 %0 to i1
  %conv1 = trunc i8 %1 to i1
  %or0 = or i1 %conv1, %conv0
  %conv3 = zext i1 %or0 to i8
  store i8 %conv3, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    orUb
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB2_ADDR:[0-9]+]], %got(ub2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    lbu     $[[UB2:[0-9]+]], 0($[[UB2_ADDR]])
; CHECK-DAG:    or      $[[RES1:[0-9]+]], $[[UB2]], $[[UB1]]
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
  ret void
}

; Function Attrs: noinline nounwind
define void @orUb0() #0 {
entry:
  %0 = load i8, i8* @ub1, align 1, !tbaa !2
  %conv = trunc i8 %0 to i1
  %or = or i1 %conv, 0
  %conv1 = zext i1 %or to i8
  store i8 %conv1, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    orUb0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK:        andi    $[[RES:[0-9]+]], $[[UB1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
; CHECK:        .end    orUb0
  ret void
}

; Function Attrs: noinline nounwind
define void @orUb1() #0 {
entry:
  %x = load i8, i8* @ub1, align 1, !tbaa !2
  %or = or i8 %x, 1
  %conv = trunc i8 %or to i1
  %conv1 = zext i1 %conv to i8
  store i8 %conv1, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    orUb1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    addiu   $[[CONST:[0-9]+]], $zero, 1
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    or      $[[RES1:[0-9]+]], $[[UB1]], $[[CONST]]
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
; CHECK:        .end    orUb1
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUb() #0 {
entry:
  %0 = load i8, i8* @ub1, align 1
  %1 = load i8, i8* @ub2, align 1
  %conv0 = trunc i8 %0 to i1
  %conv1 = trunc i8 %1 to i1
  %xor0 = xor i1 %conv1, %conv0
  %conv3 = zext i1 %xor0 to i8
  store i8 %conv3, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL: .ent    xorUb
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB2_ADDR:[0-9]+]], %got(ub2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    lbu     $[[UB2:[0-9]+]], 0($[[UB2_ADDR]])
; CHECK-DAG:    xor     $[[RES1:[0-9]+]], $[[UB2]], $[[UB1]]
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUb0() #0 {
entry:
  %0 = load i8, i8* @ub1, align 1, !tbaa !2
  %conv = trunc i8 %0 to i1
  %xor = xor i1 %conv, 0
  %conv1 = zext i1 %xor to i8
  store i8 %conv1, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    xorUb0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    xor     $[[RES1:[0-9]+]], $[[UB1]], $zero
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
; CHECK:        .end    xorUb0
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUb1() #0 {
entry:
  %x = load i8, i8* @ub1, align 1, !tbaa !2
  %xor = xor i8 1, %x
  %conv = trunc i8 %xor to i1
  %conv1 = zext i1 %conv to i8
  store i8 %conv1, i8* @ub, align 1, !tbaa !2
; CHECK-LABEL:  .ent    xorUb1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UB_ADDR:[0-9]+]], %got(ub)($[[REG_GP]])
; CHECK-DAG:    addiu   $[[CONST:[0-9]+]], $zero, 1
; CHECK-DAG:    lw      $[[UB1_ADDR:[0-9]+]], %got(ub1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UB1:[0-9]+]], 0($[[UB1_ADDR]])
; CHECK-DAG:    xor     $[[RES1:[0-9]+]], $[[UB1]], $[[CONST]]
; CHECK:        andi    $[[RES:[0-9]+]], $[[RES1]], 1
; CHECK:        sb      $[[RES]], 0($[[UB_ADDR]])
; CHECK:        .end    xorUb1
  ret void
}

; Function Attrs: noinline nounwind
define void @andUc() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %1 = load i8, i8* @uc2, align 1, !tbaa !2
  %and3 = and i8 %1, %0
  store i8 %and3, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    andUc
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC2_ADDR:[0-9]+]], %got(uc2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    lbu     $[[UC2:[0-9]+]], 0($[[UC2_ADDR]])
; CHECK-DAG:    and     $[[RES:[0-9]+]], $[[UC2]], $[[UC1]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
  ret void
}

; Function Attrs: noinline nounwind
define void @andUc0() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %and = and i8 %0, 67
  store i8 %and, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    andUc0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_67:[0-9]+]], $zero, 67
; CHECK-DAG:    and     $[[RES:[0-9]+]], $[[UC1]], $[[CONST_67]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    andUc0
  ret void
}

; Function Attrs: noinline nounwind
define void @andUc1() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %and = and i8 %0, 167
  store i8 %and, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    andUc1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_167:[0-9]+]], $zero, 167
; CHECK-DAG:    and     $[[RES:[0-9]+]], $[[UC1]], $[[CONST_167]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    andUc1
  ret void
}

; Function Attrs: noinline nounwind
define void @orUc() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %1 = load i8, i8* @uc2, align 1, !tbaa !2
  %or3 = or i8 %1, %0
  store i8 %or3, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    orUc
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC2_ADDR:[0-9]+]], %got(uc2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    lbu     $[[UC2:[0-9]+]], 0($[[UC2_ADDR]])
; CHECK-DAG:    or      $[[RES:[0-9]+]], $[[UC2]], $[[UC1]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    orUc
  ret void
}

; Function Attrs: noinline nounwind
define void @orUc0() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
   %or = or i8 %0, 69
  store i8 %or, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    orUc0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_69:[0-9]+]], $zero, 69
; CHECK-DAG:    or      $[[RES:[0-9]+]], $[[UC1]], $[[CONST_69]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    orUc0
  ret void
}

; Function Attrs: noinline nounwind
define void @orUc1() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %or = or i8 %0, 238
  store i8 %or, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    orUc1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_238:[0-9]+]], $zero, 238
; CHECK-DAG:    or      $[[RES:[0-9]+]], $[[UC1]], $[[CONST_238]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    orUc1
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUc() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %1 = load i8, i8* @uc2, align 1, !tbaa !2
  %xor3 = xor i8 %1, %0
  store i8 %xor3, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL: .ent    xorUc
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC2_ADDR:[0-9]+]], %got(uc2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    lbu     $[[UC2:[0-9]+]], 0($[[UC2_ADDR]])
; CHECK-DAG:    xor     $[[RES:[0-9]+]], $[[UC2]], $[[UC1]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    xorUc
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUc0() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %xor = xor i8 %0, 23
  store i8 %xor, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    xorUc0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_23:[0-9]+]], $zero, 23
; CHECK-DAG:    xor     $[[RES:[0-9]+]], $[[UC1]], $[[CONST_23]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    xorUc0
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUc1() #0 {
entry:
  %0 = load i8, i8* @uc1, align 1, !tbaa !2
  %xor = xor i8 %0, 120
  store i8 %xor, i8* @uc, align 1, !tbaa !2
; CHECK-LABEL:  .ent    xorUc1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[UC_ADDR:[0-9]+]], %got(uc)($[[REG_GP]])
; CHECK-DAG:    lw      $[[UC1_ADDR:[0-9]+]], %got(uc1)($[[REG_GP]])
; CHECK-DAG:    lbu     $[[UC1:[0-9]+]], 0($[[UC1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_120:[0-9]+]], $zero, 120
; CHECK-DAG:    xor     $[[RES:[0-9]+]], $[[UC1]], $[[CONST_120]]
; CHECK:        sb      $[[RES]], 0($[[UC_ADDR]])
; CHECK:        .end    xorUc1
  ret void
}

; Function Attrs: noinline nounwind
define void @andUs() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %1 = load i16, i16* @us2, align 2, !tbaa !5
  %and3 = and i16 %1, %0
  store i16 %and3, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    andUs
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US2_ADDR:[0-9]+]], %got(us2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    lhu     $[[US2:[0-9]+]], 0($[[US2_ADDR]])
; CHECK-DAG:    and     $[[RES:[0-9]+]], $[[US2]], $[[US1]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end andUs
  ret void
}

; Function Attrs: noinline nounwind
define void @andUs0() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %and = and i16 %0, 4660
  store i16 %and, i16* @us, align 2, !tbaa !5
; CHECK-LABEL: .ent    andUs0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_4660:[0-9]+]], $zero, 4660
; CHECK-DAG:    and     $[[RES:[0-9]+]], $[[US1]], $[[CONST_4660]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    andUs0
  ret void
}

; Function Attrs: noinline nounwind
define void @andUs1() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %and = and i16 %0, 61351
  store i16 %and, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    andUs1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    ori     $[[CONST_61351:[0-9]+]], $zero, 61351
; CHECK-DAG:    and     $[[RES:[0-9]+]], $[[US1]], $[[CONST_61351]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    andUs1
  ret void
}

; Function Attrs: noinline nounwind
define void @orUs() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %1 = load i16, i16* @us2, align 2, !tbaa !5
  %or3 = or i16 %1, %0
  store i16 %or3, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    orUs
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US2_ADDR:[0-9]+]], %got(us2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    lhu     $[[US2:[0-9]+]], 0($[[US2_ADDR]])
; CHECK-DAG:    or      $[[RES:[0-9]+]], $[[US2]], $[[US1]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    orUs
  ret void
}

; Function Attrs: noinline nounwind
define void @orUs0() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %or = or i16 %0, 17666
  store i16 %or, i16* @us, align 2, !tbaa !5
  ret void
}

; Function Attrs: noinline nounwind
define void @orUs1() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %or = or i16 %0, 60945
  store i16 %or, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    orUs1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    ori     $[[CONST_60945:[0-9]+]], $zero, 60945
; CHECK-DAG:    or      $[[RES:[0-9]+]], $[[US1]], $[[CONST_60945]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    orUs1
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUs() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %1 = load i16, i16* @us2, align 2, !tbaa !5
  %xor3 = xor i16 %1, %0
  store i16 %xor3, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    xorUs
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US2_ADDR:[0-9]+]], %got(us2)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    lhu     $[[US2:[0-9]+]], 0($[[US2_ADDR]])
; CHECK-DAG:    xor     $[[RES:[0-9]+]], $[[US2]], $[[US1]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    xorUs
  ret void
}

; Function Attrs: noinline nounwind
define void @xorUs0() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %xor = xor i16 %0, 6062
  store i16 %xor, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    xorUs0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    addiu   $[[CONST_6062:[0-9]+]], $zero, 6062
; CHECK-DAG:    xor     $[[RES:[0-9]+]], $[[US1]], $[[CONST_6062]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    xorUs0

  ret void
}

; Function Attrs: noinline nounwind
define void @xorUs1() #0 {
entry:
  %0 = load i16, i16* @us1, align 2, !tbaa !5
  %xor = xor i16 %0, 60024
  store i16 %xor, i16* @us, align 2, !tbaa !5
; CHECK-LABEL:  .ent    xorUs1
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK-DAG:    lw      $[[US_ADDR:[0-9]+]], %got(us)($[[REG_GP]])
; CHECK-DAG:    lw      $[[US1_ADDR:[0-9]+]], %got(us1)($[[REG_GP]])
; CHECK-DAG:    lhu     $[[US1:[0-9]+]], 0($[[US1_ADDR]])
; CHECK-DAG:    ori     $[[CONST_60024:[0-9]+]], $zero, 60024
; CHECK-DAG:    xor     $[[RES:[0-9]+]], $[[US1]], $[[CONST_60024]]
; CHECK:        sh      $[[RES]], 0($[[US_ADDR]])
; CHECK:        .end    xorUs1
  ret void
}

attributes #0 = { noinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !3, i64 0}
