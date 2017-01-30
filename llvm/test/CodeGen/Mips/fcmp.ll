; RUN: llc < %s -march=mips -mcpu=mips32 | \
; RUN:    FileCheck %s -check-prefixes=ALL,32-C
; RUN: llc < %s -march=mips -mcpu=mips32r2 | \
; RUN:    FileCheck %s -check-prefixes=ALL,32-C
; RUN: llc < %s -march=mips -mcpu=mips32r6 | \
; RUN:    FileCheck %s -check-prefixes=ALL,32-CMP
; RUN: llc < %s -march=mips64 -mcpu=mips4 | \
; RUN:    FileCheck %s -check-prefixes=ALL,64-C
; RUN: llc < %s -march=mips64 -mcpu=mips64 | \
; RUN:    FileCheck %s -check-prefixes=ALL,64-C
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | \
; RUN:    FileCheck %s -check-prefixes=ALL,64-C
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | \
; RUN:    FileCheck %s -check-prefixes=ALL,64-CMP
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MM32R3
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MMR6,MM32R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM,MMR6,MM64R6

define i32 @false_f32(float %a, float %b) nounwind {
; ALL-LABEL: false_f32:
; 32-C:          addiu $2, $zero, 0

; 32-CMP:        addiu $2, $zero, 0

; 64-C:          addiu $2, $zero, 0

; 64-CMP:        addiu $2, $zero, 0

; MM-DAG:        li16 $2, 0

  %1 = fcmp false float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @oeq_f32(float %a, float %b) nounwind {
; ALL-LABEL: oeq_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.eq.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.eq.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.eq.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp oeq float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ogt_f32(float %a, float %b) nounwind {
; ALL-LABEL: ogt_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ule.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ule.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ule.s $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ogt float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @oge_f32(float %a, float %b) nounwind {
; ALL-LABEL: oge_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ult.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ult.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ult.s $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp oge float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @olt_f32(float %a, float %b) nounwind {
; ALL-LABEL: olt_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.olt.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.olt.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.olt.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.lt.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp olt float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ole_f32(float %a, float %b) nounwind {
; ALL-LABEL: ole_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ole.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ole.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ole.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.le.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ole float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @one_f32(float %a, float %b) nounwind {
; ALL-LABEL: one_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ueq.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ueq.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 32-CMP-DAG:    andi $2, $[[T2]], 1

; 64-CMP-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 64-CMP-DAG:    andi $2, $[[T2]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ueq.s $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      not $[[T2:[0-9]+]], $[[T1]]
; MMR6-DAG:      andi16 $2, $[[T2]], 1

  %1 = fcmp one float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ord_f32(float %a, float %b) nounwind {
; ALL-LABEL: ord_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.un.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.un.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 32-CMP-DAG:    andi $2, $[[T2]], 1

; 64-CMP-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 64-CMP-DAG:    andi $2, $[[T2]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.un.s  $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      not $[[T2:[0-9]+]], $[[T1]]
; MMR6-DAG:      andi16 $2, $[[T2]], 1

  %1 = fcmp ord float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ueq_f32(float %a, float %b) nounwind {
; ALL-LABEL: ueq_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ueq.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ueq.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ueq.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ueq.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ueq float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ugt_f32(float %a, float %b) nounwind {
; ALL-LABEL: ugt_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ole.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ole.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ole.s $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ugt float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @uge_f32(float %a, float %b) nounwind {
; ALL-LABEL: uge_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.olt.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.olt.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.olt.s $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp uge float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ult_f32(float %a, float %b) nounwind {
; ALL-LABEL: ult_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ult.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ult.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ult.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ult.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ult float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ule_f32(float %a, float %b) nounwind {
; ALL-LABEL: ule_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ule.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ule.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ule.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ule.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ule float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @une_f32(float %a, float %b) nounwind {
; ALL-LABEL: une_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.eq.s $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.eq.s $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 32-CMP-DAG:    andi $2, $[[T2]], 1

; 64-CMP-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 64-CMP-DAG:    andi $2, $[[T2]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.eq.s $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.eq.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      not $[[T2:[0-9]+]], $[[T1]]
; MMR6-DAG:      andi16 $2, $[[T2]], 1

  %1 = fcmp une float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @uno_f32(float %a, float %b) nounwind {
; ALL-LABEL: uno_f32:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.un.s $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.un.s $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.un.s $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.un.s $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp uno float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @true_f32(float %a, float %b) nounwind {
; ALL-LABEL: true_f32:
; 32-C:          addiu $2, $zero, 1

; 32-CMP:        addiu $2, $zero, 1

; 64-C:          addiu $2, $zero, 1

; 64-CMP:        addiu $2, $zero, 1

; MM-DAG:        li16 $2, 1

  %1 = fcmp true float %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @false_f64(double %a, double %b) nounwind {
; ALL-LABEL: false_f64:
; 32-C:          addiu $2, $zero, 0

; 32-CMP:        addiu $2, $zero, 0

; 64-C:          addiu $2, $zero, 0

; 64-CMP:        addiu $2, $zero, 0

; MM-DAG:        li16 $2, 0

  %1 = fcmp false double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @oeq_f64(double %a, double %b) nounwind {
; ALL-LABEL: oeq_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.eq.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.eq.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.eq.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp oeq double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ogt_f64(double %a, double %b) nounwind {
; ALL-LABEL: ogt_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ule.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ule.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ule.d $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ogt double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @oge_f64(double %a, double %b) nounwind {
; ALL-LABEL: oge_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ult.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ult.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ult.d $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp oge double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @olt_f64(double %a, double %b) nounwind {
; ALL-LABEL: olt_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.olt.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.olt.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.olt.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.lt.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp olt double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ole_f64(double %a, double %b) nounwind {
; ALL-LABEL: ole_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ole.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ole.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ole.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.le.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ole double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @one_f64(double %a, double %b) nounwind {
; ALL-LABEL: one_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ueq.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ueq.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 32-CMP-DAG:    andi $2, $[[T2]], 1

; 64-CMP-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 64-CMP-DAG:    andi $2, $[[T2]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ueq.d $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      not $[[T2:[0-9]+]], $[[T1]]
; MMR6-DAG:      andi16 $2, $[[T2]], 1

  %1 = fcmp one double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ord_f64(double %a, double %b) nounwind {
; ALL-LABEL: ord_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.un.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.un.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 32-CMP-DAG:    andi $2, $[[T2]], 1

; 64-CMP-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 64-CMP-DAG:    andi $2, $[[T2]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.un.d $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      not $[[T2:[0-9]+]], $[[T1]]
; MMR6-DAG:      andi16 $2, $[[T2]], 1

  %1 = fcmp ord double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ueq_f64(double %a, double %b) nounwind {
; ALL-LABEL: ueq_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ueq.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ueq.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ueq.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ueq.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ueq double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ugt_f64(double %a, double %b) nounwind {
; ALL-LABEL: ugt_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ole.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ole.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ole.d $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ugt double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @uge_f64(double %a, double %b) nounwind {
; ALL-LABEL: uge_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.olt.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.olt.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.olt.d $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f14, $f12
; MM64R6-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f13, $f12
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp uge double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ult_f64(double %a, double %b) nounwind {
; ALL-LABEL: ult_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ult.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ult.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ult.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ult.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ult double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @ule_f64(double %a, double %b) nounwind {
; ALL-LABEL: ule_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.ule.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.ule.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.ule.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.ule.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp ule double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @une_f64(double %a, double %b) nounwind {
; ALL-LABEL: une_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.eq.d $f12, $f14
; 32-C:          movt $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.eq.d $f12, $f13
; 64-C:          movt $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 32-CMP-DAG:    andi $2, $[[T2]], 1

; 64-CMP-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    not $[[T2:[0-9]+]], $[[T1]]
; 64-CMP-DAG:    andi $2, $[[T2]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.eq.d  $f12, $f14
; MM32R3-DAG:    movt $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.eq.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      not $[[T2:[0-9]+]], $[[T1]]
; MMR6-DAG:      andi16 $2, $[[T2]], 1

  %1 = fcmp une double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @uno_f64(double %a, double %b) nounwind {
; ALL-LABEL: uno_f64:

; 32-C-DAG:      addiu $2, $zero, 1
; 32-C-DAG:      c.un.d $f12, $f14
; 32-C:          movf $2, $zero, $fcc0

; 64-C-DAG:      addiu $2, $zero, 1
; 64-C-DAG:      c.un.d $f12, $f13
; 64-C:          movf $2, $zero, $fcc0

; 32-CMP-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f14
; 32-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 32-CMP-DAG:    andi $2, $[[T1]], 1

; 64-CMP-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f13
; 64-CMP-DAG:    mfc1 $[[T1:[0-9]+]], $[[T0]]
; 64-CMP-DAG:    andi $2, $[[T1]], 1

; MM32R3-DAG:    li16 $[[T0:[0-9]+]], 0
; MM32R3-DAG:    li16 $[[T1:[0-9]+]], 1
; MM32R3-DAG:    c.un.d $f12, $f14
; MM32R3-DAG:    movf $[[T1]], $[[T0]], $fcc0

; MM32R6-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f14
; MM64R6-DAG:    cmp.un.d $[[T0:f[0-9]+]], $f12, $f13
; MMR6-DAG:      mfc1 $[[T1:[0-9]+]], $[[T0]]
; MMR6-DAG:      andi16 $2, $[[T1]], 1

  %1 = fcmp uno double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @true_f64(double %a, double %b) nounwind {
; ALL-LABEL: true_f64:
; 32-C:          addiu $2, $zero, 1

; 32-CMP:        addiu $2, $zero, 1

; 64-C:          addiu $2, $zero, 1

; 64-CMP:        addiu $2, $zero, 1

; MM-DAG:        li16 $2, 1

  %1 = fcmp true double %a, %b
  %2 = zext i1 %1 to i32
  ret i32 %2
}

; The optimizers sometimes produce setlt instead of setolt/setult.
define float @bug1_f32(float %angle, float %at) #0 {
entry:
; ALL-LABEL: bug1_f32:

; 32-C-DAG:      add.s    $[[T0:f[0-9]+]], $f14, $f12
; 32-C-DAG:      lwc1     $[[T1:f[0-9]+]], %lo($CPI32_0)(
; 32-C-DAG:      c.ole.s  $[[T0]], $[[T1]]
; 32-C-DAG:      bc1t

; 32-CMP-DAG:    add.s    $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    lwc1     $[[T1:f[0-9]+]], %lo($CPI32_0)(
; 32-CMP-DAG:    cmp.le.s $[[T2:f[0-9]+]], $[[T0]], $[[T1]]
; 32-CMP-DAG:    mfc1     $[[T3:[0-9]+]], $[[T2]]
; FIXME: This instruction is redundant.
; 32-CMP-DAG:    andi     $[[T4:[0-9]+]], $[[T3]], 1
; 32-CMP-DAG:    bnezc    $[[T4]],

; 64-C-DAG:      add.s    $[[T0:f[0-9]+]], $f13, $f12
; 64-C-DAG:      lwc1     $[[T1:f[0-9]+]], %lo(.LCPI32_0)(
; 64-C-DAG:      c.ole.s  $[[T0]], $[[T1]]
; 64-C-DAG:      bc1t

; 64-CMP-DAG:    add.s    $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    lwc1     $[[T1:f[0-9]+]], %lo(.LCPI32_0)(
; 64-CMP-DAG:    cmp.le.s $[[T2:f[0-9]+]], $[[T0]], $[[T1]]
; 64-CMP-DAG:    mfc1     $[[T3:[0-9]+]], $[[T2]]
; FIXME: This instruction is redundant.
; 64-CMP-DAG:    andi     $[[T4:[0-9]+]], $[[T3]], 1
; 64-CMP-DAG:    bnezc    $[[T4]],

; MM32R3-DAG:    add.s    $[[T0:f[0-9]+]], $f14, $f12
; MM32R3-DAG:    lui      $[[T1:[0-9]+]], %hi($CPI32_0)
; MM32R3-DAG:    lwc1     $[[T2:f[0-9]+]], %lo($CPI32_0)($[[T1]])
; MM32R3-DAG:    c.ole.s  $[[T0]], $[[T2]]
; MM32R3-DAG:    bc1t

; MM32R6-DAG:    add.s    $[[T0:f[0-9]+]], $f14, $f12
; MM32R6-DAG:    lui      $[[T1:[0-9]+]], %hi($CPI32_0)
; MM32R6-DAG:    lwc1     $[[T2:f[0-9]+]], %lo($CPI32_0)($[[T1]])
; MM32R6-DAG:    cmp.le.s $[[T3:f[0-9]+]], $[[T0]], $[[T2]]
; MM32R6-DAG:    mfc1     $[[T4:[0-9]+]], $[[T3:f[0-9]+]]
; MM32R6-DAG:    andi16   $[[T5:[0-9]+]], $[[T4]], 1
; MM32R6-DAG:    bnez     $[[T5]],

; MM64R6-DAG:    add.s    $[[T0:f[0-9]+]], $f13, $f12
; MM64R6-DAG:    lui      $[[T1:[0-9]+]], %highest(.LCPI32_0)
; MM64R6-DAG:    daddiu   $[[T2:[0-9]+]], $[[T1]], %higher(.LCPI32_0)
; MM64R6-DAG:    dsll     $[[T3:[0-9]+]], $[[T2]], 16
; MM64R6-DAG:    daddiu   $[[T4:[0-9]+]], $[[T3]], %hi(.LCPI32_0)
; MM64R6-DAG:    dsll     $[[T5:[0-9]+]], $[[T4]], 16
; MM64R6-DAG:    lwc1     $[[T6:f[0-9]+]], %lo(.LCPI32_0)($[[T5]])
; MM64R6-DAG:    cmp.le.s $[[T7:f[0-9]+]], $[[T0]], $[[T6]]
; MM64R6-DAG:    mfc1     $[[T8:[0-9]+]], $[[T7]]
; MM64R6-DAG:    andi16   $[[T9:[0-9]+]], $[[T8]], 1
; MM64R6-DAG:    bnez     $[[T9]],

  %add = fadd fast float %at, %angle
  %cmp = fcmp ogt float %add, 1.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %sub = fadd fast float %add, -1.000000e+00
  br label %if.end

if.end:
  %theta.0 = phi float [ %sub, %if.then ], [ %add, %entry ]
  ret float %theta.0
}

; The optimizers sometimes produce setlt instead of setolt/setult.
define double @bug1_f64(double %angle, double %at) #0 {
entry:
; ALL-LABEL: bug1_f64:

; 32-C-DAG:      add.d    $[[T0:f[0-9]+]], $f14, $f12
; 32-C-DAG:      ldc1     $[[T1:f[0-9]+]], %lo($CPI33_0)(
; 32-C-DAG:      c.ole.d  $[[T0]], $[[T1]]
; 32-C-DAG:      bc1t

; 32-CMP-DAG:    add.d    $[[T0:f[0-9]+]], $f14, $f12
; 32-CMP-DAG:    ldc1     $[[T1:f[0-9]+]], %lo($CPI33_0)(
; 32-CMP-DAG:    cmp.le.d $[[T2:f[0-9]+]], $[[T0]], $[[T1]]
; 32-CMP-DAG:    mfc1     $[[T3:[0-9]+]], $[[T2]]
; FIXME: This instruction is redundant.
; 32-CMP-DAG:    andi     $[[T4:[0-9]+]], $[[T3]], 1
; 32-CMP-DAG:    bnezc    $[[T4]],

; 64-C-DAG:      add.d    $[[T0:f[0-9]+]], $f13, $f12
; 64-C-DAG:      ldc1     $[[T1:f[0-9]+]], %lo(.LCPI33_0)(
; 64-C-DAG:      c.ole.d  $[[T0]], $[[T1]]
; 64-C-DAG:      bc1t

; 64-CMP-DAG:    add.d    $[[T0:f[0-9]+]], $f13, $f12
; 64-CMP-DAG:    ldc1     $[[T1:f[0-9]+]], %lo(.LCPI33_0)(
; 64-CMP-DAG:    cmp.le.d $[[T2:f[0-9]+]], $[[T0]], $[[T1]]
; 64-CMP-DAG:    mfc1     $[[T3:[0-9]+]], $[[T2]]
; FIXME: This instruction is redundant.
; 64-CMP-DAG:    andi     $[[T4:[0-9]+]], $[[T3]], 1
; 64-CMP-DAG:    bnezc    $[[T4]],

; MM32R3-DAG:    add.d    $[[T0:f[0-9]+]], $f14, $f12
; MM32R3-DAG:    lui      $[[T1:[0-9]+]], %hi($CPI33_0)
; MM32R3-DAG:    ldc1     $[[T2:f[0-9]+]], %lo($CPI33_0)($[[T1]])
; MM32R3-DAG:    c.ole.d  $[[T0]], $[[T2]]
; MM32R3-DAG:    bc1t

; MM32R6-DAG:    add.d    $[[T0:f[0-9]+]], $f14, $f12
; MM32R6-DAG:    lui      $[[T1:[0-9]+]], %hi($CPI33_0)
; MM32R6-DAG:    ldc1     $[[T2:f[0-9]+]], %lo($CPI33_0)($[[T1]])
; MM32R6-DAG:    cmp.le.d $[[T3:f[0-9]+]], $[[T0]], $[[T2]]
; MM32R6-DAG:    mfc1     $[[T4:[0-9]+]], $[[T3]]
; MM32R6-DAG:    andi16   $[[T5:[0-9]+]], $[[T4]], 1
; MM32R6-DAG:    bnez     $[[T5]],

; MM64R6-DAG:    add.d    $[[T0:f[0-9]+]], $f13, $f12
; MM64R6-DAG:    lui      $[[T1:[0-9]+]], %highest(.LCPI33_0)
; MM64R6-DAG:    daddiu   $[[T2:[0-9]+]], $[[T1]], %higher(.LCPI33_0)
; MM64R6-DAG:    dsll     $[[T3:[0-9]+]], $[[T2]], 16
; MM64R6-DAG:    daddiu   $[[T4:[0-9]+]], $[[T3]], %hi(.LCPI33_0)
; MM64R6-DAG:    dsll     $[[T5:[0-9]+]], $[[T4]], 16
; MM64R6-DAG:    ldc1     $[[T6:f[0-9]+]], %lo(.LCPI33_0)($[[T5]])
; MM64R6-DAG:    cmp.le.d $[[T7:f[0-9]+]], $[[T0]], $[[T6]]
; MM64R6-DAG:    mfc1     $[[T8:[0-9]+]], $[[T7]]
; MM64R6-DAG:    andi16   $[[T9:[0-9]+]], $[[T8]], 1
; MM64R6-DAG:    bnez     $[[T9]],

  %add = fadd fast double %at, %angle
  %cmp = fcmp ogt double %add, 1.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %sub = fadd fast double %add, -1.000000e+00
  br label %if.end

if.end:
  %theta.0 = phi double [ %sub, %if.then ], [ %add, %entry ]
  ret double %theta.0
}

attributes #0 = { nounwind readnone "no-nans-fp-math"="true" }
