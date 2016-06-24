; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,M2,M2-M3
; RUN: llc < %s -march=mips -mcpu=mips32 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R1
; RUN: llc < %s -march=mips -mcpu=mips32r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-32,CMOV-32R2-R5
; RUN: llc < %s -march=mips -mcpu=mips32r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,SEL-32,32R6
; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,M3,M2-M3
; RUN: llc < %s -march=mips64 -mcpu=mips4 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | FileCheck %s \
; RUN:    -check-prefixes=ALL,CMOV,CMOV-64
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | FileCheck %s \
; RUN:    -check-prefixes=ALL,SEL-64,64R6
; RUN: llc < %s -march=mips -mcpu=mips32r3 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM32R3
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s \
; RUN:    -check-prefixes=ALL,MM32R6,SEL-32

define float @tst_select_i1_float(i1 signext %s, float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_i1_float:

  ; M2-M3:      andi    $[[T0:[0-9]+]], $4, 1
  ; M2-M3:      bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         jr      $ra
  ; M2:         mtc1    $6, $f0
  ; M3:         mov.s   $f13, $f14
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr      $ra
  ; M2:         mtc1    $5, $f0
  ; M3:         mov.s   $f0, $f13

  ; CMOV-32:    mtc1    $6, $f0
  ; CMOV-32:    mtc1    $5, $f1
  ; CMOV-32:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-32:    movn.s  $f0, $f1, $[[T0]]

  ; SEL-32:     mtc1    $5, $[[F0:f[0-9]+]]
  ; SEL-32:     mtc1    $6, $[[F1:f[0-9]+]]
  ; SEL-32:     mtc1    $4, $f0
  ; SEL-32:     sel.s   $f0, $[[F1]], $[[F0]]

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-64:    movn.s  $f14, $f13, $[[T0]]
  ; CMOV-64:    mov.s   $f0, $f14

  ; SEL-64:     mtc1    $4, $f0
  ; SEL-64:     sel.s   $f0, $f14, $f13

  ; MM32R3:     mtc1    $6, $[[F0:f[0-9]+]]
  ; MM32R3:     mtc1    $5, $[[F1:f[0-9]+]]
  ; MM32R3:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R3:     movn.s  $f0, $[[F1]], $[[T0]]

  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_i1_float_reordered(float %x, float %y,
                                            i1 signext %s) {
entry:
  ; ALL-LABEL: tst_select_i1_float_reordered:

  ; M2-M3:      andi    $[[T0:[0-9]+]], $6, 1
  ; M2-M3:      bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s   $f12, $f14
  ; M3:         mov.s   $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr      $ra
  ; M2-M3:      mov.s   $f0, $f12

  ; CMOV-32:    andi    $[[T0:[0-9]+]], $6, 1
  ; CMOV-32:    movn.s  $f14, $f12, $[[T0]]
  ; CMOV-32:    mov.s   $f0, $f14

  ; SEL-32:     mtc1    $6, $f0
  ; SEL-32:     sel.s   $f0, $f14, $f12

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $6, 1
  ; CMOV-64:    movn.s  $f13, $f12, $[[T0]]
  ; CMOV-64:    mov.s   $f0, $f13

  ; SEL-64:     mtc1    $6, $f0
  ; SEL-64:     sel.s   $f0, $f13, $f12

  ; MM32R3:     andi16  $[[T0:[0-9]+]], $6, 1
  ; MM32R3:     movn.s  $[[F0:f[0-9]+]], $f12, $[[T0]]
  ; MM32R3:     mov.s   $f0, $[[F0]]

  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_olt_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_olt_float:

  ; M2:         c.olt.s   $f12, $f14
  ; M3:         c.olt.s   $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.olt.s   $f12, $f14
  ; CMOV-32:    movt.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.lt.s  $f0, $f12, $f14
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.olt.s   $f12, $f13
  ; CMOV-64:    movt.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.lt.s  $f0, $f12, $f13
  ; SEL-64:     sel.s     $f0, $f13, $f12

  ; MM32R3:     c.olt.s   $f12, $f14
  ; MM32R3:     movt.s    $f14, $f12, $fcc0
  ; MM32R3:     mov.s     $f0, $f14

  %s = fcmp olt float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_ole_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ole_float:

  ; M2:         c.ole.s   $f12, $f14
  ; M3:         c.ole.s   $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ole.s   $f12, $f14
  ; CMOV-32:    movt.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.le.s  $f0, $f12, $f14
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ole.s   $f12, $f13
  ; CMOV-64:    movt.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.le.s  $f0, $f12, $f13
  ; SEL-64:     sel.s     $f0, $f13, $f12

  ; MM32R3:     c.ole.s   $f12, $f14
  ; MM32R3:     movt.s    $f14, $f12, $fcc0
  ; MM32R3:     mov.s     $f0, $f14

  %s = fcmp ole float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_ogt_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ogt_float:

  ; M2:         c.ule.s   $f12, $f14
  ; M3:         c.ule.s   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ule.s   $f12, $f14
  ; CMOV-32:    movf.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.lt.s  $f0, $f14, $f12
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ule.s   $f12, $f13
  ; CMOV-64:    movf.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.lt.s  $f0, $f13, $f12
  ; SEL-64:     sel.s     $f0, $f13, $f12

  ; MM32R3:     c.ule.s   $f12, $f14
  ; MM32R3:     movf.s    $f14, $f12, $fcc0
  ; MM32R3:     mov.s     $f0, $f14

  %s = fcmp ogt float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_oge_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oge_float:

  ; M2:         c.ult.s   $f12, $f14
  ; M3:         c.ult.s   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ult.s   $f12, $f14
  ; CMOV-32:    movf.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.le.s  $f0, $f14, $f12
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ult.s   $f12, $f13
  ; CMOV-64:    movf.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.le.s  $f0, $f13, $f12
  ; SEL-64:     sel.s     $f0, $f13, $f12

  ; MM32R3:     c.ult.s   $f12, $f14
  ; MM32R3:     movf.s    $f14, $f12, $fcc0
  ; MM32R3:     mov.s     $f0, $f14

  %s = fcmp oge float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_oeq_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oeq_float:

  ; M2:         c.eq.s    $f12, $f14
  ; M3:         c.eq.s    $f12, $f13
  ; M2-M3:      bc1t      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.eq.s    $f12, $f14
  ; CMOV-32:    movt.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.eq.s  $f0, $f12, $f14
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.eq.s    $f12, $f13
  ; CMOV-64:    movt.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.eq.s  $f0, $f12, $f13
  ; SEL-64:     sel.s     $f0, $f13, $f12

  ; MM32R3:     c.eq.s    $f12, $f14
  ; MM32R3:     movt.s    $f14, $f12, $fcc0
  ; MM32R3:     mov.s     $f0, $f14

  %s = fcmp oeq float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}

define float @tst_select_fcmp_one_float(float %x, float %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_one_float:

  ; M2:         c.ueq.s   $f12, $f14
  ; M3:         c.ueq.s   $f12, $f13
  ; M2-M3:      bc1f      $[[BB0:BB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.s     $f12, $f14
  ; M3:         mov.s     $f12, $f13
  ; M2-M3:      $[[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.s     $f0, $f12

  ; CMOV-32:    c.ueq.s   $f12, $f14
  ; CMOV-32:    movf.s    $f14, $f12, $fcc0
  ; CMOV-32:    mov.s     $f0, $f14

  ; SEL-32:     cmp.ueq.s $f0, $f12, $f14
  ; SEL-32:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     not       $[[T0]], $[[T0]]
  ; SEL-32:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     sel.s     $f0, $f14, $f12

  ; CMOV-64:    c.ueq.s   $f12, $f13
  ; CMOV-64:    movf.s    $f13, $f12, $fcc0
  ; CMOV-64:    mov.s     $f0, $f13

  ; SEL-64:     cmp.ueq.s $f0, $f12, $f13
  ; SEL-64:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     not       $[[T0]], $[[T0]]
  ; SEL-64:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     sel.s     $f0, $f13, $f12

  ; MM32R3:     c.ueq.s   $f12, $f14
  ; MM32R3:     movf.s    $f14, $f12, $fcc0
  ; MM32R3:     mov.s     $f0, $f14

  %s = fcmp one float %x, %y
  %r = select i1 %s, float %x, float %y
  ret float %r
}
