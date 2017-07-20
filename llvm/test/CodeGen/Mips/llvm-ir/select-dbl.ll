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

define double @tst_select_i1_double(i1 signext %s, double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_i1_double:

  ; M2:         andi    $[[T0:[0-9]+]], $4, 1
  ; M2:         bnez    $[[T0]], $[[BB0:BB[0-9_]+]]
  ; M2:         nop
  ; M2:         ldc1    $f0, 16($sp)
  ; M2:         jr      $ra
  ; M2:         nop
  ; M2:         $[[BB0]]:
  ; M2:         mtc1    $7, $f0
  ; M2:         jr      $ra
  ; M2:         mtc1    $6, $f1

  ; CMOV-32:      mtc1    $7, $[[F0:f[0-9]+]]
  ; CMOV-32R1:    mtc1    $6, $f{{[0-9]+}}
  ; CMOV-32R2-R5: mthc1   $6, $[[F0]]
  ; CMOV-32:      andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-32:      ldc1    $f0, 16($sp)
  ; CMOV-32:      movn.d  $f0, $[[F0]], $[[T0]]

  ; SEL-32:     mtc1    $7, $[[F0:f[0-9]+]]
  ; SEL-32:     mthc1   $6, $[[F0]]
  ; SEL-32:     ldc1    $[[F1:f[0-9]+]], 16($sp)
  ; SEL-32:     mtc1    $4, $f0
  ; SEL-32:     sel.d   $f0, $[[F1]], $[[F0]]

  ; M3:         andi    $[[T0:[0-9]+]], $4, 1
  ; M3:         bnez    $[[T0]], [[BB0:.LBB[0-9_]+]]
  ; M3:         nop
  ; M3:         mov.d   $f13, $f14
  ; M3:         [[BB0]]:
  ; M3:         jr      $ra
  ; M3:         mov.d   $f0, $f13

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $4, 1
  ; CMOV-64:    movn.d  $f14, $f13, $[[T0]]
  ; CMOV-64:    mov.d   $f0, $f14

  ; SEL-64:     mtc1    $4, $f0
  ; SEL-64:     sel.d   $f0, $f14, $f13

  ; MM32R3:     mtc1    $7, $[[F0:f[0-9]+]]
  ; MM32R3:     mthc1   $6, $[[F0]]
  ; MM32R3:     andi16  $[[T0:[0-9]+]], $4, 1
  ; MM32R3:     ldc1    $f0, 16($sp)
  ; MM32R3:     movn.d  $f0, $[[F0]], $[[T0]]

  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_i1_double_reordered(double %x, double %y,
                                              i1 signext %s) {
entry:
  ; ALL-LABEL: tst_select_i1_double_reordered:

  ; M2:         lw      $[[T0:[0-9]+]], 16($sp)
  ; M2:         andi    $[[T1:[0-9]+]], $[[T0]], 1
  ; M2:         bnez    $[[T1]], $[[BB0:BB[0-9_]+]]
  ; M2:         nop
  ; M2:         mov.d   $f12, $f14
  ; M2:         $[[BB0]]:
  ; M2:         jr      $ra
  ; M2:         mov.d   $f0, $f12

  ; CMOV-32:    lw      $[[T0:[0-9]+]], 16($sp)
  ; CMOV-32:    andi    $[[T1:[0-9]+]], $[[T0]], 1
  ; CMOV-32:    movn.d  $f14, $f12, $[[T1]]
  ; CMOV-32:    mov.d   $f0, $f14

  ; SEL-32:     lw      $[[T0:[0-9]+]], 16($sp)
  ; SEL-32:     mtc1    $[[T0]], $f0
  ; SEL-32:     sel.d   $f0, $f14, $f12

  ; M3:         andi    $[[T0:[0-9]+]], $6, 1
  ; M3:         bnez    $[[T0]], [[BB0:\.LBB[0-9_]+]]
  ; M3:         nop
  ; M3:         mov.d   $f12, $f13
  ; M3:         [[BB0]]:
  ; M3:         jr      $ra
  ; M3:         mov.d   $f0, $f12

  ; CMOV-64:    andi    $[[T0:[0-9]+]], $6, 1
  ; CMOV-64:    movn.d  $f13, $f12, $[[T0]]
  ; CMOV-64:    mov.d   $f0, $f13

  ; SEL-64:     mtc1    $6, $f0
  ; SEL-64:     sel.d   $f0, $f13, $f12

  ; MM32R3:     lw      $[[T0:[0-9]+]], 16($sp)
  ; MM32R3:     andi16  $[[T1:[0-9]+]], $[[T0:[0-9]+]], 1
  ; MM32R3:     movn.d  $f14, $f12, $[[T1]]
  ; MM32R3:     mov.d   $f0, $f14

  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_olt_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_olt_double:

  ; M2:         c.olt.d   $f12, $f14
  ; M3:         c.olt.d   $f12, $f13
  ; M2:         bc1t      [[BB0:\$BB[0-9_]+]]
  ; M3:         bc1t      [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      [[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.olt.d   $f12, $f14
  ; CMOV-32:    movt.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.lt.d  $f0, $f12, $f14
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.olt.d   $f12, $f13
  ; CMOV-64:    movt.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.lt.d  $f0, $f12, $f13
  ; SEL-64:     sel.d     $f0, $f13, $f12

  ; MM32R3:     c.olt.d   $f12, $f14
  ; MM32R3:     movt.d    $f14, $f12, $fcc0
  ; MM32R3:     mov.d     $f0, $f14

  %s = fcmp olt double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_ole_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ole_double:

  ; M2:         c.ole.d   $f12, $f14
  ; M3:         c.ole.d   $f12, $f13
  ; M2:         bc1t      [[BB0:\$BB[0-9_]+]]
  ; M3:         bc1t      [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      [[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ole.d   $f12, $f14
  ; CMOV-32:    movt.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.le.d  $f0, $f12, $f14
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ole.d   $f12, $f13
  ; CMOV-64:    movt.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.le.d  $f0, $f12, $f13
  ; SEL-64:     sel.d     $f0, $f13, $f12

  ; MM32R3:     c.ole.d   $f12, $f14
  ; MM32R3:     movt.d    $f14, $f12, $fcc0
  ; MM32R3:     mov.d     $f0, $f14

  %s = fcmp ole double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_ogt_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_ogt_double:

  ; M2:         c.ule.d   $f12, $f14
  ; M3:         c.ule.d   $f12, $f13
  ; M2:         bc1f      [[BB0:\$BB[0-9_]+]]
  ; M3:         bc1f      [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      [[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ule.d   $f12, $f14
  ; CMOV-32:    movf.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.lt.d  $f0, $f14, $f12
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ule.d   $f12, $f13
  ; CMOV-64:    movf.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.lt.d  $f0, $f13, $f12
  ; SEL-64:     sel.d     $f0, $f13, $f12

  ; MM32R3:     c.ule.d   $f12, $f14
  ; MM32R3:     movf.d    $f14, $f12, $fcc0
  ; MM32R3:     mov.d     $f0, $f14

  %s = fcmp ogt double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_oge_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oge_double:

  ; M2:         c.ult.d   $f12, $f14
  ; M3:         c.ult.d   $f12, $f13
  ; M2:         bc1f      [[BB0:\$BB[0-9_]+]]
  ; M3:         bc1f      [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      [[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ult.d   $f12, $f14
  ; CMOV-32:    movf.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.le.d  $f0, $f14, $f12
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ult.d   $f12, $f13
  ; CMOV-64:    movf.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.le.d  $f0, $f13, $f12
  ; SEL-64:     sel.d     $f0, $f13, $f12

  ; MM32R3:     c.ult.d   $f12, $f14
  ; MM32R3:     movf.d    $f14, $f12, $fcc0
  ; MM32R3:     mov.d     $f0, $f14

  %s = fcmp oge double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_oeq_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_oeq_double:

  ; M2:         c.eq.d    $f12, $f14
  ; M3:         c.eq.d    $f12, $f13
  ; M2:         bc1t      [[BB0:\$BB[0-9_]+]]
  ; M3:         bc1t      [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      [[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.eq.d    $f12, $f14
  ; CMOV-32:    movt.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.eq.d  $f0, $f12, $f14
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.eq.d    $f12, $f13
  ; CMOV-64:    movt.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.eq.d  $f0, $f12, $f13
  ; SEL-64:     sel.d     $f0, $f13, $f12

  ; MM32R3:     c.eq.d    $f12, $f14
  ; MM32R3:     movt.d    $f14, $f12, $fcc0
  ; MM32R3:     mov.d     $f0, $f14

  %s = fcmp oeq double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}

define double @tst_select_fcmp_one_double(double %x, double %y) {
entry:
  ; ALL-LABEL: tst_select_fcmp_one_double:

  ; M2:         c.ueq.d   $f12, $f14
  ; M3:         c.ueq.d   $f12, $f13
  ; M2:         bc1f      [[BB0:\$BB[0-9_]+]]
  ; M3:         bc1f      [[BB0:\.LBB[0-9_]+]]
  ; M2-M3:      nop
  ; M2:         mov.d     $f12, $f14
  ; M3:         mov.d     $f12, $f13
  ; M2-M3:      [[BB0]]:
  ; M2-M3:      jr        $ra
  ; M2-M3:      mov.d     $f0, $f12

  ; CMOV-32:    c.ueq.d   $f12, $f14
  ; CMOV-32:    movf.d    $f14, $f12, $fcc0
  ; CMOV-32:    mov.d     $f0, $f14

  ; SEL-32:     cmp.ueq.d $f0, $f12, $f14
  ; SEL-32:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     not       $[[T0]], $[[T0]]
  ; SEL-32:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-32:     sel.d     $f0, $f14, $f12

  ; CMOV-64:    c.ueq.d   $f12, $f13
  ; CMOV-64:    movf.d    $f13, $f12, $fcc0
  ; CMOV-64:    mov.d     $f0, $f13

  ; SEL-64:     cmp.ueq.d $f0, $f12, $f13
  ; SEL-64:     mfc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     not       $[[T0]], $[[T0]]
  ; SEL-64:     mtc1      $[[T0:[0-9]+]], $f0
  ; SEL-64:     sel.d     $f0, $f13, $f12

  ; MM32R3:     c.ueq.d   $f12, $f14
  ; MM32R3:     movf.d    $f14, $f12, $fcc0
  ; MM32R3:     mov.d     $f0, $f14

  %s = fcmp one double %x, %y
  %r = select i1 %s, double %x, double %y
  ret double %r
}
