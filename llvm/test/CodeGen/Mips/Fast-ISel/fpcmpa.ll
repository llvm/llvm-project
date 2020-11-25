; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     -verify-machineinstrs < %s | FileCheck %s

@f1 = common global float 0.000000e+00, align 4
@f2 = common global float 0.000000e+00, align 4
@b1 = common global i32 0, align 4
@d1 = common global double 0.000000e+00, align 8
@d2 = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define void @feq1()  {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load float, float* @f2, align 4
  %cmp = fcmp oeq float %0, %1
; CHECK-LABEL:  feq1:
; CHECK:        lw      $[[REG_F1_GOT:[0-9]+]], %got(f1)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F1:[0-9]+]], 0($[[REG_F1_GOT]])
; CHECK:        lw      $[[REG_F2_GOT:[0-9]+]], %got(f2)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F2:[0-9]+]], 0($[[REG_F2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.eq.s  $f[[REG_F1]], $f[[REG_F2]]
; CHECK:        movt  $[[REG_ZERO]], $[[REG_ONE]], $fcc0

  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @fne1()  {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load float, float* @f2, align 4
  %cmp = fcmp une float %0, %1
; CHECK-LABEL:  fne1:
; CHECK:        lw      $[[REG_F1_GOT:[0-9]+]], %got(f1)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F1:[0-9]+]], 0($[[REG_F1_GOT]])
; CHECK:        lw      $[[REG_F2_GOT:[0-9]+]], %got(f2)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F2:[0-9]+]], 0($[[REG_F2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.eq.s  $f[[REG_F1]], $f[[REG_F2]]
; CHECK:        movf  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @flt1()  {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load float, float* @f2, align 4
  %cmp = fcmp olt float %0, %1
; CHECK-LABEL:  flt1:
; CHECK:        lw      $[[REG_F1_GOT:[0-9]+]], %got(f1)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F1:[0-9]+]], 0($[[REG_F1_GOT]])
; CHECK:        lw      $[[REG_F2_GOT:[0-9]+]], %got(f2)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F2:[0-9]+]], 0($[[REG_F2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.olt.s  $f[[REG_F1]], $f[[REG_F2]]
; CHECK:        movt  $[[REG_ZERO]], $[[REG_ONE]], $fcc0

  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @fgt1()  {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load float, float* @f2, align 4
  %cmp = fcmp ogt float %0, %1
; CHECK-LABEL: fgt1:
; CHECK:        lw      $[[REG_F1_GOT:[0-9]+]], %got(f1)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F1:[0-9]+]], 0($[[REG_F1_GOT]])
; CHECK:        lw      $[[REG_F2_GOT:[0-9]+]], %got(f2)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F2:[0-9]+]], 0($[[REG_F2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.ule.s  $f[[REG_F1]], $f[[REG_F2]]
; CHECK:        movf  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @fle1()  {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load float, float* @f2, align 4
  %cmp = fcmp ole float %0, %1
; CHECK-LABEL:  fle1:
; CHECK:        lw      $[[REG_F1_GOT:[0-9]+]], %got(f1)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F1:[0-9]+]], 0($[[REG_F1_GOT]])
; CHECK:        lw      $[[REG_F2_GOT:[0-9]+]], %got(f2)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F2:[0-9]+]], 0($[[REG_F2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.ole.s  $f[[REG_F1]], $f[[REG_F2]]
; CHECK:        movt  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @fge1()  {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load float, float* @f2, align 4
  %cmp = fcmp oge float %0, %1
; CHECK-LABEL:  fge1:
; CHECK:        lw      $[[REG_F1_GOT:[0-9]+]], %got(f1)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F1:[0-9]+]], 0($[[REG_F1_GOT]])
; CHECK:        lw      $[[REG_F2_GOT:[0-9]+]], %got(f2)(${{[0-9]+}})
; CHECK:        lwc1    $f[[REG_F2:[0-9]+]], 0($[[REG_F2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.ult.s  $f[[REG_F1]], $f[[REG_F2]]
; CHECK:        movf  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @deq1()  {
entry:
  %0 = load double, double* @d1, align 8
  %1 = load double, double* @d2, align 8
  %cmp = fcmp oeq double %0, %1
; CHECK-LABEL:  deq1:
; CHECK:        lw      $[[REG_D1_GOT:[0-9]+]], %got(d1)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D1:[0-9]+]], 0($[[REG_D1_GOT]])
; CHECK:        lw      $[[REG_D2_GOT:[0-9]+]], %got(d2)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D2:[0-9]+]], 0($[[REG_D2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.eq.d  $f[[REG_D1]], $f[[REG_D2]]
; CHECK:        movt  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @dne1()  {
entry:
  %0 = load double, double* @d1, align 8
  %1 = load double, double* @d2, align 8
  %cmp = fcmp une double %0, %1
; CHECK-LABEL:  dne1:
; CHECK:        lw      $[[REG_D1_GOT:[0-9]+]], %got(d1)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D1:[0-9]+]], 0($[[REG_D1_GOT]])
; CHECK:        lw      $[[REG_D2_GOT:[0-9]+]], %got(d2)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D2:[0-9]+]], 0($[[REG_D2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.eq.d  $f[[REG_D1]], $f[[REG_D2]]
; CHECK:        movf  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @dlt1()  {
entry:
  %0 = load double, double* @d1, align 8
  %1 = load double, double* @d2, align 8
  %cmp = fcmp olt double %0, %1
; CHECK-LABEL:  dlt1:
; CHECK:        lw      $[[REG_D1_GOT:[0-9]+]], %got(d1)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D1:[0-9]+]], 0($[[REG_D1_GOT]])
; CHECK:        lw      $[[REG_D2_GOT:[0-9]+]], %got(d2)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D2:[0-9]+]], 0($[[REG_D2_GOT]])
; CHECK:        addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK:        addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.olt.d  $f[[REG_D1]], $f[[REG_D2]]
; CHECK:        movt  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @dgt1()  {
entry:
  %0 = load double, double* @d1, align 8
  %1 = load double, double* @d2, align 8
  %cmp = fcmp ogt double %0, %1
; CHECK-LABEL:  dgt1:
; CHECK:        lw      $[[REG_D1_GOT:[0-9]+]], %got(d1)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D1:[0-9]+]], 0($[[REG_D1_GOT]])
; CHECK:        lw      $[[REG_D2_GOT:[0-9]+]], %got(d2)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D2:[0-9]+]], 0($[[REG_D2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.ule.d  $f[[REG_D1]], $f[[REG_D2]]
; CHECK:        movf  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @dle1()  {
entry:
  %0 = load double, double* @d1, align 8
  %1 = load double, double* @d2, align 8
  %cmp = fcmp ole double %0, %1
; CHECK-LABEL:  dle1:
; CHECK:        lw      $[[REG_D1_GOT:[0-9]+]], %got(d1)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D1:[0-9]+]], 0($[[REG_D1_GOT]])
; CHECK:        lw      $[[REG_D2_GOT:[0-9]+]], %got(d2)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D2:[0-9]+]], 0($[[REG_D2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.ole.d  $f[[REG_D1]], $f[[REG_D2]]
; CHECK:        movt  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @dge1()  {
entry:
  %0 = load double, double* @d1, align 8
  %1 = load double, double* @d2, align 8
  %cmp = fcmp oge double %0, %1
; CHECK-LABEL:  dge1:
; CHECK:        lw      $[[REG_D1_GOT:[0-9]+]], %got(d1)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D1:[0-9]+]], 0($[[REG_D1_GOT]])
; CHECK:        lw      $[[REG_D2_GOT:[0-9]+]], %got(d2)(${{[0-9]+}})
; CHECK:        ldc1    $f[[REG_D2:[0-9]+]], 0($[[REG_D2_GOT]])
; CHECK-DAG:    addiu   $[[REG_ZERO:[0-9]+]], $zero, 0
; CHECK-DAG:    addiu   $[[REG_ONE:[0-9]+]], $zero, 1
; CHECK:        c.ult.d  $f[[REG_D1]], $f[[REG_D2]]
; CHECK:        movf  $[[REG_ZERO]], $[[REG_ONE]], $fcc0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @b1, align 4
  ret void
}


