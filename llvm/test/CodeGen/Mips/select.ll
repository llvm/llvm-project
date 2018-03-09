; RUN: llc < %s -march=mipsel   -mcpu=mips32   -relocation-model=pic -verify-machineinstrs | FileCheck %s -check-prefixes=ALL,32
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2 -relocation-model=pic -verify-machineinstrs | FileCheck %s -check-prefixes=ALL,32R2
; RUN: llc < %s -march=mipsel   -mcpu=mips32r6 -relocation-model=pic -verify-machineinstrs | FileCheck %s -check-prefixes=ALL,32R6
; RUN: llc < %s -march=mips64el -mcpu=mips64   -relocation-model=pic -verify-machineinstrs | FileCheck %s -check-prefixes=ALL,64
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -relocation-model=pic -verify-machineinstrs | FileCheck %s -check-prefixes=ALL,64R2
; RUN: llc < %s -march=mips64el -mcpu=mips64r6 -relocation-model=pic -verify-machineinstrs | FileCheck %s -check-prefixes=ALL,64R6

@d2 = external global double
@d3 = external global double

define i32 @i32_icmp_ne_i32_val(i32 signext %s, i32 signext %f0, i32 signext %f1) nounwind readnone {
entry:
; ALL-LABEL: i32_icmp_ne_i32_val:

; 32:            movn $5, $6, $4
; 32:            move $2, $5

; 32R2:          movn $5, $6, $4
; 32R2:          move $2, $5

; 32R6-DAG:      seleqz $[[T0:[0-9]+]], $5, $4
; 32R6-DAG:      selnez $[[T1:[0-9]+]], $6, $4
; 32R6:          or $2, $[[T1]], $[[T0]]

; 64:            movn $5, $6, $4
; 64:            move $2, $5

; 64R2:          movn $5, $6, $4
; 64R2:          move $2, $5

; 64R6-DAG:      seleqz $[[T0:[0-9]+]], $5, $4
; 64R6-DAG:      selnez $[[T1:[0-9]+]], $6, $4
; 64R6:          or $2, $[[T1]], $[[T0]]

  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, i32 %f1, i32 %f0
  ret i32 %cond
}

define i64 @i32_icmp_ne_i64_val(i32 signext %s, i64 %f0, i64 %f1) nounwind readnone {
entry:
; ALL-LABEL: i32_icmp_ne_i64_val:

; 32-DAG:        lw $[[F1:[0-9]+]], 16($sp)
; 32-DAG:        movn $6, $[[F1]], $4
; 32-DAG:        lw $[[F1H:[0-9]+]], 20($sp)
; 32:            movn $7, $[[F1H]], $4
; 32:            move $2, $6
; 32:            move $3, $7

; 32R2-DAG:      lw $[[F1:[0-9]+]], 16($sp)
; 32R2-DAG:      movn $6, $[[F1]], $4
; 32R2-DAG:      lw $[[F1H:[0-9]+]], 20($sp)
; 32R2:          movn $7, $[[F1H]], $4
; 32R2:          move $2, $6
; 32R2:          move $3, $7

; 32R6-DAG:      lw $[[F1:[0-9]+]], 16($sp)
; 32R6-DAG:      seleqz $[[T0:[0-9]+]], $6, $4
; 32R6-DAG:      selnez $[[T1:[0-9]+]], $[[F1]], $4
; 32R6:          or $2, $[[T1]], $[[T0]]
; 32R6-DAG:      lw $[[F1H:[0-9]+]], 20($sp)
; 32R6-DAG:      seleqz $[[T0:[0-9]+]], $7, $4
; 32R6-DAG:      selnez $[[T1:[0-9]+]], $[[F1H]], $4
; 32R6:          or $3, $[[T1]], $[[T0]]

; 64:            movn $5, $6, $4
; 64:            move $2, $5

; 64R2:          movn $5, $6, $4
; 64R2:          move $2, $5

; FIXME: This sll works around an implementation detail in the code generator
;        (setcc's result is i32 so bits 32-63 are undefined). It's not really
;        needed.
; 64R6-DAG:      sll $[[CC:[0-9]+]], $4, 0
; 64R6-DAG:      seleqz $[[T0:[0-9]+]], $5, $[[CC]]
; 64R6-DAG:      selnez $[[T1:[0-9]+]], $6, $[[CC]]
; 64R6:          or $2, $[[T1]], $[[T0]]

  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, i64 %f1, i64 %f0
  ret i64 %cond
}

define i64 @i64_icmp_ne_i64_val(i64 %s, i64 %f0, i64 %f1) nounwind readnone {
entry:
; ALL-LABEL: i64_icmp_ne_i64_val:

; 32-DAG:        or $[[CC:[0-9]+]], $4
; 32-DAG:        lw $[[F1:[0-9]+]], 16($sp)
; 32-DAG:        movn $6, $[[F1]], $[[CC]]
; 32-DAG:        lw $[[F1H:[0-9]+]], 20($sp)
; 32:            movn $7, $[[F1H]], $[[CC]]
; 32:            move $2, $6
; 32:            move $3, $7

; 32R2-DAG:      or $[[CC:[0-9]+]], $4
; 32R2-DAG:      lw $[[F1:[0-9]+]], 16($sp)
; 32R2-DAG:      movn $6, $[[F1]], $[[CC]]
; 32R2-DAG:      lw $[[F1H:[0-9]+]], 20($sp)
; 32R2:          movn $7, $[[F1H]], $[[CC]]
; 32R2:          move $2, $6
; 32R2:          move $3, $7

; 32R6-DAG:      lw $[[F1:[0-9]+]], 16($sp)
; 32R6-DAG:      or $[[T2:[0-9]+]], $4, $5
; 32R6-DAG:      seleqz $[[T0:[0-9]+]], $6, $[[T2]]
; 32R6-DAG:      selnez $[[T1:[0-9]+]], $[[F1]], $[[T2]]
; 32R6:          or $2, $[[T1]], $[[T0]]
; 32R6-DAG:      lw $[[F1H:[0-9]+]], 20($sp)
; 32R6-DAG:      seleqz $[[T0:[0-9]+]], $7, $[[T2]]
; 32R6-DAG:      selnez $[[T1:[0-9]+]], $[[F1H]], $[[T2]]
; 32R6:          or $3, $[[T1]], $[[T0]]

; 64:            movn $5, $6, $4
; 64:            move $2, $5

; 64R2:          movn $5, $6, $4
; 64R2:          move $2, $5

; 64R6-DAG:      seleqz $[[T0:[0-9]+]], $5, $4
; 64R6-DAG:      selnez $[[T1:[0-9]+]], $6, $4
; 64R6:          or $2, $[[T1]], $[[T0]]

  %tobool = icmp ne i64 %s, 0
  %cond = select i1 %tobool, i64 %f1, i64 %f0
  ret i64 %cond
}

define float @i32_icmp_ne_f32_val(i32 signext %s, float %f0, float %f1) nounwind readnone {
entry:
; ALL-LABEL: i32_icmp_ne_f32_val:

; 32-DAG:        mtc1 $5, $[[F0:f[0-9]+]]
; 32-DAG:        mtc1 $6, $[[F1:f0]]
; 32:            movn.s $[[F1]], $[[F0]], $4

; 32R2-DAG:      mtc1 $5, $[[F0:f[0-9]+]]
; 32R2-DAG:      mtc1 $6, $[[F1:f0]]
; 32R2:          movn.s $[[F1]], $[[F0]], $4

; 32R6:          sltu $[[T0:[0-9]+]], $zero, $4
; 32R6:          negu $[[T0]], $[[T0]]
; 32R6-DAG:      mtc1 $5, $[[F0:f[0-9]+]]
; 32R6-DAG:      mtc1 $6, $[[F1:f[0-9]+]]
; 32R6:          mtc1 $[[T0]], $[[CC:f0]]
; 32R6:          sel.s $[[CC]], $[[F1]], $[[F0]]

; 64:            movn.s $f14, $f13, $4
; 64:            mov.s $f0, $f14

; 64R2:          movn.s $f14, $f13, $4
; 64R2:          mov.s $f0, $f14

; 64R6:          sltu $[[T0:[0-9]+]], $zero, $4
; 64R6:          mtc1 $[[T0]], $[[CC:f0]]
; 64R6:          sel.s $[[CC]], $f14, $f13

  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, float %f0, float %f1
  ret float %cond
}

define double @i32_icmp_ne_f64_val(i32 signext %s, double %f0, double %f1) nounwind readnone {
entry:
; ALL-LABEL: i32_icmp_ne_f64_val:

; 32-DAG:        mtc1 $6, $[[F0:f[1-3]*[02468]+]]
; 32-DAG:        mtc1 $7, $[[F0H:f[1-3]*[13579]+]]
; 32-DAG:        ldc1 $[[F1:f0]], 16($sp)
; 32:            movn.d $[[F1]], $[[F0]], $4

; 32R2-DAG:      mtc1 $6, $[[F0:f[0-9]+]]
; 32R2-DAG:      mthc1 $7, $[[F0]]
; 32R2-DAG:      ldc1 $[[F1:f0]], 16($sp)
; 32R2:          movn.d $[[F1]], $[[F0]], $4

; 32R6-DAG:      mtc1 $6, $[[F0:f[0-9]+]]
; 32R6-DAG:      mthc1 $7, $[[F0]]
; 32R6-DAG:      sltu $[[T0:[0-9]+]], $zero, $4
; 32R6-DAG:      mtc1 $[[T0]], $[[CC:f0]]
; 32R6-DAG:      ldc1 $[[F1:f[0-9]+]], 16($sp)
; 32R6:          sel.d $[[CC]], $[[F1]], $[[F0]]

; 64:            movn.d $f14, $f13, $4
; 64:            mov.d $f0, $f14

; 64R2:          movn.d $f14, $f13, $4
; 64R2:          mov.d $f0, $f14

; 64R6-DAG:      sltu $[[T0:[0-9]+]], $zero, $4
; 64R6-DAG:      mtc1 $[[T0]], $[[CC:f0]]
; 64R6:          sel.d $[[CC]], $f14, $f13

  %tobool = icmp ne i32 %s, 0
  %cond = select i1 %tobool, double %f0, double %f1
  ret double %cond
}

define float @f32_fcmp_oeq_f32_val(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_oeq_f32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[0-9]+]]
; 32-DAG:        mtc1 $7, $[[F3:f[0-9]+]]
; 32:            c.eq.s $[[F2]], $[[F3]]
; 32:            movt.s $f14, $f12, $fcc0
; 32:            mov.s $f0, $f14

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R2:          c.eq.s $[[F2]], $[[F3]]
; 32R2:          movt.s $f14, $f12, $fcc0
; 32R2:          mov.s $f0, $f14

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R6:          cmp.eq.s $[[CC:f0]], $[[F2]], $[[F3]]
; 32R6:          sel.s $[[CC]], $f14, $f12

; 64:            c.eq.s $f14, $f15
; 64:            movt.s $f13, $f12, $fcc0
; 64:            mov.s $f0, $f13

; 64R2:          c.eq.s $f14, $f15
; 64R2:          movt.s $f13, $f12, $fcc0
; 64R2:          mov.s $f0, $f13

; 64R6:          cmp.eq.s $[[CC:f0]], $f14, $f15
; 64R6:          sel.s $[[CC]], $f13, $f12

  %cmp = fcmp oeq float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define float @f32_fcmp_olt_f32_val(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_olt_f32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[0-9]+]]
; 32-DAG:        mtc1 $7, $[[F3:f[0-9]+]]
; 32:            c.olt.s $[[F2]], $[[F3]]
; 32:            movt.s $f14, $f12, $fcc0
; 32:            mov.s $f0, $f14

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R2:          c.olt.s $[[F2]], $[[F3]]
; 32R2:          movt.s $f14, $f12, $fcc0
; 32R2:          mov.s $f0, $f14

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R6:          cmp.lt.s $[[CC:f0]], $[[F2]], $[[F3]]
; 32R6:          sel.s $[[CC]], $f14, $f12

; 64:            c.olt.s $f14, $f15
; 64:            movt.s $f13, $f12, $fcc0
; 64:            mov.s $f0, $f13

; 64R2:          c.olt.s $f14, $f15
; 64R2:          movt.s $f13, $f12, $fcc0
; 64R2:          mov.s $f0, $f13

; 64R6:          cmp.lt.s $[[CC:f0]], $f14, $f15
; 64R6:          sel.s $[[CC]], $f13, $f12

  %cmp = fcmp olt float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define float @f32_fcmp_ogt_f32_val(float %f0, float %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_ogt_f32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[0-9]+]]
; 32-DAG:        mtc1 $7, $[[F3:f[0-9]+]]
; 32:            c.ule.s $[[F2]], $[[F3]]
; 32:            movf.s $f14, $f12, $fcc0
; 32:            mov.s $f0, $f14

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R2:          c.ule.s $[[F2]], $[[F3]]
; 32R2:          movf.s $f14, $f12, $fcc0
; 32R2:          mov.s $f0, $f14

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R6:          cmp.lt.s $[[CC:f0]], $[[F3]], $[[F2]]
; 32R6:          sel.s $[[CC]], $f14, $f12

; 64:            c.ule.s $f14, $f15
; 64:            movf.s $f13, $f12, $fcc0
; 64:            mov.s $f0, $f13

; 64R2:          c.ule.s $f14, $f15
; 64R2:          movf.s $f13, $f12, $fcc0
; 64R2:          mov.s $f0, $f13

; 64R6:          cmp.lt.s $[[CC:f0]], $f15, $f14
; 64R6:          sel.s $[[CC]], $f13, $f12

  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define double @f32_fcmp_ogt_f64_val(double %f0, double %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_ogt_f64_val:

; 32-DAG:        lwc1 $[[F2:f[0-9]+]], 16($sp)
; 32-DAG:        lwc1 $[[F3:f[0-9]+]], 20($sp)
; 32:            c.ule.s $[[F2]], $[[F3]]
; 32:            movf.d $f14, $f12, $fcc0
; 32:            mov.d $f0, $f14

; 32R2-DAG:      lwc1 $[[F2:f[0-9]+]], 16($sp)
; 32R2-DAG:      lwc1 $[[F3:f[0-9]+]], 20($sp)
; 32R2:          c.ule.s $[[F2]], $[[F3]]
; 32R2:          movf.d $f14, $f12, $fcc0
; 32R2:          mov.d $f0, $f14

; 32R6-DAG:      lwc1 $[[F2:f[0-9]+]], 16($sp)
; 32R6-DAG:      lwc1 $[[F3:f[0-9]+]], 20($sp)
; 32R6:          cmp.lt.s $[[CC:f0]], $[[F3]], $[[F2]]
; 32R6:          sel.d $[[CC]], $f14, $f12

; 64:            c.ule.s $f14, $f15
; 64:            movf.d $f13, $f12, $fcc0
; 64:            mov.d $f0, $f13

; 64R2:          c.ule.s $f14, $f15
; 64R2:          movf.d $f13, $f12, $fcc0
; 64R2:          mov.d $f0, $f13

; 64R6:          cmp.lt.s $[[CC:f0]], $f15, $f14
; 64R6:          sel.d $[[CC]], $f13, $f12

  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @f64_fcmp_oeq_f64_val(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; ALL-LABEL: f64_fcmp_oeq_f64_val:

; 32-DAG:        ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32-DAG:        ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32:            c.eq.d $[[F2]], $[[F3]]
; 32:            movt.d $f14, $f12, $fcc0
; 32:            mov.d $f0, $f14

; 32R2-DAG:      ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32R2-DAG:      ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32R2:          c.eq.d $[[F2]], $[[F3]]
; 32R2:          movt.d $f14, $f12, $fcc0
; 32R2:          mov.d $f0, $f14

; 32R6-DAG:      ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32R6-DAG:      ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32R6:          cmp.eq.d $[[CC:f0]], $[[F2]], $[[F3]]
; 32R6:          sel.d $[[CC]], $f14, $f12

; 64:            c.eq.d $f14, $f15
; 64:            movt.d $f13, $f12, $fcc0
; 64:            mov.d $f0, $f13

; 64R2:          c.eq.d $f14, $f15
; 64R2:          movt.d $f13, $f12, $fcc0
; 64R2:          mov.d $f0, $f13

; 64R6:          cmp.eq.d $[[CC:f0]], $f14, $f15
; 64R6:          sel.d $[[CC]], $f13, $f12

  %cmp = fcmp oeq double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @f64_fcmp_olt_f64_val(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; ALL-LABEL: f64_fcmp_olt_f64_val:

; 32-DAG:        ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32-DAG:        ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32:            c.olt.d $[[F2]], $[[F3]]
; 32:            movt.d $f14, $f12, $fcc0
; 32:            mov.d $f0, $f14

; 32R2-DAG:      ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32R2-DAG:      ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32R2:          c.olt.d $[[F2]], $[[F3]]
; 32R2:          movt.d $f14, $f12, $fcc0
; 32R2:          mov.d $f0, $f14

; 32R6-DAG:      ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32R6-DAG:      ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32R6:          cmp.lt.d $[[CC:f0]], $[[F2]], $[[F3]]
; 32R6:          sel.d $[[CC]], $f14, $f12

; 64:            c.olt.d $f14, $f15
; 64:            movt.d $f13, $f12, $fcc0
; 64:            mov.d $f0, $f13

; 64R2:          c.olt.d $f14, $f15
; 64R2:          movt.d $f13, $f12, $fcc0
; 64R2:          mov.d $f0, $f13

; 64R6:          cmp.lt.d $[[CC:f0]], $f14, $f15
; 64R6:          sel.d $[[CC]], $f13, $f12

  %cmp = fcmp olt double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define double @f64_fcmp_ogt_f64_val(double %f0, double %f1, double %f2, double %f3) nounwind readnone {
entry:
; ALL-LABEL: f64_fcmp_ogt_f64_val:

; 32-DAG:        ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32-DAG:        ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32:            c.ule.d $[[F2]], $[[F3]]
; 32:            movf.d $f14, $f12, $fcc0
; 32:            mov.d $f0, $f14

; 32R2-DAG:      ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32R2-DAG:      ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32R2:          c.ule.d $[[F2]], $[[F3]]
; 32R2:          movf.d $f14, $f12, $fcc0
; 32R2:          mov.d $f0, $f14

; 32R6-DAG:      ldc1 $[[F2:f[0-9]+]], 16($sp)
; 32R6-DAG:      ldc1 $[[F3:f[0-9]+]], 24($sp)
; 32R6:          cmp.lt.d $[[CC:f0]], $[[F3]], $[[F2]]
; 32R6:          sel.d $[[CC]], $f14, $f12

; 64:            c.ule.d $f14, $f15
; 64:            movf.d $f13, $f12, $fcc0
; 64:            mov.d $f0, $f13

; 64R2:          c.ule.d $f14, $f15
; 64R2:          movf.d $f13, $f12, $fcc0
; 64R2:          mov.d $f0, $f13

; 64R6:          cmp.lt.d $[[CC:f0]], $f15, $f14
; 64R6:          sel.d $[[CC]], $f13, $f12

  %cmp = fcmp ogt double %f2, %f3
  %cond = select i1 %cmp, double %f0, double %f1
  ret double %cond
}

define float @f64_fcmp_ogt_f32_val(float %f0, float %f1, double %f2, double %f3) nounwind readnone {
entry:
; ALL-LABEL: f64_fcmp_ogt_f32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[1-3]*[02468]+]]
; 32-DAG:        mtc1 $7, $[[F2H:f[1-3]*[13579]+]]
; 32-DAG:        ldc1 $[[F3:f[0-9]+]], 16($sp)
; 32:            c.ule.d $[[F2]], $[[F3]]
; 32:            movf.s $f14, $f12, $fcc0
; 32:            mov.s $f0, $f14

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mthc1 $7, $[[F2]]
; 32R2-DAG:      ldc1 $[[F3:f[0-9]+]], 16($sp)
; 32R2:          c.ule.d $[[F2]], $[[F3]]
; 32R2:          movf.s $f14, $f12, $fcc0
; 32R2:          mov.s $f0, $f14

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mthc1 $7, $[[F2]]
; 32R6-DAG:      ldc1 $[[F3:f[0-9]+]], 16($sp)
; 32R6:          cmp.lt.d $[[CC:f0]], $[[F3]], $[[F2]]
; 32R6:          sel.s $[[CC]], $f14, $f12

; 64:            c.ule.d $f14, $f15
; 64:            movf.s $f13, $f12, $fcc0
; 64:            mov.s $f0, $f13

; 64R2:          c.ule.d $f14, $f15
; 64R2:          movf.s $f13, $f12, $fcc0
; 64R2:          mov.s $f0, $f13

; 64R6:          cmp.lt.d $[[CC:f0]], $f15, $f14
; 64R6:          sel.s $[[CC]], $f13, $f12

  %cmp = fcmp ogt double %f2, %f3
  %cond = select i1 %cmp, float %f0, float %f1
  ret float %cond
}

define i32 @f32_fcmp_oeq_i32_val(i32 signext %f0, i32 signext %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_oeq_i32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[0-9]+]]
; 32-DAG:        mtc1 $7, $[[F3:f[0-9]+]]
; 32:            c.eq.s $[[F2]], $[[F3]]
; 32:            movt $5, $4, $fcc0
; 32:            move $2, $5

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R2:          c.eq.s $[[F2]], $[[F3]]
; 32R2:          movt $5, $4, $fcc0
; 32R2:          move $2, $5

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R6:          cmp.eq.s $[[CC:f[0-9]+]], $[[F2]], $[[F3]]
; 32R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 32R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 32R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 32R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 32R6:          or $2, $[[NE]], $[[EQ]]

; 64:            c.eq.s $f14, $f15
; 64:            movt $5, $4, $fcc0
; 64:            move $2, $5

; 64R2:          c.eq.s $f14, $f15
; 64R2:          movt $5, $4, $fcc0
; 64R2:          move $2, $5

; 64R6:          cmp.eq.s $[[CC:f[0-9]+]], $f14, $f15
; 64R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 64R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 64R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 64R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 64R6:          or $2, $[[NE]], $[[EQ]]

  %cmp = fcmp oeq float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @f32_fcmp_olt_i32_val(i32 signext %f0, i32 signext %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_olt_i32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[0-9]+]]
; 32-DAG:        mtc1 $7, $[[F3:f[0-9]+]]
; 32:            c.olt.s $[[F2]], $[[F3]]
; 32:            movt $5, $4, $fcc0
; 32:            move $2, $5

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R2:          c.olt.s $[[F2]], $[[F3]]
; 32R2:          movt $5, $4, $fcc0
; 32R2:          move $2, $5

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R6:          cmp.lt.s $[[CC:f[0-9]+]], $[[F2]], $[[F3]]
; 32R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 32R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 32R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 32R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 32R6:          or $2, $[[NE]], $[[EQ]]

; 64:            c.olt.s $f14, $f15
; 64:            movt $5, $4, $fcc0
; 64:            move $2, $5

; 64R2:          c.olt.s $f14, $f15
; 64R2:          movt $5, $4, $fcc0
; 64R2:          move $2, $5

; 64R6:          cmp.lt.s $[[CC:f[0-9]+]], $f14, $f15
; 64R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 64R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 64R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 64R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 64R6:          or $2, $[[NE]], $[[EQ]]
  %cmp = fcmp olt float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @f32_fcmp_ogt_i32_val(i32 signext %f0, i32 signext %f1, float %f2, float %f3) nounwind readnone {
entry:
; ALL-LABEL: f32_fcmp_ogt_i32_val:

; 32-DAG:        mtc1 $6, $[[F2:f[0-9]+]]
; 32-DAG:        mtc1 $7, $[[F3:f[0-9]+]]
; 32:            c.ule.s $[[F2]], $[[F3]]
; 32:            movf $5, $4, $fcc0
; 32:            move $2, $5

; 32R2-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R2-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R2:          c.ule.s $[[F2]], $[[F3]]
; 32R2:          movf $5, $4, $fcc0
; 32R2:          move $2, $5

; 32R6-DAG:      mtc1 $6, $[[F2:f[0-9]+]]
; 32R6-DAG:      mtc1 $7, $[[F3:f[0-9]+]]
; 32R6:          cmp.lt.s $[[CC:f[0-9]+]], $[[F3]], $[[F2]]
; 32R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 32R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 32R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 32R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 32R6:          or $2, $[[NE]], $[[EQ]]

; 64:            c.ule.s $f14, $f15
; 64:            movf $5, $4, $fcc0
; 64:            move $2, $5

; 64R2:          c.ule.s $f14, $f15
; 64R2:          movf $5, $4, $fcc0
; 64R2:          move $2, $5

; 64R6:          cmp.lt.s $[[CC:f[0-9]+]], $f15, $f14
; 64R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 64R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 64R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 64R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 64R6:          or $2, $[[NE]], $[[EQ]]

  %cmp = fcmp ogt float %f2, %f3
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @f64_fcmp_oeq_i32_val(i32 signext %f0, i32 signext %f1) nounwind readonly {
entry:
; ALL-LABEL: f64_fcmp_oeq_i32_val:

; 32-DAG:        addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32-DAG:        addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32-DAG:        lw $[[D2:[0-9]+]], %got(d2)($1)
; 32-DAG:        ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32-DAG:        lw $[[D3:[0-9]+]], %got(d3)($1)
; 32-DAG:        ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32:            c.eq.d $[[TMP]], $[[TMP1]]
; 32:            movt $5, $4, $fcc0
; 32:            move $2, $5

; 32R2-DAG:      addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32R2-DAG:      addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32R2-DAG:      lw $[[D2:[0-9]+]], %got(d2)($1)
; 32R2-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32R2-DAG:      lw $[[D3:[0-9]+]], %got(d3)($1)
; 32R2-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32R2:          c.eq.d $[[TMP]], $[[TMP1]]
; 32R2:          movt $5, $4, $fcc0
; 32R2:          move $2, $5

; 32R6-DAG:      addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32R6-DAG:      addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32R6-DAG:      lw $[[D2:[0-9]+]], %got(d2)($1)
; 32R6-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32R6-DAG:      lw $[[D3:[0-9]+]], %got(d3)($1)
; 32R6-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32R6:          cmp.eq.d $[[CC:f[0-9]+]], $[[TMP]], $[[TMP1]]
; 32R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 32R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 32R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 32R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 32R6:          or $2, $[[NE]], $[[EQ]]

; 64-DAG:        daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_oeq_i32_val)))
; 64-DAG:        daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64-DAG:        ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64-DAG:        ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64-DAG:        ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64-DAG:        ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64:            c.eq.d $[[TMP]], $[[TMP1]]
; 64:            movt $5, $4, $fcc0
; 64:            move $2, $5

; 64R2-DAG:      daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_oeq_i32_val)))
; 64R2-DAG:      daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64R2-DAG:      ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64R2-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64R2-DAG:      ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64R2-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64R2:          c.eq.d $[[TMP]], $[[TMP1]]
; 64R2:          movt $5, $4, $fcc0
; 64R2:          move $2, $5

; 64R6-DAG:      daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_oeq_i32_val)))
; 64R6-DAG:      daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64R6-DAG:      ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64R6-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64R6-DAG:      ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64R6-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64R6:          cmp.eq.d $[[CC:f[0-9]+]], $[[TMP]], $[[TMP1]]
; 64R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 64R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 64R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 64R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 64R6:          or $2, $[[NE]], $[[EQ]]

  %tmp = load double, double* @d2, align 8
  %tmp1 = load double, double* @d3, align 8
  %cmp = fcmp oeq double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @f64_fcmp_olt_i32_val(i32 signext %f0, i32 signext %f1) nounwind readonly {
entry:
; ALL-LABEL: f64_fcmp_olt_i32_val:

; 32-DAG:        addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32-DAG:        addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32-DAG:        lw $[[D2:[0-9]+]], %got(d2)($1)
; 32-DAG:        ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32-DAG:        lw $[[D3:[0-9]+]], %got(d3)($1)
; 32-DAG:        ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32:            c.olt.d $[[TMP]], $[[TMP1]]
; 32:            movt $5, $4, $fcc0
; 32:            move $2, $5

; 32R2-DAG:      addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32R2-DAG:      addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32R2-DAG:      lw $[[D2:[0-9]+]], %got(d2)($1)
; 32R2-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32R2-DAG:      lw $[[D3:[0-9]+]], %got(d3)($1)
; 32R2-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32R2:          c.olt.d $[[TMP]], $[[TMP1]]
; 32R2:          movt $5, $4, $fcc0
; 32R2:          move $2, $5

; 32R6-DAG:      addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32R6-DAG:      addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32R6-DAG:      lw $[[D2:[0-9]+]], %got(d2)($1)
; 32R6-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32R6-DAG:      lw $[[D3:[0-9]+]], %got(d3)($1)
; 32R6-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32R6:          cmp.lt.d $[[CC:f[0-9]+]], $[[TMP]], $[[TMP1]]
; 32R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 32R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 32R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 32R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 32R6:          or $2, $[[NE]], $[[EQ]]

; 64-DAG:        daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_olt_i32_val)))
; 64-DAG:        daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64-DAG:        ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64-DAG:        ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64-DAG:        ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64-DAG:        ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64:            c.olt.d $[[TMP]], $[[TMP1]]
; 64:            movt $5, $4, $fcc0
; 64:            move $2, $5

; 64R2-DAG:      daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_olt_i32_val)))
; 64R2-DAG:      daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64R2-DAG:      ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64R2-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64R2-DAG:      ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64R2-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64R2:          c.olt.d $[[TMP]], $[[TMP1]]
; 64R2:          movt $5, $4, $fcc0
; 64R2:          move $2, $5

; 64R6-DAG:      daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_olt_i32_val)))
; 64R6-DAG:      daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64R6-DAG:      ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64R6-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64R6-DAG:      ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64R6-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64R6:          cmp.lt.d $[[CC:f[0-9]+]], $[[TMP]], $[[TMP1]]
; 64R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 64R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 64R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 64R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 64R6:          or $2, $[[NE]], $[[EQ]]

  %tmp = load double, double* @d2, align 8
  %tmp1 = load double, double* @d3, align 8
  %cmp = fcmp olt double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}

define i32 @f64_fcmp_ogt_i32_val(i32 signext %f0, i32 signext %f1) nounwind readonly {
entry:
; ALL-LABEL: f64_fcmp_ogt_i32_val:

; 32-DAG:        addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32-DAG:        addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32-DAG:        lw $[[D2:[0-9]+]], %got(d2)($1)
; 32-DAG:        ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32-DAG:        lw $[[D3:[0-9]+]], %got(d3)($1)
; 32-DAG:        ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32:            c.ule.d $[[TMP]], $[[TMP1]]
; 32:            movf $5, $4, $fcc0
; 32:            move $2, $5

; 32R2-DAG:      addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32R2-DAG:      addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32R2-DAG:      lw $[[D2:[0-9]+]], %got(d2)($1)
; 32R2-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32R2-DAG:      lw $[[D3:[0-9]+]], %got(d3)($1)
; 32R2-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32R2:          c.ule.d $[[TMP]], $[[TMP1]]
; 32R2:          movf $5, $4, $fcc0
; 32R2:          move $2, $5

; 32R6-DAG:      addiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(_gp_disp)
; 32R6-DAG:      addu $[[GOT:[0-9]+]], $[[T0]], $25
; 32R6-DAG:      lw $[[D2:[0-9]+]], %got(d2)($1)
; 32R6-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 32R6-DAG:      lw $[[D3:[0-9]+]], %got(d3)($1)
; 32R6-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 32R6:          cmp.lt.d $[[CC:f[0-9]+]], $[[TMP1]], $[[TMP]]
; 32R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 32R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 32R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 32R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 32R6:          or $2, $[[NE]], $[[EQ]]

; 64-DAG:        daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_ogt_i32_val)))
; 64-DAG:        daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64-DAG:        ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64-DAG:        ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64-DAG:        ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64-DAG:        ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64:            c.ule.d $[[TMP]], $[[TMP1]]
; 64:            movf $5, $4, $fcc0
; 64:            move $2, $5

; 64R2-DAG:      daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_ogt_i32_val)))
; 64R2-DAG:      daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64R2-DAG:      ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64R2-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64R2-DAG:      ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64R2-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64R2:          c.ule.d $[[TMP]], $[[TMP1]]
; 64R2:          movf $5, $4, $fcc0
; 64R2:          move $2, $5

; 64R6-DAG:      daddiu $[[T0:[0-9]+]], ${{[0-9]+}}, %lo(%neg(%gp_rel(f64_fcmp_ogt_i32_val)))
; 64R6-DAG:      daddu $[[GOT:[0-9]+]], $[[T0]], $25
; 64R6-DAG:      ld $[[D2:[0-9]+]], %got_disp(d2)($1)
; 64R6-DAG:      ldc1 $[[TMP:f[0-9]+]], 0($[[D2]])
; 64R6-DAG:      ld $[[D3:[0-9]+]], %got_disp(d3)($1)
; 64R6-DAG:      ldc1 $[[TMP1:f[0-9]+]], 0($[[D3]])
; 64R6:          cmp.lt.d $[[CC:f[0-9]+]], $[[TMP1]], $[[TMP]]
; 64R6:          mfc1 $[[CCGPR:[0-9]+]], $[[CC]]
; 64R6:          andi $[[CCGPR]], $[[CCGPR]], 1
; 64R6:          seleqz $[[EQ:[0-9]+]], $5, $[[CCGPR]]
; 64R6:          selnez $[[NE:[0-9]+]], $4, $[[CCGPR]]
; 64R6:          or $2, $[[NE]], $[[EQ]]

  %tmp = load double, double* @d2, align 8
  %tmp1 = load double, double* @d3, align 8
  %cmp = fcmp ogt double %tmp, %tmp1
  %cond = select i1 %cmp, i32 %f0, i32 %f1
  ret i32 %cond
}
