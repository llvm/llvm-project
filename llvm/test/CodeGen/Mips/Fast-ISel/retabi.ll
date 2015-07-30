; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s

@i = global i32 75, align 4
@s = global i16 -345, align 2
@c = global i8 118, align 1
@f = global float 0x40BE623360000000, align 4
@d = global double 1.298330e+03, align 8

; Function Attrs: nounwind
define i32 @reti() {
entry:
; CHECK-LABEL: reti:
  %0 = load i32, i32* @i, align 4
  ret i32 %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_I_ADDR:[0-9]+]], %got(i)($[[REG_GP]])
; CHECK:        lw      $2, 0($[[REG_I_ADDR]])
; CHECK:        jr      $ra
}

; Function Attrs: nounwind
define i16 @retus() {
entry:
; CHECK-LABEL: retus:
  %0 = load i16, i16* @s, align 2
  ret i16 %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_S_ADDR:[0-9]+]], %got(s)($[[REG_GP]])
; CHECK:        lhu     $2, 0($[[REG_S_ADDR]])
; CHECK:        jr      $ra
}

; Function Attrs: nounwind
define signext i16 @rets() {
entry:
; CHECK-LABEL: rets:
  %0 = load i16, i16* @s, align 2
  ret i16 %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_S_ADDR:[0-9]+]], %got(s)($[[REG_GP]])
; CHECK:        lhu     $[[REG_S:[0-9]+]], 0($[[REG_S_ADDR]])
; CHECK:        seh     $2, $[[REG_S]]
; CHECK:        jr      $ra
}

; Function Attrs: nounwind
define i8 @retuc() {
entry:
; CHECK-LABEL: retuc:
  %0 = load i8, i8* @c, align 1
  ret i8 %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_C_ADDR:[0-9]+]], %got(c)($[[REG_GP]])
; CHECK:        lbu     $2, 0($[[REG_C_ADDR]])
; CHECK:        jr      $ra
}

; Function Attrs: nounwind
define signext i8 @retc() {
entry:
; CHECK-LABEL: retc:
  %0 = load i8, i8* @c, align 1
  ret i8 %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_C_ADDR:[0-9]+]], %got(c)($[[REG_GP]])
; CHECK:        lbu     $[[REG_C:[0-9]+]], 0($[[REG_C_ADDR]])
; CHECK:        seb     $2, $[[REG_C]]
; CHECK:        jr      $ra
}

; Function Attrs: nounwind
define float @retf() {
entry:
; CHECK-LABEL: retf:
  %0 = load float, float* @f, align 4
  ret float %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_F_ADDR:[0-9]+]], %got(f)($[[REG_GP]])
; CHECK:        lwc1    $f0, 0($[[REG_F_ADDR]])
; CHECK:        jr      $ra
}

; Function Attrs: nounwind
define double @retd() {
entry:
; CHECK-LABEL: retd:
  %0 = load double, double* @d, align 8
  ret double %0
; CHECK:        lui     $[[REG_GPa:[0-9]+]], %hi(_gp_disp)
; CHECK:        addiu   $[[REG_GPb:[0-9]+]], $[[REG_GPa]], %lo(_gp_disp)
; CHECK:        addu    $[[REG_GP:[0-9]+]], $[[REG_GPb]], $25
; CHECK:        lw      $[[REG_D_ADDR:[0-9]+]], %got(d)($[[REG_GP]])
; CHECK:        ldc1    $f0, 0($[[REG_D_ADDR]])
; CHECK:        jr      $ra
}
