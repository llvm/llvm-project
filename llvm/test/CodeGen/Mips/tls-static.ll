; RUN: llc -mtriple=mipsel-- -disable-mips-delay-filler \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC32
; RUN: llc -mtriple=mips64el-- -disable-mips-delay-filler \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC64

; RUN: llc -mtriple=mipsel-- -disable-mips-delay-filler -mips-fix-global-base-reg=false \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATICGP32
; RUN: llc -mtriple=mips64el-- -disable-mips-delay-filler -mips-fix-global-base-reg=false \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATICGP64

@t1 = dso_local thread_local global i32 0, align 4

define dso_local i32 @f1() nounwind {
entry:
  %tmp = load i32, i32* @t1, align 4
  ret i32 %tmp

; STATIC32-LABEL:   f1:
; STATIC32:   lui     $[[R0:[0-9]+]], %tprel_hi(t1)
; STATIC32:   addiu   $[[R1:[0-9]+]], $[[R0]], %tprel_lo(t1)
; STATIC32:   rdhwr   $3, $29{{$}}
; STATIC32:   addu    $[[R2:[0-9]+]], $3, $[[R1]]
; STATIC32:   lw      $2, 0($[[R2]])

; STATIC64-LABEL:   f1:
; STATIC64:   lui     $[[R0:[0-9]+]], %tprel_hi(t1)
; STATIC64:   daddiu  $[[R1:[0-9]+]], $[[R0]], %tprel_lo(t1)
; STATIC64:   rdhwr   $3, $29{{$}}
; STATIC64:   daddu   $[[R2:[0-9]+]], $3, $[[R0]]
; STATIC64:   lw      $2, 0($[[R2]])
}

@t2 = external thread_local global i32

define dso_local i32 @f2() nounwind {
entry:
  %tmp = load i32, i32* @t2, align 4
  ret i32 %tmp

; STATICGP32-LABEL: f2:
; STATICGP32: lui     $[[R0:[0-9]+]], %hi(__gnu_local_gp)
; STATICGP32: addiu   $[[GP:[0-9]+]], $[[R0]], %lo(__gnu_local_gp)
; STATICGP32: lw      ${{[0-9]+}}, %gottprel(t2)($[[GP]])

; STATICGP64-LABEL: f2:
; STATICGP64: lui     $[[R0:[0-9]+]], %hi(%neg(%gp_rel(f2)))
; STATICGP64: daddiu  $[[GP:[0-9]+]], $[[R0]], %lo(%neg(%gp_rel(f2)))
; STATICGP64: ld      $1, %gottprel(t2)($[[GP]])

; STATIC32-LABEL:   f2:
; STATIC32:   lui     $[[R0:[0-9]+]], %hi(__gnu_local_gp)
; STATIC32:   addiu   $[[GP:[0-9]+]], $[[R0]], %lo(__gnu_local_gp)
; STATIC32:   rdhwr   $3, $29{{$}}
; STATIC32:   lw      $[[R0:[0-9]+]], %gottprel(t2)($[[GP]])
; STATIC32:   addu    $[[R1:[0-9]+]], $3, $[[R0]]
; STATIC32:   lw      $2, 0($[[R1]])

; STATIC64-LABEL:   f2:
; STATIC64:   lui     $[[R0:[0-9]+]], %hi(%neg(%gp_rel(f2)))
; STATIC64:   daddiu  $[[GP:[0-9]+]], $[[R0]], %lo(%neg(%gp_rel(f2)))
; STATIC64:   rdhwr   $3, $29{{$}}
; STATIC64:   ld      $[[R0:[0-9]+]], %gottprel(t2)($[[GP]])
; STATIC64:   daddu   $[[R1:[0-9]+]], $3, $[[R0]]
; STATIC64:   lw      $2, 0($[[R1]])
}

@f3.i = internal thread_local unnamed_addr global i32 1, align 4

define dso_local i32 @f3() nounwind {
entry:
; MM-LABEL:       f3:
; MM:   addiu   $4, ${{[a-z0-9]+}}, %tlsldm(f3.i)
; MM:   jalr    $25
; MM:   lui     $[[R0:[0-9]+]], %dtprel_hi(f3.i)
; MM:   addu16  $[[R1:[0-9]+]], $[[R0]], $2
; MM:   lw      ${{[0-9]+}}, %dtprel_lo(f3.i)($[[R1]])

  %0 = load i32, i32* @f3.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @f3.i, align 4
  ret i32 %inc
}
