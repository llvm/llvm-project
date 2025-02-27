; RUN: llc -mtriple=mipsel-- -disable-mips-delay-filler \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC32
; RUN: llc -mtriple=mips64el-- -disable-mips-delay-filler \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC64

; RUN: llc -mtriple=mipsel-- -mattr=+micromips -disable-mips-delay-filler \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=MM

@t1 = dso_preemptable thread_local global i32 0, align 4

define dso_preemptable i32 @f1() nounwind {
entry:
  %tmp = load i32, ptr @t1, align 4
  ret i32 %tmp

; PIC32-LABEL:       f1:
; PIC32-DAG:   addu    $[[R0:[a-z0-9]+]], $2, $25
; PIC32-DAG:   addiu   $4, $[[R0]], %tlsgd(t1)
; PIC32-DAG:   lw      $25, %call16(__tls_get_addr)($[[R0]])
; PIC32-DAG:   jalr    $25
; PIC32-DAG:   lw      $2, 0($2)

; PIC64-LABEL:       f1:
; PIC64-DAG:   daddiu  $[[R0:[a-z0-9]+]], $1, %lo(%neg(%gp_rel(f1)))
; PIC64-DAG:   daddiu  $4, $[[R0]], %tlsgd(t1)
; PIC64-DAG:   ld      $25, %call16(__tls_get_addr)($[[R0]])
; PIC64-DAG:   jalr    $25
; PIC64-DAG:   lw      $2, 0($2)

; MM-LABEL:       f1:
; MM-DAG:   addu    $gp, $2, $25
; MM-DAG:   addiu   $4, $gp, %tlsgd(t1)
; MM-DAG:   lw      $25, %call16(__tls_get_addr)($gp)
; MM-DAG:   jalr    $25
; MM-DAG:   lw16    $2, 0($2)
}

@t2 = external thread_local global i32

define dso_preemptable i32 @f2() nounwind {
entry:
  %tmp = load i32, ptr @t2, align 4
  ret i32 %tmp

; PIC32-LABEL:       f2:
; PIC32-DAG:   addu    $[[R0:[a-z0-9]+]], $2, $25
; PIC32-DAG:   addiu   $4, $[[R0]], %tlsgd(t2)
; PIC32-DAG:   lw      $25, %call16(__tls_get_addr)($[[R0]])
; PIC32-DAG:   jalr    $25
; PIC32-DAG:   lw      $2, 0($2)

; PIC64-LABEL:       f2:
; PIC64-DAG:   daddiu  $[[R0:[a-z0-9]+]], $1, %lo(%neg(%gp_rel(f2)))
; PIC64-DAG:   daddiu  $4, $[[R0]], %tlsgd(t2)
; PIC64-DAG:   ld      $25, %call16(__tls_get_addr)($[[R0]])
; PIC64-DAG:   jalr    $25
; PIC64-DAG:   lw      $2, 0($2)

; MM-LABEL:       f2:
; MM-DAG:   addu    $[[R0:[a-z0-9]+]], $2, $25
; MM-DAG:   lw      $25, %call16(__tls_get_addr)($[[R0]])
; MM-DAG:   addiu   $4, $[[R0]], %tlsgd(t2)
; MM-DAG:   jalr    $25
; MM-DAG:   lw16    $2, 0($2)
}

@f3.i = internal thread_local unnamed_addr global i32 1, align 4

define dso_preemptable i32 @f3() nounwind {
entry:
; PIC32-LABEL:      f3:
; PIC32:   addu    $[[R0:[a-z0-9]+]], $2, $25
; PIC32:   addiu   $4, $[[R0]], %tlsldm(f3.i)
; PIC32:   lw      $25, %call16(__tls_get_addr)($[[R0]])
; PIC32:   jalr    $25
; PIC32:   lui     $[[R0:[0-9]+]], %dtprel_hi(f3.i)
; PIC32:   addu    $[[R1:[0-9]+]], $[[R0]], $2
; PIC32:   lw      $[[R3:[0-9]+]], %dtprel_lo(f3.i)($[[R1]])
; PIC32:   addiu   $[[R3]], $[[R3]], 1
; PIC32:   sw      $[[R3]], %dtprel_lo(f3.i)($[[R1]])

; PIC64-LABEL:      f3:
; PIC64:   lui     $[[R0:[a-z0-9]+]], %hi(%neg(%gp_rel(f3)))
; PIC64:   daddu   $[[R0]], $[[R0]], $25
; PIC64:   daddiu  $[[R1:[a-z0-9]+]], $[[R0]], %lo(%neg(%gp_rel(f3)))
; PIC64:   daddiu  $4, $[[R1]], %tlsldm(f3.i)
; PIC64:   ld      $25, %call16(__tls_get_addr)($[[R1]])
; PIC64:   jalr    $25
; PIC64:   lui     $[[R0:[0-9]+]], %dtprel_hi(f3.i)
; PIC64:   daddu   $[[R1:[0-9]+]], $[[R0]], $2
; PIC64:   lw      $[[R2:[0-9]+]], %dtprel_lo(f3.i)($[[R1]])
; PIC64:   addiu   $[[R2]], $[[R2]], 1
; PIC64:   sw      $[[R2]], %dtprel_lo(f3.i)($[[R1]])

; MM-LABEL:       f3:
; MM:   addiu   $4, ${{[a-z0-9]+}}, %tlsldm(f3.i)
; MM:   jalr    $25
; MM:   lui     $[[R0:[0-9]+]], %dtprel_hi(f3.i)
; MM:   addu16  $[[R1:[0-9]+]], $[[R0]], $2
; MM:   lw      ${{[0-9]+}}, %dtprel_lo(f3.i)($[[R1]])

  %0 = load i32, ptr @f3.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @f3.i, align 4
  ret i32 %inc
}
