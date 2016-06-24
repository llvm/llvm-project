; RUN: llc -march=mipsel -disable-mips-delay-filler -relocation-model=pic < %s | \
; RUN:     FileCheck %s -check-prefixes=PIC,CHECK
; RUN: llc -march=mipsel -relocation-model=static -disable-mips-delay-filler < \
; RUN:     %s | FileCheck %s -check-prefixes=STATIC,CHECK
; RUN: llc -march=mipsel -relocation-model=static -disable-mips-delay-filler \
; RUN:     -mips-fix-global-base-reg=false < %s  | \
; RUN:     FileCheck %s -check-prefixes=STATICGP,CHECK

@t1 = thread_local global i32 0, align 4

define i32 @f1() nounwind {
entry:
  %tmp = load i32, i32* @t1, align 4
  ret i32 %tmp

; PIC-LABEL:       f1:
; PIC-DAG:   addu    $[[R0:[a-z0-9]+]], $2, $25
; PIC-DAG:   lw      $25, %call16(__tls_get_addr)($[[R0]])
; PIC-DAG:   addiu   $4, $[[R0]], %tlsgd(t1)
; PIC-DAG:   jalr    $25
; PIC-DAG:   lw      $2, 0($2)

; STATIC-LABEL:   f1:
; STATIC:   lui     $[[R0:[0-9]+]], %tprel_hi(t1)
; STATIC:   addiu   $[[R1:[0-9]+]], $[[R0]], %tprel_lo(t1)
; STATIC:   rdhwr   $3, $29
; STATIC:   addu    $[[R2:[0-9]+]], $3, $[[R1]]
; STATIC:   lw      $2, 0($[[R2]])
}


@t2 = external thread_local global i32

define i32 @f2() nounwind {
entry:
  %tmp = load i32, i32* @t2, align 4
  ret i32 %tmp

; PIC-LABEL:       f2:
; PIC-DAG:   addu    $[[R0:[a-z0-9]+]], $2, $25
; PIC-DAG:   lw      $25, %call16(__tls_get_addr)($[[R0]])
; PIC-DAG:   addiu   $4, $[[R0]], %tlsgd(t2)
; PIC-DAG:   jalr    $25
; PIC-DAG:   lw      $2, 0($2)

; STATICGP-LABEL: f2:
; STATICGP: lui     $[[R0:[0-9]+]], %hi(__gnu_local_gp)
; STATICGP: addiu   $[[GP:[0-9]+]], $[[R0]], %lo(__gnu_local_gp)
; STATICGP: lw      ${{[0-9]+}}, %gottprel(t2)($[[GP]])

; STATIC-LABEL:   f2:
; STATIC:   lui     $[[R0:[0-9]+]], %hi(__gnu_local_gp)
; STATIC:   addiu   $[[GP:[0-9]+]], $[[R0]], %lo(__gnu_local_gp)
; STATIC:   rdhwr   $3, $29
; STATIC:   lw      $[[R0:[0-9]+]], %gottprel(t2)($[[GP]])
; STATIC:   addu    $[[R1:[0-9]+]], $3, $[[R0]]
; STATIC:   lw      $2, 0($[[R1]])
}

@f3.i = internal thread_local unnamed_addr global i32 1, align 4

define i32 @f3() nounwind {
entry:
; CHECK-LABEL: f3:

; PIC:   addiu   $4, ${{[a-z0-9]+}}, %tlsldm(f3.i)
; PIC:   jalr    $25
; PIC:   lui     $[[R0:[0-9]+]], %dtprel_hi(f3.i)
; PIC:   addu    $[[R1:[0-9]+]], $[[R0]], $2
; PIC:   lw      ${{[0-9]+}}, %dtprel_lo(f3.i)($[[R1]])

  %0 = load i32, i32* @f3.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @f3.i, align 4
  ret i32 %inc
}

