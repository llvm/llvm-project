; Test that the FP64A ABI performs double precision moves via a spill/reload.
; The requirement is really that odd-numbered double precision registers do not
; use mfc1/mtc1 to move the bottom 32-bits (because the hardware will redirect
; this to the top 32-bits of the even register) but we have to make the decision
; before register allocation so we do this for all double-precision values.

; We don't test MIPS32r1 since support for 64-bit coprocessors (such as a 64-bit
; FPU) on a 32-bit architecture was added in MIPS32r2.

; RUN: not llc -march=mips -mcpu=mips32 -mattr=fp64 < %s 2>&1 | FileCheck %s -check-prefix=32R1-FP64
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=fp64 < %s | FileCheck %s -check-prefixes=ALL,32R2-NO-FP64A-BE
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=fp64,nooddspreg < %s | FileCheck %s -check-prefixes=ALL,32R2-FP64A
; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=fp64 < %s | FileCheck %s -check-prefixes=ALL,32R2-NO-FP64A-LE
; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=fp64,nooddspreg < %s | FileCheck %s -check-prefixes=ALL,32R2-FP64A

; RUN: llc -march=mips64 -mcpu=mips64 -mattr=fp64 < %s | FileCheck %s -check-prefixes=ALL,64-NO-FP64A
; RUN: not llc -march=mips64 -mcpu=mips64 -mattr=fp64,nooddspreg < %s 2>&1 | FileCheck %s -check-prefix=64-FP64A
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=fp64 < %s | FileCheck %s -check-prefixes=ALL,64-NO-FP64A
; RUN: not llc -march=mips64el -mcpu=mips64 -mattr=fp64,nooddspreg < %s 2>&1 | FileCheck %s -check-prefix=64-FP64A

; 32R1-FP64: LLVM ERROR: FPU with 64-bit registers is not available on MIPS32 pre revision 2. Use -mcpu=mips32r2 or greater.
; 64-FP64A: LLVM ERROR: -mattr=+nooddspreg requires the O32 ABI.

declare double @dbl();

define double @call1(double %d, ...) {
  ret double %d

; ALL-LABEL:            call1:

; 32R2-NO-FP64A-LE-NOT:     addiu   $sp, $sp
; 32R2-NO-FP64A-LE:         mtc1    $4, $f0
; 32R2-NO-FP64A-LE:         mthc1   $5, $f0

; 32R2-NO-FP64A-BE-NOT:     addiu   $sp, $sp
; 32R2-NO-FP64A-BE:         mtc1    $5, $f0
; 32R2-NO-FP64A-BE:         mthc1   $4, $f0

; 32R2-FP64A:               addiu   $sp, $sp, -8
; 32R2-FP64A:               sw      $4, 0($sp)
; 32R2-FP64A:               sw      $5, 4($sp)
; 32R2-FP64A:               ldc1    $f0, 0($sp)

; 64-NO-FP64A:              daddiu  $sp, $sp, -64
; 64-NO-FP64A:              mov.d   $f0, $f12
}

define double @call2(i32 %i, double %d) {
  ret double %d

; ALL-LABEL:        call2:

; 32R2-NO-FP64A-LE:     mtc1    $6, $f0
; 32R2-NO-FP64A-LE:     mthc1   $7, $f0

; 32R2-NO-FP64A-BE:     mtc1    $7, $f0
; 32R2-NO-FP64A-BE:     mthc1   $6, $f0

; 32R2-FP64A:           addiu   $sp, $sp, -8
; 32R2-FP64A:           sw      $6, 0($sp)
; 32R2-FP64A:           sw      $7, 4($sp)
; 32R2-FP64A:           ldc1    $f0, 0($sp)

; 64-NO-FP64A-NOT:      daddiu  $sp, $sp
; 64-NO-FP64A:          mov.d   $f0, $f13
}

define double @call3(float %f1, float %f2, double %d) {
  ret double %d

; ALL-LABEL:        call3:

; 32R2-NO-FP64A-LE:     mtc1    $6, $f0
; 32R2-NO-FP64A-LE:     mthc1   $7, $f0

; 32R2-NO-FP64A-BE:     mtc1    $7, $f0
; 32R2-NO-FP64A-BE:     mthc1   $6, $f0

; 32R2-FP64A:           addiu   $sp, $sp, -8
; 32R2-FP64A:           sw      $6, 0($sp)
; 32R2-FP64A:           sw      $7, 4($sp)
; 32R2-FP64A:           ldc1    $f0, 0($sp)

; 64-NO-FP64A-NOT:      daddiu  $sp, $sp
; 64-NO-FP64A:          mov.d   $f0, $f14
}

define double @call4(float %f, double %d, ...) {
  ret double %d

; ALL-LABEL:        call4:

; 32R2-NO-FP64A-LE:     mtc1    $6, $f0
; 32R2-NO-FP64A-LE:     mthc1   $7, $f0

; 32R2-NO-FP64A-BE:     mtc1    $7, $f0
; 32R2-NO-FP64A-BE:     mthc1   $6, $f0

; 32R2-FP64A:           addiu   $sp, $sp, -8
; 32R2-FP64A:           sw      $6, 0($sp)
; 32R2-FP64A:           sw      $7, 4($sp)
; 32R2-FP64A:           ldc1    $f0, 0($sp)

; 64-NO-FP64A:          daddiu  $sp, $sp, -48
; 64-NO-FP64A:          mov.d   $f0, $f13
}

define double @call5(double %a, double %b, ...) {
  %1 = fsub double %a, %b
  ret double %1

; ALL-LABEL:            call5:

; 32R2-NO-FP64A-LE-DAG:     mtc1    $4, $[[T0:f[0-9]+]]
; 32R2-NO-FP64A-LE-DAG:     mthc1   $5, $[[T0:f[0-9]+]]
; 32R2-NO-FP64A-LE-DAG:     mtc1    $6, $[[T1:f[0-9]+]]
; 32R2-NO-FP64A-LE-DAG:     mthc1   $7, $[[T1:f[0-9]+]]
; 32R2-NO-FP64A-LE:         sub.d   $f0, $[[T0]], $[[T1]]

; 32R2-NO-FP64A-BE-DAG:     mtc1    $5, $[[T0:f[0-9]+]]
; 32R2-NO-FP64A-BE-DAG:     mthc1   $4, $[[T0:f[0-9]+]]
; 32R2-NO-FP64A-BE-DAG:     mtc1    $7, $[[T1:f[0-9]+]]
; 32R2-NO-FP64A-BE-DAG:     mthc1   $6, $[[T1:f[0-9]+]]
; 32R2-NO-FP64A-BE:         sub.d   $f0, $[[T0]], $[[T1]]

; 32R2-FP64A:               addiu   $sp, $sp, -8
; 32R2-FP64A:               sw      $6, 0($sp)
; 32R2-FP64A:               sw      $7, 4($sp)
; 32R2-FP64A:               ldc1    $[[T1:f[0-9]+]], 0($sp)
; 32R2-FP64A:               sw      $4, 0($sp)
; 32R2-FP64A:               sw      $5, 4($sp)
; 32R2-FP64A:               ldc1    $[[T0:f[0-9]+]], 0($sp)
; 32R2-FP64A:               sub.d   $f0, $[[T0]], $[[T1]]

; 64-NO-FP64A:              sub.d   $f0, $f12, $f13
}

define double @move_from(double %d) {
  %1 = call double @dbl()
  %2 = call double @call2(i32 0, double %1)
  ret double %2

; ALL-LABEL:        move_from:

; 32R2-NO-FP64A-LE-DAG: mfc1    $6, $f0
; 32R2-NO-FP64A-LE-DAG: mfhc1   $7, $f0

; 32R2-NO-FP64A-BE-DAG: mfc1    $7, $f0
; 32R2-NO-FP64A-BE-DAG: mfhc1   $6, $f0

; 32R2-FP64A:           addiu   $sp, $sp, -32
; 32R2-FP64A:           sdc1    $f0, 16($sp)
; 32R2-FP64A:           lw      $6, 16($sp)
; FIXME: This store is redundant
; 32R2-FP64A:           sdc1    $f0, 16($sp)
; 32R2-FP64A:           lw      $7, 20($sp)

; 64-NO-FP64A:          mov.d   $f13, $f0
}
