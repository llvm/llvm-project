; RUN: llc -march=mipsel -mcpu=mips32 < %s | FileCheck %s -check-prefixes=ALL,32-NOFPXX
; RUN: llc -march=mipsel -mcpu=mips32 -mattr=fpxx < %s | FileCheck %s -check-prefixes=ALL,32-FPXX

; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck %s -check-prefixes=ALL,32R2-NOFPXX
; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=fpxx < %s | FileCheck %s -check-prefixes=ALL,32R2-FPXX

; RUN: llc -march=mips64 -mcpu=mips4 < %s | FileCheck %s -check-prefixes=ALL,4-NOFPXX
; RUN: not llc -march=mips64 -mcpu=mips4 -mattr=fpxx < %s 2>&1 | FileCheck %s -check-prefix=4-FPXX

; RUN: llc -march=mips64 -mcpu=mips64 < %s | FileCheck %s -check-prefixes=ALL,64-NOFPXX
; RUN: not llc -march=mips64 -mcpu=mips64 -mattr=fpxx < %s 2>&1 | FileCheck %s -check-prefix=64-FPXX

; RUN-TODO: llc -march=mips64 -mcpu=mips4 -target-abi o32 < %s | FileCheck %s -check-prefixes=ALL,4-O32-NOFPXX
; RUN-TODO: llc -march=mips64 -mcpu=mips4 -target-abi o32 -mattr=fpxx < %s | FileCheck %s -check-prefixes=ALL,4-O32-FPXX

; RUN-TODO: llc -march=mips64 -mcpu=mips64 -target-abi o32 < %s | FileCheck %s -check-prefixes=ALL,64-O32-NOFPXX
; RUN-TODO: llc -march=mips64 -mcpu=mips64 -target-abi o32 -mattr=fpxx < %s | FileCheck %s -check-prefixes=ALL,64-O32-FPXX

declare double @dbl();

; 4-FPXX:  LLVM ERROR: FPXX is not permitted for the N32/N64 ABI's.
; 64-FPXX: LLVM ERROR: FPXX is not permitted for the N32/N64 ABI's.

define double @test1(double %d, ...) {
  ret double %d

; ALL-LABEL: test1:

; 32-NOFPXX:     mtc1    $4, $f0
; 32-NOFPXX:     mtc1    $5, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $4, 0($sp)
; 32-FPXX:       sw      $5, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:   mtc1    $4, $f0
; 32R2-NOFPXX:   mthc1   $5, $f0

; 32R2-FPXX:     mtc1    $4, $f0
; 32R2-FPXX:     mthc1   $5, $f0

; floats/doubles are not passed in integer registers for n64, so dmtc1 is not used.
; 4-NOFPXX:      mov.d   $f0, $f12

; 64-NOFPXX:     mov.d   $f0, $f12
}

define double @test2(i32 %i, double %d) {
  ret double %d

; ALL-LABEL: test2:

; 32-NOFPXX:     mtc1    $6, $f0
; 32-NOFPXX:     mtc1    $7, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $6, 0($sp)
; 32-FPXX:       sw      $7, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:   mtc1    $6, $f0
; 32R2-NOFPXX:   mthc1   $7, $f0

; 32R2-FPXX:     mtc1    $6, $f0
; 32R2-FPXX:     mthc1   $7, $f0

; 4-NOFPXX:      mov.d   $f0, $f13

; 64-NOFPXX:     mov.d   $f0, $f13
}

define double @test3(float %f1, float %f2, double %d) {
  ret double %d

; ALL-LABEL: test3:

; 32-NOFPXX:     mtc1    $6, $f0
; 32-NOFPXX:     mtc1    $7, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $6, 0($sp)
; 32-FPXX:       sw      $7, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:   mtc1    $6, $f0
; 32R2-NOFPXX:   mthc1   $7, $f0

; 32R2-FPXX:     mtc1    $6, $f0
; 32R2-FPXX:     mthc1   $7, $f0

; 4-NOFPXX:      mov.d   $f0, $f14

; 64-NOFPXX:     mov.d   $f0, $f14
}

define double @test4(float %f, double %d, ...) {
  ret double %d

; ALL-LABEL: test4:

; 32-NOFPXX:     mtc1    $6, $f0
; 32-NOFPXX:     mtc1    $7, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $6, 0($sp)
; 32-FPXX:       sw      $7, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:   mtc1    $6, $f0
; 32R2-NOFPXX:   mthc1   $7, $f0

; 32R2-FPXX:     mtc1    $6, $f0
; 32R2-FPXX:     mthc1   $7, $f0

; 4-NOFPXX:      mov.d   $f0, $f13

; 64-NOFPXX:     mov.d   $f0, $f13
}

define double @test5() {
  ret double 0.000000e+00

; ALL-LABEL: test5:

; 32-NOFPXX:     mtc1    $zero, $f0
; 32-NOFPXX:     mtc1    $zero, $f1

; 32-FPXX:       addiu   $sp, $sp, -8
; 32-FPXX:       sw      $zero, 0($sp)
; 32-FPXX:       sw      $zero, 4($sp)
; 32-FPXX:       ldc1    $f0, 0($sp)

; 32R2-NOFPXX:   mtc1    $zero, $f0
; 32R2-NOFPXX:   mthc1   $zero, $f0

; 32R2-FPXX:     mtc1    $zero, $f0
; 32R2-FPXX:     mthc1   $zero, $f0

; 4-NOFPXX:      dmtc1 $zero, $f0

; 64-NOFPXX:     dmtc1 $zero, $f0
}

define double @test6(double %a, double %b, ...) {
  %1 = fsub double %a, %b
  ret double %1

; ALL-LABEL:     test6:

; 32-NOFPXX-DAG:     mtc1    $4, $[[T0:f[0-9]+]]
; 32-NOFPXX-DAG:     mtc1    $5, ${{f[0-9]*[13579]}}
; 32-NOFPXX-DAG:     mtc1    $6, $[[T1:f[0-9]+]]
; 32-NOFPXX-DAG:     mtc1    $7, ${{f[0-9]*[13579]}}
; 32-NOFPXX:         sub.d   $f0, $[[T0]], $[[T1]]

; 32-FPXX:           addiu   $sp, $sp, -8
; 32-FPXX:           sw      $6, 0($sp)
; 32-FPXX:           sw      $7, 4($sp)
; 32-FPXX:           ldc1    $[[T1:f[0-9]+]], 0($sp)
; 32-FPXX:           sw      $4, 0($sp)
; 32-FPXX:           sw      $5, 4($sp)
; 32-FPXX:           ldc1    $[[T0:f[0-9]+]], 0($sp)
; 32-FPXX:           sub.d   $f0, $[[T0]], $[[T1]]

; 32R2-NOFPXX-DAG:   mtc1    $4, $[[T0:f[0-9]+]]
; 32R2-NOFPXX-DAG:   mthc1   $5, $[[T0]]
; 32R2-NOFPXX-DAG:   mtc1    $6, $[[T1:f[0-9]+]]
; 32R2-NOFPXX-DAG:   mthc1   $7, $[[T1]]
; 32R2-NOFPXX:       sub.d   $f0, $[[T0]], $[[T1]]

; 32R2-FPXX-DAG:     mtc1    $4, $[[T0:f[0-9]+]]
; 32R2-FPXX-DAG:     mthc1   $5, $[[T0]]
; 32R2-FPXX-DAG:     mtc1    $6, $[[T1:f[0-9]+]]
; 32R2-FPXX-DAG:     mthc1   $7, $[[T1]]
; 32R2-FPXX:         sub.d   $f0, $[[T0]], $[[T1]]

; floats/doubles are not passed in integer registers for n64, so dmtc1 is not used.
; 4-NOFPXX:          sub.d   $f0, $f12, $f13

; floats/doubles are not passed in integer registers for n64, so dmtc1 is not used.
; 64-NOFPXX:         sub.d   $f0, $f12, $f13
}

define double @move_from1(double %d) {
  %1 = call double @dbl()
  %2 = call double @test2(i32 0, double %1)
  ret double %2

; ALL-LABEL:   move_from1:

; 32-NOFPXX-DAG:   mfc1    $6, $f0
; 32-NOFPXX-DAG:   mfc1    $7, $f1

; 32-FPXX:         addiu   $sp, $sp, -32
; 32-FPXX:         sdc1    $f0, 16($sp)
; 32-FPXX:         lw      $6, 16($sp)
; FIXME: This store is redundant
; 32-FPXX:         sdc1    $f0, 16($sp)
; 32-FPXX:         lw      $7, 20($sp)

; 32R2-NOFPXX-DAG: mfc1    $6, $f0
; 32R2-NOFPXX-DAG: mfhc1   $7, $f0

; 32R2-FPXX-DAG:   mfc1    $6, $f0
; 32R2-FPXX-DAG:   mfhc1   $7, $f0

; floats/doubles are not passed in integer registers for n64, so dmfc1 is not used.
; We can't use inline assembly to force a copy either because trying to force
; a copy to a GPR this way fails with ; "couldn't allocate input reg for
; constraint 'r'". It therefore seems impossible to test the generation of dmfc1
; in a simple test.
; 4-NOFPXX:        mov.d   $f13, $f0

; floats/doubles are not passed in integer registers for n64, so dmfc1 is not used.
; We can't use inline assembly to force a copy either because trying to force
; a copy to a GPR this way fails with ; "couldn't allocate input reg for
; constraint 'r'". It therefore seems impossible to test the generation of dmfc1
; in a simple test.
; 64-NOFPXX:       mov.d   $f13, $f0
}
