; Check that madd.[ds], msub.[ds], nmadd.[ds], and nmsub.[ds] are supported
; correctly.
; The spec for nmadd.[ds], and nmsub.[ds] does not state that they obey the
; the Has2008 and ABS2008 configuration bits which govern the conformance to
; IEEE 754 (1985) and IEEE 754 (2008). These instructions are therefore only
; available when -enable-no-nans-fp-math is given.

; RUN: llc < %s -march=mipsel   -mcpu=mips32              -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,32-NOMADD,32-NONAN-NOMADD
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2            -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,32R2,32R2-NONAN
; RUN: llc < %s -march=mipsel   -mcpu=mips32r6            -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,32R6-NOMADD,32R6-NONAN-NOMADD
; RUN: llc < %s -march=mips64el -mcpu=mips64   -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,64,64-NONAN
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,64R2,64R2-NONAN
; RUN: llc < %s -march=mips64el -mcpu=mips64r6 -target-abi=n64 -enable-no-nans-fp-math | FileCheck %s -check-prefixes=ALL,64R6-NOMADD,64R6-NONAN-NOMADD
; RUN: llc < %s -march=mipsel   -mcpu=mips32              | FileCheck %s -check-prefixes=ALL,32-NOMADD,32-NAN-NOMADD
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2            | FileCheck %s -check-prefixes=ALL,32R2,32R2-NAN
; RUN: llc < %s -march=mipsel   -mcpu=mips32r6            | FileCheck %s -check-prefixes=ALL,32R6-NOMADD,32R6-NAN-NOMADD
; RUN: llc < %s -march=mips64el -mcpu=mips64   -target-abi=n64 | FileCheck %s -check-prefixes=ALL,64,64-NAN
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 | FileCheck %s -check-prefixes=ALL,64R2,64R2-NAN
; RUN: llc < %s -march=mips64el -mcpu=mips64r6 -target-abi=n64 | FileCheck %s -check-prefixes=ALL,64R6-NOMADD,64R6-NAN-NOMADD

; Check that madd.[ds], msub.[ds], nmadd.[ds], and nmsub.[ds] are not generated
; when +nomadd attribute is specified.
; Output for mips32 and mips64r6 reused since aforementioned instructions are
; not generated in those cases.
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2            -enable-no-nans-fp-math -mattr=+nomadd4 | FileCheck %s -check-prefixes=ALL,32-NOMADD,32-NONAN-NOMADD
; RUN: llc < %s -march=mips64el -mcpu=mips64   -target-abi=n64 -enable-no-nans-fp-math -mattr=+nomadd4 | FileCheck %s -check-prefixes=ALL,64R6-NOMADD,64R6-NONAN-NOMADD
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 -enable-no-nans-fp-math -mattr=+nomadd4 | FileCheck %s -check-prefixes=ALL,64R6-NOMADD,64R6-NONAN-NOMADD
; RUN: llc < %s -march=mipsel   -mcpu=mips32r2            -mattr=+nomadd4 | FileCheck %s -check-prefixes=ALL,32-NOMADD,32-NAN-NOMADD
; RUN: llc < %s -march=mips64el -mcpu=mips64   -target-abi=n64 -mattr=+nomadd4 | FileCheck %s -check-prefixes=ALL,64R6-NOMADD,64R6-NAN-NOMADD
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 -mattr=+nomadd4 | FileCheck %s -check-prefixes=ALL,64R6-NOMADD,64R6-NAN-NOMADD

define float @FOO0float(float %a, float %b, float %c) nounwind readnone {
entry:
; ALL-LABEL: FOO0float:

; 32-NOMADD-DAG:        mtc1 $6, $[[T0:f[0-9]+]]
; 32-NOMADD-DAG:        mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        add.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        add.s $f0, $[[T1]], $[[T2]]

; 32R2:                 mtc1 $6, $[[T0:f[0-9]+]]
; 32R2:                 madd.s $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2:                 mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2:                 add.s $f0, $[[T1]], $[[T2]]

; 32R6-NOMADD-DAG:      mtc1 $6, $[[T0:f[0-9]+]]
; 32R6-NOMADD-DAG:      mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      add.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      add.s $f0, $[[T1]], $[[T2]]

; 64-DAG:               madd.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-DAG:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-DAG:               add.s $f0, $[[T0]], $[[T1]]

; 64R2:                 madd.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2:                 mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2:                 add.s $f0, $[[T0]], $[[T1]]

; 64R6-NOMADD-DAG:      mul.s $[[T0:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      add.s $[[T1:f[0-9]+]], $[[T0]], $f14
; 64R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      add.s $f0, $[[T1]], $[[T2]]

  %mul = fmul float %a, %b
  %add = fadd float %mul, %c
  %add1 = fadd float %add, 0.000000e+00
  ret float %add1
}

define float @FOO1float(float %a, float %b, float %c) nounwind readnone {
entry:
; ALL-LABEL: FOO1float:

; 32-NOMADD-DAG:        mtc1 $6, $[[T0:f[0-9]+]]
; 32-NOMADD-DAG:        mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        sub.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        add.s $f0, $[[T1]], $[[T2]]

; 32R2:                 mtc1 $6, $[[T0:f[0-9]+]]
; 32R2:                 msub.s $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2:                 mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2:                 add.s $f0, $[[T1]], $[[T2]]

; 32R6-NOMADD-DAG:      mtc1 $6, $[[T0:f[0-9]+]]
; 32R6-NOMADD-DAG:      mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      sub.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      add.s $f0, $[[T1]], $[[T2]]

; 64-DAG:               msub.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-DAG:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-DAG:               add.s $f0, $[[T0]], $[[T1]]

; 64R2:                 msub.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2:                 mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2:                 add.s $f0, $[[T0]], $[[T1]]

; 64R6-NOMADD-DAG:      mul.s $[[T0:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      sub.s $[[T1:f[0-9]+]], $[[T0]], $f14
; 64R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      add.s $f0, $[[T1]], $[[T2]]

  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c
  %add = fadd float %sub, 0.000000e+00
  ret float %add
}

define float @FOO2float(float %a, float %b, float %c) nounwind readnone {
entry:
; ALL-LABEL: FOO2float:

; 32-NOMADD-DAG:        mtc1 $6, $[[T0:f[0-9]+]]
; 32-NOMADD-DAG:        mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        add.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        sub.s $f0, $[[T2]], $[[T1]]

; 32R2-NONAN:           mtc1 $6, $[[T0:f[0-9]+]]
; 32R2-NONAN:           nmadd.s $f0, $[[T0]], $f12, $f14

; 32R2-NAN:             mtc1 $6, $[[T0:f[0-9]+]]
; 32R2-NAN:             madd.s $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2-NAN:             mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2-NAN:             sub.s  $f0, $[[T2]], $[[T1]]

; 32R6-NOMADD-DAG:      mtc1 $6, $[[T0:f[0-9]+]]
; 32R6-NOMADD-DAG:      mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      add.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      sub.s $f0, $[[T2]], $[[T1]]

; 64-NONAN:             nmadd.s $f0, $f14, $f12, $f13

; 64-NAN:               madd.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-NAN:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-NAN:               sub.s  $f0, $[[T1]], $[[T0]]

; 64R2-NONAN:           nmadd.s $f0, $f14, $f12, $f13

; 64R2-NAN:             madd.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2-NAN:             mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2-NAN:             sub.s  $f0, $[[T1]], $[[T0]]

; 64R6-NOMADD-DAG:      mul.s $[[T1:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      add.s $[[T2:f[0-9]+]], $[[T1]], $f14
; 64R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      sub.s $f0, $[[T2]], $[[T1]]

  %mul = fmul float %a, %b
  %add = fadd float %mul, %c
  %sub = fsub float 0.000000e+00, %add
  ret float %sub
}

define float @FOO3float(float %a, float %b, float %c) nounwind readnone {
entry:
; ALL-LABEL: FOO3float:

; 32-NOMADD-DAG:        mtc1 $6, $[[T0:f[0-9]+]]
; 32-NOMADD-DAG:        mul.s $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        sub.s $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        sub.s $f0, $[[T2]], $[[T1]]

; 32R2-NONAN:           mtc1 $6, $[[T0:f[0-9]+]]
; 32R2-NONAN:           nmsub.s $f0, $[[T0]], $f12, $f14

; 32R2-NAN:             mtc1 $6, $[[T0:f[0-9]+]]
; 32R2-NAN:             msub.s $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2-NAN:             mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2-NAN:             sub.s  $f0, $[[T2]], $[[T1]]

; 64-NAN:               msub.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-NAN:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-NAN:               sub.s  $f0, $[[T1]], $[[T0]]

; 64-NONAN:             nmsub.s $f0, $f14, $f12, $f13

; 64R2-NAN:             msub.s $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2-NAN:             mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2-NAN:             sub.s  $f0, $[[T1]], $[[T0]]

; 64R6-NOMADD-DAG:      mul.s $[[T1:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      sub.s $[[T2:f[0-9]+]], $[[T1]], $f14
; 64R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      sub.s $f0, $[[T2]], $[[T1]]

  %mul = fmul float %a, %b
  %sub = fsub float %mul, %c
  %sub1 = fsub float 0.000000e+00, %sub
  ret float %sub1
}

define double @FOO10double(double %a, double %b, double %c) nounwind readnone {
entry:
; ALL-LABEL: FOO10double:

; 32-NOMADD-DAG:        ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32-NOMADD-DAG:        mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        add.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        add.d $f0, $[[T1]], $[[T2]]

; 32R2:                 ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R2:                 madd.d $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2:                 mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2:                 mthc1 $zero, $[[T2]]
; 32R2:                 add.d $f0, $[[T1]], $[[T2]]

; 32R6-NOMADD-DAG:      ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      add.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      add.d $f0, $[[T1]], $[[T2]]

; 64-DAG:               madd.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-DAG:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-DAG:               add.d $f0, $[[T0]], $[[T1]]

; 64R2:                 madd.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2:                 mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2:                 add.d $f0, $[[T0]], $[[T1]]

; 64R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      add.d $[[T2:f[0-9]+]], $[[T1]], $f14
; 64R6-NOMADD-DAG:      dmtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      add.d $f0, $[[T1]], $[[T2]]

  %mul = fmul double %a, %b
  %add = fadd double %mul, %c
  %add1 = fadd double %add, 0.000000e+00
  ret double %add1
}

define double @FOO11double(double %a, double %b, double %c) nounwind readnone {
entry:
; ALL-LABEL: FOO11double:

; 32-NOMADD-DAG:        ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32-NOMADD-DAG:        mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        sub.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        add.d $f0, $[[T1]], $[[T2]]

; 32R2:                 ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R2:                 msub.d $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2:                 mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2:                 mthc1 $zero, $[[T2]]
; 32R2:                 add.d $f0, $[[T1]], $[[T2]]

; 32R6-NOMADD-DAG:      ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      sub.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      add.d $f0, $[[T1]], $[[T2]]

; 64-DAG:               msub.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-DAG:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-DAG:               add.d $f0, $[[T0]], $[[T1]]

; 64R2:                 msub.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2:                 mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2:                 add.d $f0, $[[T0]], $[[T1]]

; 64R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      sub.d $[[T2:f[0-9]+]], $[[T1]], $f14
; 64R6-NOMADD-DAG:      dmtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      add.d $f0, $[[T1]], $[[T2]]

  %mul = fmul double %a, %b
  %sub = fsub double %mul, %c
  %add = fadd double %sub, 0.000000e+00
  ret double %add
}

define double @FOO12double(double %a, double %b, double %c) nounwind readnone {
entry:
; ALL-LABEL: FOO12double:

; 32-NOMADD-DAG:        ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32-NOMADD-DAG:        mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        add.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        sub.d $f0, $[[T2]], $[[T1]]

; 32R2-NONAN:           ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R2-NONAN:           nmadd.d $f0, $[[T0]], $f12, $f14

; 32R2-NAN:             ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R2-NAN:             madd.d $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2-NAN:             mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2-NAN:             mthc1 $zero, $[[T2]]
; 32R2-NAN:             sub.d $f0, $[[T2]], $[[T1]]

; 32R6-NOMADD-DAG:      ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      add.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      sub.d $f0, $[[T2]], $[[T1]]

; 64-NONAN:             nmadd.d $f0, $f14, $f12, $f13

; 64-NAN:               madd.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-NAN:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-NAN:               sub.d $f0, $[[T1]], $[[T0]]

; 64R2-NONAN:           nmadd.d $f0, $f14, $f12, $f13

; 64R2-NAN:             madd.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2-NAN:             mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2-NAN:             sub.d $f0, $[[T1]], $[[T0]]

; 64R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      add.d $[[T2:f[0-9]+]], $[[T1]], $f14
; 64R6-NOMADD-DAG:      dmtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      sub.d $f0, $[[T2]], $[[T1]]

  %mul = fmul double %a, %b
  %add = fadd double %mul, %c
  %sub = fsub double 0.000000e+00, %add
  ret double %sub
}

define double @FOO13double(double %a, double %b, double %c) nounwind readnone {
entry:
; ALL-LABEL: FOO13double:

; 32-NOMADD-DAG:        ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32-NOMADD-DAG:        mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32-NOMADD-DAG:        sub.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32-NOMADD-DAG:        mtc1 $zero, $[[T2:f[0-9]+]]
; 32-NOMADD-DAG:        sub.d $f0, $[[T2]], $[[T1]]

; 32R2-NONAN:           ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R2-NONAN:           nmsub.d $f0, $[[T0]], $f12, $f14

; 32R2-NAN:             ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R2-NAN:             msub.d $[[T1:f[0-9]+]], $[[T0]], $f12, $f14
; 32R2-NAN:             mtc1 $zero, $[[T2:f[0-9]+]]
; 32R2-NAN:             mthc1 $zero, $[[T2]]
; 32R2-NAN:             sub.d $f0, $[[T2]], $[[T1]]

; 32R6-NOMADD-DAG:      ldc1 $[[T0:f[0-9]+]], 16($sp)
; 32R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f14
; 32R6-NOMADD-DAG:      sub.d $[[T2:f[0-9]+]], $[[T1]], $[[T0]]
; 32R6-NOMADD-DAG:      mtc1 $zero, $[[T2:f[0-9]+]]
; 32R6-NOMADD-DAG:      sub.d $f0, $[[T2]], $[[T1]]

; 64-NONAN:             nmsub.d $f0, $f14, $f12, $f13

; 64-NAN:               msub.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64-NAN:               mtc1 $zero, $[[T1:f[0-9]+]]
; 64-NAN:               sub.d $f0, $[[T1]], $[[T0]]

; 64R2-NONAN:           nmsub.d $f0, $f14, $f12, $f13

; 64R2-NAN:             msub.d $[[T0:f[0-9]+]], $f14, $f12, $f13
; 64R2-NAN:             mtc1 $zero, $[[T1:f[0-9]+]]
; 64R2-NAN:             sub.d $f0, $[[T1]], $[[T0]]

; 64R6-NOMADD-DAG:      mul.d $[[T1:f[0-9]+]], $f12, $f13
; 64R6-NOMADD-DAG:      sub.d $[[T2:f[0-9]+]], $[[T1]], $f14
; 64R6-NOMADD-DAG:      dmtc1 $zero, $[[T2:f[0-9]+]]
; 64R6-NOMADD-DAG:      sub.d $f0, $[[T2]], $[[T1]]

  %mul = fmul double %a, %b
  %sub = fsub double %mul, %c
  %sub1 = fsub double 0.000000e+00, %sub
  ret double %sub1
}
