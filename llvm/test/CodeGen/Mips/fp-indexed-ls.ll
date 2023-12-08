; RUN: llc -march=mipsel   -mcpu=mips32   -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS32R1
; RUN: llc -march=mipsel   -mcpu=mips32r2 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS32R2
; RUN: llc -march=mipsel   -mcpu=mips32r6 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS32R6
; RUN: llc -march=mips64el -mcpu=mips4    -target-abi=n64 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS4
; RUN: llc -march=mips64el -mcpu=mips64   -target-abi=n64 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS4
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi=n64 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS4
; RUN: llc -march=mips64el -mcpu=mips64r6 -target-abi=n64 -relocation-model=pic < %s | FileCheck %s -check-prefixes=ALL,MIPS64R6

; Check that [ls][dwu]xc1 are not emitted for nacl.
; RUN: llc -mtriple=mipsel-none-nacl-gnu -mcpu=mips32r2 < %s | FileCheck %s -check-prefix=CHECK-NACL

%struct.S = type <{ [4 x float] }>
%struct.S2 = type <{ [4 x double] }>
%struct.S3 = type <{ i8, float }>

@s = external global [4 x %struct.S]
@gf = external global float
@gd = external global double
@s2 = external global [4 x %struct.S2]
@s3 = external global %struct.S3

define float @foo0(ptr nocapture %b, i32 %o) nounwind readonly {
entry:
; ALL-LABEL: foo0:

; MIPS32R1:      sll $[[T1:[0-9]+]], $5, 2
; MIPS32R1:      addu $[[T3:[0-9]+]], $4, $[[T1]]
; MIPS32R1:      lwc1 $f0, 0($[[T3]])

; MIPS32R2:      sll $[[T1:[0-9]+]], $5, 2
; MIPS32R2:      lwxc1 $f0, $[[T1]]($4)

; MIPS32R6:      sll $[[T1:[0-9]+]], $5, 2
; MIPS32R6:      addu $[[T3:[0-9]+]], $4, $[[T1]]
; MIPS32R6:      lwc1 $f0, 0($[[T3]])

; MIPS4:         sll $[[T0:[0-9]+]], $5, 0
; MIPS4:         dsll $[[T1:[0-9]+]], $[[T0]], 2
; MIPS4:         lwxc1 $f0, $[[T1]]($4)

; MIPS64R6:      sll $[[T0:[0-9]+]], $5, 0
; MIPS64R6:      dsll $[[T1:[0-9]+]], $[[T0]], 2
; MIPS64R6:      daddu $[[T3:[0-9]+]], $4, $[[T1]]
; MIPS64R6:      lwc1 $f0, 0($[[T3]])

; CHECK-NACL-NOT: lwxc1

  %arrayidx = getelementptr inbounds float, ptr %b, i32 %o
  %0 = load float, ptr %arrayidx, align 4
  ret float %0
}

define double @foo1(ptr nocapture %b, i32 %o) nounwind readonly {
entry:
; ALL-LABEL: foo1:

; MIPS32R1:      sll $[[T1:[0-9]+]], $5, 3
; MIPS32R1:      addu $[[T3:[0-9]+]], $4, $[[T1]]
; MIPS32R1:      ldc1 $f0, 0($[[T3]])

; MIPS32R2:      sll $[[T1:[0-9]+]], $5, 3
; MIPS32R2:      ldxc1 $f0, $[[T1]]($4)

; MIPS32R6:      sll $[[T1:[0-9]+]], $5, 3
; MIPS32R6:      addu $[[T3:[0-9]+]], $4, $[[T1]]
; MIPS32R6:      ldc1 $f0, 0($[[T3]])

; MIPS4:         sll $[[T0:[0-9]+]], $5, 0
; MIPS4:         dsll $[[T1:[0-9]+]], $[[T0]], 3
; MIPS4:         ldxc1 $f0, $[[T1]]($4)

; MIPS64R6:      sll $[[T0:[0-9]+]], $5, 0
; MIPS64R6:      dsll $[[T1:[0-9]+]], $[[T0]], 3
; MIPS64R6:      daddu $[[T3:[0-9]+]], $4, $[[T1]]
; MIPS64R6:      ldc1 $f0, 0($[[T3]])

; CHECK-NACL-NOT: ldxc1

  %arrayidx = getelementptr inbounds double, ptr %b, i32 %o
  %0 = load double, ptr %arrayidx, align 8
  ret double %0
}

define float @foo2(i32 %b, i32 %c) nounwind readonly {
entry:
; ALL-LABEL: foo2:

; luxc1 did not exist in MIPS32r1
; MIPS32R1-NOT:  luxc1

; luxc1 is a misnomer since it aligns the given pointer downwards and performs
; an aligned load. We mustn't use it to handle unaligned loads.
; MIPS32R2-NOT:  luxc1

; luxc1 was removed in MIPS32r6
; MIPS32R6-NOT:  luxc1

; MIPS4-NOT:     luxc1

; luxc1 was removed in MIPS64r6
; MIPS64R6-NOT:  luxc1

  %arrayidx1 = getelementptr inbounds [4 x %struct.S], ptr @s, i32 0, i32 %b, i32 0, i32 %c
  %0 = load float, ptr %arrayidx1, align 1
  ret float %0
}

define void @foo3(ptr nocapture %b, i32 %o) nounwind {
entry:
; ALL-LABEL: foo3:

; MIPS32R1-DAG:  lwc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS32R1-DAG:  addu $[[T1:[0-9]+]], $4, ${{[0-9]+}}
; MIPS32R1-DAG:  swc1 $[[T0]], 0($[[T1]])

; MIPS32R2:      lwc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS32R2:      swxc1 $[[T0]], ${{[0-9]+}}($4)

; MIPS32R6-DAG:  lwc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS32R6-DAG:  addu $[[T1:[0-9]+]], $4, ${{[0-9]+}}
; MIPS32R6-DAG:  swc1 $[[T0]], 0($[[T1]])

; MIPS4:         lwc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS4:         swxc1 $[[T0]], ${{[0-9]+}}($4)

; MIPS64R6-DAG:  lwc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS64R6-DAG:  daddu $[[T1:[0-9]+]], $4, ${{[0-9]+}}
; MIPS64R6-DAG:  swc1 $[[T0]], 0($[[T1]])

; CHECK-NACL-NOT: swxc1

  %0 = load float, ptr @gf, align 4
  %arrayidx = getelementptr inbounds float, ptr %b, i32 %o
  store float %0, ptr %arrayidx, align 4
  ret void
}

define void @foo4(ptr nocapture %b, i32 %o) nounwind {
entry:
; ALL-LABEL: foo4:

; MIPS32R1-DAG:  ldc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS32R1-DAG:  addu $[[T1:[0-9]+]], $4, ${{[0-9]+}}
; MIPS32R1-DAG:  sdc1 $[[T0]], 0($[[T1]])

; MIPS32R2:      ldc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS32R2:      sdxc1 $[[T0]], ${{[0-9]+}}($4)

; MIPS32R6-DAG:  ldc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS32R6-DAG:  addu $[[T1:[0-9]+]], $4, ${{[0-9]+}}
; MIPS32R6-DAG:  sdc1 $[[T0]], 0($[[T1]])

; MIPS4:         ldc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS4:         sdxc1 $[[T0]], ${{[0-9]+}}($4)

; MIPS64R6-DAG:  ldc1 $[[T0:f0]], 0(${{[0-9]+}})
; MIPS64R6-DAG:  daddu $[[T1:[0-9]+]], $4, ${{[0-9]+}}
; MIPS64R6-DAG:  sdc1 $[[T0]], 0($[[T1]])

; CHECK-NACL-NOT: sdxc1

  %0 = load double, ptr @gd, align 8
  %arrayidx = getelementptr inbounds double, ptr %b, i32 %o
  store double %0, ptr %arrayidx, align 8
  ret void
}

define void @foo5(i32 %b, i32 %c) nounwind {
entry:
; ALL-LABEL: foo5:

; MIPS32R1-NOT:  suxc1

; MIPS32R2-NOT:  suxc1

; MIPS32R6-NOT:  suxc1

; MIPS4-NOT:     suxc1

; MIPS64R6-NOT:  suxc1

  %0 = load float, ptr @gf, align 4
  %arrayidx1 = getelementptr inbounds [4 x %struct.S], ptr @s, i32 0, i32 %b, i32 0, i32 %c
  store float %0, ptr %arrayidx1, align 1
  ret void
}

define double @foo6(i32 %b, i32 %c) nounwind readonly {
entry:
; ALL-LABEL: foo6:

; MIPS32R1-NOT:  luxc1

; MIPS32R2-NOT:  luxc1

; MIPS32R6-NOT:  luxc1

; MIPS4-NOT:     luxc1

; MIPS64R6-NOT:  luxc1

  %arrayidx1 = getelementptr inbounds [4 x %struct.S2], ptr @s2, i32 0, i32 %b, i32 0, i32 %c
  %0 = load double, ptr %arrayidx1, align 1
  ret double %0
}

define void @foo7(i32 %b, i32 %c) nounwind {
entry:
; ALL-LABEL: foo7:

; MIPS32R1-NOT:  suxc1

; MIPS32R2-NOT:  suxc1

; MIPS32R6-NOT:  suxc1

; MIPS4-NOT:     suxc1

; MIPS64R6-NOT:  suxc1

  %0 = load double, ptr @gd, align 8
  %arrayidx1 = getelementptr inbounds [4 x %struct.S2], ptr @s2, i32 0, i32 %b, i32 0, i32 %c
  store double %0, ptr %arrayidx1, align 1
  ret void
}

define float @foo8() nounwind readonly {
entry:
; ALL-LABEL: foo8:

; MIPS32R1-NOT:  luxc1

; MIPS32R2-NOT:  luxc1

; MIPS32R6-NOT:  luxc1

; MIPS4-NOT:     luxc1

; MIPS64R6-NOT:  luxc1

  %0 = load float, ptr getelementptr inbounds (%struct.S3, ptr @s3, i32 0, i32 1), align 1
  ret float %0
}

define void @foo9(float %f) nounwind {
entry:
; ALL-LABEL: foo9:

; MIPS32R1-NOT:  suxc1

; MIPS32R2-NOT:  suxc1

; MIPS32R6-NOT:  suxc1

; MIPS4-NOT:     suxc1

; MIPS64R6-NOT:  suxc1

  store float %f, ptr getelementptr inbounds (%struct.S3, ptr @s3, i32 0, i32 1), align 1
  ret void
}

