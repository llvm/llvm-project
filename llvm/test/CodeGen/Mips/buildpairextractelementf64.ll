; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefixes=NO-MFHC1,ALL
; RUN: llc -march=mips  < %s | FileCheck %s -check-prefixes=NO-MFHC1,ALL
; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck %s -check-prefixes=HAS-MFHC1,ALL
; RUN: llc -march=mips -mcpu=mips32r2 < %s | FileCheck %s -check-prefixes=HAS-MFHC1,ALL
; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+fp64 < %s | FileCheck %s -check-prefixes=HAS-MFHC1,ALL
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=+fp64 < %s | FileCheck %s -check-prefixes=HAS-MFHC1,ALL

@a = external global i32

; ALL-LABEL: f:

; NO-MFHC1: mtc1
; NO-MFHC1: mtc1

; HAS-MFHC1-DAG: mtc1
; HAS-MFHC1-DAG: mthc1

define double @f(i32 %a1, double %d) nounwind {
entry:
  store i32 %a1, ptr @a, align 4
  %add = fadd double %d, 2.000000e+00
  ret double %add
}

; ALL-LABEL: f3:

; NO-MFHC1: mfc1
; NO-MFHC1: mfc1

; HAS-MFHC1-DAG: mfc1
; HAS-MFHC1-DAG: mfhc1

define void @f3(double %d, i32 %a1) nounwind {
entry:
  tail call void @f2(i32 %a1, double %d) nounwind
  ret void
}

declare void @f2(i32, double)

