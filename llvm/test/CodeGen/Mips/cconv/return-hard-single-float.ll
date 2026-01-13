; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static -mattr=+single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static -mattr=+single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n32 -mattr=+single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n32 -mattr=+single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n64 -mattr=+single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n64 -mattr=+single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,N64 %s

@float = global float zeroinitializer
@double = global double zeroinitializer

define float @retfloat() nounwind {
entry:
        %0 = load volatile float, ptr @float
        ret float %0
}

; ALL-LABEL: retfloat:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(float)
; O32-DAG:           lwc1 $f0, %lo(float)([[R1]])
; N32-DAG:           lui [[R1:\$[0-9]+]], %hi(float)
; N32-DAG:           lwc1 $f0, %lo(float)([[R1]])
; N64-DAG:           lwc1 $f0, %lo(float)([[R1:\$[0-9+]]])

define double @retdouble() nounwind {
entry:
        %0 = load volatile double, ptr @double
        ret double %0
}

; ALL-LABEL: retdouble:
; O32-DAG:           lw $2, %lo(double)([[R1:\$[0-9]+]])
; O32-DAG:           addiu [[R2:\$[0-9]+]], [[R1]], %lo(double)
; O32-DAG:           lw $3, 4([[R2]])
; N32-DAG:           ld $2, %lo(double)([[R1:\$[0-9]+]])
; N64-DAG:           ld $2, %lo(double)([[R1:\$[0-9]+]])
