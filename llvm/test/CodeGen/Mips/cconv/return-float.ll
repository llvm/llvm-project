; RUN: llc -mtriple=mips-linux-gnu -mattr=+soft-float -relocation-model=static < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel-linux-gnu -mattr=+soft-float -relocation-model=static < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN-TODO: llc -mtriple=mips64-linux-gnu -mattr=+soft-float -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -mtriple=mips64el-linux-gnu -mattr=+soft-float -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64-linux-gnu -mattr=+soft-float -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el-linux-gnu -mattr=+soft-float -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64-linux-gnu -mattr=+soft-float -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el-linux-gnu -mattr=+soft-float -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; Test the float returns for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).

; We only test Linux because other OS's use different relocations and I don't
; know if this is correct.

@float = global float zeroinitializer
@double = global double zeroinitializer

define float @retfloat() nounwind {
entry:
        %0 = load volatile float, ptr @float
        ret float %0
}

; ALL-LABEL: retfloat:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(float)
; O32-DAG:           lw $2, %lo(float)([[R1]])
; N32-DAG:           lui [[R1:\$[0-9]+]], %hi(float)
; N32-DAG:           lw $2, %lo(float)([[R1]])
; N64-DAG:           lw $2, %lo(float)([[R1:\$[0-9+]]])

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
