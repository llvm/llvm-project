; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN-TODO: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static -mattr=+o32,+fp64 < %s | FileCheck --check-prefixes=ALL,032FP64 %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static -mattr=+o32,+fp64 < %s | FileCheck --check-prefixes=ALL,032FP64 %s

; Test the float returns for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).

; We only test Linux because other OS's use different relocations and I don't
; know if this is correct.

@float = global float zeroinitializer
@double = global double zeroinitializer

define float @retfloat() nounwind {
entry:
        %0 = load volatile float, float* @float
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
        %0 = load volatile double, double* @double
        ret double %0
}

; ALL-LABEL: retdouble:
; O32-DAG:           ldc1 $f0, %lo(double)([[R1:\$[0-9]+]])
; N32-DAG:           ldc1 $f0, %lo(double)([[R1:\$[0-9]+]])
; N64-DAG:           ldc1 $f0, %lo(double)([[R1:\$[0-9]+]])

define { double, double } @retComplexDouble() #0 {
  %retval = alloca { double, double }, align 8
  %1 = load { double, double }, { double, double }* %retval
  ret { double, double } %1
}

; ALL-LABEL: retComplexDouble:
; 032FP64-DAG:      ldc1     $f0, 0($sp)
; 032FP64-DAG:      ldc1     $f2, 8($sp)
