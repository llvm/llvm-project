; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN-TODO: llc -mtriple=mips64-linux-gnu -relocation-model=static -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN-TODO: llc -mtriple=mips64el-linux-gnu -relocation-model=static -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 %s

; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static -mattr=+o32,+fp64 < %s | FileCheck --check-prefix=ALL --check-prefix=032FP64 %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static -mattr=+o32,+fp64 < %s | FileCheck --check-prefix=ALL --check-prefix=032FP64 %s

; Test the float returns for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).

; We only test Linux because other OS's use different relocations and I don't
; know if this is correct.

@float = global float zeroinitializer
@double = global double zeroinitializer

define float @retfloat() nounwind {
entry:
        %0 = load volatile float* @float
        ret float %0
}

; ALL-LABEL: retfloat:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(float)
; O32-DAG:           lwc1 $f0, %lo(float)([[R1]])
; N32-DAG:           lui [[R1:\$[0-9]+]], %hi(float)
; N32-DAG:           lwc1 $f0, %lo(float)([[R1]])
; N64-DAG:           ld  [[R1:\$[0-9]+]], %got_disp(float)(
; N64-DAG:           lwc1 $f0, 0([[R1]])

define double @retdouble() nounwind {
entry:
        %0 = load volatile double* @double
        ret double %0
}

; ALL-LABEL: retdouble:
; O32-DAG:           ldc1 $f0, %lo(double)([[R1:\$[0-9]+]])
; N32-DAG:           ldc1 $f0, %lo(double)([[R1:\$[0-9]+]])
; N64-DAG:           ld  [[R1:\$[0-9]+]], %got_disp(double)(
; N64-DAG:           ldc1 $f0, 0([[R1]])

define { double, double } @retComplexDouble() #0 {
  %retval = alloca { double, double }, align 8
  %1 = load { double, double }* %retval
  ret { double, double } %1
}

; ALL-LABEL: retComplexDouble:
; 032FP64-DAG:      ldc1     $f0, 0($sp)
; 032FP64-DAG:      ldc1     $f2, 8($sp)
