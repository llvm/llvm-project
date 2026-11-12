; RUN: llc -mtriple=mips -mcpu=mips32                                    -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips -mcpu=mips32r2 -mattr=+abs2008,+fp64            -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32FP64
; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=+abs2008,+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MM
; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=+abs2008,+micromips,+fp64 -asm-show-inst < %s | FileCheck %s --check-prefix=MMFP64
; RUN: llc -mtriple=mips -mcpu=mips32r6 -mattr=+micromips                -asm-show-inst < %s | FileCheck %s --check-prefix=MMR6

define float @abs_s(float %a) {
; MIPS32: {{(ori|ins)}}
; MIPS32-NOT: abs.s
; MIPS32FP64: abs.s  {{.*}}               # <MCInst #{{[0-9]+}} FABS_S
; MM:         abs.s  {{.*}}               # <MCInst #{{[0-9]+}} FABS_S_MM
; MMFP64:     abs.s  {{.*}}               # <MCInst #{{[0-9]+}} FABS_S_MM
; MMR6:       abs.s  {{.*}}               # <MCInst #{{[0-9]+}} FABS_S_MM
    %ret = call float @llvm.fabs.f32(float %a)
    ret float %ret
}

define double @abs_d(double %a) {
; MIPS32: {{(ori|ins|dsll)}}
; MIPS32-NOT: abs.d
; MIPS32FP64: abs.d  {{.*}}               # <MCInst #{{[0-9]+}} FABS_D64
; MM:         abs.d  {{.*}}               # <MCInst #{{[0-9]+}} FABS_D32_MM
; MMFP64:     abs.d  {{.*}}               # <MCInst #{{[0-9]+}} FABS_D64_MM
; MMR6:       abs.d  {{.*}}               # <MCInst #{{[0-9]+}} FABS_D64_MM
    %ret = call double @llvm.fabs.f64(double %a)
    ret double %ret
}

declare float @llvm.fabs.f32(float %a)
declare double @llvm.fabs.f64(double %a)
