; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -mattr=+micromips | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r2 -mattr=+micromips | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32                           -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=+fp64            -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32FP64
; RUN: llc -march=mips -mcpu=mips32r3 -mattr=+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MM
; RUN: llc -march=mips -mcpu=mips32r3 -mattr=+micromips,+fp64 -asm-show-inst < %s | FileCheck %s --check-prefix=MMFP64
; RUN: llc -march=mips -mcpu=mips32r6 -mattr=+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MMR6

define float @sqrt_fn(float %value) #0 {
entry:
  %sqrtf = tail call float @sqrtf(float %value) #0
  ret float %sqrtf
}

declare float @sqrtf(float)

; CHECK: sqrt.s $f0, $f12


define float @sqrt_s(float %a) {
; MIPS32:     sqrt.s  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_S
; MIPS32FP64: sqrt.s  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_S
; MM:         sqrt.s  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_S_MM
; MMFP64:     sqrt.s  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_S_MM
; MMR6:       sqrt.s  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_S_MM
  %ret = call float @llvm.sqrt.f32(float %a)
  ret float %ret
}

define double @sqrt_d(double %a) {
; MIPS32:     sqrt.d  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_D32
; MIPS32FP64: sqrt.d  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_D64
; MM:         sqrt.d  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_D32_MM
; MMFP64:     sqrt.d  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_D64_MM
; MMR6:       sqrt.d  {{.*}}               # <MCInst #{{[0-9]+}} FSQRT_D64_MM
  %ret = call double @llvm.sqrt.f64(double %a)
  ret double %ret
}

declare float @llvm.sqrt.f32(float %a)
declare double @llvm.sqrt.f64(double %a)
