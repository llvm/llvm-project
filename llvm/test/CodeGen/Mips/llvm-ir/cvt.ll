; RUN: llc -mtriple=mips -mcpu=mips32                           -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips -mcpu=mips32r2 -mattr=+fp64            -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32FP64
; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MM
; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=+micromips,+fp64 -asm-show-inst < %s | FileCheck %s --check-prefix=MMFP64
; RUN: llc -mtriple=mips -mcpu=mips32r6 -mattr=+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MMR6

; TODO: Test for cvt_w_d is missing, could not generate instruction

define double @cvt_d_s(float %a) {
; MIPS32:     cvt.d.s {{.*}}               # <MCInst #{{[0-9]+}} CVT_D32_S
; MIPS32FP64: cvt.d.s {{.*}}               # <MCInst #{{[0-9]+}} CVT_D64_S
; MM:         cvt.d.s {{.*}}               # <MCInst #{{[0-9]+}} CVT_D32_S_MM
; MMFP64:     cvt.d.s {{.*}}               # <MCInst #{{[0-9]+}} CVT_D64_S_MM
; MMR6:       cvt.d.s {{.*}}               # <MCInst #{{[0-9]+}} CVT_D64_S_MM
    %1 = fpext float %a to double
    ret double %1
}

define double @cvt_d_w(i32 %a) {
; MIPS32:     cvt.d.w {{.*}}                # <MCInst #{{[0-9]+}} CVT_D32_W
; MIPS32FP64: cvt.d.w {{.*}}                # <MCInst #{{[0-9]+}} CVT_D64_W
; MM:         cvt.d.w {{.*}}                # <MCInst #{{[0-9]+}} CVT_D32_W_MM
; MMFP64:     cvt.d.w {{.*}}                # <MCInst #{{[0-9]+}} CVT_D64_W_MM
; MMR6:       cvt.d.w {{.*}}                # <MCInst #{{[0-9]+}} CVT_D64_W_MM
    %1 = sitofp i32 %a to double
    ret double %1
}

define float @cvt_s_d(double %a) {
; MIPS32:     cvt.s.d {{.*}}               # <MCInst #{{[0-9]+}} CVT_S_D32
; MIPS32FP64: cvt.s.d {{.*}}               # <MCInst #{{[0-9]+}} CVT_S_D64
; MM:         cvt.s.d {{.*}}               # <MCInst #{{[0-9]+}} CVT_S_D32_MM
; MMFP64:     cvt.s.d {{.*}}               # <MCInst #{{[0-9]+}} CVT_S_D64_MM
; MMR6:       cvt.s.d {{.*}}               # <MCInst #{{[0-9]+}} CVT_S_D64_MM
     %1 = fptrunc double %a to float
     ret float %1
}
