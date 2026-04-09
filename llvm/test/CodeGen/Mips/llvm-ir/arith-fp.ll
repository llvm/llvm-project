; RUN: llc -mtriple=mips -mcpu=mips32                           -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips -mcpu=mips32r2 -mattr=+fp64            -asm-show-inst < %s | FileCheck %s --check-prefix=MIPS32FP64
; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MM
; RUN: llc -mtriple=mips -mcpu=mips32r3 -mattr=+micromips,+fp64 -asm-show-inst < %s | FileCheck %s --check-prefix=MMFP64
; RUN: llc -mtriple=mips -mcpu=mips32r6 -mattr=+micromips       -asm-show-inst < %s | FileCheck %s --check-prefix=MMR6

define double @add_d(double %a, double %b) {
; MIPS32:     add.d   {{.*}}         # <MCInst #{{[0-9]+}} FADD_D32
; MIPS32FP64: add.d   {{.*}}         # <MCInst #{{[0-9]+}} FADD_D64
; MM:         add.d   {{.*}}         # <MCInst #{{[0-9]+}} FADD_D32_MM
; MMFP64:     add.d   {{.*}}         # <MCInst #{{[0-9]+}} FADD_D64_MM
; MMR6:       add.d   {{.*}}         # <MCInst #{{[0-9]+}} FADD_D64_MM
    %1 = fadd double %a, %b
    ret double %1
}

define double @sub_d(double %a, double %b) {
; MIPS32:     sub.d   {{.*}}         # <MCInst #{{[0-9]+}} FSUB_D32
; MIPS32FP64: sub.d   {{.*}}         # <MCInst #{{[0-9]+}} FSUB_D64
; MM:         sub.d   {{.*}}         # <MCInst #{{[0-9]+}} FSUB_D32_MM
; MMFP64:     sub.d   {{.*}}         # <MCInst #{{[0-9]+}} FSUB_D64_MM
; MMR6:       sub.d   {{.*}}         # <MCInst #{{[0-9]+}} FSUB_D64_MM
    %1 = fsub double %a, %b
    ret double %1
}

define double @mul_d(double %a, double %b) {
; MIPS32:     mul.d   {{.*}}         # <MCInst #{{[0-9]+}} FMUL_D32
; MIPS32FP64: mul.d   {{.*}}         # <MCInst #{{[0-9]+}} FMUL_D64
; MM:         mul.d   {{.*}}         # <MCInst #{{[0-9]+}} FMUL_D32_MM
; MMFP64:     mul.d   {{.*}}         # <MCInst #{{[0-9]+}} FMUL_D64_MM
; MMR6:       mul.d   {{.*}}         # <MCInst #{{[0-9]+}} FMUL_D64_MM
    %1 = fmul double %a, %b
    ret double %1
}

define double @div_d(double %a, double %b) {
; MIPS32:     div.d   {{.*}}         # <MCInst #{{[0-9]+}} FDIV_D32
; MIPS32FP64: div.d   {{.*}}         # <MCInst #{{[0-9]+}} FDIV_D64
; MM:         div.d   {{.*}}         # <MCInst #{{[0-9]+}} FDIV_D32_MM
; MMFP64:     div.d   {{.*}}         # <MCInst #{{[0-9]+}} FDIV_D64_MM
; MMR6:       div.d   {{.*}}         # <MCInst #{{[0-9]+}} FDIV_D64_MM
    %1 = fdiv double %a, %b
    ret double %1
}

define double @fneg(double %a) {
; MIPS32:     neg.d   {{.*}}         # <MCInst #{{[0-9]+}} FNEG_D32
; MIPS32FP64: neg.d   {{.*}}         # <MCInst #{{[0-9]+}} FNEG_D64
; MM:         neg.d   {{.*}}         # <MCInst #{{[0-9]+}} FNEG_D32_MM
; MMFP64:     neg.d   {{.*}}         # <MCInst #{{[0-9]+}} FNEG_D64_MM
; MMR6:       neg.d   {{.*}}         # <MCInst #{{[0-9]+}} FNEG_D64_MM
    %1 = fsub double -0.000000e+00, %a
    ret double %1
}
