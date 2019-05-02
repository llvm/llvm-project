; RUN: llc -march=mips -mcpu=mips32r2                         -asm-show-inst \
; RUN: < %s | FileCheck %s --check-prefix=MIPS32R2
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=+fp64            -asm-show-inst \
; RUN: < %s | FileCheck %s --check-prefix=MIPS32FP64
; RUN: llc -march=mips -mcpu=mips32r3 -mattr=+micromips       -asm-show-inst \
; RUN: < %s | FileCheck %s --check-prefix=MM
; RUN: llc -march=mips -mcpu=mips32r3 -mattr=+micromips,+fp64 -asm-show-inst \
; RUN: < %s | FileCheck %s --check-prefix=MMFP64
; RUN: llc -march=mips -mcpu=mips32r6 -mattr=+micromips       -asm-show-inst \
; RUN: < %s | FileCheck %s --check-prefix=MMR6

define double @mthc1(i64 %a) {
; MIPS32R2:   mthc1   {{.*}}                 # <MCInst #{{[0-9]+}} MTHC1_D32
; MIPS32FP64: mthc1   {{.*}}                 # <MCInst #{{[0-9]+}} MTHC1_D64
; MM:         mthc1   {{.*}}                 # <MCInst #{{[0-9]+}} MTHC1_D32_MM
; MMFP64:     mthc1   {{.*}}                 # <MCInst #{{[0-9]+}} MTHC1_D64_MM
; MMR6:       mthc1   {{.*}}                 # <MCInst #{{[0-9]+}} MTHC1_D64_MM
    %1 = bitcast i64 %a to double
    ret double %1
}

define i64 @mfhc1(double %a) {
; MIPS32R2:   mfhc1   {{.*}}                 # <MCInst #{{[0-9]+}} MFHC1_D32
; MIPS32FP64: mfhc1   {{.*}}                 # <MCInst #{{[0-9]+}} MFHC1_D64
; MM:         mfhc1   {{.*}}                 # <MCInst #{{[0-9]+}} MFHC1_D32_MM
; MMFP64:     mfhc1   {{.*}}                 # <MCInst #{{[0-9]+}} MFHC1_D64_MM
; MMR6:       mfhc1   {{.*}}                 # <MCInst #{{[0-9]+}} MFHC1_D64_MM
    %1 = bitcast double %a to i64
    ret i64 %1
}
