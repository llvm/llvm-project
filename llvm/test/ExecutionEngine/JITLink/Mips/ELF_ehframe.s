# REQUIRES: asserts
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t/o32el.o
# RUN: llvm-mc -triple=mips-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t/o32be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r2 -filetype=obj %s -o %t/n64el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64r2 -filetype=obj %s -o %t/n64be.o
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec -phony-externals \
# RUN:   -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 \
# RUN:   %t/o32el.o 2>&1 | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec -phony-externals \
# RUN:   -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 \
# RUN:   %t/o32be.o 2>&1 | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec -phony-externals \
# RUN:   -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 \
# RUN:   %t/n64el.o 2>&1 | FileCheck %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec -phony-externals \
# RUN:   -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 \
# RUN:   %t/n64be.o 2>&1 | FileCheck %s

# CHECK: DWARFRecordSectionSplitter: Processing .eh_frame...
# CHECK: Processing CFI record at
# CHECK: Processing CFI record at
# CHECK: Processing CFI record at
# CHECK: EHFrameEdgeFixer: Processing .eh_frame in "{{.*}}"...
# CHECK: Record is CIE
# CHECK: Record is FDE
# CHECK: Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK: Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK: Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}
# CHECK: Record is FDE
# CHECK: Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK: Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK: Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        .cfi_startproc
        nop
        jr $ra
        nop
        .cfi_endproc
        .size main, .-main

        .globl dup
        .type dup,@function
dup:
        .cfi_startproc
        nop
        jr $ra
        nop
        .cfi_endproc
        .size dup, .-dup
