# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r6 -filetype=obj %s -o %t/o32el.o
# RUN: llvm-mc -triple=mips-unknown-linux-gnu -mcpu=mips32r6 -filetype=obj %s -o %t/o32be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r6 -filetype=obj %s -o %t/n64el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64r6 -filetype=obj %s -o %t/n64be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-o32el --abs=external_func=0x30000000 %t/o32el.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-o32be --abs=external_func=0x30000000 %t/o32be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-n64el --abs=external_func=0x30000000 %t/n64el.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-n64be --abs=external_func=0x30000000 %t/n64be.o

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        balc external_func
        jrc $ra
        .size main, .-main

# R_MIPS_PC26_S2 carries an addend of -4 because its architectural base is
# the instruction following the branch. An out-of-range absolute call must be
# redirected to a reachable stub, whose pointer entry contains the callee.
# check-o32el: decode_operand(main, 0) = stub_addr(o32el.o, external_func) - main
# check-o32el: *{4}(got_addr(o32el.o, external_func)) = external_func
# check-o32be: decode_operand(main, 0) = stub_addr(o32be.o, external_func) - main
# check-o32be: *{4}(got_addr(o32be.o, external_func)) = external_func
# check-n64el: decode_operand(main, 0) = stub_addr(n64el.o, external_func) - main
# check-n64el: *{8}(got_addr(n64el.o, external_func)) = external_func
# check-n64be: decode_operand(main, 0) = stub_addr(n64be.o, external_func) - main
# check-n64be: *{8}(got_addr(n64be.o, external_func)) = external_func
