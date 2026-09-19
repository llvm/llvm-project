# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t/o32el.o
# RUN: llvm-mc -triple=mips-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t/o32be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r2 -defsym PTR64=1 -filetype=obj %s -o %t/n64el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64r2 -defsym PTR64=1 -filetype=obj %s -o %t/n64be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-o32el --abs=ext0=0x20000000 --abs=ext1=0x20000004 --abs=ext2=0x20000008 --abs=ext3=0x2000000c --abs=ext4=0x20000010 --abs=ext5=0x20000014 --abs=ext6=0x20000018 --abs=ext7=0x2000001c %t/o32el.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-o32be --abs=ext0=0x20000000 --abs=ext1=0x20000004 --abs=ext2=0x20000008 --abs=ext3=0x2000000c --abs=ext4=0x20000010 --abs=ext5=0x20000014 --abs=ext6=0x20000018 --abs=ext7=0x2000001c %t/o32be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-n64el --abs=ext0=0x20000000 --abs=ext1=0x20000004 --abs=ext2=0x20000008 --abs=ext3=0x2000000c --abs=ext4=0x20000010 --abs=ext5=0x20000014 --abs=ext6=0x20000018 --abs=ext7=0x2000001c %t/n64el.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s -check-name=check-n64be --abs=ext0=0x20000000 --abs=ext1=0x20000004 --abs=ext2=0x20000008 --abs=ext3=0x2000000c --abs=ext4=0x20000010 --abs=ext5=0x20000014 --abs=ext6=0x20000018 --abs=ext7=0x2000001c %t/n64be.o

# Regression test for GP-region ordering. More than four pointer-sized entries
# used to be sorted before the 0x7ff0-byte GOT prefix, placing the first entry
# below the signed 16-bit range of _gp.

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        jr $ra
        nop

        .globl load_got
        .type load_got,@function
load_got:
        .ifdef PTR64
        ld $2, %got(ext0)($gp)
        ld $3, %got(ext1)($gp)
        ld $4, %got(ext2)($gp)
        ld $5, %got(ext3)($gp)
        ld $6, %got(ext4)($gp)
        ld $7, %got(ext5)($gp)
        ld $8, %got(ext6)($gp)
        ld $9, %got(ext7)($gp)
        .else
        lw $2, %got(ext0)($gp)
        lw $3, %got(ext1)($gp)
        lw $4, %got(ext2)($gp)
        lw $5, %got(ext3)($gp)
        lw $6, %got(ext4)($gp)
        lw $7, %got(ext5)($gp)
        lw $8, %got(ext6)($gp)
        lw $9, %got(ext7)($gp)
        .endif
        jr $ra
        nop

# check-o32el: *{4}(got_addr(o32el.o, ext0)) = 0x20000000
# check-o32el: *{4}(got_addr(o32el.o, ext7)) = 0x2000001c
# check-o32be: *{4}(got_addr(o32be.o, ext0)) = 0x20000000
# check-o32be: *{4}(got_addr(o32be.o, ext7)) = 0x2000001c
# check-n64el: *{8}(got_addr(n64el.o, ext0)) = 0x20000000
# check-n64el: *{8}(got_addr(n64el.o, ext7)) = 0x2000001c
# check-n64be: *{8}(got_addr(n64be.o, ext0)) = 0x20000000
# check-n64be: *{8}(got_addr(n64be.o, ext7)) = 0x2000001c
