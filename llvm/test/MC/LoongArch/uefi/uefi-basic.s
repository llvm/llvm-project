# Test basic LoongArch64 UEFI assembly and COFF object generation

# RUN: llvm-mc -triple=loongarch64-unknown-uefi -filetype=obj -o %t.coff %s
# RUN: llvm-readobj --file-headers %t.coff | FileCheck %s

.text
.globl _start
_start:
    addi.d $a0, $zero, 42
    b _exit

_exit:
    addi.d $a7, $zero, 93
    syscall 0

# CHECK: Machine: IMAGE_FILE_MACHINE_LOONGARCH64 (0x6264)
