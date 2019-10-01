// RUN: llvm-mc -filetype=obj -triple aarch64-windows-gnu %s | llvm-readobj -S --sd | FileCheck %s
    .text
    .align 5
f0:
    ret

// CHECK: IMAGE_SCN_ALIGN_32BYTES
