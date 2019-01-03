// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-arm-none-eabi %s -o - | llvm-dwarfdump - -v | FileCheck %s
#CHECK:   Augmentation:          "zRB"
f1:
        .cfi_startproc
        .cfi_b_key_frame
        .cfi_endproc
