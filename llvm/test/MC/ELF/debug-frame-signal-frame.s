// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o
// RUN: llvm-dwarfdump --debug-frame %t.o | FileCheck %s

        .cfi_sections .debug_frame
sig:
        .cfi_startproc
        .cfi_signal_frame
        .cfi_endproc
normal:
        .cfi_startproc
        .cfi_endproc

// CHECK-DAG: Augmentation:          "S"
// CHECK-DAG: Augmentation:          ""
