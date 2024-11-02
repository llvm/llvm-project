# RUN: llvm-mc -filetype=obj --triple=arm64-apple-darwin22.1.0 %s -o %t.o
# RUN: llvm-dwarfdump -debug-frame %t.o | FileCheck %s

# CHECK: .debug_frame contents:
# CHECK-EMPTY:
# CHECK-NEXT: 00000000 00000014 ffffffff CIE
# CHECK: .eh_frame contents:
# CHECK-EMPTY:

 .cfi_startproc
 .cfi_signal_frame
 .cfi_def_cfa x28, 0x340
 .cfi_endproc
 .cfi_sections .debug_frame
