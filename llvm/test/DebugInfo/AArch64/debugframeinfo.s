# RUN: llvm-mc -filetype=obj --triple=arm64-apple-darwin22.1.0 %s -o %t.o
# RUN: llvm-dwarfdump -debug-frame %t.o | FileCheck %s

# CHECK: .debug_frame contents:
# CHECK-EMPTY:
# CHECK-NEXT: {{.+}}

# CHECK: .eh_frame contents:
# CHECK-EMPTY:
# CHECK-EMPTY:

        .cfi_sections .debug_frame
        .cfi_startproc
        .cfi_personality 0x9b, g
        .cfi_lsda 0x1b, h
        .cfi_endproc
        .global g
g:
        .global h
h:
