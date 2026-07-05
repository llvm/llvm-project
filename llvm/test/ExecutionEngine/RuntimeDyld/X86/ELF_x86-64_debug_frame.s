# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %t/ELF_x86-64_debug_frame.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify -check=%s %t/ELF_x86-64_debug_frame.o

        .text
        .file   "debug_frame_test.c"
        .align  16, 0x90
        .type   foo,@function
foo:
        .cfi_startproc
        retq
.Ltmp0:
        .size   foo, .Ltmp0-foo
        .cfi_endproc
        .cfi_sections .debug_frame

# Check that .debug_frame is mapped to 0.
# rtdyld-check: section_addr(ELF_x86-64_debug_frame.o, .debug_frame) = 0

# Check that The relocated FDE's CIE offset also points to zero.
# rtdyld-check: *{4}(section_addr(ELF_x86-64_debug_frame.o, .debug_frame) + 0x1C) = 0
