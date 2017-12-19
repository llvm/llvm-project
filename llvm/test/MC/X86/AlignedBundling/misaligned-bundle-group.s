# RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump -disassemble -no-show-raw-insn - \
# RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-OPT %s
# RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -disassemble -no-show-raw-insn - \
# RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-RELAX %s

        .text
foo:
        .bundle_align_mode 5
        push    %ebp # 1 byte
        .align  16
        .bundle_lock align_to_end
# CHECK:            1:  nopw %cs:(%eax,%eax)
# CHECK:            10: nopw %cs:(%eax,%eax)
# CHECK-RELAX:      1f: nop
# CHECK-RELAX:      20: nopw %cs:(%eax,%eax)
# CHECK-RELAX:      2f: nopw %cs:(%eax,%eax)
# CHECK-OPT:        1b: calll -4
# CHECK-RELAX:      3b: calll -4
        calll   bar # 5 bytes
        .bundle_unlock
        ret         # 1 byte
