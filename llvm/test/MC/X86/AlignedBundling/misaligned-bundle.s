# RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d --no-show-raw-insn - \
# RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-OPT %s
# RUN: llvm-mc -filetype=obj -triple i686-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d --no-show-raw-insn - \
# RUN:   | FileCheck --check-prefixes=CHECK,CHECK-OPT %s

        .text
foo:
        .bundle_align_mode 5
        push    %ebp          # 1 byte
        .align  16
# CHECK:            1:  nopw %cs:(%eax,%eax)
# CHECK-OPT:        10: movl $1, (%esp)
        movl $0x1, (%esp)     # 7 bytes
        movl $0x1, (%esp)     # 7 bytes
# CHECK-OPT:        1e: nop
        movl $0x2, 0x1(%esp)  # 8 bytes
        movl $0x2, 0x1(%esp)  # 8 bytes
        movl $0x2, 0x1(%esp)  # 8 bytes
        movl $0x2, (%esp)     # 7 bytes
# CHECK-OPT:        3f: nop
# CHECK-OPT:        40: movl $3, (%esp)
        movl $0x3, (%esp)     # 7 bytes
        movl $0x3, (%esp)     # 7 bytes
        ret
