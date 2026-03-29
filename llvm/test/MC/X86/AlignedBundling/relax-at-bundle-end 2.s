# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# Test that an instruction near a bundle end gets properly padded
# after it is relaxed.
.text
foo:
        .bundle_align_mode 5
        .rept 29
        push %rax
        .endr
# CHECK: 1c: push
# CHECK: 1d: nop
# CHECK: 20: jne
        jne 0x100

