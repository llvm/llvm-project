# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-OPT %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck --check-prefix=CHECK --check-prefix=CHECK-RELAX %s

# Test simple NOP insertion for single instructions.

  .text
foo:
  # Will be bundle-aligning to 16 byte boundaries
  .bundle_align_mode 4
  pushq   %rbp
  pushq   %r14
  pushq   %rbx

  movl    %edi, %ebx
  callq   bar
  movl    %eax, %r14d

  imull   $17, %ebx, %ebp
# This imull is 3 bytes long and should have started at 0xe, so two bytes
# of nop padding are inserted instead and it starts at 0x10
# CHECK:          nop
# CHECK-NEXT:     10: imull

  movl    %ebx, %edi
  callq   bar
  cmpl    %r14d, %ebp
# CHECK-RELAX:   nopl
  jle     .L_ELSE
# Due to the padding that's inserted before the addl, the jump target
# becomes farther by one byte.
# CHECK-OPT:     jle 0x24
# CHECK-RELAX:   jle 0x2d

  addl    %ebp, %eax
# CHECK-OPT:     nop
# CHECK-OPT-NEXT:20: addl
# CHECK-RELAX:   26: addl

  jmp     .L_RET
.L_ELSE:
  imull   %ebx, %eax
.L_RET:
  ret

# Just sanity checking that data fills don't drive bundling crazy
  .data
  .byte 40
  .byte 98


