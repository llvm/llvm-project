# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu -mcpu=pentiumpro -mc-relax-all %s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

# Test two different executable sections with bundling.

  .bundle_align_mode 3
  .section text1, "x"
# CHECK: section text1
  imull $17, %ebx, %ebp
  imull $17, %ebx, %ebp

  imull $17, %ebx, %ebp
# CHECK:      6: nop
# CHECK-NEXT: 8: imull

  .section text2, "x"
# CHECK: section text2
  imull $17, %ebx, %ebp
  imull $17, %ebx, %ebp

  imull $17, %ebx, %ebp
# CHECK:      6: nop
# CHECK-NEXT: 8: imull


