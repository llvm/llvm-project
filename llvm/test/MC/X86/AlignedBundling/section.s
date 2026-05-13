# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple x86_64 %t/two-sections.s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %t/two-sections.s
# RUN: llvm-mc -filetype=obj -triple x86_64 -mc-relax-all %t/two-sections.s -o - \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %t/two-sections.s
# RUN: llvm-mc -filetype=obj -triple x86_64 %t/section-alignment.s -o - \
# RUN:   | llvm-readobj --sections - | FileCheck %t/section-alignment.s
# RUN: llvm-mc -filetype=obj -triple x86_64 %t/data-section-alignment.s -o - \
# RUN:   | llvm-readobj --sections - | FileCheck %t/data-section-alignment.s

## Test two different executable sections with bundling.
#--- two-sections.s
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

## Test that bundle-aligned sections with instructions are aligned
#--- section-alignment.s
  .bundle_align_mode 5
# CHECK: Sections
## Check that the empty .text section has the default alignment
# CHECK-LABEL: Name: .text
# CHECK-NOT: Name
# CHECK: AddressAlignment: 4

  .section text1, "x"
  imull $17, %ebx, %ebp
# CHECK-LABEL: Name: text1
# CHECK-NOT: Name
# CHECK: AddressAlignment: 32

  .section text2, "x"
  imull $17, %ebx, %ebp
# CHECK-LABEL: Name: text2
# CHECK-NOT: Name
# CHECK: AddressAlignment: 32

## Test that bundle alignment is only applied to executable sections.
#--- data-section-alignment.s
  .bundle_align_mode 5
  .text
  imull $17, %ebx, %ebp
# CHECK-LABEL: Name: .text
# CHECK-NOT: Name
# CHECK: AddressAlignment: 32

  .section .init_array,"aw"
  .p2align 3
  .quad 0
# CHECK-LABEL: Name: .init_array
# CHECK-NOT: Name
# CHECK: AddressAlignment: 8

  .section .data.rel.ro,"aw"
  .p2align 3
  .quad 0
# CHECK-LABEL: Name: .data.rel.ro
# CHECK-NOT: Name
# CHECK: AddressAlignment: 8
