# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - \
# RUN:   | llvm-readobj --sections - | FileCheck %s

# Test that bundle-aligned sections with instructions are aligned

  .bundle_align_mode 5
# CHECK: Sections
# Check that the empty .text section has the default alignment
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
