# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -dr --no-addresses %t | sed 's/#.*//' | FileCheck %s

# CHECK:      62 f5 7f 08 69 05 00 00 00 00 vcvtbf162ibs    (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7f 08 6b 05 00 00 00 00 vcvtbf162iubs   (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7c 08 69 05 00 00 00 00 vcvtph2ibs      (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7c 08 6b 05 00 00 00 00 vcvtph2iubs     (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7d 08 69 05 00 00 00 00 vcvtps2ibs      (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7d 08 6b 05 00 00 00 00 vcvtps2iubs     (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7f 08 68 05 00 00 00 00 vcvttbf162ibs   (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7f 08 6a 05 00 00 00 00 vcvttbf162iubs  (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7c 08 68 05 00 00 00 00 vcvttph2ibs     (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7c 08 6a 05 00 00 00 00 vcvttph2iubs    (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7d 08 68 05 00 00 00 00 vcvttps2ibs     (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 62 f5 7d 08 6a 05 00 00 00 00 vcvttps2iubs    (%rip), %xmm0
# CHECK-NEXT:          R_X86_64_PC32   foo-0x4

vcvtbf162ibs foo(%rip), %xmm0
vcvtbf162iubs foo(%rip), %xmm0
vcvtph2ibs foo(%rip), %xmm0
vcvtph2iubs foo(%rip), %xmm0
vcvtps2ibs foo(%rip), %xmm0
vcvtps2iubs foo(%rip), %xmm0
vcvttbf162ibs foo(%rip), %xmm0
vcvttbf162iubs foo(%rip), %xmm0
vcvttph2ibs foo(%rip), %xmm0
vcvttph2iubs foo(%rip), %xmm0
vcvttps2ibs foo(%rip), %xmm0
vcvttps2iubs foo(%rip), %xmm0
