# RUN: llvm-mc -triple=x86_64 -disassemble -show-source-loc %s | FileCheck %s

# CHECK: nop
# CHECK-NEXT: # <SourceLoc: {{.*}}disassemble-show-source-loc.s:5:1>
0x90

# CHECK: xorl %ecx, %ecx
# CHECK-NEXT: # <SourceLoc: {{.*}}disassemble-show-source-loc.s:9:1>
0x31 0xc9
