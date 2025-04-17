# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: {evex}	popcnt	ax, dx
# CHECK: encoding: [0x62,0xf4,0x7d,0x08,0x88,0xc2]
         {evex}	popcnt	ax, dx
# CHECK: {nf}	popcnt	ax, dx
# CHECK: encoding: [0x62,0xf4,0x7d,0x0c,0x88,0xc2]
         {nf}	popcnt	ax, dx
# CHECK: {evex}	popcnt	edx, ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x08,0x88,0xd1]
         {evex}	popcnt	edx, ecx
# CHECK: {nf}	popcnt	edx, ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x0c,0x88,0xd1]
         {nf}	popcnt	edx, ecx
# CHECK: {evex}	popcnt	r15, r9
# CHECK: encoding: [0x62,0x54,0xfc,0x08,0x88,0xf9]
         {evex}	popcnt	r15, r9
# CHECK: {nf}	popcnt	r15, r9
# CHECK: encoding: [0x62,0x54,0xfc,0x0c,0x88,0xf9]
         {nf}	popcnt	r15, r9
# CHECK: {evex}	popcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7d,0x08,0x88,0x54,0x80,0x7b]
         {evex}	popcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: {nf}	popcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7d,0x0c,0x88,0x54,0x80,0x7b]
         {nf}	popcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: {evex}	popcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7c,0x08,0x88,0x4c,0x80,0x7b]
         {evex}	popcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: {nf}	popcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7c,0x0c,0x88,0x4c,0x80,0x7b]
         {nf}	popcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: {evex}	popcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0x54,0xfc,0x08,0x88,0x4c,0x80,0x7b]
         {evex}	popcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: {nf}	popcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0x54,0xfc,0x0c,0x88,0x4c,0x80,0x7b]
         {nf}	popcnt	r9, qword ptr [r8 + 4*rax + 123]
