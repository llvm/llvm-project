# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: {evex}	tzcnt	ax, dx
# CHECK: encoding: [0x62,0xf4,0x7d,0x08,0xf4,0xc2]
         {evex}	tzcnt	ax, dx
# CHECK: {nf}	tzcnt	ax, dx
# CHECK: encoding: [0x62,0xf4,0x7d,0x0c,0xf4,0xc2]
         {nf}	tzcnt	ax, dx
# CHECK: {evex}	tzcnt	edx, ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x08,0xf4,0xd1]
         {evex}	tzcnt	edx, ecx
# CHECK: {nf}	tzcnt	edx, ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x0c,0xf4,0xd1]
         {nf}	tzcnt	edx, ecx
# CHECK: {evex}	tzcnt	r15, r9
# CHECK: encoding: [0x62,0x54,0xfc,0x08,0xf4,0xf9]
         {evex}	tzcnt	r15, r9
# CHECK: {nf}	tzcnt	r15, r9
# CHECK: encoding: [0x62,0x54,0xfc,0x0c,0xf4,0xf9]
         {nf}	tzcnt	r15, r9
# CHECK: {evex}	tzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7d,0x08,0xf4,0x54,0x80,0x7b]
         {evex}	tzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: {nf}	tzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7d,0x0c,0xf4,0x54,0x80,0x7b]
         {nf}	tzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: {evex}	tzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7c,0x08,0xf4,0x4c,0x80,0x7b]
         {evex}	tzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: {nf}	tzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7c,0x0c,0xf4,0x4c,0x80,0x7b]
         {nf}	tzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: {evex}	tzcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0x54,0xfc,0x08,0xf4,0x4c,0x80,0x7b]
         {evex}	tzcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: {nf}	tzcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0x54,0xfc,0x0c,0xf4,0x4c,0x80,0x7b]
         {nf}	tzcnt	r9, qword ptr [r8 + 4*rax + 123]
