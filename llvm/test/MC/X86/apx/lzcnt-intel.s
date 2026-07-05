# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: {evex}	lzcnt	ax, dx
# CHECK: encoding: [0x62,0xf4,0x7d,0x08,0xf5,0xc2]
         {evex}	lzcnt	ax, dx
# CHECK: {nf}	lzcnt	ax, dx
# CHECK: encoding: [0x62,0xf4,0x7d,0x0c,0xf5,0xc2]
         {nf}	lzcnt	ax, dx
# CHECK: {evex}	lzcnt	edx, ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x08,0xf5,0xd1]
         {evex}	lzcnt	edx, ecx
# CHECK: {nf}	lzcnt	edx, ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x0c,0xf5,0xd1]
         {nf}	lzcnt	edx, ecx
# CHECK: {evex}	lzcnt	r15, r9
# CHECK: encoding: [0x62,0x54,0xfc,0x08,0xf5,0xf9]
         {evex}	lzcnt	r15, r9
# CHECK: {nf}	lzcnt	r15, r9
# CHECK: encoding: [0x62,0x54,0xfc,0x0c,0xf5,0xf9]
         {nf}	lzcnt	r15, r9
# CHECK: {evex}	lzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7d,0x08,0xf5,0x54,0x80,0x7b]
         {evex}	lzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: {nf}	lzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7d,0x0c,0xf5,0x54,0x80,0x7b]
         {nf}	lzcnt	dx, word ptr [r8 + 4*rax + 123]
# CHECK: {evex}	lzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7c,0x08,0xf5,0x4c,0x80,0x7b]
         {evex}	lzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: {nf}	lzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0xd4,0x7c,0x0c,0xf5,0x4c,0x80,0x7b]
         {nf}	lzcnt	ecx, dword ptr [r8 + 4*rax + 123]
# CHECK: {evex}	lzcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0x54,0xfc,0x08,0xf5,0x4c,0x80,0x7b]
         {evex}	lzcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: {nf}	lzcnt	r9, qword ptr [r8 + 4*rax + 123]
# CHECK: encoding: [0x62,0x54,0xfc,0x0c,0xf5,0x4c,0x80,0x7b]
         {nf}	lzcnt	r9, qword ptr [r8 + 4*rax + 123]
