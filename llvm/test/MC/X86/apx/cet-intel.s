# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

## wrssd

# CHECK: {evex}	wrssd	dword ptr [rax + 4*rbx + 123], ecx
# CHECK: encoding: [0x62,0xf4,0x7c,0x08,0x66,0x4c,0x98,0x7b]
         {evex}	wrssd	dword ptr [rax + 4*rbx + 123], ecx

# CHECK: wrssd	dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8c,0x78,0x08,0x66,0x94,0xac,0x23,0x01,0x00,0x00]
         wrssd	dword ptr [r28 + 4*r29 + 291], r18d

## wrssq

# CHECK: {evex}	wrssq	qword ptr [rax + 4*rbx + 123], r9
# CHECK: encoding: [0x62,0x74,0xfc,0x08,0x66,0x4c,0x98,0x7b]
         {evex}	wrssq	qword ptr [rax + 4*rbx + 123], r9

# CHECK: wrssq	qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8c,0xf8,0x08,0x66,0x9c,0xac,0x23,0x01,0x00,0x00]
         wrssq	qword ptr [r28 + 4*r29 + 291], r19

## wrussd

# CHECK: {evex}	wrussd	dword ptr [rax + 4*rbx + 123], ecx
# CHECK: encoding: [0x62,0xf4,0x7d,0x08,0x65,0x4c,0x98,0x7b]
         {evex}	wrussd	dword ptr [rax + 4*rbx + 123], ecx

# CHECK: wrussd	dword ptr [r28 + 4*r29 + 291], r18d
# CHECK: encoding: [0x62,0x8c,0x79,0x08,0x65,0x94,0xac,0x23,0x01,0x00,0x00]
         wrussd	dword ptr [r28 + 4*r29 + 291], r18d

## wrussq

# CHECK: {evex}	wrussq	qword ptr [rax + 4*rbx + 123], r9
# CHECK: encoding: [0x62,0x74,0xfd,0x08,0x65,0x4c,0x98,0x7b]
         {evex}	wrussq	qword ptr [rax + 4*rbx + 123], r9

# CHECK: wrussq	qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8c,0xf9,0x08,0x65,0x9c,0xac,0x23,0x01,0x00,0x00]
         wrussq	qword ptr [r28 + 4*r29 + 291], r19
