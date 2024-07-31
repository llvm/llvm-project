# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

# CHECK: {evex}	cmpnbexadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe7,0x54,0x98,0x7b]
         {evex}	cmpnbexadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnbexadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe7,0x7c,0x98,0x7b]
         {evex}	cmpnbexadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnbexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe7,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnbexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnbexadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe7,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnbexadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpbexadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe6,0x54,0x98,0x7b]
         {evex}	cmpbexadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpbexadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe6,0x7c,0x98,0x7b]
         {evex}	cmpbexadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpbexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe6,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpbexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpbexadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe6,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpbexadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpbxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe2,0x54,0x98,0x7b]
         {evex}	cmpbxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpbxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe2,0x7c,0x98,0x7b]
         {evex}	cmpbxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpbxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe2,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpbxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpbxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe2,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpbxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpzxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe4,0x54,0x98,0x7b]
         {evex}	cmpzxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpzxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe4,0x7c,0x98,0x7b]
         {evex}	cmpzxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpzxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe4,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpzxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpzxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe4,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpzxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpnlxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xed,0x54,0x98,0x7b]
         {evex}	cmpnlxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnlxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xed,0x7c,0x98,0x7b]
         {evex}	cmpnlxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnlxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xed,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnlxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnlxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xed,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnlxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpnlexadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xef,0x54,0x98,0x7b]
         {evex}	cmpnlexadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnlexadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xef,0x7c,0x98,0x7b]
         {evex}	cmpnlexadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnlexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xef,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnlexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnlexadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xef,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnlexadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmplexadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xee,0x54,0x98,0x7b]
         {evex}	cmplexadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmplexadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xee,0x7c,0x98,0x7b]
         {evex}	cmplexadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmplexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xee,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmplexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmplexadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xee,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmplexadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmplxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xec,0x54,0x98,0x7b]
         {evex}	cmplxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmplxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xec,0x7c,0x98,0x7b]
         {evex}	cmplxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmplxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xec,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmplxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmplxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xec,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmplxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpnzxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe5,0x54,0x98,0x7b]
         {evex}	cmpnzxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnzxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe5,0x7c,0x98,0x7b]
         {evex}	cmpnzxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnzxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe5,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnzxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnzxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe5,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnzxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpnoxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe1,0x54,0x98,0x7b]
         {evex}	cmpnoxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnoxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe1,0x7c,0x98,0x7b]
         {evex}	cmpnoxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnoxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe1,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnoxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnoxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe1,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnoxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpnpxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xeb,0x54,0x98,0x7b]
         {evex}	cmpnpxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnpxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xeb,0x7c,0x98,0x7b]
         {evex}	cmpnpxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnpxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xeb,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnpxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnpxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xeb,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnpxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpnsxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe9,0x54,0x98,0x7b]
         {evex}	cmpnsxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpnsxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe9,0x7c,0x98,0x7b]
         {evex}	cmpnsxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpnsxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe9,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpnsxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpnsxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe9,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpnsxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpoxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe0,0x54,0x98,0x7b]
         {evex}	cmpoxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpoxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe0,0x7c,0x98,0x7b]
         {evex}	cmpoxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpoxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe0,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpoxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpoxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe0,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpoxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmppxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xea,0x54,0x98,0x7b]
         {evex}	cmppxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmppxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xea,0x7c,0x98,0x7b]
         {evex}	cmppxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmppxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xea,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmppxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmppxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xea,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmppxadd	qword ptr [r28 + 4*r29 + 291], r23, r19

# CHECK: {evex}	cmpsxadd	dword ptr [rax + 4*rbx + 123], edx, ecx
# CHECK: encoding: [0x62,0xf2,0x75,0x08,0xe8,0x54,0x98,0x7b]
         {evex}	cmpsxadd	dword ptr [rax + 4*rbx + 123], edx, ecx

# CHECK: {evex}	cmpsxadd	qword ptr [rax + 4*rbx + 123], r15, r9
# CHECK: encoding: [0x62,0x72,0xb5,0x08,0xe8,0x7c,0x98,0x7b]
         {evex}	cmpsxadd	qword ptr [rax + 4*rbx + 123], r15, r9

# CHECK: cmpsxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe8,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpsxadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

# CHECK: cmpsxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
# CHECK: encoding: [0x62,0x8a,0xe1,0x00,0xe8,0xbc,0xac,0x23,0x01,0x00,0x00]
         cmpsxadd	qword ptr [r28 + 4*r29 + 291], r23, r19
