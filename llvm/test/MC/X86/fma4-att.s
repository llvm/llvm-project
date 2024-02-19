// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// vfmadd
// CHECK: vfmaddss  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6a,0x01,0x10]
          vfmaddss  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddss   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6a,0x01,0x10]
          vfmaddss   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddss   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6a,0xc2,0x10]
          vfmaddss   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6b,0x01,0x10]
          vfmaddsd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6b,0x01,0x10]
          vfmaddsd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddsd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6b,0xc2,0x10]
          vfmaddsd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsd   %xmm10, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xc3,0xf9,0x6b,0xc2,0x10]
          vfmaddsd   %xmm10, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddps  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x68,0x01,0x10]
          vfmaddps  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddps   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x68,0x01,0x10]
          vfmaddps   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddps   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x68,0xc2,0x10]
          vfmaddps   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddpd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x69,0x01,0x10]
          vfmaddpd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddpd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x69,0x01,0x10]
          vfmaddpd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddpd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x69,0xc2,0x10]
          vfmaddpd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddps  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x68,0x01,0x10]
          vfmaddps  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmaddps   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x68,0x01,0x10]
          vfmaddps   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmaddps   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x68,0xc2,0x10]
          vfmaddps   %ymm2, %ymm1, %ymm0, %ymm0

// CHECK: vfmaddpd  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x69,0x01,0x10]
          vfmaddpd  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmaddpd   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x69,0x01,0x10]
          vfmaddpd   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmaddpd   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x69,0xc2,0x10]
          vfmaddpd   %ymm2, %ymm1, %ymm0, %ymm0

// PR15040
// CHECK: vfmaddss  foo(%rip), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6a,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddss  foo(%rip), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddss   %xmm1, foo(%rip), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6a,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddss   %xmm1, foo(%rip),%xmm0, %xmm0

// CHECK: vfmaddsd  foo(%rip), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6b,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddsd  foo(%rip), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsd   %xmm1, foo(%rip), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6b,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddsd   %xmm1, foo(%rip),%xmm0, %xmm0

// CHECK: vfmaddps  foo(%rip), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x68,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddps  foo(%rip), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddps   %xmm1, foo(%rip), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x68,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddps   %xmm1, foo(%rip),%xmm0, %xmm0

// CHECK: vfmaddpd  foo(%rip), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x69,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddpd  foo(%rip), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddpd   %xmm1, foo(%rip), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x69,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddpd   %xmm1, foo(%rip),%xmm0, %xmm0

// CHECK: vfmaddps  foo(%rip), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x68,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddps  foo(%rip), %ymm1, %ymm0, %ymm0

// CHECK: vfmaddps   %ymm1, foo(%rip), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x68,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddps   %ymm1, foo(%rip),%ymm0, %ymm0

// CHECK: vfmaddpd  foo(%rip), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x69,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddpd  foo(%rip), %ymm1, %ymm0, %ymm0

// CHECK: vfmaddpd   %ymm1, foo(%rip), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x69,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: foo-5, kind: reloc_riprel_4byte
          vfmaddpd   %ymm1, foo(%rip),%ymm0, %ymm0

// vfmsub
// CHECK: vfmsubss  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6e,0x01,0x10]
          vfmsubss  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmsubss   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6e,0x01,0x10]
          vfmsubss   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmsubss   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6e,0xc2,0x10]
          vfmsubss   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmsubsd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6f,0x01,0x10]
          vfmsubsd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmsubsd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6f,0x01,0x10]
          vfmsubsd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmsubsd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6f,0xc2,0x10]
          vfmsubsd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmsubps  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6c,0x01,0x10]
          vfmsubps  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmsubps   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6c,0x01,0x10]
          vfmsubps   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmsubps   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6c,0xc2,0x10]
          vfmsubps   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmsubpd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6d,0x01,0x10]
          vfmsubpd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmsubpd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x6d,0x01,0x10]
          vfmsubpd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmsubpd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x6d,0xc2,0x10]
          vfmsubpd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmsubps  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x6c,0x01,0x10]
          vfmsubps  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmsubps   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x6c,0x01,0x10]
          vfmsubps   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmsubps   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x6c,0xc2,0x10]
          vfmsubps   %ymm2, %ymm1, %ymm0, %ymm0

// CHECK: vfmsubpd  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x6d,0x01,0x10]
          vfmsubpd  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmsubpd   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x6d,0x01,0x10]
          vfmsubpd   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmsubpd   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x6d,0xc2,0x10]
          vfmsubpd   %ymm2, %ymm1, %ymm0, %ymm0

// vfnmadd
// CHECK: vfnmaddss  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7a,0x01,0x10]
          vfnmaddss  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddss   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x7a,0x01,0x10]
          vfnmaddss   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmaddss   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7a,0xc2,0x10]
          vfnmaddss   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddsd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7b,0x01,0x10]
          vfnmaddsd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddsd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x7b,0x01,0x10]
          vfnmaddsd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmaddsd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7b,0xc2,0x10]
          vfnmaddsd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddps  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x78,0x01,0x10]
          vfnmaddps  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddps   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x78,0x01,0x10]
          vfnmaddps   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmaddps   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x78,0xc2,0x10]
          vfnmaddps   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddpd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x79,0x01,0x10]
          vfnmaddpd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddpd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x79,0x01,0x10]
          vfnmaddpd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmaddpd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x79,0xc2,0x10]
          vfnmaddpd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmaddps  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x78,0x01,0x10]
          vfnmaddps  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfnmaddps   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x78,0x01,0x10]
          vfnmaddps   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfnmaddps   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x78,0xc2,0x10]
          vfnmaddps   %ymm2, %ymm1, %ymm0, %ymm0

// CHECK: vfnmaddpd  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x79,0x01,0x10]
          vfnmaddpd  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfnmaddpd   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x79,0x01,0x10]
          vfnmaddpd   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfnmaddpd   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x79,0xc2,0x10]
          vfnmaddpd   %ymm2, %ymm1, %ymm0, %ymm0

// vfnmsub
// CHECK: vfnmsubss  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7e,0x01,0x10]
          vfnmsubss  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubss   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x7e,0x01,0x10]
          vfnmsubss   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmsubss   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7e,0xc2,0x10]
          vfnmsubss   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubsd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7f,0x01,0x10]
          vfnmsubsd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubsd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x7f,0x01,0x10]
          vfnmsubsd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmsubsd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7f,0xc2,0x10]
          vfnmsubsd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubps  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7c,0x01,0x10]
          vfnmsubps  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubps   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x7c,0x01,0x10]
          vfnmsubps   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmsubps   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7c,0xc2,0x10]
          vfnmsubps   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubpd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7d,0x01,0x10]
          vfnmsubpd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubpd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x7d,0x01,0x10]
          vfnmsubpd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfnmsubpd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x7d,0xc2,0x10]
          vfnmsubpd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfnmsubps  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x7c,0x01,0x10]
          vfnmsubps  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfnmsubps   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x7c,0x01,0x10]
          vfnmsubps   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfnmsubps   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x7c,0xc2,0x10]
          vfnmsubps   %ymm2, %ymm1, %ymm0, %ymm0

// CHECK: vfnmsubpd  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x7d,0x01,0x10]
          vfnmsubpd  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfnmsubpd   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x7d,0x01,0x10]
          vfnmsubpd   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfnmsubpd   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x7d,0xc2,0x10]
          vfnmsubpd   %ymm2, %ymm1, %ymm0, %ymm0

// vfmaddsub
// CHECK: vfmaddsubps  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5c,0x01,0x10]
          vfmaddsubps  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsubps   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x5c,0x01,0x10]
          vfmaddsubps   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddsubps   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5c,0xc2,0x10]
          vfmaddsubps   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsubpd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5d,0x01,0x10]
          vfmaddsubpd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsubpd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x5d,0x01,0x10]
          vfmaddsubpd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmaddsubpd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5d,0xc2,0x10]
          vfmaddsubpd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmaddsubps  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5c,0x01,0x10]
          vfmaddsubps  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmaddsubps   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x5c,0x01,0x10]
          vfmaddsubps   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmaddsubps   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5c,0xc2,0x10]
          vfmaddsubps   %ymm2, %ymm1, %ymm0, %ymm0

// CHECK: vfmaddsubpd  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5d,0x01,0x10]
          vfmaddsubpd  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmaddsubpd   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x5d,0x01,0x10]
          vfmaddsubpd   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmaddsubpd   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5d,0xc2,0x10]
          vfmaddsubpd   %ymm2, %ymm1, %ymm0, %ymm0

// vfmsubadd
// CHECK: vfmsubaddps  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5e,0x01,0x10]
          vfmsubaddps  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmsubaddps   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x5e,0x01,0x10]
          vfmsubaddps   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmsubaddps   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5e,0xc2,0x10]
          vfmsubaddps   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmsubaddpd  (%rcx), %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5f,0x01,0x10]
          vfmsubaddpd  (%rcx), %xmm1, %xmm0, %xmm0

// CHECK: vfmsubaddpd   %xmm1, (%rcx), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0x79,0x5f,0x01,0x10]
          vfmsubaddpd   %xmm1, (%rcx),%xmm0, %xmm0

// CHECK: vfmsubaddpd   %xmm2, %xmm1, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe3,0xf9,0x5f,0xc2,0x10]
          vfmsubaddpd   %xmm2, %xmm1, %xmm0, %xmm0

// CHECK: vfmsubaddps  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5e,0x01,0x10]
          vfmsubaddps  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmsubaddps   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x5e,0x01,0x10]
          vfmsubaddps   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmsubaddps   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5e,0xc2,0x10]
          vfmsubaddps   %ymm2, %ymm1, %ymm0, %ymm0

// CHECK: vfmsubaddpd  (%rcx), %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5f,0x01,0x10]
          vfmsubaddpd  (%rcx), %ymm1, %ymm0, %ymm0

// CHECK: vfmsubaddpd   %ymm1, (%rcx), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x5f,0x01,0x10]
          vfmsubaddpd   %ymm1, (%rcx),%ymm0, %ymm0

// CHECK: vfmsubaddpd   %ymm2, %ymm1, %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0xfd,0x5f,0xc2,0x10]
          vfmsubaddpd   %ymm2, %ymm1, %ymm0, %ymm0
