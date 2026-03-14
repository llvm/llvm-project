// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// PR8283

// CHECK: pavgusb %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xbf]
pavgusb	%mm2, %mm1

// CHECK: pavgusb 9(%esi,%edx), %mm3 # encoding: [0x67,0x0f,0x0f,0x5c,0x16,0x09,0xbf]
pavgusb	9(%esi,%edx), %mm3

        
// CHECK: pf2id %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x1d]
pf2id	%mm2, %mm1

// CHECK: pf2id 9(%esi,%edx), %mm3 # encoding: [0x67,0x0f,0x0f,0x5c,0x16,0x09,0x1d]
pf2id	9(%esi,%edx), %mm3

// CHECK: pfacc %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xae]
pfacc	%mm2, %mm1

// CHECK: pfadd %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x9e]
pfadd	%mm2, %mm1

// CHECK: pfcmpeq %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xb0]
pfcmpeq	%mm2, %mm1

// CHECK: pfcmpge %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x90]
pfcmpge	%mm2, %mm1

// CHECK: pfcmpgt %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xa0]
pfcmpgt	%mm2, %mm1

// CHECK: pfmax %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xa4]
pfmax	%mm2, %mm1

// CHECK: pfmin %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x94]
pfmin	%mm2, %mm1

// CHECK: pfmul %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xb4]
pfmul	%mm2, %mm1

// CHECK: pfrcp %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x96]
pfrcp	%mm2, %mm1

// CHECK: pfrcpit1 %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xa6]
pfrcpit1	%mm2, %mm1

// CHECK: pfrcpit2 %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xb6]
pfrcpit2	%mm2, %mm1

// CHECK: pfrsqit1 %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xa7]
pfrsqit1	%mm2, %mm1

// CHECK: pfrsqrt %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x97]
pfrsqrt	%mm2, %mm1

// CHECK: pfsub %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x9a]
pfsub	%mm2, %mm1

// CHECK: pfsubr %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xaa]
pfsubr	%mm2, %mm1

// CHECK: pi2fd %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x0d]
pi2fd	%mm2, %mm1

// CHECK: pmulhrw %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xb7]
pmulhrw	%mm2, %mm1


// CHECK: femms # encoding: [0x0f,0x0e]
femms

// CHECK: prefetch (%rax)   # encoding: [0x0f,0x0d,0x00]
// CHECK: prefetchw (%rax)  # encoding: [0x0f,0x0d,0x08]
// CHECK: prefetchwt1 (%rax)  # encoding: [0x0f,0x0d,0x10]
prefetch (%rax)
prefetchw (%rax)
prefetchwt1 (%rax)


// CHECK: pf2iw %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x1c]
pf2iw %mm2, %mm1

// CHECK: pi2fw %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x0c]
pi2fw %mm2, %mm1

// CHECK: pfnacc %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x8a]
pfnacc %mm2, %mm1

// CHECK: pfpnacc %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0x8e]
pfpnacc %mm2, %mm1

// CHECK: pswapd %mm2, %mm1  # encoding: [0x0f,0x0f,0xca,0xbb]
pswapd %mm2, %mm1
