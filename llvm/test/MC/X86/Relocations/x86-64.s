# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -dr --no-addresses %t | sed 's/#.*//;/^ *$/d' | FileCheck %s

# CHECK:      8b 05 00 00 00 00                      movl    (%rip), %eax
# CHECK-NEXT:                R_X86_64_PC32   foo-0x4
# CHECK-NEXT: c6 05 00 00 00 00 0c                   movb    $0xc, (%rip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: 66 c7 05 00 00 00 00 0c 00             movw    $0xc, (%rip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x6
# CHECK-NEXT: c7 05 00 00 00 00 0c 00 00 00          movl    $0xc, (%rip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x8
# CHECK-NEXT: 48 c7 05 00 00 00 00 0c 00 00 00       movq    $0xc, (%rip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x8
# CHECK-NEXT: 67 8b 05 00 00 00 00                   movl    (%eip), %eax
# CHECK-NEXT:                R_X86_64_PC32   foo-0x4
# CHECK-NEXT: 67 c6 05 00 00 00 00 0c                movb    $0xc, (%eip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: 67 66 c7 05 00 00 00 00 0c 00          movw    $0xc, (%eip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x6
# CHECK-NEXT: 67 c7 05 00 00 00 00 0c 00 00 00       movl    $0xc, (%eip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x8
# CHECK-NEXT: 67 48 c7 05 00 00 00 00 0c 00 00 00    movq    $0xc, (%eip)
# CHECK-NEXT:                R_X86_64_PC32   foo-0x8
movl	foo(%rip), %eax
movb	$12, foo(%rip)
movw	$12, foo(%rip)
movl	$12, foo(%rip)
movq	$12, foo(%rip)

movl	foo(%eip), %eax
movb	$12, foo(%eip)
movw	$12, foo(%eip)
movl	$12, foo(%eip)
movq	$12, foo(%eip)

# CHECK-NEXT: 48 8b 05 00 00 00 00                   movq    (%rip), %rax
# CHECK-NEXT:                R_X86_64_REX_GOTPCRELX  foo-0x4
# CHECK-NEXT: 4c 8b 35 00 00 00 00                   movq    (%rip), %r14
# CHECK-NEXT:                R_X86_64_REX_GOTPCRELX  foo-0x4
# CHECK-NEXT: 67 48 8b 05 00 00 00 00                movq    (%eip), %rax
# CHECK-NEXT:                R_X86_64_REX_GOTPCRELX  foo-0x4
# CHECK-NEXT: 67 4c 8b 35 00 00 00 00                movq    (%eip), %r14
# CHECK-NEXT:                R_X86_64_REX_GOTPCRELX  foo-0x4
movq foo@GOTPCREL(%rip), %rax
movq foo@GOTPCREL(%rip), %r14
movq foo@GOTPCREL(%eip), %rax
movq foo@GOTPCREL(%eip), %r14

# CHECK-NEXT: 66 0f 38 00 0d 00 00 00 00             pshufb  (%rip), %xmm1
# CHECK-NEXT:                R_X86_64_PC32   foo-0x4
pshufb foo(%rip), %xmm1

## PR15040
# CHECK-NEXT: c4 e3 f9 6a 05 00 00 00 00 10          vfmaddss        (%rip), %xmm1, %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 79 6a 05 00 00 00 00 10          vfmaddss        %xmm1, (%rip), %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 f9 6b 05 00 00 00 00 10          vfmaddsd        (%rip), %xmm1, %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 79 6b 05 00 00 00 00 10          vfmaddsd        %xmm1, (%rip), %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 f9 68 05 00 00 00 00 10          vfmaddps        (%rip), %xmm1, %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 79 68 05 00 00 00 00 10          vfmaddps        %xmm1, (%rip), %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 f9 69 05 00 00 00 00 10          vfmaddpd        (%rip), %xmm1, %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 79 69 05 00 00 00 00 10          vfmaddpd        %xmm1, (%rip), %xmm0, %xmm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 fd 68 05 00 00 00 00 10          vfmaddps        (%rip), %ymm1, %ymm0, %ymm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 7d 68 05 00 00 00 00 10          vfmaddps        %ymm1, (%rip), %ymm0, %ymm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 fd 69 05 00 00 00 00 10          vfmaddpd        (%rip), %ymm1, %ymm0, %ymm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
# CHECK-NEXT: c4 e3 7d 69 05 00 00 00 00 10          vfmaddpd        %ymm1, (%rip), %ymm0, %ymm0
# CHECK-NEXT:                R_X86_64_PC32   foo-0x5
vfmaddss  foo(%rip), %xmm1, %xmm0, %xmm0
vfmaddss   %xmm1, foo(%rip),%xmm0, %xmm0
vfmaddsd  foo(%rip), %xmm1, %xmm0, %xmm0
vfmaddsd   %xmm1, foo(%rip),%xmm0, %xmm0
vfmaddps  foo(%rip), %xmm1, %xmm0, %xmm0
vfmaddps   %xmm1, foo(%rip),%xmm0, %xmm0
vfmaddpd  foo(%rip), %xmm1, %xmm0, %xmm0
vfmaddpd   %xmm1, foo(%rip),%xmm0, %xmm0
vfmaddps  foo(%rip), %ymm1, %ymm0, %ymm0
vfmaddps   %ymm1, foo(%rip),%ymm0, %ymm0
vfmaddpd  foo(%rip), %ymm1, %ymm0, %ymm0
vfmaddpd   %ymm1, foo(%rip),%ymm0, %ymm0

# CHECK-NEXT:<l1>:
# CHECK-NEXT: 90                                     nop
# CHECK-NEXT: c4 e2 79 00 05 f6 ff ff ff             vpshufb -0xa(%rip), %xmm0, %xmm0
# CHECK-NEXT:<l2>:
# CHECK-NEXT: 90                                     nop
# CHECK-NEXT: c4 e3 7d 4a 05 f5 ff ff ff 10          vblendvps       %ymm1, -0xb(%rip), %ymm0, %ymm0
l1:
  nop
  vpshufb l1(%rip), %xmm0, %xmm0
l2:
  nop
  vblendvps %ymm1, l2(%rip), %ymm0, %ymm0
