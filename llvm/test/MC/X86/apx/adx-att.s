# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-16: error:
# ERROR-NOT: error:
# CHECK: adcxl	%r16d, %r17d
# CHECK: encoding: [0x62,0xec,0x7d,0x08,0x66,0xc8]
         adcxl	%r16d, %r17d
# CHECK: adcxl	%r16d, %r17d, %r18d
# CHECK: encoding: [0x62,0xec,0x6d,0x10,0x66,0xc8]
         adcxl	%r16d, %r17d, %r18d
# CHECK: adcxq	%r16, %r17
# CHECK: encoding: [0x62,0xec,0xfd,0x08,0x66,0xc8]
         adcxq	%r16, %r17
# CHECK: adcxq	%r16, %r17, %r18
# CHECK: encoding: [0x62,0xec,0xed,0x10,0x66,0xc8]
         adcxq	%r16, %r17, %r18
# CHECK: adcxl	(%r16), %r17d
# CHECK: encoding: [0x62,0xec,0x7d,0x08,0x66,0x08]
         adcxl	(%r16), %r17d
# CHECK: adcxl	(%r16), %r17d, %r18d
# CHECK: encoding: [0x62,0xec,0x6d,0x10,0x66,0x08]
         adcxl	(%r16), %r17d, %r18d
# CHECK: adcxq	(%r16), %r17
# CHECK: encoding: [0x62,0xec,0xfd,0x08,0x66,0x08]
         adcxq	(%r16), %r17
# CHECK: adcxq	(%r16), %r17, %r18
# CHECK: encoding: [0x62,0xec,0xed,0x10,0x66,0x08]
         adcxq	(%r16), %r17, %r18
# CHECK: adoxl	%r16d, %r17d
# CHECK: encoding: [0x62,0xec,0x7e,0x08,0x66,0xc8]
         adoxl	%r16d, %r17d
# CHECK: adoxl	%r16d, %r17d, %r18d
# CHECK: encoding: [0x62,0xec,0x6e,0x10,0x66,0xc8]
         adoxl	%r16d, %r17d, %r18d
# CHECK: adoxq	%r16, %r17
# CHECK: encoding: [0x62,0xec,0xfe,0x08,0x66,0xc8]
         adoxq	%r16, %r17
# CHECK: adoxq	%r16, %r17, %r18
# CHECK: encoding: [0x62,0xec,0xee,0x10,0x66,0xc8]
         adoxq	%r16, %r17, %r18
# CHECK: adoxl	(%r16), %r17d
# CHECK: encoding: [0x62,0xec,0x7e,0x08,0x66,0x08]
         adoxl	(%r16), %r17d
# CHECK: adoxl	(%r16), %r17d, %r18d
# CHECK: encoding: [0x62,0xec,0x6e,0x10,0x66,0x08]
         adoxl	(%r16), %r17d, %r18d
# CHECK: adoxq	(%r16), %r17
# CHECK: encoding: [0x62,0xec,0xfe,0x08,0x66,0x08]
         adoxq	(%r16), %r17
# CHECK: adoxq	(%r16), %r17, %r18
# CHECK: encoding: [0x62,0xec,0xee,0x10,0x66,0x08]
         adoxq	(%r16), %r17, %r18
