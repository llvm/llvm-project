# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple x86_64 -show-encoding -output-asm-variant=1 %s | FileCheck %s --check-prefix=INTEL

## The NDD shift/rotate by 1 forms take an explicit $1, as in GNU binutils, so
## an 8-bit source of %cl doesn't clash with the legacy shift by %cl.

# CHECK: shlb	$1, %cl, %dl
# INTEL: shl	dl, cl, 1
# CHECK: encoding: [0x62,0xf4,0x6c,0x18,0xd0,0xe1]
         shlb	$1, %cl, %dl
# CHECK: {nf}	shlb	$1, %cl, %dl
# INTEL: {nf}	shl	dl, cl, 1
# CHECK: encoding: [0x62,0xf4,0x6c,0x1c,0xd0,0xe1]
         {nf}	shlb	$1, %cl, %dl
# CHECK: shlb	%cl, %dl
# INTEL: shl	dl, cl
# CHECK: encoding: [0xd2,0xe2]
         shlb	%cl, %dl
# CHECK: {nf}	shlb	%cl, %dl
# INTEL: {nf}	shl	dl, cl
# CHECK: encoding: [0x62,0xf4,0x7c,0x0c,0xd2,0xe2]
         {nf}	shlb	%cl, %dl
# CHECK: rclb	$1, %cl, %dl
# INTEL: rcl	dl, cl, 1
# CHECK: encoding: [0x62,0xf4,0x6c,0x18,0xd0,0xd1]
         rclb	$1, %cl, %dl
# CHECK: rcrb	$1, %cl, %dl
# INTEL: rcr	dl, cl, 1
# CHECK: encoding: [0x62,0xf4,0x6c,0x18,0xd0,0xd9]
         rcrb	$1, %cl, %dl

## The two-operand spelling is still accepted where it isn't ambiguous.

# CHECK: {nf}	sarw	$1, %cx, %dx
# INTEL: {nf}	sar	dx, cx, 1
# CHECK: encoding: [0x62,0xf4,0x6d,0x1c,0xd1,0xf9]
         {nf}	sarw	%cx, %dx
# CHECK: rorq	$1, (%rax), %rdx
# INTEL: ror	rdx, qword ptr [rax], 1
# CHECK: encoding: [0x62,0xf4,0xec,0x18,0xd1,0x08]
         rorq	(%rax), %rdx
