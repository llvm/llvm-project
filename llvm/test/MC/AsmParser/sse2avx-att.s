# RUN: llvm-mc -triple x86_64 -x86-sse2avx %s | FileCheck %s
# RUN: llvm-mc -triple=x86_64 -output-asm-variant=1 %s | llvm-mc -triple=x86_64 -x86-asm-syntax=intel -x86-sse2avx
	.text
# CHECK: vmovsd  -352(%rbp), %xmm0
	movsd	-352(%rbp), %xmm0               # xmm0 = mem[0],zero
# CHECK-NEXT: vunpcklpd       %xmm1, %xmm0, %xmm0     # xmm0 = xmm0[0],xmm1[0]
	unpcklpd	%xmm1, %xmm0                    # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT: vmovapd %xmm0, -368(%rbp)
	movapd	%xmm0, -368(%rbp)
# CHECK-NEXT: vmovapd -368(%rbp), %xmm0
	movapd	-368(%rbp), %xmm0
# CHECK-NEXT: vmovsd  -376(%rbp), %xmm1
	movsd	-376(%rbp), %xmm1               # xmm1 = mem[0],zero
# CHECK-NEXT: vmovsd  -384(%rbp), %xmm0
	movsd	-384(%rbp), %xmm0               # xmm0 = mem[0],zero
# CHECK-NEXT: vunpcklpd       %xmm1, %xmm0, %xmm0     # xmm0 = xmm0[0],xmm1[0]
	unpcklpd	%xmm1, %xmm0                    # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT: vaddpd  %xmm1, %xmm0, %xmm0
	addpd	%xmm1, %xmm0
# CHECK-NEXT: vmovapd %xmm0, -464(%rbp)
	movapd	%xmm0, -464(%rbp)
# CHECK-NEXT: vmovaps -304(%rbp), %xmm1
	movaps	-304(%rbp), %xmm1
# CHECK-NEXT: vpandn  %xmm1, %xmm0, %xmm0
	pandn	%xmm1, %xmm0
# CHECK-NEXT: vmovaps %xmm0, -480(%rbp)
	movaps	%xmm0, -480(%rbp)
# CHECK-NEXT: vmovss  -220(%rbp), %xmm1
	movss	-220(%rbp), %xmm1               # xmm1 = mem[0],zero,zero,zero
# CHECK-NEXT: vinsertps       $16, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	$16, %xmm1, %xmm0               # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
# CHECK-NEXT: vmovaps %xmm0, -496(%rbp)
	movaps	%xmm0, -496(%rbp)
# CHECK-NEXT: vmovss  -256(%rbp), %xmm0
	movss	-256(%rbp), %xmm0               # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT: vmovaps -192(%rbp), %xmm0
	movaps	-192(%rbp), %xmm0
# CHECK-NEXT: vdivss  %xmm1, %xmm0, %xmm0
	divss	%xmm1, %xmm0
# CHECK-NEXT: vmovaps %xmm0, -192(%rbp)
	movaps	%xmm0, -192(%rbp)
# CHECK-NEXT: vmovd   -128(%rbp), %xmm0
	movd	-128(%rbp), %xmm0               # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT: vpinsrd $1, %edx, %xmm0, %xmm0
	pinsrd	$1, %edx, %xmm0
# CHECK-NEXT: vmovaps %xmm0, -144(%rbp)
	movaps	%xmm0, -144(%rbp)
# CHECK-NEXT: vmovd   -160(%rbp), %xmm0
	movd	-160(%rbp), %xmm0               # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT: vpblendw        $170, %xmm1, %xmm0, %xmm0       # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
	pblendw	$170, %xmm1, %xmm0              # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
# CHECK-NEXT: vmovdqa %xmm0, -576(%rbp)
	movdqa	%xmm0, -576(%rbp)
# CHECK-NEXT: vphsubw %xmm1, %xmm0, %xmm0
	phsubw	%xmm1, %xmm0
# CHECK-NEXT: vmovdqa %xmm0, -592(%rbp)
	movdqa	%xmm0, -592(%rbp)
# CHECK-NEXT: vmovaps -496(%rbp), %xmm0
	movaps	-496(%rbp), %xmm0
# CHECK-NEXT: vroundps        $8, %xmm0, %xmm0
	roundps	$8, %xmm0, %xmm0
# CHECK-NEXT: vmovaps %xmm0, -608(%rbp)
	movaps	%xmm0, -608(%rbp)
# CHECK-NEXT: vmovapd -432(%rbp), %xmm0
	movapd	-432(%rbp), %xmm0
# CHECK-NEXT: vpxor   %xmm1, %xmm0, %xmm0
	pxor	%xmm1, %xmm0
# CHECK-NEXT: vmovaps %xmm0, -640(%rbp)
	movaps	%xmm0, -640(%rbp)
# CHECK-NEXT: vmovapd -32(%rbp), %xmm0
	movapd	-32(%rbp), %xmm0
# CHECK-NEXT: vmovupd %xmm0, (%rax)
	movupd	%xmm0, (%rax)
# CHECK-NEXT: vmovsd  -656(%rbp), %xmm0
	movsd	-656(%rbp), %xmm0               # xmm0 = mem[0],zero
# CHECK-NEXT: extrq   $16, $8, %xmm0                  # xmm0 = xmm0[2],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
        extrq   $16, $8, %xmm0
# CHECK-NEXT: insertq $16, $8, %xmm1, %xmm0           # xmm0 = xmm0[0,1],xmm1[0],xmm0[3,4,5,6,7,u,u,u,u,u,u,u,u]
	insertq $16, $8, %xmm1, %xmm0
# CHECK-NEXT: pshufw  $1, %mm0, %mm2                  # mm2 = mm0[1,0,0,0]
	pshufw  $1, %mm0, %mm2
# CHECK-NEXT: vpblendvb       %xmm2, %xmm2, %xmm1, %xmm1
	pblendvb   %xmm0, %xmm2, %xmm1
# CHECK-NEXT: vblendvps       %xmm0, %xmm0, %xmm2, %xmm2
	blendvps   %xmm0, %xmm0, %xmm2
# CHECK-NEXT: vblendvpd       %xmm0, %xmm0, %xmm2, %xmm2
	blendvpd   %xmm0, %xmm0, %xmm2
# CHECK-NEXT: vblendvpd       %xmm0, %xmm0, %xmm2, %xmm2
	blendvpd   %xmm0, %xmm2
