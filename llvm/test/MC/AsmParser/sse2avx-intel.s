# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -x86-sse2avx %s | FileCheck %s
	.text
# CHECK: vmovsd  -352(%rbp), %xmm0
	movsd	xmm0, qword ptr [rbp - 352]     # xmm0 = mem[0],zero
# CHECK-NEXT: vunpcklpd       %xmm1, %xmm0, %xmm0     # xmm0 = xmm0[0],xmm1[0]
	unpcklpd	xmm0, xmm1                      # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT: vmovapd %xmm0, -368(%rbp)
	movapd	xmmword ptr [rbp - 368], xmm0
# CHECK-NEXT: vmovapd -368(%rbp), %xmm0
	movapd	xmm0, xmmword ptr [rbp - 368]
# CHECK-NEXT: vmovapd %xmm0, -432(%rbp)
	movapd	xmmword ptr [rbp - 432], xmm0
# CHECK-NEXT: movabsq $4613937818241073152, %rax      # imm = 0x4008000000000000
	movabs	rax, 4613937818241073152
# CHECK-NEXT: vunpcklpd       %xmm1, %xmm0, %xmm0     # xmm0 = xmm0[0],xmm1[0]
	unpcklpd	xmm0, xmm1                      # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT: vaddpd  %xmm1, %xmm0, %xmm0
	addpd	xmm0, xmm1
# CHECK-NEXT: vmovapd %xmm0, -464(%rbp)
	movapd	xmmword ptr [rbp - 464], xmm0
# CHECK-NEXT: vmovaps -304(%rbp), %xmm1
	movaps	xmm1, xmmword ptr [rbp - 304]
# CHECK-NEXT: vpandn  %xmm1, %xmm0, %xmm0
	pandn	xmm0, xmm1
# CHECK-NEXT: vmovaps %xmm0, -480(%rbp)
	movaps	xmmword ptr [rbp - 480], xmm0
# CHECK-NEXT: vmovss  -220(%rbp), %xmm1
	movss	xmm1, dword ptr [rbp - 220]     # xmm1 = mem[0],zero,zero,zero
# CHECK-NEXT: vinsertps       $16, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
	insertps	xmm0, xmm1, 16                  # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
# CHECK-NEXT: vmovaps %xmm0, -496(%rbp)
	movaps	xmmword ptr [rbp - 496], xmm0
# CHECK-NEXT: vmovss  -252(%rbp), %xmm1
	movss	xmm1, dword ptr [rbp - 252]     # xmm1 = mem[0],zero,zero,zero
# CHECK-NEXT: vmovaps %xmm1, -192(%rbp)
	movaps	xmmword ptr [rbp - 192], xmm1
# CHECK-NEXT: vdivss  %xmm1, %xmm0, %xmm0
	divss	xmm0, xmm1
# CHECK-NEXT: vmovaps %xmm0, -192(%rbp)
	movaps	xmmword ptr [rbp - 192], xmm0
# CHECK-NEXT: vmovd   -128(%rbp), %xmm0
	movd	xmm0, dword ptr [rbp - 128]     # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT: vpinsrd $1, %edx, %xmm0, %xmm0
	pinsrd	xmm0, edx, 1
# CHECK-NEXT: vmovaps %xmm0, -144(%rbp)
	movaps	xmmword ptr [rbp - 144], xmm0
# CHECK-NEXT: vmovd   -160(%rbp), %xmm0
	movd	xmm0, dword ptr [rbp - 160]     # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT: vpblendw        $170, %xmm1, %xmm0, %xmm0       # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
	pblendw	xmm0, xmm1, 170                 # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
# CHECK-NEXT: vmovdqa %xmm0, -576(%rbp)
	movdqa	xmmword ptr [rbp - 576], xmm0
# CHECK-NEXT: vphsubw %xmm1, %xmm0, %xmm0
	phsubw	xmm0, xmm1
# CHECK-NEXT: vmovdqa %xmm0, -592(%rbp)
	movdqa	xmmword ptr [rbp - 592], xmm0
# CHECK-NEXT: vmovaps -496(%rbp), %xmm0
	movaps	xmm0, xmmword ptr [rbp - 496]
# CHECK-NEXT: vroundps        $8, %xmm0, %xmm0
	roundps	xmm0, xmm0, 8
# CHECK-NEXT: vmovaps %xmm0, -608(%rbp)
	movaps	xmmword ptr [rbp - 608], xmm0
# CHECK-NEXT: vmovapd -432(%rbp), %xmm0
	movapd	xmm0, xmmword ptr [rbp - 432]
# CHECK-NEXT: vpxor   %xmm1, %xmm0, %xmm0
	pxor	xmm0, xmm1
# CHECK-NEXT: vmovaps %xmm0, -640(%rbp)
	movaps	xmmword ptr [rbp - 640], xmm0
# CHECK-NEXT: vmovapd %xmm0, -32(%rbp)
	movapd	xmmword ptr [rbp - 32], xmm0
# CHECK-NEXT: vmovupd %xmm0, (%rax)
	movupd	xmmword ptr [rax], xmm0
# CHECK-NEXT: vmovsd  -656(%rbp), %xmm0
	movsd	xmm0, qword ptr [rbp - 656]     # xmm0 = mem[0],zero
# CHECK-NEXT: extrq   $8, $16, %xmm0                  # xmm0 = xmm0[1,2],zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
	extrq xmm0, 16, 8
# CHECK-NEXT: insertq $8, $16, %xmm1, %xmm0           # xmm0 = xmm0[0],xmm1[0,1],xmm0[3,4,5,6,7,u,u,u,u,u,u,u,u]
	insertq xmm0, xmm1, 16, 8
# CHECK-NEXT: pshufw  $1, %mm0, %mm2                  # mm2 = mm0[1,0,0,0]
	pshufw mm2, mm0, 1
# CHECK-NEXT: vpblendvb       %xmm2, %xmm2, %xmm1, %xmm1
        pblendvb xmm1, xmm2, xmm0
# CHECK-NEXT: vblendvps       %xmm0, %xmm0, %xmm2, %xmm2
        blendvps xmm2, xmm0, xmm0
# CHECK-NEXT: vblendvpd       %xmm0, %xmm0, %xmm2, %xmm2
        blendvpd xmm2, xmm0, xmm0
# CHECK-NEXT: vblendvpd       %xmm0, %xmm0, %xmm2, %xmm2
        blendvpd xmm2, xmm0
