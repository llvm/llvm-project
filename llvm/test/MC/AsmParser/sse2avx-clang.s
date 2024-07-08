# RUN: clang -march=x86-64 -msse2avx %s -c -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# CHECK:       0:       vmovsd  -0x160(%rbp), %xmm0
		movsd   -352(%rbp), %xmm0               # xmm0 = mem[0],zero
# CHECK-NEXT:       8:       vunpcklpd       %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0]
		unpcklpd        %xmm1, %xmm0                    # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT:       c:       vmovapd %xmm0, -0x170(%rbp)
		movapd  %xmm0, -368(%rbp)
# CHECK-NEXT:      14:       vmovapd -0x170(%rbp), %xmm0
		movapd  -368(%rbp), %xmm0
# CHECK-NEXT:      1c:       vmovsd  -0x178(%rbp), %xmm1
		movsd   -376(%rbp), %xmm1               # xmm1 = mem[0],zero
# CHECK-NEXT:      24:       vmovsd  -0x180(%rbp), %xmm0
		movsd   -384(%rbp), %xmm0               # xmm0 = mem[0],zero
# CHECK-NEXT:      2c:       vunpcklpd       %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0]
		unpcklpd        %xmm1, %xmm0                    # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT:      30:       vaddpd  %xmm1, %xmm0, %xmm0
        addpd   %xmm1, %xmm0		
# CHECK-NEXT:      34:       vmovapd %xmm0, -0x1d0(%rbp)
        movapd  %xmm0, -464(%rbp)
# CHECK-NEXT:      3c:       vmovaps -0x130(%rbp), %xmm1
        movaps  -304(%rbp), %xmm1
# CHECK-NEXT:      44:       vpandn  %xmm1, %xmm0, %xmm0
        pandn   %xmm1, %xmm0
# CHECK-NEXT:      48:       vmovaps %xmm0, -0x1e0(%rbp)
        movaps  %xmm0, -480(%rbp)
# CHECK-NEXT:      50:       vmovss  -0xdc(%rbp), %xmm1
        movss   -220(%rbp), %xmm1               # xmm1 = mem[0],zero,zero,zero
# CHECK-NEXT:      58:       vinsertps       $0x10, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
        insertps        $16, %xmm1, %xmm0               # xmm0 = xmm0[0],xmm1[0]
# CHECK-NEXT:      5e:       vmovaps %xmm0, -0x1f0(%rbp)
        movaps  %xmm0, -496(%rbp)
# CHECK-NEXT:      66:       vmovss  -0x100(%rbp), %xmm0
        movss   -256(%rbp), %xmm0               # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT:      6e:       vmovaps -0xc0(%rbp), %xmm0
        movaps  -192(%rbp), %xmm0
# CHECK-NEXT:      76:       vdivss  %xmm1, %xmm0, %xmm0
        divss   %xmm1, %xmm0
# CHECK-NEXT:      7a:       vmovaps %xmm0, -0xc0(%rbp)
        movaps  %xmm0, -192(%rbp)
# CHECK-NEXT:      82:       vmovd   -0x80(%rbp), %xmm0
        movd    -128(%rbp), %xmm0               # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT:      87:       vpinsrd $0x1, %edx, %xmm0, %xmm0
        pinsrd  $1, %edx, %xmm0
# CHECK-NEXT:      8d:       vmovaps %xmm0, -0x90(%rbp)
        movaps  %xmm0, -144(%rbp)
# CHECK-NEXT:      95:       vmovd   -0xa0(%rbp), %xmm0
        movd    -160(%rbp), %xmm0               # xmm0 = mem[0],zero,zero,zero
# CHECK-NEXT:      9d:       vpblendw        $0xaa, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
        pblendw $170, %xmm1, %xmm0              # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
# CHECK-NEXT:      a3:       vmovdqa %xmm0, -0x240(%rbp)
        movdqa  %xmm0, -576(%rbp)
# CHECK-NEXT:      ab:       vphsubw %xmm1, %xmm0, %xmm0
        phsubw  %xmm1, %xmm0
# CHECK-NEXT:      b0:       vmovdqa %xmm0, -0x250(%rbp)
        movdqa  %xmm0, -592(%rbp)
# CHECK-NEXT:      b8:       vmovaps -0x1f0(%rbp), %xmm0
        movaps  -496(%rbp), %xmm0
# CHECK-NEXT:      c0:       vroundps        $0x8, %xmm0, %xmm0
        roundps $8, %xmm0, %xmm0
# CHECK-NEXT:      c6:       vmovaps %xmm0, -0x260(%rbp)
        movaps  %xmm0, -608(%rbp)
# CHECK-NEXT:      ce:       vmovapd -0x1b0(%rbp), %xmm0
        movapd  -432(%rbp), %xmm0
# CHECK-NEXT:      d6:       vpxor   %xmm1, %xmm0, %xmm0
        pxor    %xmm1, %xmm0
# CHECK-NEXT:      da:       vmovaps %xmm0, -0x280(%rbp)
        movaps  %xmm0, -640(%rbp)
# CHECK-NEXT:      e2:       vmovapd -0x20(%rbp), %xmm0
        movapd  -32(%rbp), %xmm0
# CHECK-NEXT:      e7:       vmovupd %xmm0, (%rax)
        movupd  %xmm0, (%rax)
# CHECK-NEXT:      eb:       vmovsd  -0x290(%rbp), %xmm0
        movsd   -656(%rbp), %xmm0               # xmm0 = mem[0],zero
# CHECK-NEXT:      f3:       extrq   $0x10, $0x8, %xmm0      # xmm0 = xmm0[2],zero,zero,zero,zero,zero,zero,zero,xmm0[u,u,u,u,u,u,u,u]
        extrq   $16, $8, %xmm0
# CHECK-NEXT:      f9:       insertq $0x10, $0x8, %xmm1, %xmm0 # xmm0 = xmm0[0,1],xmm1[0],xmm0[3,4,5,6,7,u,u,u,u,u,u,u,u]
        insertq $16, $8, %xmm1, %xmm0
# CHECK-NEXT:      ff:       pshufw  $0x1, %mm0, %mm2        # mm2 = mm0[1,0,0,0]
        pshufw  $1, %mm0, %mm2
# CHECK-NEXT:     103:       vpblendvb       %xmm2, %xmm2, %xmm1, %xmm1
        pblendvb   %xmm0, %xmm2, %xmm1
# CHECK-NEXT:     109:       vblendvps       %xmm0, %xmm0, %xmm2, %xmm2
        blendvps   %xmm0, %xmm0, %xmm2
# CHECK-NEXT:     10f:       vblendvpd       %xmm0, %xmm0, %xmm2, %xmm2
        blendvpd   %xmm0, %xmm0, %xmm2
# CHECK-NEXT:     115:       vblendvpd       %xmm0, %xmm0, %xmm2, %xmm2
        blendvpd   %xmm0, %xmm2
