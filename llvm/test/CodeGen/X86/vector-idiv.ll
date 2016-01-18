; RUN: llc -march=x86-64 -mcpu=core2 -mattr=+sse4.1 < %s | FileCheck %s --check-prefix=SSE41
; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s --check-prefix=SSE
; RUN: llc -march=x86-64 -mcpu=core-avx2 < %s | FileCheck %s --check-prefix=AVX

target triple = "x86_64-unknown-unknown"

define <4 x i32> @test1(<4 x i32> %a) #0 {
; SSE41-LABEL: test1:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm1 = [613566757,613566757,613566757,613566757]
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuludq %xmm2, %xmm3
; SSE41-NEXT:    pmuludq %xmm0, %xmm1
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1],xmm3[2,3],xmm1[4,5],xmm3[6,7]
; SSE41-NEXT:    psubd %xmm1, %xmm0
; SSE41-NEXT:    psrld $1, %xmm0
; SSE41-NEXT:    paddd %xmm1, %xmm0
; SSE41-NEXT:    psrld $2, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: test1:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [613566757,613566757,613566757,613566757]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pmuludq %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm1, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm3[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE-NEXT:    psubd %xmm2, %xmm0
; SSE-NEXT:    psrld $1, %xmm0
; SSE-NEXT:    paddd %xmm2, %xmm0
; SSE-NEXT:    psrld $2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: test1:
; AVX:       # BB#0:
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX-NEXT:    vpshufd {{.*#+}} xmm2 = xmm1[1,1,3,3]
; AVX-NEXT:    vpshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; AVX-NEXT:    vpmuludq %xmm2, %xmm3, %xmm2
; AVX-NEXT:    vpmuludq %xmm1, %xmm0, %xmm1
; AVX-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; AVX-NEXT:    vpblendd {{.*#+}} xmm1 = xmm1[0],xmm2[1],xmm1[2],xmm2[3]
; AVX-NEXT:    vpsubd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrld $1, %xmm0, %xmm0
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrld $2, %xmm0, %xmm0
; AVX-NEXT:    retq
  %div = udiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div
}

define <8 x i32> @test2(<8 x i32> %a) #0 {
; SSE41-LABEL: test2:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [613566757,613566757,613566757,613566757]
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm2[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuludq %xmm3, %xmm4
; SSE41-NEXT:    movdqa %xmm0, %xmm5
; SSE41-NEXT:    pmuludq %xmm2, %xmm5
; SSE41-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm5 = xmm5[0,1],xmm4[2,3],xmm5[4,5],xmm4[6,7]
; SSE41-NEXT:    psubd %xmm5, %xmm0
; SSE41-NEXT:    psrld $1, %xmm0
; SSE41-NEXT:    paddd %xmm5, %xmm0
; SSE41-NEXT:    psrld $2, %xmm0
; SSE41-NEXT:    pshufd {{.*#+}} xmm4 = xmm1[1,1,3,3]
; SSE41-NEXT:    pmuludq %xmm3, %xmm4
; SSE41-NEXT:    pmuludq %xmm1, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1],xmm4[2,3],xmm2[4,5],xmm4[6,7]
; SSE41-NEXT:    psubd %xmm2, %xmm1
; SSE41-NEXT:    psrld $1, %xmm1
; SSE41-NEXT:    paddd %xmm2, %xmm1
; SSE41-NEXT:    psrld $2, %xmm1
; SSE41-NEXT:    retq
;
; SSE-LABEL: test2:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [613566757,613566757,613566757,613566757]
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    pmuludq %xmm2, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm2[1,1,3,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm0[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm5
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm3 = xmm3[0],xmm5[0],xmm3[1],xmm5[1]
; SSE-NEXT:    psubd %xmm3, %xmm0
; SSE-NEXT:    psrld $1, %xmm0
; SSE-NEXT:    paddd %xmm3, %xmm0
; SSE-NEXT:    psrld $2, %xmm0
; SSE-NEXT:    pmuludq %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    psubd %xmm2, %xmm1
; SSE-NEXT:    psrld $1, %xmm1
; SSE-NEXT:    paddd %xmm2, %xmm1
; SSE-NEXT:    psrld $2, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test2:
; AVX:       # BB#0:
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm2 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpshufd {{.*#+}} ymm3 = ymm0[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpmuludq %ymm2, %ymm3, %ymm2
; AVX-NEXT:    vpmuludq %ymm1, %ymm0, %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpblendd {{.*#+}} ymm1 = ymm1[0],ymm2[1],ymm1[2],ymm2[3],ymm1[4],ymm2[5],ymm1[6],ymm2[7]
; AVX-NEXT:    vpsubd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpsrld $1, %ymm0, %ymm0
; AVX-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpsrld $2, %ymm0, %ymm0
; AVX-NEXT:    retq
  %div = udiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div
}

define <8 x i16> @test3(<8 x i16> %a) #0 {
; SSE41-LABEL: test3:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm1 = [9363,9363,9363,9363,9363,9363,9363,9363]
; SSE41-NEXT:    pmulhuw %xmm0, %xmm1
; SSE41-NEXT:    psubw %xmm1, %xmm0
; SSE41-NEXT:    psrlw $1, %xmm0
; SSE41-NEXT:    paddw %xmm1, %xmm0
; SSE41-NEXT:    psrlw $2, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: test3:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [9363,9363,9363,9363,9363,9363,9363,9363]
; SSE-NEXT:    pmulhuw %xmm0, %xmm1
; SSE-NEXT:    psubw %xmm1, %xmm0
; SSE-NEXT:    psrlw $1, %xmm0
; SSE-NEXT:    paddw %xmm1, %xmm0
; SSE-NEXT:    psrlw $2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: test3:
; AVX:       # BB#0:
; AVX-NEXT:    vpmulhuw {{.*}}(%rip), %xmm0, %xmm1
; AVX-NEXT:    vpsubw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $1, %xmm0, %xmm0
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $2, %xmm0, %xmm0
; AVX-NEXT:    retq
  %div = udiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div
}

define <16 x i16> @test4(<16 x i16> %a) #0 {
; SSE41-LABEL: test4:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [9363,9363,9363,9363,9363,9363,9363,9363]
; SSE41-NEXT:    movdqa %xmm0, %xmm3
; SSE41-NEXT:    pmulhuw %xmm2, %xmm3
; SSE41-NEXT:    psubw %xmm3, %xmm0
; SSE41-NEXT:    psrlw $1, %xmm0
; SSE41-NEXT:    paddw %xmm3, %xmm0
; SSE41-NEXT:    psrlw $2, %xmm0
; SSE41-NEXT:    pmulhuw %xmm1, %xmm2
; SSE41-NEXT:    psubw %xmm2, %xmm1
; SSE41-NEXT:    psrlw $1, %xmm1
; SSE41-NEXT:    paddw %xmm2, %xmm1
; SSE41-NEXT:    psrlw $2, %xmm1
; SSE41-NEXT:    retq
;
; SSE-LABEL: test4:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [9363,9363,9363,9363,9363,9363,9363,9363]
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    pmulhuw %xmm2, %xmm3
; SSE-NEXT:    psubw %xmm3, %xmm0
; SSE-NEXT:    psrlw $1, %xmm0
; SSE-NEXT:    paddw %xmm3, %xmm0
; SSE-NEXT:    psrlw $2, %xmm0
; SSE-NEXT:    pmulhuw %xmm1, %xmm2
; SSE-NEXT:    psubw %xmm2, %xmm1
; SSE-NEXT:    psrlw $1, %xmm1
; SSE-NEXT:    paddw %xmm2, %xmm1
; SSE-NEXT:    psrlw $2, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test4:
; AVX:       # BB#0:
; AVX-NEXT:    vpmulhuw {{.*}}(%rip), %ymm0, %ymm1
; AVX-NEXT:    vpsubw %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpsrlw $1, %ymm0, %ymm0
; AVX-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpsrlw $2, %ymm0, %ymm0
; AVX-NEXT:    retq
  %div = udiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div
}

define <8 x i16> @test5(<8 x i16> %a) #0 {
; SSE41-LABEL: test5:
; SSE41:       # BB#0:
; SSE41-NEXT:    pmulhw {{.*}}(%rip), %xmm0
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    psrlw $15, %xmm1
; SSE41-NEXT:    psraw $1, %xmm0
; SSE41-NEXT:    paddw %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: test5:
; SSE:       # BB#0:
; SSE-NEXT:    pmulhw {{.*}}(%rip), %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlw $15, %xmm1
; SSE-NEXT:    psraw $1, %xmm0
; SSE-NEXT:    paddw %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: test5:
; AVX:       # BB#0:
; AVX-NEXT:    vpmulhw {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $15, %xmm0, %xmm1
; AVX-NEXT:    vpsraw $1, %xmm0, %xmm0
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %div = sdiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div
}

define <16 x i16> @test6(<16 x i16> %a) #0 {
; SSE41-LABEL: test6:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [18725,18725,18725,18725,18725,18725,18725,18725]
; SSE41-NEXT:    pmulhw %xmm2, %xmm0
; SSE41-NEXT:    movdqa %xmm0, %xmm3
; SSE41-NEXT:    psrlw $15, %xmm3
; SSE41-NEXT:    psraw $1, %xmm0
; SSE41-NEXT:    paddw %xmm3, %xmm0
; SSE41-NEXT:    pmulhw %xmm2, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm2
; SSE41-NEXT:    psrlw $15, %xmm2
; SSE41-NEXT:    psraw $1, %xmm1
; SSE41-NEXT:    paddw %xmm2, %xmm1
; SSE41-NEXT:    retq
;
; SSE-LABEL: test6:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [18725,18725,18725,18725,18725,18725,18725,18725]
; SSE-NEXT:    pmulhw %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    psrlw $15, %xmm3
; SSE-NEXT:    psraw $1, %xmm0
; SSE-NEXT:    paddw %xmm3, %xmm0
; SSE-NEXT:    pmulhw %xmm2, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm2
; SSE-NEXT:    psrlw $15, %xmm2
; SSE-NEXT:    psraw $1, %xmm1
; SSE-NEXT:    paddw %xmm2, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test6:
; AVX:       # BB#0:
; AVX-NEXT:    vpmulhw {{.*}}(%rip), %ymm0, %ymm0
; AVX-NEXT:    vpsrlw $15, %ymm0, %ymm1
; AVX-NEXT:    vpsraw $1, %ymm0, %ymm0
; AVX-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %div = sdiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div
}

define <16 x i8> @test7(<16 x i8> %a) #0 {
; SSE41-LABEL: test7:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrb $1, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pextrb $0, %xmm0, %ecx
; SSE41-NEXT:    movsbl %cl, %ecx
; SSE41-NEXT:    imull $-109, %ecx, %edx
; SSE41-NEXT:    shrl $8, %edx
; SSE41-NEXT:    addb %dl, %cl
; SSE41-NEXT:    movb %cl, %dl
; SSE41-NEXT:    shrb $7, %dl
; SSE41-NEXT:    sarb $2, %cl
; SSE41-NEXT:    addb %dl, %cl
; SSE41-NEXT:    movzbl %cl, %ecx
; SSE41-NEXT:    movd %ecx, %xmm1
; SSE41-NEXT:    pinsrb $1, %eax, %xmm1
; SSE41-NEXT:    pextrb $2, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $2, %eax, %xmm1
; SSE41-NEXT:    pextrb $3, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $3, %eax, %xmm1
; SSE41-NEXT:    pextrb $4, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $4, %eax, %xmm1
; SSE41-NEXT:    pextrb $5, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $5, %eax, %xmm1
; SSE41-NEXT:    pextrb $6, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $6, %eax, %xmm1
; SSE41-NEXT:    pextrb $7, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $7, %eax, %xmm1
; SSE41-NEXT:    pextrb $8, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $8, %eax, %xmm1
; SSE41-NEXT:    pextrb $9, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $9, %eax, %xmm1
; SSE41-NEXT:    pextrb $10, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $10, %eax, %xmm1
; SSE41-NEXT:    pextrb $11, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $11, %eax, %xmm1
; SSE41-NEXT:    pextrb $12, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $12, %eax, %xmm1
; SSE41-NEXT:    pextrb $13, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $13, %eax, %xmm1
; SSE41-NEXT:    pextrb $14, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $14, %eax, %xmm1
; SSE41-NEXT:    pextrb $15, %xmm0, %eax
; SSE41-NEXT:    movsbl %al, %eax
; SSE41-NEXT:    imull $-109, %eax, %ecx
; SSE41-NEXT:    shrl $8, %ecx
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $7, %cl
; SSE41-NEXT:    sarb $2, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movzbl %al, %eax
; SSE41-NEXT:    pinsrb $15, %eax, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: test7:
; SSE:       # BB#0:
; SSE-NEXT:    pushq %rbp
; SSE-NEXT:    pushq %r14
; SSE-NEXT:    pushq %rbx
; SSE-NEXT:    movaps %xmm0, -{{[0-9]+}}(%rsp)
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %eax
; SSE-NEXT:    imull $-109, %eax, %ecx
; SSE-NEXT:    shrl $8, %ecx
; SSE-NEXT:    addb %al, %cl
; SSE-NEXT:    movb %cl, %al
; SSE-NEXT:    shrb $7, %al
; SSE-NEXT:    sarb $2, %cl
; SSE-NEXT:    addb %al, %cl
; SSE-NEXT:    movzbl %cl, %eax
; SSE-NEXT:    movd %eax, %xmm0
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %r14d
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %edx
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %r9d
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %eax
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %r11d
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %ecx
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %r8d
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %esi
; SSE-NEXT:    imull $-109, %esi, %edi
; SSE-NEXT:    shrl $8, %edi
; SSE-NEXT:    addb %sil, %dil
; SSE-NEXT:    movb %dil, %bl
; SSE-NEXT:    shrb $7, %bl
; SSE-NEXT:    sarb $2, %dil
; SSE-NEXT:    addb %bl, %dil
; SSE-NEXT:    movzbl %dil, %esi
; SSE-NEXT:    movd %esi, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE-NEXT:    imull $-109, %eax, %esi
; SSE-NEXT:    shrl $8, %esi
; SSE-NEXT:    addb %al, %sil
; SSE-NEXT:    movb %sil, %al
; SSE-NEXT:    shrb $7, %al
; SSE-NEXT:    sarb $2, %sil
; SSE-NEXT:    addb %al, %sil
; SSE-NEXT:    movzbl %sil, %eax
; SSE-NEXT:    movd %eax, %xmm2
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %ebp
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %esi
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %r10d
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %edi
; SSE-NEXT:    imull $-109, %edi, %ebx
; SSE-NEXT:    shrl $8, %ebx
; SSE-NEXT:    addb %dil, %bl
; SSE-NEXT:    movb %bl, %al
; SSE-NEXT:    shrb $7, %al
; SSE-NEXT:    sarb $2, %bl
; SSE-NEXT:    addb %al, %bl
; SSE-NEXT:    movzbl %bl, %eax
; SSE-NEXT:    movd %eax, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    imull $-109, %edx, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %dl, %al
; SSE-NEXT:    movb %al, %dl
; SSE-NEXT:    shrb $7, %dl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %dl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm1
; SSE-NEXT:    imull $-109, %esi, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %sil, %al
; SSE-NEXT:    movb %al, %dl
; SSE-NEXT:    shrb $7, %dl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %dl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm2
; SSE-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE-NEXT:    imull $-109, %ecx, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movb %al, %cl
; SSE-NEXT:    shrb $7, %cl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm3
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %ecx
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %eax
; SSE-NEXT:    imull $-109, %eax, %edx
; SSE-NEXT:    shrl $8, %edx
; SSE-NEXT:    addb %al, %dl
; SSE-NEXT:    movb %dl, %al
; SSE-NEXT:    shrb $7, %al
; SSE-NEXT:    sarb $2, %dl
; SSE-NEXT:    addb %al, %dl
; SSE-NEXT:    movzbl %dl, %eax
; SSE-NEXT:    movd %eax, %xmm1
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE-NEXT:    imull $-109, %r14d, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %r14b, %al
; SSE-NEXT:    movb %al, %dl
; SSE-NEXT:    shrb $7, %dl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %dl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm2
; SSE-NEXT:    imull $-109, %ebp, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %bpl, %al
; SSE-NEXT:    movb %al, %dl
; SSE-NEXT:    shrb $7, %dl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %dl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE-NEXT:    imull $-109, %r11d, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %r11b, %al
; SSE-NEXT:    movb %al, %dl
; SSE-NEXT:    shrb $7, %dl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %dl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm3
; SSE-NEXT:    imull $-109, %ecx, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movb %al, %cl
; SSE-NEXT:    shrb $7, %cl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm2
; SSE-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE-NEXT:    imull $-109, %r9d, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %r9b, %al
; SSE-NEXT:    movb %al, %cl
; SSE-NEXT:    shrb $7, %cl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm0
; SSE-NEXT:    imull $-109, %r10d, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %r10b, %al
; SSE-NEXT:    movb %al, %cl
; SSE-NEXT:    shrb $7, %cl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm3
; SSE-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE-NEXT:    imull $-109, %r8d, %eax
; SSE-NEXT:    shrl $8, %eax
; SSE-NEXT:    addb %r8b, %al
; SSE-NEXT:    movb %al, %cl
; SSE-NEXT:    shrb $7, %cl
; SSE-NEXT:    sarb $2, %al
; SSE-NEXT:    addb %cl, %al
; SSE-NEXT:    movzbl %al, %eax
; SSE-NEXT:    movd %eax, %xmm4
; SSE-NEXT:    movsbl -{{[0-9]+}}(%rsp), %eax
; SSE-NEXT:    imull $-109, %eax, %ecx
; SSE-NEXT:    shrl $8, %ecx
; SSE-NEXT:    addb %al, %cl
; SSE-NEXT:    movb %cl, %al
; SSE-NEXT:    shrb $7, %al
; SSE-NEXT:    sarb $2, %cl
; SSE-NEXT:    addb %al, %cl
; SSE-NEXT:    movzbl %cl, %eax
; SSE-NEXT:    movd %eax, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE-NEXT:    popq %rbx
; SSE-NEXT:    popq %r14
; SSE-NEXT:    popq %rbp
; SSE-NEXT:    retq
;
; AVX-LABEL: test7:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-NEXT:    movsbl %al, %eax
; AVX-NEXT:    imull $-109, %eax, %ecx
; AVX-NEXT:    shrl $8, %ecx
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $7, %cl
; AVX-NEXT:    sarb $2, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movzbl %al, %eax
; AVX-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %dl
; AVX-NEXT:    shrb $7, %dl
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movzbl %cl, %ecx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpextrb $2, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $3, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $4, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $3, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $5, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $4, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $6, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $5, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $7, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $6, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $8, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $7, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $9, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $8, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $10, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $9, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $11, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $10, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $12, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $11, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $13, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $12, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $14, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $13, %eax, %xmm1, %xmm1
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $15, %xmm0, %ecx
; AVX-NEXT:    movsbl %cl, %ecx
; AVX-NEXT:    imull $-109, %ecx, %edx
; AVX-NEXT:    vpinsrb $14, %eax, %xmm1, %xmm0
; AVX-NEXT:    shrl $8, %edx
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %al
; AVX-NEXT:    shrb $7, %al
; AVX-NEXT:    sarb $2, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $15, %eax, %xmm0, %xmm0
; AVX-NEXT:    retq
  %div = sdiv <16 x i8> %a, <i8 7, i8 7, i8 7, i8 7,i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7,i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %div
}

define <4 x i32> @test8(<4 x i32> %a) #0 {
; SSE41-LABEL: test8:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm1 = [2454267027,2454267027,2454267027,2454267027]
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuldq %xmm2, %xmm3
; SSE41-NEXT:    pmuldq %xmm0, %xmm1
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1],xmm3[2,3],xmm1[4,5],xmm3[6,7]
; SSE41-NEXT:    paddd %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    psrld $31, %xmm0
; SSE41-NEXT:    psrad $2, %xmm1
; SSE41-NEXT:    paddd %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: test8:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [2454267027,2454267027,2454267027,2454267027]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    psrad $31, %xmm2
; SSE-NEXT:    pand %xmm1, %xmm2
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    pmuludq %xmm1, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm1[1,1,3,3]
; SSE-NEXT:    psrad $31, %xmm1
; SSE-NEXT:    pand %xmm0, %xmm1
; SSE-NEXT:    paddd %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm3[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1]
; SSE-NEXT:    psubd %xmm2, %xmm1
; SSE-NEXT:    paddd %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    psrld $31, %xmm0
; SSE-NEXT:    psrad $2, %xmm1
; SSE-NEXT:    paddd %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: test8:
; AVX:       # BB#0:
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX-NEXT:    vpshufd {{.*#+}} xmm2 = xmm1[1,1,3,3]
; AVX-NEXT:    vpshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; AVX-NEXT:    vpmuldq %xmm2, %xmm3, %xmm2
; AVX-NEXT:    vpmuldq %xmm1, %xmm0, %xmm1
; AVX-NEXT:    vpshufd {{.*#+}} xmm1 = xmm1[1,1,3,3]
; AVX-NEXT:    vpblendd {{.*#+}} xmm1 = xmm1[0],xmm2[1],xmm1[2],xmm2[3]
; AVX-NEXT:    vpaddd %xmm0, %xmm1, %xmm0
; AVX-NEXT:    vpsrld $31, %xmm0, %xmm1
; AVX-NEXT:    vpsrad $2, %xmm0, %xmm0
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %div = sdiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div
}

define <8 x i32> @test9(<8 x i32> %a) #0 {
; SSE41-LABEL: test9:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm3 = [2454267027,2454267027,2454267027,2454267027]
; SSE41-NEXT:    pshufd {{.*#+}} xmm4 = xmm3[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm5 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuldq %xmm4, %xmm5
; SSE41-NEXT:    movdqa %xmm0, %xmm2
; SSE41-NEXT:    pmuldq %xmm3, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1],xmm5[2,3],xmm2[4,5],xmm5[6,7]
; SSE41-NEXT:    paddd %xmm0, %xmm2
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    psrld $31, %xmm0
; SSE41-NEXT:    psrad $2, %xmm2
; SSE41-NEXT:    paddd %xmm0, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[1,1,3,3]
; SSE41-NEXT:    pmuldq %xmm4, %xmm0
; SSE41-NEXT:    pmuldq %xmm1, %xmm3
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm3 = xmm3[0,1],xmm0[2,3],xmm3[4,5],xmm0[6,7]
; SSE41-NEXT:    paddd %xmm1, %xmm3
; SSE41-NEXT:    movdqa %xmm3, %xmm0
; SSE41-NEXT:    psrld $31, %xmm0
; SSE41-NEXT:    psrad $2, %xmm3
; SSE41-NEXT:    paddd %xmm0, %xmm3
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    movdqa %xmm3, %xmm1
; SSE41-NEXT:    retq
;
; SSE-LABEL: test9:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    movdqa {{.*#+}} xmm3 = [2454267027,2454267027,2454267027,2454267027]
; SSE-NEXT:    movdqa %xmm3, %xmm4
; SSE-NEXT:    psrad $31, %xmm4
; SSE-NEXT:    movdqa %xmm4, %xmm0
; SSE-NEXT:    pand %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm2, %xmm5
; SSE-NEXT:    psrad $31, %xmm5
; SSE-NEXT:    pand %xmm3, %xmm5
; SSE-NEXT:    paddd %xmm0, %xmm5
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    pmuludq %xmm3, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm3[1,1,3,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm7 = xmm2[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm6, %xmm7
; SSE-NEXT:    pshufd {{.*#+}} xmm7 = xmm7[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm7[0],xmm0[1],xmm7[1]
; SSE-NEXT:    psubd %xmm5, %xmm0
; SSE-NEXT:    paddd %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    psrld $31, %xmm2
; SSE-NEXT:    psrad $2, %xmm0
; SSE-NEXT:    paddd %xmm2, %xmm0
; SSE-NEXT:    pand %xmm1, %xmm4
; SSE-NEXT:    movdqa %xmm1, %xmm5
; SSE-NEXT:    psrad $31, %xmm5
; SSE-NEXT:    pand %xmm3, %xmm5
; SSE-NEXT:    paddd %xmm4, %xmm5
; SSE-NEXT:    pmuludq %xmm1, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm3[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm6, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    psubd %xmm5, %xmm2
; SSE-NEXT:    paddd %xmm1, %xmm2
; SSE-NEXT:    movdqa %xmm2, %xmm1
; SSE-NEXT:    psrld $31, %xmm1
; SSE-NEXT:    psrad $2, %xmm2
; SSE-NEXT:    paddd %xmm1, %xmm2
; SSE-NEXT:    movdqa %xmm2, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test9:
; AVX:       # BB#0:
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm2 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpshufd {{.*#+}} ymm3 = ymm0[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpmuldq %ymm2, %ymm3, %ymm2
; AVX-NEXT:    vpmuldq %ymm1, %ymm0, %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpblendd {{.*#+}} ymm1 = ymm1[0],ymm2[1],ymm1[2],ymm2[3],ymm1[4],ymm2[5],ymm1[6],ymm2[7]
; AVX-NEXT:    vpaddd %ymm0, %ymm1, %ymm0
; AVX-NEXT:    vpsrld $31, %ymm0, %ymm1
; AVX-NEXT:    vpsrad $2, %ymm0, %ymm0
; AVX-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %div = sdiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div
}

define <8 x i32> @test10(<8 x i32> %a) #0 {
; SSE41-LABEL: test10:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [613566757,613566757,613566757,613566757]
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm2[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuludq %xmm3, %xmm4
; SSE41-NEXT:    movdqa %xmm0, %xmm5
; SSE41-NEXT:    pmuludq %xmm2, %xmm5
; SSE41-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm5 = xmm5[0,1],xmm4[2,3],xmm5[4,5],xmm4[6,7]
; SSE41-NEXT:    movdqa %xmm0, %xmm4
; SSE41-NEXT:    psubd %xmm5, %xmm4
; SSE41-NEXT:    psrld $1, %xmm4
; SSE41-NEXT:    paddd %xmm5, %xmm4
; SSE41-NEXT:    psrld $2, %xmm4
; SSE41-NEXT:    movdqa {{.*#+}} xmm5 = [7,7,7,7]
; SSE41-NEXT:    pmulld %xmm5, %xmm4
; SSE41-NEXT:    psubd %xmm4, %xmm0
; SSE41-NEXT:    pshufd {{.*#+}} xmm4 = xmm1[1,1,3,3]
; SSE41-NEXT:    pmuludq %xmm3, %xmm4
; SSE41-NEXT:    pmuludq %xmm1, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1],xmm4[2,3],xmm2[4,5],xmm4[6,7]
; SSE41-NEXT:    movdqa %xmm1, %xmm3
; SSE41-NEXT:    psubd %xmm2, %xmm3
; SSE41-NEXT:    psrld $1, %xmm3
; SSE41-NEXT:    paddd %xmm2, %xmm3
; SSE41-NEXT:    psrld $2, %xmm3
; SSE41-NEXT:    pmulld %xmm5, %xmm3
; SSE41-NEXT:    psubd %xmm3, %xmm1
; SSE41-NEXT:    retq
;
; SSE-LABEL: test10:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm3 = [613566757,613566757,613566757,613566757]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pmuludq %xmm3, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm3[1,1,3,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm0[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm5
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm5[0],xmm2[1],xmm5[1]
; SSE-NEXT:    movdqa %xmm0, %xmm5
; SSE-NEXT:    psubd %xmm2, %xmm5
; SSE-NEXT:    psrld $1, %xmm5
; SSE-NEXT:    paddd %xmm2, %xmm5
; SSE-NEXT:    psrld $2, %xmm5
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [7,7,7,7]
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm5[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm2, %xmm5
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[0,2,2,3]
; SSE-NEXT:    pmuludq %xmm2, %xmm6
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm6[0,2,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm5 = xmm5[0],xmm6[0],xmm5[1],xmm6[1]
; SSE-NEXT:    psubd %xmm5, %xmm0
; SSE-NEXT:    pmuludq %xmm1, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm1[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm5
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm5[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm3 = xmm3[0],xmm4[0],xmm3[1],xmm4[1]
; SSE-NEXT:    movdqa %xmm1, %xmm4
; SSE-NEXT:    psubd %xmm3, %xmm4
; SSE-NEXT:    psrld $1, %xmm4
; SSE-NEXT:    paddd %xmm3, %xmm4
; SSE-NEXT:    psrld $2, %xmm4
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm4[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm2, %xmm4
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm4[0,2,2,3]
; SSE-NEXT:    pmuludq %xmm2, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm3[0,2,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm4 = xmm4[0],xmm2[0],xmm4[1],xmm2[1]
; SSE-NEXT:    psubd %xmm4, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test10:
; AVX:       # BB#0:
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm2 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpshufd {{.*#+}} ymm3 = ymm0[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpmuludq %ymm2, %ymm3, %ymm2
; AVX-NEXT:    vpmuludq %ymm1, %ymm0, %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpblendd {{.*#+}} ymm1 = ymm1[0],ymm2[1],ymm1[2],ymm2[3],ymm1[4],ymm2[5],ymm1[6],ymm2[7]
; AVX-NEXT:    vpsubd %ymm1, %ymm0, %ymm2
; AVX-NEXT:    vpsrld $1, %ymm2, %ymm2
; AVX-NEXT:    vpaddd %ymm1, %ymm2, %ymm1
; AVX-NEXT:    vpsrld $2, %ymm1, %ymm1
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX-NEXT:    vpmulld %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vpsubd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %rem = urem <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %rem
}

define <8 x i32> @test11(<8 x i32> %a) #0 {
; SSE41-LABEL: test11:
; SSE41:       # BB#0:
; SSE41-NEXT:    movdqa {{.*#+}} xmm2 = [2454267027,2454267027,2454267027,2454267027]
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm2[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuldq %xmm3, %xmm4
; SSE41-NEXT:    movdqa %xmm0, %xmm5
; SSE41-NEXT:    pmuldq %xmm2, %xmm5
; SSE41-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm5 = xmm5[0,1],xmm4[2,3],xmm5[4,5],xmm4[6,7]
; SSE41-NEXT:    paddd %xmm0, %xmm5
; SSE41-NEXT:    movdqa %xmm5, %xmm4
; SSE41-NEXT:    psrld $31, %xmm4
; SSE41-NEXT:    psrad $2, %xmm5
; SSE41-NEXT:    paddd %xmm4, %xmm5
; SSE41-NEXT:    movdqa {{.*#+}} xmm4 = [7,7,7,7]
; SSE41-NEXT:    pmulld %xmm4, %xmm5
; SSE41-NEXT:    psubd %xmm5, %xmm0
; SSE41-NEXT:    pshufd {{.*#+}} xmm5 = xmm1[1,1,3,3]
; SSE41-NEXT:    pmuldq %xmm3, %xmm5
; SSE41-NEXT:    pmuldq %xmm1, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1],xmm5[2,3],xmm2[4,5],xmm5[6,7]
; SSE41-NEXT:    paddd %xmm1, %xmm2
; SSE41-NEXT:    movdqa %xmm2, %xmm3
; SSE41-NEXT:    psrld $31, %xmm3
; SSE41-NEXT:    psrad $2, %xmm2
; SSE41-NEXT:    paddd %xmm3, %xmm2
; SSE41-NEXT:    pmulld %xmm4, %xmm2
; SSE41-NEXT:    psubd %xmm2, %xmm1
; SSE41-NEXT:    retq
;
; SSE-LABEL: test11:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [2454267027,2454267027,2454267027,2454267027]
; SSE-NEXT:    movdqa %xmm2, %xmm3
; SSE-NEXT:    psrad $31, %xmm3
; SSE-NEXT:    movdqa %xmm3, %xmm4
; SSE-NEXT:    pand %xmm0, %xmm4
; SSE-NEXT:    movdqa %xmm0, %xmm6
; SSE-NEXT:    psrad $31, %xmm6
; SSE-NEXT:    pand %xmm2, %xmm6
; SSE-NEXT:    paddd %xmm4, %xmm6
; SSE-NEXT:    movdqa %xmm0, %xmm4
; SSE-NEXT:    pmuludq %xmm2, %xmm4
; SSE-NEXT:    pshufd {{.*#+}} xmm7 = xmm4[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm5 = xmm2[1,1,3,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm5, %xmm4
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm4[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm7 = xmm7[0],xmm4[0],xmm7[1],xmm4[1]
; SSE-NEXT:    psubd %xmm6, %xmm7
; SSE-NEXT:    paddd %xmm0, %xmm7
; SSE-NEXT:    movdqa %xmm7, %xmm4
; SSE-NEXT:    psrld $31, %xmm4
; SSE-NEXT:    psrad $2, %xmm7
; SSE-NEXT:    paddd %xmm4, %xmm7
; SSE-NEXT:    movdqa {{.*#+}} xmm4 = [7,7,7,7]
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm7[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm7
; SSE-NEXT:    pshufd {{.*#+}} xmm7 = xmm7[0,2,2,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm6
; SSE-NEXT:    pshufd {{.*#+}} xmm6 = xmm6[0,2,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm7 = xmm7[0],xmm6[0],xmm7[1],xmm6[1]
; SSE-NEXT:    psubd %xmm7, %xmm0
; SSE-NEXT:    pand %xmm1, %xmm3
; SSE-NEXT:    movdqa %xmm1, %xmm6
; SSE-NEXT:    psrad $31, %xmm6
; SSE-NEXT:    pand %xmm2, %xmm6
; SSE-NEXT:    paddd %xmm3, %xmm6
; SSE-NEXT:    pmuludq %xmm1, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm1[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm5, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    psubd %xmm6, %xmm2
; SSE-NEXT:    paddd %xmm1, %xmm2
; SSE-NEXT:    movdqa %xmm2, %xmm3
; SSE-NEXT:    psrld $31, %xmm3
; SSE-NEXT:    psrad $2, %xmm2
; SSE-NEXT:    paddd %xmm3, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm2[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm2
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[0,2,2,3]
; SSE-NEXT:    pmuludq %xmm4, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[0,2,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
; SSE-NEXT:    psubd %xmm2, %xmm1
; SSE-NEXT:    retq
;
; AVX-LABEL: test11:
; AVX:       # BB#0:
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm2 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpshufd {{.*#+}} ymm3 = ymm0[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpmuldq %ymm2, %ymm3, %ymm2
; AVX-NEXT:    vpmuldq %ymm1, %ymm0, %ymm1
; AVX-NEXT:    vpshufd {{.*#+}} ymm1 = ymm1[1,1,3,3,5,5,7,7]
; AVX-NEXT:    vpblendd {{.*#+}} ymm1 = ymm1[0],ymm2[1],ymm1[2],ymm2[3],ymm1[4],ymm2[5],ymm1[6],ymm2[7]
; AVX-NEXT:    vpaddd %ymm0, %ymm1, %ymm1
; AVX-NEXT:    vpsrld $31, %ymm1, %ymm2
; AVX-NEXT:    vpsrad $2, %ymm1, %ymm1
; AVX-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX-NEXT:    vpmulld %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vpsubd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %rem = srem <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %rem
}

define <2 x i16> @test12() #0 {
; SSE41-LABEL: test12:
; SSE41:       # BB#0:
; SSE41-NEXT:    xorps %xmm0, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: test12:
; SSE:       # BB#0:
; SSE-NEXT:    xorps %xmm0, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: test12:
; AVX:       # BB#0:
; AVX-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    retq
  %I8 = insertelement <2 x i16> zeroinitializer, i16 -1, i32 0
  %I9 = insertelement <2 x i16> %I8, i16 -1, i32 1
  %B9 = urem <2 x i16> %I9, %I9
  ret <2 x i16> %B9
}

define <4 x i32> @PR20355(<4 x i32> %a) #0 {
; SSE41-LABEL: PR20355:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa {{.*#+}} xmm1 = [1431655766,1431655766,1431655766,1431655766]
; SSE41-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[1,1,3,3]
; SSE41-NEXT:    pshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; SSE41-NEXT:    pmuldq %xmm2, %xmm3
; SSE41-NEXT:    pmuldq %xmm1, %xmm0
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[1,1,3,3]
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1],xmm3[2,3],xmm1[4,5],xmm3[6,7]
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    psrld $31, %xmm0
; SSE41-NEXT:    paddd %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; SSE-LABEL: PR20355:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [1431655766,1431655766,1431655766,1431655766]
; SSE-NEXT:    movdqa %xmm1, %xmm2
; SSE-NEXT:    psrad $31, %xmm2
; SSE-NEXT:    pand %xmm0, %xmm2
; SSE-NEXT:    movdqa %xmm0, %xmm3
; SSE-NEXT:    psrad $31, %xmm3
; SSE-NEXT:    pand %xmm1, %xmm3
; SSE-NEXT:    paddd %xmm2, %xmm3
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm4 = xmm0[1,3,2,3]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[1,1,3,3]
; SSE-NEXT:    pmuludq %xmm2, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[1,3,2,3]
; SSE-NEXT:    punpckldq {{.*#+}} xmm4 = xmm4[0],xmm0[0],xmm4[1],xmm0[1]
; SSE-NEXT:    psubd %xmm3, %xmm4
; SSE-NEXT:    movdqa %xmm4, %xmm0
; SSE-NEXT:    psrld $31, %xmm0
; SSE-NEXT:    paddd %xmm4, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: PR20355:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX-NEXT:    vpshufd {{.*#+}} xmm2 = xmm1[1,1,3,3]
; AVX-NEXT:    vpshufd {{.*#+}} xmm3 = xmm0[1,1,3,3]
; AVX-NEXT:    vpmuldq %xmm2, %xmm3, %xmm2
; AVX-NEXT:    vpmuldq %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[1,1,3,3]
; AVX-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX-NEXT:    vpsrld $31, %xmm0, %xmm1
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
entry:
  %sdiv = sdiv <4 x i32> %a, <i32 3, i32 3, i32 3, i32 3>
  ret <4 x i32> %sdiv
}

attributes #0 = { nounwind }
