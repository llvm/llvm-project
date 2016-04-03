; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck --check-prefix=KNL --check-prefix=CHECK %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck --check-prefix=SKX --check-prefix=CHECK %s

;CHECK-LABEL: test1:
;CHECK: vinsertps
;CHECK: vinsertf32x4
;CHECK: ret
define <16 x float> @test1(<16 x float> %x, float* %br, float %y) nounwind {
  %rrr = load float, float* %br
  %rrr2 = insertelement <16 x float> %x, float %rrr, i32 1
  %rrr3 = insertelement <16 x float> %rrr2, float %y, i32 14
  ret <16 x float> %rrr3
}

define <8 x double> @test2(<8 x double> %x, double* %br, double %y) nounwind {
; KNL-LABEL: test2:
; KNL:       ## BB#0:
; KNL-NEXT:    vmovhpd (%rdi), %xmm0, %xmm2
; KNL-NEXT:    vinsertf32x4 $0, %xmm2, %zmm0, %zmm0
; KNL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; KNL-NEXT:    vmovsd %xmm1, %xmm2, %xmm1
; KNL-NEXT:    vinsertf32x4 $3, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test2:
; SKX:       ## BB#0:
; SKX-NEXT:    vmovhpd (%rdi), %xmm0, %xmm2
; SKX-NEXT:    vinsertf64x2 $0, %xmm2, %zmm0, %zmm0
; SKX-NEXT:    vextractf64x2 $3, %zmm0, %xmm2
; SKX-NEXT:    vmovsd %xmm1, %xmm2, %xmm1
; SKX-NEXT:    vinsertf64x2 $3, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %rrr = load double, double* %br
  %rrr2 = insertelement <8 x double> %x, double %rrr, i32 1
  %rrr3 = insertelement <8 x double> %rrr2, double %y, i32 6
  ret <8 x double> %rrr3
}

;CHECK-LABEL: test3:
;CHECK: vextractf32x4 $1
;CHECK: vinsertf32x4 $0
;CHECK: ret
define <16 x float> @test3(<16 x float> %x) nounwind {
  %eee = extractelement <16 x float> %x, i32 4
  %rrr2 = insertelement <16 x float> %x, float %eee, i32 1
  ret <16 x float> %rrr2
}

define <8 x i64> @test4(<8 x i64> %x) nounwind {
; KNL-LABEL: test4:
; KNL:       ## BB#0:
; KNL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; KNL-NEXT:    vmovq %xmm1, %rax
; KNL-NEXT:    vpinsrq $1, %rax, %xmm0, %xmm1
; KNL-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test4:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti64x2 $2, %zmm0, %xmm1
; SKX-NEXT:    vmovq %xmm1, %rax
; SKX-NEXT:    vpinsrq $1, %rax, %xmm0, %xmm1
; SKX-NEXT:    vinserti64x2 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %eee = extractelement <8 x i64> %x, i32 4
  %rrr2 = insertelement <8 x i64> %x, i64 %eee, i32 1
  ret <8 x i64> %rrr2
}

;CHECK-LABEL: test5:
;CHECK: vextractps
;CHECK: ret
define i32 @test5(<4 x float> %x) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  %ei = bitcast float %ef to i32
  ret i32 %ei
}

;CHECK-LABEL: test6:
;CHECK: vextractps {{.*}}, (%rdi)
;CHECK: ret
define void @test6(<4 x float> %x, float* %out) nounwind {
  %ef = extractelement <4 x float> %x, i32 3
  store float %ef, float* %out, align 4
  ret void
}

;CHECK-LABEL: test7
;CHECK: vmovd
;CHECK: vpermps %zmm
;CHECK: ret
define float @test7(<16 x float> %x, i32 %ind) nounwind {
  %e = extractelement <16 x float> %x, i32 %ind
  ret float %e
}

;CHECK-LABEL: test8
;CHECK: vmovq
;CHECK: vpermpd %zmm
;CHECK: ret
define double @test8(<8 x double> %x, i32 %ind) nounwind {
  %e = extractelement <8 x double> %x, i32 %ind
  ret double %e
}

;CHECK-LABEL: test9
;CHECK: vmovd
;CHECK: vpermps %ymm
;CHECK: ret
define float @test9(<8 x float> %x, i32 %ind) nounwind {
  %e = extractelement <8 x float> %x, i32 %ind
  ret float %e
}

;CHECK-LABEL: test10
;CHECK: vmovd
;CHECK: vpermd %zmm
;CHECK: vmovd  %xmm0, %eax
;CHECK: ret
define i32 @test10(<16 x i32> %x, i32 %ind) nounwind {
  %e = extractelement <16 x i32> %x, i32 %ind
  ret i32 %e
}

;CHECK-LABEL: test11
;CHECK: vpcmpltud
;CHECK: kshiftlw $11
;CHECK: kshiftrw $15
;CHECK: testb
;CHECK: je
;CHECK: ret
;CHECK: ret
define <16 x i32> @test11(<16 x i32>%a, <16 x i32>%b) {
  %cmp_res = icmp ult <16 x i32> %a, %b
  %ia = extractelement <16 x i1> %cmp_res, i32 4
  br i1 %ia, label %A, label %B
  A:
    ret <16 x i32>%b
  B:
   %c = add <16 x i32>%b, %a
   ret <16 x i32>%c
}

;CHECK-LABEL: test12
;CHECK: vpcmpgtq
;CHECK: kshiftlw $15
;CHECK: kshiftrw $15
;CHECK: testb
;CHECK: ret

define i64 @test12(<16 x i64>%a, <16 x i64>%b, i64 %a1, i64 %b1) {

  %cmpvector_func.i = icmp slt <16 x i64> %a, %b
  %extract24vector_func.i = extractelement <16 x i1> %cmpvector_func.i, i32 0
  %res = select i1 %extract24vector_func.i, i64 %a1, i64 %b1
  ret i64 %res
}

;CHECK-LABEL: test13
;CHECK: cmpl    %esi, %edi
;CHECK: setb    %al
;CHECK: andl    $1, %eax
;CHECK: kmovw   %eax, %k0
;CHECK: movw    $-4
;CHECK: korw
define i16 @test13(i32 %a, i32 %b) {
  %cmp_res = icmp ult i32 %a, %b
  %maskv = insertelement <16 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, i1 %cmp_res, i32 0
  %res = bitcast <16 x i1> %maskv to i16
  ret i16 %res
}

;CHECK-LABEL: test14
;CHECK: vpcmpgtq
;KNL: kshiftlw $11
;KNL: kshiftrw $15
;KNL: testb
;SKX: kshiftlb $3
;SKX: kshiftrb $7
;SKX: testb
;CHECK: ret

define i64 @test14(<8 x i64>%a, <8 x i64>%b, i64 %a1, i64 %b1) {

  %cmpvector_func.i = icmp slt <8 x i64> %a, %b
  %extract24vector_func.i = extractelement <8 x i1> %cmpvector_func.i, i32 4
  %res = select i1 %extract24vector_func.i, i64 %a1, i64 %b1
  ret i64 %res
}

;CHECK-LABEL: test15
;CHECK: movb (%rdi), %al
;CHECK: movw    $-1, %ax
;CHECK: cmovew
define i16 @test15(i1 *%addr) {
  %x = load i1 , i1 * %addr, align 1
  %x1 = insertelement <16 x i1> undef, i1 %x, i32 10
  %x2 = bitcast <16 x i1>%x1 to i16
  ret i16 %x2
}

;CHECK-LABEL: test16
;CHECK: movzbw (%rdi), %ax
;CHECK: kmovw
;CHECK: kshiftlw        $10
;CHECK: korw
;CHECK: ret
define i16 @test16(i1 *%addr, i16 %a) {
  %x = load i1 , i1 * %addr, align 128
  %a1 = bitcast i16 %a to <16 x i1>
  %x1 = insertelement <16 x i1> %a1, i1 %x, i32 10
  %x2 = bitcast <16 x i1>%x1 to i16
  ret i16 %x2
}

;CHECK-LABEL: test17
;KNL: movzbw (%rdi), %ax
;KNL: andw $1, %ax
;KNL: kshiftlw $4
;KNL: korw
;SKX: kshiftlb $4
;SKX: korb
;CHECK: ret
define i8 @test17(i1 *%addr, i8 %a) {
  %x = load i1 , i1 * %addr, align 128
  %a1 = bitcast i8 %a to <8 x i1>
  %x1 = insertelement <8 x i1> %a1, i1 %x, i32 4
  %x2 = bitcast <8 x i1>%x1 to i8
  ret i8 %x2
}

define i64 @extract_v8i64(<8 x i64> %x, i64* %dst) {
; SKX-LABEL: extract_v8i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrq $1, %xmm0, %rax
; SKX-NEXT:    vextracti64x2 $1, %zmm0, %xmm0
; SKX-NEXT:    vpextrq $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <8 x i64> %x, i32 1
  %r2 = extractelement <8 x i64> %x, i32 3
  store i64 %r2, i64* %dst, align 1
  ret i64 %r1
}

define i64 @extract_v4i64(<4 x i64> %x, i64* %dst) {
; SKX-LABEL: extract_v4i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrq $1, %xmm0, %rax
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm0
; SKX-NEXT:    vpextrq $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <4 x i64> %x, i32 1
  %r2 = extractelement <4 x i64> %x, i32 3
  store i64 %r2, i64* %dst, align 1
  ret i64 %r1
}

define i64 @extract_v2i64(<2 x i64> %x, i64* %dst) {
; SKX-LABEL: extract_v2i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vmovq %xmm0, %rax
; SKX-NEXT:    vpextrq $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <2 x i64> %x, i32 0
  %r2 = extractelement <2 x i64> %x, i32 1
  store i64 %r2, i64* %dst, align 1
  ret i64 %r1
}

define i32 @extract_v16i32(<16 x i32> %x, i32* %dst) {
; SKX-LABEL: extract_v16i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrd $1, %xmm0, %eax
; SKX-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; SKX-NEXT:    vpextrd $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <16 x i32> %x, i32 1
  %r2 = extractelement <16 x i32> %x, i32 5
  store i32 %r2, i32* %dst, align 1
  ret i32 %r1
}

define i32 @extract_v8i32(<8 x i32> %x, i32* %dst) {
; SKX-LABEL: extract_v8i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrd $1, %xmm0, %eax
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm0
; SKX-NEXT:    vpextrd $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <8 x i32> %x, i32 1
  %r2 = extractelement <8 x i32> %x, i32 5
  store i32 %r2, i32* %dst, align 1
  ret i32 %r1
}

define i32 @extract_v4i32(<4 x i32> %x, i32* %dst) {
; SKX-LABEL: extract_v4i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrd $1, %xmm0, %eax
; SKX-NEXT:    vpextrd $3, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <4 x i32> %x, i32 1
  %r2 = extractelement <4 x i32> %x, i32 3
  store i32 %r2, i32* %dst, align 1
  ret i32 %r1
}

define i16 @extract_v32i16(<32 x i16> %x, i16* %dst) {
; SKX-LABEL: extract_v32i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrw $1, %xmm0, %eax
; SKX-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; SKX-NEXT:    vpextrw $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <32 x i16> %x, i32 1
  %r2 = extractelement <32 x i16> %x, i32 9
  store i16 %r2, i16* %dst, align 1
  ret i16 %r1
}

define i16 @extract_v16i16(<16 x i16> %x, i16* %dst) {
; SKX-LABEL: extract_v16i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrw $1, %xmm0, %eax
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm0
; SKX-NEXT:    vpextrw $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <16 x i16> %x, i32 1
  %r2 = extractelement <16 x i16> %x, i32 9
  store i16 %r2, i16* %dst, align 1
  ret i16 %r1
}

define i16 @extract_v8i16(<8 x i16> %x, i16* %dst) {
; SKX-LABEL: extract_v8i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrw $1, %xmm0, %eax
; SKX-NEXT:    vpextrw $3, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <8 x i16> %x, i32 1
  %r2 = extractelement <8 x i16> %x, i32 3
  store i16 %r2, i16* %dst, align 1
  ret i16 %r1
}

define i8 @extract_v64i8(<64 x i8> %x, i8* %dst) {
; SKX-LABEL: extract_v64i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrb $1, %xmm0, %eax
; SKX-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; SKX-NEXT:    vpextrb $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <64 x i8> %x, i32 1
  %r2 = extractelement <64 x i8> %x, i32 17
  store i8 %r2, i8* %dst, align 1
  ret i8 %r1
}

define i8 @extract_v32i8(<32 x i8> %x, i8* %dst) {
; SKX-LABEL: extract_v32i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrb $1, %xmm0, %eax
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm0
; SKX-NEXT:    vpextrb $1, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <32 x i8> %x, i32 1
  %r2 = extractelement <32 x i8> %x, i32 17
  store i8 %r2, i8* %dst, align 1
  ret i8 %r1
}

define i8 @extract_v16i8(<16 x i8> %x, i8* %dst) {
; SKX-LABEL: extract_v16i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpextrb $1, %xmm0, %eax
; SKX-NEXT:    vpextrb $3, %xmm0, (%rdi)
; SKX-NEXT:    retq
  %r1 = extractelement <16 x i8> %x, i32 1
  %r2 = extractelement <16 x i8> %x, i32 3
  store i8 %r2, i8* %dst, align 1
  ret i8 %r1
}

define <8 x i64> @insert_v8i64(<8 x i64> %x, i64 %y , i64* %ptr) {
; SKX-LABEL: insert_v8i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrq $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vinserti64x2 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    vextracti64x2 $1, %zmm0, %xmm1
; SKX-NEXT:    vpinsrq $1, %rdi, %xmm1, %xmm1
; SKX-NEXT:    vinserti64x2 $1, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %val = load i64, i64* %ptr
  %r1 = insertelement <8 x i64> %x, i64 %val, i32 1
  %r2 = insertelement <8 x i64> %r1, i64 %y, i32 3
  ret <8 x i64> %r2
}

define <4 x i64> @insert_v4i64(<4 x i64> %x, i64 %y , i64* %ptr) {
; SKX-LABEL: insert_v4i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrq $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm1
; SKX-NEXT:    vpinsrq $1, %rdi, %xmm1, %xmm1
; SKX-NEXT:    vinserti64x2 $1, %xmm1, %ymm0, %ymm0
; SKX-NEXT:    retq
  %val = load i64, i64* %ptr
  %r1 = insertelement <4 x i64> %x, i64 %val, i32 1
  %r2 = insertelement <4 x i64> %r1, i64 %y, i32 3
  ret <4 x i64> %r2
}

define <2 x i64> @insert_v2i64(<2 x i64> %x, i64 %y , i64* %ptr) {
; SKX-LABEL: insert_v2i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrq $1, (%rsi), %xmm0, %xmm0
; SKX-NEXT:    vpinsrq $3, %rdi, %xmm0, %xmm0
; SKX-NEXT:    retq
  %val = load i64, i64* %ptr
  %r1 = insertelement <2 x i64> %x, i64 %val, i32 1
  %r2 = insertelement <2 x i64> %r1, i64 %y, i32 3
  ret <2 x i64> %r2
}

define <16 x i32> @insert_v16i32(<16 x i32> %x, i32 %y, i32* %ptr) {
; SKX-LABEL: insert_v16i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrd $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; SKX-NEXT:    vpinsrd $1, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %val = load i32, i32* %ptr
  %r1 = insertelement <16 x i32> %x, i32 %val, i32 1
  %r2 = insertelement <16 x i32> %r1, i32 %y, i32 5
  ret <16 x i32> %r2
}

define <8 x i32> @insert_v8i32(<8 x i32> %x, i32 %y, i32* %ptr) {
; KNL-LABEL: insert_v8i32:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrd $1, (%rsi), %xmm0, %xmm1
; KNL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; KNL-NEXT:    vextracti128 $1, %ymm0, %xmm1
; KNL-NEXT:    vpinsrd $1, %edi, %xmm1, %xmm1
; KNL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v8i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrd $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm1
; SKX-NEXT:    vpinsrd $1, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; SKX-NEXT:    retq
  %val = load i32, i32* %ptr
  %r1 = insertelement <8 x i32> %x, i32 %val, i32 1
  %r2 = insertelement <8 x i32> %r1, i32 %y, i32 5
  ret <8 x i32> %r2
}

define <4 x i32> @insert_v4i32(<4 x i32> %x, i32 %y, i32* %ptr) {
; KNL-LABEL: insert_v4i32:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrd $1, (%rsi), %xmm0, %xmm0
; KNL-NEXT:    vpinsrd $3, %edi, %xmm0, %xmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v4i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrd $1, (%rsi), %xmm0, %xmm0
; SKX-NEXT:    vpinsrd $3, %edi, %xmm0, %xmm0
; SKX-NEXT:    retq
  %val = load i32, i32* %ptr
  %r1 = insertelement <4 x i32> %x, i32 %val, i32 1
  %r2 = insertelement <4 x i32> %r1, i32 %y, i32 3
  ret <4 x i32> %r2
}

define <32 x i16> @insert_v32i16(<32 x i16> %x, i16 %y, i16* %ptr) {
; KNL-LABEL: insert_v32i16:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrw $1, (%rsi), %xmm0, %xmm2
; KNL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm2[0,1,2,3],ymm0[4,5,6,7]
; KNL-NEXT:    vextracti128 $1, %ymm0, %xmm2
; KNL-NEXT:    vpinsrw $1, %edi, %xmm2, %xmm2
; KNL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v32i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrw $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; SKX-NEXT:    vpinsrw $1, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %val = load i16, i16* %ptr
  %r1 = insertelement <32 x i16> %x, i16 %val, i32 1
  %r2 = insertelement <32 x i16> %r1, i16 %y, i32 9
  ret <32 x i16> %r2
}

define <16 x i16> @insert_v16i16(<16 x i16> %x, i16 %y, i16* %ptr) {
; KNL-LABEL: insert_v16i16:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrw $1, (%rsi), %xmm0, %xmm1
; KNL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; KNL-NEXT:    vextracti128 $1, %ymm0, %xmm1
; KNL-NEXT:    vpinsrw $1, %edi, %xmm1, %xmm1
; KNL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v16i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrw $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm1
; SKX-NEXT:    vpinsrw $1, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; SKX-NEXT:    retq
  %val = load i16, i16* %ptr
  %r1 = insertelement <16 x i16> %x, i16 %val, i32 1
  %r2 = insertelement <16 x i16> %r1, i16 %y, i32 9
  ret <16 x i16> %r2
}

define <8 x i16> @insert_v8i16(<8 x i16> %x, i16 %y, i16* %ptr) {
; KNL-LABEL: insert_v8i16:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrw $1, (%rsi), %xmm0, %xmm0
; KNL-NEXT:    vpinsrw $5, %edi, %xmm0, %xmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v8i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrw $1, (%rsi), %xmm0, %xmm0
; SKX-NEXT:    vpinsrw $5, %edi, %xmm0, %xmm0
; SKX-NEXT:    retq
  %val = load i16, i16* %ptr
  %r1 = insertelement <8 x i16> %x, i16 %val, i32 1
  %r2 = insertelement <8 x i16> %r1, i16 %y, i32 5
  ret <8 x i16> %r2
}

define <64 x i8> @insert_v64i8(<64 x i8> %x, i8 %y, i8* %ptr) {
; KNL-LABEL: insert_v64i8:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrb $1, (%rsi), %xmm0, %xmm2
; KNL-NEXT:    vpblendd {{.*#+}} ymm0 = ymm2[0,1,2,3],ymm0[4,5,6,7]
; KNL-NEXT:    vextracti128 $1, %ymm1, %xmm2
; KNL-NEXT:    vpinsrb $2, %edi, %xmm2, %xmm2
; KNL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v64i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrb $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; SKX-NEXT:    vpinsrb $2, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $3, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %val = load i8, i8* %ptr
  %r1 = insertelement <64 x i8> %x, i8 %val, i32 1
  %r2 = insertelement <64 x i8> %r1, i8 %y, i32 50
  ret <64 x i8> %r2
}

define <32 x i8> @insert_v32i8(<32 x i8> %x, i8 %y, i8* %ptr) {
; SKX-LABEL: insert_v32i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrb $1, (%rsi), %xmm0, %xmm1
; SKX-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm1
; SKX-NEXT:    vpinsrb $1, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; SKX-NEXT:    retq
  %val = load i8, i8* %ptr
  %r1 = insertelement <32 x i8> %x, i8 %val, i32 1
  %r2 = insertelement <32 x i8> %r1, i8 %y, i32 17
  ret <32 x i8> %r2
}

define <16 x i8> @insert_v16i8(<16 x i8> %x, i8 %y, i8* %ptr) {
; KNL-LABEL: insert_v16i8:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrb $3, (%rsi), %xmm0, %xmm0
; KNL-NEXT:    vpinsrb $10, %edi, %xmm0, %xmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: insert_v16i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrb $3, (%rsi), %xmm0, %xmm0
; SKX-NEXT:    vpinsrb $10, %edi, %xmm0, %xmm0
; SKX-NEXT:    retq
  %val = load i8, i8* %ptr
  %r1 = insertelement <16 x i8> %x, i8 %val, i32 3
  %r2 = insertelement <16 x i8> %r1, i8 %y, i32 10
  ret <16 x i8> %r2
}

define <8 x i64> @test_insert_128_v8i64(<8 x i64> %x, i64 %y) {
; KNL-LABEL: test_insert_128_v8i64:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrq $1, %rdi, %xmm0, %xmm1
; KNL-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test_insert_128_v8i64:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrq $1, %rdi, %xmm0, %xmm1
; SKX-NEXT:    vinserti64x2 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %r = insertelement <8 x i64> %x, i64 %y, i32 1
  ret <8 x i64> %r
}

define <16 x i32> @test_insert_128_v16i32(<16 x i32> %x, i32 %y) {
; KNL-LABEL: test_insert_128_v16i32:
; KNL:       ## BB#0:
; KNL-NEXT:    vpinsrd $1, %edi, %xmm0, %xmm1
; KNL-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test_insert_128_v16i32:
; SKX:       ## BB#0:
; SKX-NEXT:    vpinsrd $1, %edi, %xmm0, %xmm1
; SKX-NEXT:    vinserti32x4 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %r = insertelement <16 x i32> %x, i32 %y, i32 1
  ret <16 x i32> %r
}

define <8 x double> @test_insert_128_v8f64(<8 x double> %x, double %y) {
; KNL-LABEL: test_insert_128_v8f64:
; KNL:       ## BB#0:
; KNL-NEXT:    vunpcklpd {{.*#+}} xmm1 = xmm0[0],xmm1[0]
; KNL-NEXT:    vinsertf32x4 $0, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test_insert_128_v8f64:
; SKX:       ## BB#0:
; SKX-NEXT:    vunpcklpd %xmm1, %xmm0, %xmm1
; SKX-NEXT:    vinsertf64x2 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %r = insertelement <8 x double> %x, double %y, i32 1
  ret <8 x double> %r
}

define <16 x float> @test_insert_128_v16f32(<16 x float> %x, float %y) {
; KNL-LABEL: test_insert_128_v16f32:
; KNL:       ## BB#0:
; KNL-NEXT:    vinsertps $16, %xmm1, %xmm0, %xmm1
; KNL-NEXT:    vinsertf32x4 $0, %xmm1, %zmm0, %zmm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test_insert_128_v16f32:
; SKX:       ## BB#0:
; SKX-NEXT:    vinsertps $16, %xmm1, %xmm0, %xmm1
; SKX-NEXT:    vinsertf32x4 $0, %xmm1, %zmm0, %zmm0
; SKX-NEXT:    retq
  %r = insertelement <16 x float> %x, float %y, i32 1
  ret <16 x float> %r
}

define <16 x i16> @test_insert_128_v16i16(<16 x i16> %x, i16 %y) {
; KNL-LABEL: test_insert_128_v16i16:
; KNL:       ## BB#0:
; KNL-NEXT:    vextracti128 $1, %ymm0, %xmm1
; KNL-NEXT:    vpinsrw $2, %edi, %xmm1, %xmm1
; KNL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test_insert_128_v16i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm1
; SKX-NEXT:    vpinsrw $2, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; SKX-NEXT:    retq
  %r = insertelement <16 x i16> %x, i16 %y, i32 10
  ret <16 x i16> %r
}

define <32 x i8> @test_insert_128_v32i8(<32 x i8> %x, i8 %y) {
; KNL-LABEL: test_insert_128_v32i8:
; KNL:       ## BB#0:
; KNL-NEXT:    vextracti128 $1, %ymm0, %xmm1
; KNL-NEXT:    vpinsrb $4, %edi, %xmm1, %xmm1
; KNL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; KNL-NEXT:    retq
;
; SKX-LABEL: test_insert_128_v32i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti128 $1, %ymm0, %xmm1
; SKX-NEXT:    vpinsrb $4, %edi, %xmm1, %xmm1
; SKX-NEXT:    vinserti32x4 $1, %xmm1, %ymm0, %ymm0
; SKX-NEXT:    retq
  %r = insertelement <32 x i8> %x, i8 %y, i32 20
  ret <32 x i8> %r
}
