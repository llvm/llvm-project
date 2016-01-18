; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL --check-prefix=CHECK
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=SKX --check-prefix=CHECK

; CHECK-LABEL: mask16
; CHECK: kmovw
; CHECK-NEXT: knotw
; CHECK-NEXT: kmovw
define i16 @mask16(i16 %x) {
  %m0 = bitcast i16 %x to <16 x i1>
  %m1 = xor <16 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <16 x i1> %m1 to i16
  ret i16 %ret
}

; CHECK-LABEL: mask8
; KNL: kmovw
; KNL-NEXT: knotw
; KNL-NEXT: kmovw
; SKX: kmovb
; SKX-NEXT: knotb
; SKX-NEXT: kmovb

define i8 @mask8(i8 %x) {
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  ret i8 %ret
}

; CHECK-LABEL: mask16_mem
; CHECK: kmovw ([[ARG1:%rdi|%rcx]]), %k{{[0-7]}}
; CHECK-NEXT: knotw
; CHECK-NEXT: kmovw %k{{[0-7]}}, ([[ARG1]])
; CHECK: ret

define void @mask16_mem(i16* %ptr) {
  %x = load i16, i16* %ptr, align 4
  %m0 = bitcast i16 %x to <16 x i1>
  %m1 = xor <16 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <16 x i1> %m1 to i16
  store i16 %ret, i16* %ptr, align 4
  ret void
}

; CHECK-LABEL: mask8_mem
; KNL: kmovw ([[ARG1]]), %k{{[0-7]}}
; KNL-NEXT: knotw
; KNL-NEXT: kmovw %k{{[0-7]}}, ([[ARG1]])
; SKX: kmovb ([[ARG1]]), %k{{[0-7]}}
; SKX-NEXT: knotb
; SKX-NEXT: kmovb %k{{[0-7]}}, ([[ARG1]])

define void @mask8_mem(i8* %ptr) {
  %x = load i8, i8* %ptr, align 4
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  store i8 %ret, i8* %ptr, align 4
  ret void
}

; CHECK-LABEL: mand16
; CHECK: kandw
; CHECK: kxorw
; CHECK: korw
define i16 @mand16(i16 %x, i16 %y) {
  %ma = bitcast i16 %x to <16 x i1>
  %mb = bitcast i16 %y to <16 x i1>
  %mc = and <16 x i1> %ma, %mb
  %md = xor <16 x i1> %ma, %mb
  %me = or <16 x i1> %mc, %md
  %ret = bitcast <16 x i1> %me to i16
  ret i16 %ret
}

; CHECK-LABEL: shuf_test1
; CHECK: kshiftrw        $8
define i8 @shuf_test1(i16 %v) nounwind {
   %v1 = bitcast i16 %v to <16 x i1>
   %mask = shufflevector <16 x i1> %v1, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
   %mask1 = bitcast <8 x i1> %mask to i8
   ret i8 %mask1
}

; CHECK-LABEL: zext_test1
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw

define i32 @zext_test1(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i32
  ret i32 %res
}

; CHECK-LABEL: zext_test2
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw

define i16 @zext_test2(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i16
  ret i16 %res
}

; CHECK-LABEL: zext_test3
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw

define i8 @zext_test3(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i8
  ret i8 %res
}

; CHECK-LABEL: conv1
; KNL: kmovw   %k0, %eax
; KNL: movb    %al, (%rdi)
; SKX: kmovb   %k0, (%rdi)
define i8 @conv1(<8 x i1>* %R) {
entry:
  store <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <8 x i1>* %R

  %maskPtr = alloca <8 x i1>
  store <8 x i1> <i1 0, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <8 x i1>* %maskPtr
  %mask = load <8 x i1>, <8 x i1>* %maskPtr
  %mask_convert = bitcast <8 x i1> %mask to i8
  ret i8 %mask_convert
}

; SKX-LABEL: test4
; SKX: vpcmpgt
; SKX: knot
; SKX: vpcmpgt
; SKX: vpmovm2d
define <4 x i32> @test4(<4 x i64> %x, <4 x i64> %y, <4 x i64> %x1, <4 x i64> %y1) {
  %x_gt_y = icmp sgt <4 x i64> %x, %y
  %x1_gt_y1 = icmp sgt <4 x i64> %x1, %y1
  %res = icmp sgt <4 x i1>%x_gt_y, %x1_gt_y1
  %resse = sext <4 x i1>%res to <4 x i32>
  ret <4 x i32> %resse
}

; SKX-LABEL: test5
; SKX: vpcmpgt
; SKX: knot
; SKX: vpcmpgt
; SKX: vpmovm2q
define <2 x i64> @test5(<2 x i64> %x, <2 x i64> %y, <2 x i64> %x1, <2 x i64> %y1) {
  %x_gt_y = icmp slt <2 x i64> %x, %y
  %x1_gt_y1 = icmp sgt <2 x i64> %x1, %y1
  %res = icmp slt <2 x i1>%x_gt_y, %x1_gt_y1
  %resse = sext <2 x i1>%res to <2 x i64>
  ret <2 x i64> %resse
}

; KNL-LABEL: test6
; KNL: vpmovsxbd
; KNL: vpandd
; KNL: kmovw   %eax, %k1
; KNL vptestmd {{.*}}, %k0 {%k1}

; SKX-LABEL: test6
; SKX: vpmovb2m
; SKX: kmovw   %eax, %k1
; SKX: kandw
define void @test6(<16 x i1> %mask)  {
allocas:
  %a= and <16 x i1> %mask, <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>
  %b = bitcast <16 x i1> %a to i16
  %c = icmp eq i16 %b, 0
  br i1 %c, label %true, label %false

true:
  ret void

false:
  ret void
}

; KNL-LABEL: test7
; KNL: vpmovsxwq
; KNL: vpandq
; KNL: vptestmq {{.*}}, %k0
; KNL: korw

; SKX-LABEL: test7
; SKX: vpmovw2m
; SKX: kmovb   %eax, %k1
; SKX: korb

define void @test7(<8 x i1> %mask)  {
allocas:
  %a= or <8 x i1> %mask, <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>
  %b = bitcast <8 x i1> %a to i8
  %c = icmp eq i8 %b, 0
  br i1 %c, label %true, label %false

true:
  ret void

false:
  ret void
}

; KNL-LABEL: test8
; KNL: vpxord  %zmm2, %zmm2, %zmm2
; KNL: jg
; KNL: vpcmpltud       %zmm2, %zmm1, %k1
; KNL: jmp
; KNL: vpcmpgtd        %zmm2, %zmm0, %k1

; SKX-LABEL: test8
; SKX: jg
; SKX: vpcmpltud {{.*}}, %k0
; SKX: vpmovm2b
; SKX: vpcmpgtd {{.*}}, %k0
; SKX: vpmovm2b

define <16 x i8> @test8(<16 x i32>%a, <16 x i32>%b, i32 %a1, i32 %b1) {
  %cond = icmp sgt i32 %a1, %b1
  %cmp1 = icmp sgt <16 x i32> %a, zeroinitializer
  %cmp2 = icmp ult <16 x i32> %b, zeroinitializer
  %mix = select i1 %cond, <16 x i1> %cmp1, <16 x i1> %cmp2
  %res = sext <16 x i1> %mix to <16 x i8>
  ret <16 x i8> %res
}

; KNL-LABEL: test9
; KNL: jg
; KNL: vpmovsxbd       %xmm1, %zmm0
; KNL: jmp
; KNL: vpmovsxbd       %xmm0, %zmm0

; SKX-LABEL: test9
; SKX: vpmovb2m        %xmm1, %k0
; SKX: vpmovm2b        %k0, %xmm0
; SKX: retq
; SKX: vpmovb2m        %xmm0, %k0
; SKX: vpmovm2b        %k0, %xmm0

define <16 x i1> @test9(<16 x i1>%a, <16 x i1>%b, i32 %a1, i32 %b1) {
  %mask = icmp sgt i32 %a1, %b1
  %c = select i1 %mask, <16 x i1>%a, <16 x i1>%b
  ret <16 x i1>%c
}

; KNL-LABEL: test10
; KNL: jg
; KNL: vpmovsxwq       %xmm1, %zmm0
; KNL: jmp
; KNL: vpmovsxwq       %xmm0, %zmm0

; SKX-LABEL: test10
; SKX: jg
; SKX: vpmovw2m        %xmm1, %k0
; SKX: vpmovm2w        %k0, %xmm0
; SKX: retq
; SKX: vpmovw2m        %xmm0, %k0
; SKX: vpmovm2w        %k0, %xmm0
define <8 x i1> @test10(<8 x i1>%a, <8 x i1>%b, i32 %a1, i32 %b1) {
  %mask = icmp sgt i32 %a1, %b1
  %c = select i1 %mask, <8 x i1>%a, <8 x i1>%b
  ret <8 x i1>%c
}

; SKX-LABEL: test11
; SKX: jg
; SKX: vpmovd2m        %xmm1, %k0
; SKX: vpmovm2d        %k0, %xmm0
; SKX: retq
; SKX: vpmovd2m        %xmm0, %k0
; SKX: vpmovm2d        %k0, %xmm0
define <4 x i1> @test11(<4 x i1>%a, <4 x i1>%b, i32 %a1, i32 %b1) {
  %mask = icmp sgt i32 %a1, %b1
  %c = select i1 %mask, <4 x i1>%a, <4 x i1>%b
  ret <4 x i1>%c
}

; KNL-LABEL: test12
; KNL: movl    %edi, %eax
define i32 @test12(i32 %x, i32 %y)  {
  %a = bitcast i16 21845 to <16 x i1>
  %b = extractelement <16 x i1> %a, i32 0
  %c = select i1 %b, i32 %x, i32 %y
  ret i32 %c
}

; KNL-LABEL: test13
; KNL: movl    %esi, %eax
define i32 @test13(i32 %x, i32 %y)  {
  %a = bitcast i16 21845 to <16 x i1>
  %b = extractelement <16 x i1> %a, i32 3
  %c = select i1 %b, i32 %x, i32 %y
  ret i32 %c
}

; SKX-LABEL: test14
; SKX: movb     $11, %al
; SKX: kmovb    %eax, %k0
; SKX: vpmovm2d %k0, %xmm0

define <4 x i1> @test14()  {
  %a = bitcast i16 21845 to <16 x i1>
  %b = extractelement <16 x i1> %a, i32 2
  %c = insertelement <4 x i1> <i1 true, i1 false, i1 false, i1 true>, i1 %b, i32 1
  ret <4 x i1> %c
}

; KNL-LABEL: test15
; KNL: cmovgw
define <16 x i1> @test15(i32 %x, i32 %y)  {
  %a = bitcast i16 21845 to <16 x i1>
  %b = bitcast i16 1 to <16 x i1>
  %mask = icmp sgt i32 %x, %y
  %c = select i1 %mask, <16 x i1> %a, <16 x i1> %b
  ret <16 x i1> %c
}

; SKX-LABEL: test16
; SKX: kxnorw  %k1, %k1, %k1
; SKX: kshiftrw        $15, %k1, %k1
; SKX: kshiftlq        $5, %k1, %k1
; SKX: korq    %k1, %k0, %k0
; SKX: vpmovm2b        %k0, %zmm0
define <64 x i8> @test16(i64 %x) {
  %a = bitcast i64 %x to <64 x i1>
  %b = insertelement <64 x i1>%a, i1 true, i32 5
  %c = sext <64 x i1>%b to <64 x i8>
  ret <64 x i8>%c
}

; SKX-LABEL: test17
; SKX: setg    %al
; SKX: andl    $1, %eax
; SKX: kmovw   %eax, %k1
; SKX: kshiftlq        $5, %k1, %k1
; SKX: korq    %k1, %k0, %k0
; SKX: vpmovm2b        %k0, %zmm0
define <64 x i8> @test17(i64 %x, i32 %y, i32 %z) {
  %a = bitcast i64 %x to <64 x i1>
  %b = icmp sgt i32 %y, %z
  %c = insertelement <64 x i1>%a, i1 %b, i32 5
  %d = sext <64 x i1>%c to <64 x i8>
  ret <64 x i8>%d
}

; KNL-LABEL: test18
define <8 x i1> @test18(i8 %a, i16 %y) {
  %b = bitcast i8 %a to <8 x i1>
  %b1 = bitcast i16 %y to <16 x i1>
  %el1 = extractelement <16 x i1>%b1, i32 8
  %el2 = extractelement <16 x i1>%b1, i32 9
  %c = insertelement <8 x i1>%b, i1 %el1, i32 7
  %d = insertelement <8 x i1>%c, i1 %el2, i32 6
  ret <8 x i1>%d
}

; KNL-LABEL: test21
; KNL: vpand %ymm
; KNL: vextracti128    $1, %ymm2
; KNL: vpand %ymm

; SKX-LABEL: test21
; SKX: vpmovb2m
; SKX: vmovdqu16 {{.*}}%k1

define <32 x i16> @test21(<32 x i16> %x , <32 x i1> %mask) nounwind readnone {
  %ret = select <32 x i1> %mask, <32 x i16> %x, <32 x i16> zeroinitializer
  ret <32 x i16> %ret
}

; SKX-LABEL: test22
; SKX: kmovb
define void @test22(<4 x i1> %a, <4 x i1>* %addr) {
  store <4 x i1> %a, <4 x i1>* %addr
  ret void
}

; SKX-LABEL: test23
; SKX: kmovb
define void @test23(<2 x i1> %a, <2 x i1>* %addr) {
  store <2 x i1> %a, <2 x i1>* %addr
  ret void
}
