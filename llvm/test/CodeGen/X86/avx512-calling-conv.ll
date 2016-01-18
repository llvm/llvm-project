; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=SKX
; RUN: llc < %s -mtriple=i686-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL_X32

; KNL-LABEL: test1
; KNL: vxorps
define <16 x i1> @test1() {
  ret <16 x i1> zeroinitializer
}

; SKX-LABEL: test2
; SKX: vpmovb2m
; SKX: vpmovb2m
; SKX: kandw
; SKX: vpmovm2b
; KNL-LABEL: test2
; KNL: vpmovsxbd
; KNL: vpmovsxbd
; KNL: vpandd
; KNL: vpmovdb
define <16 x i1> @test2(<16 x i1>%a, <16 x i1>%b) {
  %c = and <16 x i1>%a, %b
  ret <16 x i1> %c
}

; SKX-LABEL: test3
; SKX: vpmovw2m
; SKX: vpmovw2m
; SKX: kandb
; SKX: vpmovm2w
define <8 x i1> @test3(<8 x i1>%a, <8 x i1>%b) {
  %c = and <8 x i1>%a, %b
  ret <8 x i1> %c
}

; SKX-LABEL: test4
; SKX: vpmovd2m
; SKX: vpmovd2m
; SKX: kandw
; SKX: vpmovm2d
define <4 x i1> @test4(<4 x i1>%a, <4 x i1>%b) {
  %c = and <4 x i1>%a, %b
  ret <4 x i1> %c
}

; SKX-LABEL: test5
; SKX: vpcmpgtd
; SKX: vpmovm2w
; SKX: call
; SKX: vpmovzxwd
declare <8 x i1> @func8xi1(<8 x i1> %a)
define <8 x i32> @test5(<8 x i32>%a, <8 x i32>%b) {
  %cmpRes = icmp sgt <8 x i32>%a, %b
  %resi = call <8 x i1> @func8xi1(<8 x i1> %cmpRes)
  %res = sext <8 x i1>%resi to <8 x i32>
  ret <8 x i32> %res
}

declare <16 x i1> @func16xi1(<16 x i1> %a)

; KNL-LABEL: test6
; KNL: vpbroadcastd
; KNL: vpmovdb
; KNL: call
; KNL: vpmovzxbd
; KNL: vpslld  $31, %zmm
; KNL: vpsrad  $31, %zmm
define <16 x i32> @test6(<16 x i32>%a, <16 x i32>%b) {
  %cmpRes = icmp sgt <16 x i32>%a, %b
  %resi = call <16 x i1> @func16xi1(<16 x i1> %cmpRes)
  %res = sext <16 x i1>%resi to <16 x i32>
  ret <16 x i32> %res
}

declare <4 x i1> @func4xi1(<4 x i1> %a)
; SKX-LABEL: test7
; SKX: vpmovm2d
; SKX: call
; SKX: vpslld  $31, %xmm
; SKX: vpsrad  $31, %xmm

define <4 x i32> @test7(<4 x i32>%a, <4 x i32>%b) {
  %cmpRes = icmp sgt <4 x i32>%a, %b
  %resi = call <4 x i1> @func4xi1(<4 x i1> %cmpRes)
  %res = sext <4 x i1>%resi to <4 x i32>
  ret <4 x i32> %res
}

; SKX-LABEL: test7a
; SKX: call
; SKX: vpmovw2m  %xmm0, %k0
; SKX: kandb
define <8 x i1> @test7a(<8 x i32>%a, <8 x i32>%b) {
  %cmpRes = icmp sgt <8 x i32>%a, %b
  %resi = call <8 x i1> @func8xi1(<8 x i1> %cmpRes)
  %res = and <8 x i1>%resi,  <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>
  ret <8 x i1> %res
}


; KNL_X32-LABEL: test8
; KNL_X32: testb $1, 4(%esp)
; KNL_X32:jne

; KNL-LABEL: test8
; KNL: testb   $1, %dil
; KNL:jne

define <16 x i8> @test8(<16 x i8> %a1, <16 x i8> %a2, i1 %cond) {
  %res = select i1 %cond, <16 x i8> %a1, <16 x i8> %a2
  ret <16 x i8> %res
}

; KNL-LABEL: test9
; KNL: vucomisd
; KNL: setb
define i1 @test9(double %a, double %b) {
  %c = fcmp ugt double %a, %b
  ret i1 %c
}

; KNL_X32-LABEL: test10
; KNL_X32: testb $1, 12(%esp)
; KNL_X32: cmovnel

; KNL-LABEL: test10
; KNL: testb   $1, %dl
; KNL: cmovel
define i32 @test10(i32 %a, i32 %b, i1 %cond) {
  %c = select i1 %cond, i32 %a, i32 %b
  ret i32 %c
}

; KNL-LABEL: test11
; KNL: cmp
; KNL: setg
define i1 @test11(i32 %a, i32 %b) {
  %c = icmp sgt i32 %a, %b
  ret i1 %c
}

; KNL-LABEL: test12
; KNL: callq _test11
;; return value in %al
; KNL: movzbl	%al, %ebx
; KNL: callq _test10
; KNL: testb   $1, %bl

define i32 @test12(i32 %a1, i32 %a2, i32 %b1) {
  %cond = call i1 @test11(i32 %a1, i32 %b1)
  %res = call i32 @test10(i32 %a1, i32 %a2, i1 %cond)
  %res1 = select i1 %cond, i32 %res, i32 0
  ret i32 %res1
}