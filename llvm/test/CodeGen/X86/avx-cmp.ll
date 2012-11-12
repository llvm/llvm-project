; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vcmpltps %ymm
; CHECK-NOT: vucomiss
define <8 x i32> @cmp00(<8 x float> %a, <8 x float> %b) nounwind readnone {
  %bincmp = fcmp olt <8 x float> %a, %b
  %s = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %s
}

; CHECK: vcmpltpd %ymm
; CHECK-NOT: vucomisd
define <4 x i64> @cmp01(<4 x double> %a, <4 x double> %b) nounwind readnone {
  %bincmp = fcmp olt <4 x double> %a, %b
  %s = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %s
}

declare void @scale() nounwind uwtable

; CHECK: vucomisd
define void @render() nounwind uwtable {
entry:
  br i1 undef, label %for.cond5, label %for.end52

for.cond5:
  %or.cond = and i1 undef, false
  br i1 %or.cond, label %for.body33, label %for.cond5

for.cond30:
  br i1 false, label %for.body33, label %for.cond5

for.body33:
  %tobool = fcmp une double undef, 0.000000e+00
  br i1 %tobool, label %if.then, label %for.cond30

if.then:
  call void @scale()
  br label %for.cond30

for.end52:
  ret void
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpgtd  %xmm
; CHECK-NEXT: vpcmpgtd  %xmm
; CHECK-NEXT: vinsertf128 $1
define <8 x i32> @int256-cmp(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp slt <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpgtq  %xmm
; CHECK-NEXT: vpcmpgtq  %xmm
; CHECK-NEXT: vinsertf128 $1
define <4 x i64> @v4i64-cmp(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %bincmp = icmp slt <4 x i64> %i, %j
  %x = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpgtw  %xmm
; CHECK-NEXT: vpcmpgtw  %xmm
; CHECK-NEXT: vinsertf128 $1
define <16 x i16> @v16i16-cmp(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp slt <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpgtb  %xmm
; CHECK-NEXT: vpcmpgtb  %xmm
; CHECK-NEXT: vinsertf128 $1
define <32 x i8> @v32i8-cmp(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp slt <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpeqd  %xmm
; CHECK-NEXT: vpcmpeqd  %xmm
; CHECK-NEXT: vinsertf128 $1
define <8 x i32> @int256-cmpeq(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %bincmp = icmp eq <8 x i32> %i, %j
  %x = sext <8 x i1> %bincmp to <8 x i32>
  ret <8 x i32> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpeqq  %xmm
; CHECK-NEXT: vpcmpeqq  %xmm
; CHECK-NEXT: vinsertf128 $1
define <4 x i64> @v4i64-cmpeq(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %bincmp = icmp eq <4 x i64> %i, %j
  %x = sext <4 x i1> %bincmp to <4 x i64>
  ret <4 x i64> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpeqw  %xmm
; CHECK-NEXT: vpcmpeqw  %xmm
; CHECK-NEXT: vinsertf128 $1
define <16 x i16> @v16i16-cmpeq(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %bincmp = icmp eq <16 x i16> %i, %j
  %x = sext <16 x i1> %bincmp to <16 x i16>
  ret <16 x i16> %x
}

; CHECK: vextractf128  $1
; CHECK: vextractf128  $1
; CHECK-NEXT: vpcmpeqb  %xmm
; CHECK-NEXT: vpcmpeqb  %xmm
; CHECK-NEXT: vinsertf128 $1
define <32 x i8> @v32i8-cmpeq(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %bincmp = icmp eq <32 x i8> %i, %j
  %x = sext <32 x i1> %bincmp to <32 x i8>
  ret <32 x i8> %x
}

;; Scalar comparison

; CHECK: scalarcmpA
; CHECK: vcmpeqsd
define i32 @scalarcmpA() uwtable ssp {
  %cmp29 = fcmp oeq double undef, 0.000000e+00
  %res = zext i1 %cmp29 to i32
  ret i32 %res
}

; CHECK: scalarcmpB
; CHECK: vcmpeqss
define i32 @scalarcmpB() uwtable ssp {
  %cmp29 = fcmp oeq float undef, 0.000000e+00
  %res = zext i1 %cmp29 to i32
  ret i32 %res
}

