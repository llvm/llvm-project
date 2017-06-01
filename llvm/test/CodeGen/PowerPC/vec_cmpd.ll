; Test the doubleword comparison instructions that were added in POWER8
;
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s

define <2 x i64> @v2si64_cmp(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
       %cmp = icmp eq <2 x i64> %x, %y
       %result = sext <2 x i1> %cmp to <2 x i64>
       ret <2 x i64> %result
; CHECK-LABEL: v2si64_cmp:
; CHECK: vcmpequd 2, 2, 3
}

define <4 x i64> @v4si64_cmp(<4 x i64> %x, <4 x i64> %y) nounwind readnone {
       %cmp = icmp eq <4 x i64> %x, %y
       %result = sext <4 x i1> %cmp to <4 x i64>
       ret <4 x i64> %result
; CHECK-LABEL: v4si64_cmp
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <8 x i64> @v8si64_cmp(<8 x i64> %x, <8 x i64> %y) nounwind readnone {
       %cmp = icmp eq <8 x i64> %x, %y
       %result = sext <8 x i1> %cmp to <8 x i64>
       ret <8 x i64> %result
; CHECK-LABEL: v8si64_cmp
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <16 x i64> @v16si64_cmp(<16 x i64> %x, <16 x i64> %y) nounwind readnone {
       %cmp = icmp eq <16 x i64> %x, %y
       %result = sext <16 x i1> %cmp to <16 x i64>
       ret <16 x i64> %result
; CHECK-LABEL: v16si64_cmp
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <32 x i64> @v32si64_cmp(<32 x i64> %x, <32 x i64> %y) nounwind readnone {
       %cmp = icmp eq <32 x i64> %x, %y
       %result = sext <32 x i1> %cmp to <32 x i64>
       ret <32 x i64> %result
; CHECK-LABEL: v32si64_cmp
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Greater than signed
define <2 x i64> @v2si64_cmp_gt(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
       %cmp = icmp sgt <2 x i64> %x, %y
       %result = sext <2 x i1> %cmp to <2 x i64>
       ret <2 x i64> %result
; CHECK-LABEL: v2si64_cmp_gt
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <4 x i64> @v4si64_cmp_gt(<4 x i64> %x, <4 x i64> %y) nounwind readnone {
       %cmp = icmp sgt <4 x i64> %x, %y
       %result = sext <4 x i1> %cmp to <4 x i64>
       ret <4 x i64> %result
; CHECK-LABEL: v4si64_cmp_gt
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <8 x i64> @v8si64_cmp_gt(<8 x i64> %x, <8 x i64> %y) nounwind readnone {
       %cmp = icmp sgt <8 x i64> %x, %y
       %result = sext <8 x i1> %cmp to <8 x i64>
       ret <8 x i64> %result
; CHECK-LABEL: v8si64_cmp_gt
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <16 x i64> @v16si64_cmp_gt(<16 x i64> %x, <16 x i64> %y) nounwind readnone {
       %cmp = icmp sgt <16 x i64> %x, %y
       %result = sext <16 x i1> %cmp to <16 x i64>
       ret <16 x i64> %result
; CHECK-LABEL: v16si64_cmp_gt
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <32 x i64> @v32si64_cmp_gt(<32 x i64> %x, <32 x i64> %y) nounwind readnone {
       %cmp = icmp sgt <32 x i64> %x, %y
       %result = sext <32 x i1> %cmp to <32 x i64>
       ret <32 x i64> %result
; CHECK-LABEL: v32si64_cmp_gt
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Greater than unsigned
define <2 x i64> @v2ui64_cmp_gt(<2 x i64> %x, <2 x i64> %y) nounwind readnone {
       %cmp = icmp ugt <2 x i64> %x, %y
       %result = sext <2 x i1> %cmp to <2 x i64>
       ret <2 x i64> %result
; CHECK-LABEL: v2ui64_cmp_gt
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <4 x i64> @v4ui64_cmp_gt(<4 x i64> %x, <4 x i64> %y) nounwind readnone {
       %cmp = icmp ugt <4 x i64> %x, %y
       %result = sext <4 x i1> %cmp to <4 x i64>
       ret <4 x i64> %result
; CHECK-LABEL: v4ui64_cmp_gt
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <8 x i64> @v8ui64_cmp_gt(<8 x i64> %x, <8 x i64> %y) nounwind readnone {
       %cmp = icmp ugt <8 x i64> %x, %y
       %result = sext <8 x i1> %cmp to <8 x i64>
       ret <8 x i64> %result
; CHECK-LABEL: v8ui64_cmp_gt
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <16 x i64> @v16ui64_cmp_gt(<16 x i64> %x, <16 x i64> %y) nounwind readnone {
       %cmp = icmp ugt <16 x i64> %x, %y
       %result = sext <16 x i1> %cmp to <16 x i64>
       ret <16 x i64> %result
; CHECK-LABEL: v16ui64_cmp_gt
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <32 x i64> @v32ui64_cmp_gt(<32 x i64> %x, <32 x i64> %y) nounwind readnone {
       %cmp = icmp ugt <32 x i64> %x, %y
       %result = sext <32 x i1> %cmp to <32 x i64>
       ret <32 x i64> %result
; CHECK-LABEL: v32ui64_cmp_gt
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Check the intrinsics also
declare <2 x i64> @llvm.ppc.altivec.vcmpequd(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpequd.p(i32, <2 x i64>, <2 x i64>) nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpgtsd.p(i32, <2 x i64>, <2 x i64>) nounwind readnone
declare <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64>, <2 x i64>) nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpgtud.p(i32, <2 x i64>, <2 x i64>) nounwind readnone

define <2 x i64> @test_vcmpequd(<2 x i64> %x, <2 x i64> %y) {
       %tmp = tail call <2 x i64> @llvm.ppc.altivec.vcmpequd(<2 x i64> %x, <2 x i64> %y)
       ret <2 x i64> %tmp
; CHECK-LABEL: test_vcmpequd:
; CHECK: vcmpequd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define i32 @test_vcmpequd_p(<2 x i64> %x, <2 x i64> %y) {
       %tmp = tail call i32 @llvm.ppc.altivec.vcmpequd.p(i32 2, <2 x i64> %x, <2 x i64> %y)
       ret i32 %tmp
; CHECK-LABEL: test_vcmpequd_p:
; CHECK: vcmpequd. {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <2 x i64> @test_vcmpgtsd(<2 x i64> %x, <2 x i64> %y) {
       %tmp = tail call <2 x i64> @llvm.ppc.altivec.vcmpgtsd(<2 x i64> %x, <2 x i64> %y)
       ret <2 x i64> %tmp
; CHECK-LABEL: test_vcmpgtsd
; CHECK: vcmpgtsd {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define i32 @test_vcmpgtsd_p(<2 x i64> %x, <2 x i64> %y) {
       %tmp = tail call i32 @llvm.ppc.altivec.vcmpgtsd.p(i32 2, <2 x i64> %x, <2 x i64> %y)
       ret i32 %tmp
; CHECK-LABEL: test_vcmpgtsd_p
; CHECK: vcmpgtsd. {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define <2 x i64> @test_vcmpgtud(<2 x i64> %x, <2 x i64> %y) {
       %tmp = tail call <2 x i64> @llvm.ppc.altivec.vcmpgtud(<2 x i64> %x, <2 x i64> %y)
       ret <2 x i64> %tmp
; CHECK-LABEL: test_vcmpgtud
; CHECK: vcmpgtud {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

define i32 @test_vcmpgtud_p(<2 x i64> %x, <2 x i64> %y) {
       %tmp = tail call i32 @llvm.ppc.altivec.vcmpgtud.p(i32 2, <2 x i64> %x, <2 x i64> %y)
       ret i32 %tmp
; CHECK-LABEL: test_vcmpgtud_p
; CHECK: vcmpgtud. {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}




