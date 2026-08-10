; Test the quadword comparison instructions that were added in POWER10.
;
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -mcpu=pwr10 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -mcpu=pwr10 -mattr=-vsx < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -mcpu=pwr10 < %s | FileCheck %s
define <1 x i128> @v1si128_cmp(<1 x i128> %x, <1 x i128> %y) nounwind readnone {
       %cmp = icmp eq <1 x i128> %x, %y
       %result = sext <1 x i1> %cmp to <1 x i128>
       ret <1 x i128> %result
; CHECK-LABEL: v1si128_cmp:
; CHECK: vcmpequq 2, 2, 3
}

define <2 x i128> @v2si128_cmp(<2 x i128> %x, <2 x i128> %y) nounwind readnone {
       %cmp = icmp eq <2 x i128> %x, %y
       %result = sext <2 x i1> %cmp to <2 x i128>
       ret <2 x i128> %result
; CHECK-LABEL: v2si128_cmp
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <4 x i128> @v4si128_cmp(<4 x i128> %x, <4 x i128> %y) nounwind readnone {
       %cmp = icmp eq <4 x i128> %x, %y
       %result = sext <4 x i1> %cmp to <4 x i128>
       ret <4 x i128> %result
; CHECK-LABEL: v4si128_cmp
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <8 x i128> @v8si128_cmp(<8 x i128> %x, <8 x i128> %y) nounwind readnone {
       %cmp = icmp eq <8 x i128> %x, %y
       %result = sext <8 x i1> %cmp to <8 x i128>
       ret <8 x i128> %result
; CHECK-LABEL: v8si128_cmp
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <16 x i128> @v16si128_cmp(<16 x i128> %x, <16 x i128> %y) nounwind readnone {
       %cmp = icmp eq <16 x i128> %x, %y
       %result = sext <16 x i1> %cmp to <16 x i128>
       ret <16 x i128> %result
; CHECK-LABEL: v16si128_cmp
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

; Greater than signed
define <1 x i128> @v1si128_cmp_gt(<1 x i128> %x, <1 x i128> %y) nounwind readnone {
       %cmp = icmp sgt <1 x i128> %x, %y
       %result = sext <1 x i1> %cmp to <1 x i128>
       ret <1 x i128> %result
; CHECK-LABEL: v1si128_cmp_gt
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <2 x i128> @v2si128_cmp_gt(<2 x i128> %x, <2 x i128> %y) nounwind readnone {
       %cmp = icmp sgt <2 x i128> %x, %y
       %result = sext <2 x i1> %cmp to <2 x i128>
       ret <2 x i128> %result
; CHECK-LABEL: v2si128_cmp_gt
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <4 x i128> @v4si128_cmp_gt(<4 x i128> %x, <4 x i128> %y) nounwind readnone {
       %cmp = icmp sgt <4 x i128> %x, %y
       %result = sext <4 x i1> %cmp to <4 x i128>
       ret <4 x i128> %result
; CHECK-LABEL: v4si128_cmp_gt
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <8 x i128> @v8si128_cmp_gt(<8 x i128> %x, <8 x i128> %y) nounwind readnone {
       %cmp = icmp sgt <8 x i128> %x, %y
       %result = sext <8 x i1> %cmp to <8 x i128>
       ret <8 x i128> %result
; CHECK-LABEL: v8si128_cmp_gt
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <16 x i128> @v16si128_cmp_gt(<16 x i128> %x, <16 x i128> %y) nounwind readnone {
       %cmp = icmp sgt <16 x i128> %x, %y
       %result = sext <16 x i1> %cmp to <16 x i128>
       ret <16 x i128> %result
; CHECK-LABEL: v16si128_cmp_gt
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

; Greater than unsigned
define <1 x i128> @v1ui128_cmp_gt(<1 x i128> %x, <1 x i128> %y) nounwind readnone {
       %cmp = icmp ugt <1 x i128> %x, %y
       %result = sext <1 x i1> %cmp to <1 x i128>
       ret <1 x i128> %result
; CHECK-LABEL: v1ui128_cmp_gt
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <2 x i128> @v2ui128_cmp_gt(<2 x i128> %x, <2 x i128> %y) nounwind readnone {
       %cmp = icmp ugt <2 x i128> %x, %y
       %result = sext <2 x i1> %cmp to <2 x i128>
       ret <2 x i128> %result
; CHECK-LABEL: v2ui128_cmp_gt
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <4 x i128> @v4ui128_cmp_gt(<4 x i128> %x, <4 x i128> %y) nounwind readnone {
       %cmp = icmp ugt <4 x i128> %x, %y
       %result = sext <4 x i1> %cmp to <4 x i128>
       ret <4 x i128> %result
; CHECK-LABEL: v4ui128_cmp_gt
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <8 x i128> @v8ui128_cmp_gt(<8 x i128> %x, <8 x i128> %y) nounwind readnone {
       %cmp = icmp ugt <8 x i128> %x, %y
       %result = sext <8 x i1> %cmp to <8 x i128>
       ret <8 x i128> %result
; CHECK-LABEL: v8ui128_cmp_gt
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <16 x i128> @v16ui128_cmp_gt(<16 x i128> %x, <16 x i128> %y) nounwind readnone {
       %cmp = icmp ugt <16 x i128> %x, %y
       %result = sext <16 x i1> %cmp to <16 x i128>
       ret <16 x i128> %result
; CHECK-LABEL: v16ui128_cmp_gt
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

; Check the intrinsics also
declare <1 x i128> @llvm.ppc.altivec.vcmpequq(<1 x i128>, <1 x i128>) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vcmpgtsq(<1 x i128>, <1 x i128>) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vcmpgtuq(<1 x i128>, <1 x i128>) nounwind readnone

define <1 x i128> @test_vcmpequq(<1 x i128> %x, <1 x i128> %y) {
       %tmp = tail call <1 x i128> @llvm.ppc.altivec.vcmpequq(<1 x i128> %x, <1 x i128> %y)
       ret <1 x i128> %tmp
; CHECK-LABEL: test_vcmpequq:
; CHECK: vcmpequq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <1 x i128> @test_vcmpgtsq(<1 x i128> %x, <1 x i128> %y) {
       %tmp = tail call <1 x i128> @llvm.ppc.altivec.vcmpgtsq(<1 x i128> %x, <1 x i128> %y)
       ret <1 x i128> %tmp
; CHECK-LABEL: test_vcmpgtsq
; CHECK: vcmpgtsq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define <1 x i128> @test_vcmpgtuq(<1 x i128> %x, <1 x i128> %y) {
       %tmp = tail call <1 x i128> @llvm.ppc.altivec.vcmpgtuq(<1 x i128> %x, <1 x i128> %y)
       ret <1 x i128> %tmp
; CHECK-LABEL: test_vcmpgtuq
; CHECK: vcmpgtuq {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

declare i32 @llvm.ppc.altivec.vcmpequq.p(i32, <1 x i128>, <1 x i128>) nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpgtsq.p(i32, <1 x i128>, <1 x i128>) nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpgtuq.p(i32, <1 x i128>, <1 x i128>) nounwind readnone

define i32 @test_vcmpequq_p(<1 x i128> %x, <1 x i128> %y) {
      %tmp = tail call i32 @llvm.ppc.altivec.vcmpequq.p(i32 2, <1 x i128> %x, <1 x i128> %y)
      ret i32 %tmp
; CHECK-LABEL: test_vcmpequq_p:
; CHECK: vcmpequq. {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define i32 @test_vcmpgtsq_p(<1 x i128> %x, <1 x i128> %y) {
      %tmp = tail call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 2, <1 x i128> %x, <1 x i128> %y)
      ret i32 %tmp
; CHECK-LABEL: test_vcmpgtsq_p
; CHECK: vcmpgtsq. {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}

define i32 @test_vcmpgtuq_p(<1 x i128> %x, <1 x i128> %y) {
      %tmp = tail call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 2, <1 x i128> %x, <1 x i128> %y)
      ret i32 %tmp
; CHECK-LABEL: test_vcmpgtuq_p
; CHECK: vcmpgtuq. {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
}
