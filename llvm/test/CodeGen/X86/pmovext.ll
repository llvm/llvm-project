; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7 | FileCheck %s

; rdar://11897677

;CHECK-LABEL: intrin_pmov:
;CHECK: pmovzxbw  (%{{.*}}), %xmm0
;CHECK-NEXT: movdqu
;CHECK-NEXT: ret
define void @intrin_pmov(ptr noalias %dest, ptr noalias %src) nounwind uwtable ssp {
  %1 = load <2 x i64>, ptr %src, align 16
  %2 = bitcast <2 x i64> %1 to <16 x i8>
  %3 = tail call <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8> %2) nounwind
  %4 = bitcast <8 x i16> %3 to <16 x i8>
  tail call void @llvm.x86.sse2.storeu.dq(ptr %dest, <16 x i8> %4) nounwind
  ret void
}

declare <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8>) nounwind readnone
declare void @llvm.x86.sse2.storeu.dq(ptr, <16 x i8>) nounwind

; rdar://15245794

define <4 x i32> @foo0(double %v.coerce) nounwind ssp {
; CHECK-LABEL: foo0
; CHECK: pmovzxwd %xmm0, %xmm0
; CHECK-NEXT: ret
  %tmp = bitcast double %v.coerce to <4 x i16>
  %tmp1 = shufflevector <4 x i16> %tmp, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %tmp2 = tail call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> %tmp1) nounwind
  ret <4 x i32> %tmp2
}

define <8 x i16> @foo1(double %v.coerce) nounwind ssp {
; CHECK-LABEL: foo1
; CHECK: pmovzxbw %xmm0, %xmm0
; CHECK-NEXT: ret
  %tmp = bitcast double %v.coerce to <8 x i8>
  %tmp1 = shufflevector <8 x i8> %tmp, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %tmp2 = tail call <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8> %tmp1)
  ret <8 x i16> %tmp2
}

declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>) nounwind readnone
