; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

; rdar://11897677

;CHECK: intrin_pmov
;CHECK: pmovzxbw  (%{{.*}}), %xmm0
;CHECK-NEXT: movdqu
;CHECK-NEXT: ret
define void @intrin_pmov(i16* noalias %dest, i8* noalias %src) nounwind uwtable ssp {
  %1 = bitcast i8* %src to <2 x i64>*
  %2 = load <2 x i64>* %1, align 16
  %3 = bitcast <2 x i64> %2 to <16 x i8>
  %4 = tail call <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8> %3) nounwind
  %5 = bitcast i16* %dest to i8*
  %6 = bitcast <8 x i16> %4 to <16 x i8>
  tail call void @llvm.x86.sse2.storeu.dq(i8* %5, <16 x i8> %6) nounwind
  ret void
}

declare <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8>) nounwind readnone

declare void @llvm.x86.sse2.storeu.dq(i8*, <16 x i8>) nounwind
