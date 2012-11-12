; RUN: opt -S -tbaa -basicaa -gvn < %s | FileCheck -check-prefix=BASICAA %s
; RUN: opt -S -tbaa -gvn < %s | FileCheck %s
; rdar://8875631, rdar://8875069

; BasicAA should notice that the store stores to the entire %u object,
; so the %tmp5 load is PartialAlias with the store and suppress TBAA.
; Without BasicAA, TBAA should say that %tmp5 is NoAlias with the store.

target datalayout = "e-p:64:64:64"

%union.anon = type { double }

@u = global %union.anon { double -2.500000e-01 }, align 8
@endianness_test = global i64 1, align 8

define i32 @signbit(double %x) nounwind {
; BASICAA: ret i32 %tmp5.lobit
; CHECK:   ret i32 0
entry:
  %u = alloca %union.anon, align 8
  %tmp9 = getelementptr inbounds %union.anon* %u, i64 0, i32 0
  store double %x, double* %tmp9, align 8, !tbaa !0
  %tmp2 = load i32* bitcast (i64* @endianness_test to i32*), align 8, !tbaa !3
  %idxprom = sext i32 %tmp2 to i64
  %tmp4 = bitcast %union.anon* %u to [2 x i32]*
  %arrayidx = getelementptr inbounds [2 x i32]* %tmp4, i64 0, i64 %idxprom
  %tmp5 = load i32* %arrayidx, align 4, !tbaa !3
  %tmp5.lobit = lshr i32 %tmp5, 31
  ret i32 %tmp5.lobit
}

!0 = metadata !{metadata !"double", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"int", metadata !1}
