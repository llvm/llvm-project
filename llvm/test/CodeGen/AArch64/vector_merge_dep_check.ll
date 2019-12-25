; RUN: llc < %s | FileCheck %s

; This test checks that we do not merge stores together which have
; dependencies through their non-chain operands (e.g. one store is the
; chain ancestor of a load whose value is used in as the data for the
; other store). Merging in such cases creates a loop in the DAG.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

%"class.std::__1::complex.0.20.56.60.64.72.76.88.92.112.140.248" = type { float, float }

; Function Attrs: noinline norecurse nounwind ssp uwtable
define void @fn(<2 x i64>* %argA, <2 x i64>* %argB, i64* %a) #0 align 2 {
  %_p_vec_full = load <2 x i64>, <2 x i64>* %argA, align 4, !alias.scope !1, !noalias !3
  %x = extractelement <2 x i64> %_p_vec_full, i32 1
  store i64 %x, i64* %a, align 8, !alias.scope !4, !noalias !9
  %_p_vec_full155 = load <2 x i64>, <2 x i64>* %argB, align 4, !alias.scope !1, !noalias !3
  %y = extractelement <2 x i64> %_p_vec_full155, i32 0
  %scevgep41 = getelementptr i64, i64* %a, i64 -1
  store i64 %y, i64* %scevgep41, align 8, !alias.scope !4, !noalias !9
  ret void
}

; CHECK: ret

attributes #0 = { noinline norecurse nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "polly-optimized" "stack-protector-buffer-size"="8" "target-features"="+crc,+crypto,+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"Snapdragon LLVM ARM Compiler 3.8.0 (based on LLVM 3.8.0)"}
!1 = distinct !{!1, !2, !"polly.alias.scope.rhs"}
!2 = distinct !{!2, !"polly.alias.scope.domain"}
!3 = !{!4, !5, !6, !7, !8}
!4 = distinct !{!4, !2, !"polly.alias.scope.blockB"}
!5 = distinct !{!5, !2, !"polly.alias.scope.add28.lcssa.reg2mem"}
!6 = distinct !{!6, !2, !"polly.alias.scope.count.0.lcssa.reg2mem"}
!7 = distinct !{!7, !2, !"polly.alias.scope.mul"}
!8 = distinct !{!8, !2, !"polly.alias.scope.add28.us.lcssa.reg2mem"}
!9 = !{!1, !5, !6, !7, !8}
