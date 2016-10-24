; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -mcpu=pwr8 < %s | FileCheck %s

%class.PB2 = type { [1 x i32], %class.PB1* }
%class.PB1 = type { [1 x i32], i64, i64, i32 }

; Function Attrs: norecurse nounwind readonly
define zeroext i1 @foo(%class.PB2* %s_a, %class.PB2* %s_b) local_unnamed_addr {
entry:
  %arrayidx.i6 = bitcast %class.PB2* %s_a to i32*
  %0 = load i32, i32* %arrayidx.i6, align 8, !tbaa !1
  %and.i = and i32 %0, 8
  %cmp.i = icmp ne i32 %and.i, 0
  %arrayidx.i37 = bitcast %class.PB2* %s_b to i32*
  %1 = load i32, i32* %arrayidx.i37, align 8, !tbaa !1
  %and.i4 = and i32 %1, 8
  %cmp.i5 = icmp ne i32 %and.i4, 0
  %cmp = xor i1 %cmp.i, %cmp.i5
  ret i1 %cmp
; CHECK-LABEL: @foo
; CHECK: rldicl  {{[0-9]+}}, {{[0-9]+}}, 61, 63

}

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}

