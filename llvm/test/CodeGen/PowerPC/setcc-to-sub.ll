; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN: -mcpu=pwr8 < %s | FileCheck %s

%class.PB2 = type { [1 x i32], %class.PB1* }
%class.PB1 = type { [1 x i32], i64, i64, i32 }

; Function Attrs: norecurse nounwind readonly
define zeroext i1 @test1(%class.PB2* %s_a, %class.PB2* %s_b) local_unnamed_addr #0 {
entry:
  %arrayidx.i6 = bitcast %class.PB2* %s_a to i32*
  %0 = load i32, i32* %arrayidx.i6, align 8, !tbaa !1
  %and.i = and i32 %0, 8
  %arrayidx.i37 = bitcast %class.PB2* %s_b to i32*
  %1 = load i32, i32* %arrayidx.i37, align 8, !tbaa !1
  %and.i4 = and i32 %1, 8
  %cmp.i5 = icmp ult i32 %and.i, %and.i4
  ret i1 %cmp.i5

; CHECK-LABEL: @test1
; CHECK: rlwinm [[REG1:[0-9]*]]
; CHECK-NEXT: rlwinm [[REG2:[0-9]*]]
; CHECK-NEXT: sub [[REG3:[0-9]*]], [[REG1]], [[REG2]]
; CHECK-NEXT: rldicl 3, [[REG3]]
; CHECK: blr

}

; Function Attrs: norecurse nounwind readonly
define zeroext i1 @test2(%class.PB2* %s_a, %class.PB2* %s_b) local_unnamed_addr #0 {
entry:
  %arrayidx.i6 = bitcast %class.PB2* %s_a to i32*
  %0 = load i32, i32* %arrayidx.i6, align 8, !tbaa !1
  %and.i = and i32 %0, 8
  %arrayidx.i37 = bitcast %class.PB2* %s_b to i32*
  %1 = load i32, i32* %arrayidx.i37, align 8, !tbaa !1
  %and.i4 = and i32 %1, 8
  %cmp.i5 = icmp ule i32 %and.i, %and.i4
  ret i1 %cmp.i5

; CHECK-LABEL: @test2
; CHECK: rlwinm [[REG1:[0-9]*]]
; CHECK-NEXT: rlwinm [[REG2:[0-9]*]]
; CHECK-NEXT: sub [[REG3:[0-9]*]], [[REG2]], [[REG1]]
; CHECK-NEXT: rldicl [[REG4:[0-9]*]], [[REG3]]
; CHECK-NEXT: xori 3, [[REG4]], 1
; CHECK: blr

}

; Function Attrs: norecurse nounwind readonly
define zeroext i1 @test3(%class.PB2* %s_a, %class.PB2* %s_b) local_unnamed_addr #0 {
entry:
  %arrayidx.i6 = bitcast %class.PB2* %s_a to i32*
  %0 = load i32, i32* %arrayidx.i6, align 8, !tbaa !1
  %and.i = and i32 %0, 8
  %arrayidx.i37 = bitcast %class.PB2* %s_b to i32*
  %1 = load i32, i32* %arrayidx.i37, align 8, !tbaa !1
  %and.i4 = and i32 %1, 8
  %cmp.i5 = icmp ugt i32 %and.i, %and.i4
  ret i1 %cmp.i5

; CHECK-LABEL: @test3
; CHECK: rlwinm [[REG1:[0-9]*]]
; CHECK-NEXT: rlwinm [[REG2:[0-9]*]]
; CHECK-NEXT: sub [[REG3:[0-9]*]], [[REG2]], [[REG1]]
; CHECK-NEXT: rldicl 3, [[REG3]]
; CHECK: blr

}

; Function Attrs: norecurse nounwind readonly
define zeroext i1 @test4(%class.PB2* %s_a, %class.PB2* %s_b) local_unnamed_addr #0 {
entry:
  %arrayidx.i6 = bitcast %class.PB2* %s_a to i32*
  %0 = load i32, i32* %arrayidx.i6, align 8, !tbaa !1
  %and.i = and i32 %0, 8
  %arrayidx.i37 = bitcast %class.PB2* %s_b to i32*
  %1 = load i32, i32* %arrayidx.i37, align 8, !tbaa !1
  %and.i4 = and i32 %1, 8
  %cmp.i5 = icmp uge i32 %and.i, %and.i4
  ret i1 %cmp.i5

; CHECK-LABEL: @test4
; CHECK: rlwinm [[REG1:[0-9]*]]
; CHECK-NEXT: rlwinm [[REG2:[0-9]*]]
; CHECK-NEXT: sub [[REG3:[0-9]*]], [[REG1]], [[REG2]]
; CHECK-NEXT: rldicl [[REG4:[0-9]*]], [[REG3]]
; CHECK-NEXT: xori 3, [[REG4]], 1
; CHECK: blr

}

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
