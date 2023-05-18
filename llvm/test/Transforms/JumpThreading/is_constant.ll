; RUN: opt -S -passes=jump-threading,verify < %s | FileCheck %s

; This is reduced from the Linux net/ipv4/tcp.c file built with ASAN. We
; don't want jump threading to split up a basic block which has a PHI node with
; at least one constant incoming value, whose value is used by an is.constant
; intrinsic with non-local uses. It could lead to later passes no DCE'ing
; invalid paths.

; CHECK-LABEL:    define void @test1(
; CHECK-LABEL:    bb_cond:
; CHECK-NOT:        %sext = phi i64 [ %var, %entry ]
; CHECK-NEXT:       %sext = phi i64 [ 24, %bb_constant ], [ %var, %entry ]
; CHECK-NEXT:       %cond2 = icmp
; CHECK-NEXT:       call i1 @llvm.is.constant.i64(

define void @test1(i32 %a, i64 %var) {
entry:
  %cond1 = icmp ugt i32 %a, 24
  br i1 %cond1, label %bb_constant, label %bb_cond

bb_constant:
  br label %bb_cond

bb_cond:
  %sext = phi i64 [ 24, %bb_constant ], [ %var, %entry ]
  %cond2 = icmp ugt i64 %sext, 24
  %is_constant = call i1 @llvm.is.constant.i64(i64 %sext)
  br i1 %cond2, label %bb_then, label %bb_else

bb_then:
  unreachable

bb_else:
  unreachable
}

; Function Attrs: nounwind readnone willreturn
declare i1 @llvm.is.constant.i64(i64) #0

attributes #0 = { convergent nounwind readnone willreturn }
