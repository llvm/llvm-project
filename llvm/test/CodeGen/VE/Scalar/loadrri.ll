; RUN: llc < %s -mtriple=ve | FileCheck %s

%struct.data = type { [4 x i8] }

;;; Check basic usage of rri format load instructions.
;;; Our target is DAG selection mechanism for LD1BSXrri.
;;; We prepared following three styles.
;;;   1. LD1BSXrri with %reg1 + %reg2
;;;   2. LD1BSXrri with %frame-index + %reg
;;;   3. LD1BSXrri with %reg + %frame-index

; Function Attrs: norecurse nounwind readonly
define signext i8 @func_rr(ptr nocapture readonly %0, i32 signext %1) {
; CHECK-LABEL: func_rr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s1, %s1, 2
; CHECK-NEXT:    ld1b.sx %s0, (%s1, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sext i32 %1 to i64
  %4 = getelementptr inbounds %struct.data, ptr %0, i64 %3, i32 0, i64 0
  %5 = load i8, ptr %4, align 1
  ret i8 %5
}

; Function Attrs: nounwind
define signext i8 @func_fr(ptr readonly %0, i32 signext %1) {
; CHECK-LABEL: func_fr:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s1, %s1, 2
; CHECK-NEXT:    ldl.sx %s0, (%s1, %s0)
; CHECK-NEXT:    stl %s0, 8(%s1, %s11)
; CHECK-NEXT:    ld1b.sx %s0, 8(%s1, %s11)
; CHECK-NEXT:    adds.l %s11, 48, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = alloca [10 x %struct.data], align 1
  call void @llvm.lifetime.start.p0(i64 40, ptr nonnull %3)
  %4 = sext i32 %1 to i64
  %5 = getelementptr inbounds [10 x %struct.data], ptr %3, i64 0, i64 %4, i32 0, i64 0
  %6 = getelementptr inbounds %struct.data, ptr %0, i64 %4, i32 0, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %5, ptr align 1 %6, i64 4, i1 true)
  %7 = load volatile i8, ptr %5, align 1
  call void @llvm.lifetime.end.p0(i64 40, ptr nonnull %3)
  ret i8 %7
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

%"basic_string" = type { %union.anon.3, [23 x i8] }
%union.anon.3 = type { i8 }

define signext i8 @func_rf(ptr readonly %0, i64 %1, i32 signext %2) {
; CHECK-LABEL: func_rf:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 8(%s1, %s11)
; CHECK-NEXT:    adds.l %s11, 32, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %buf = alloca %"basic_string", align 8

  %sub631 = add nsw i64 %1, -1
  %add.ptr.i = getelementptr inbounds %"basic_string", ptr %buf, i64 0, i32 1, i64 %sub631
  %ret = load i8, ptr %add.ptr.i, align 1
  ret i8 %ret
}
