; RUN: llc < %s -mtriple=ve | FileCheck %s

%struct.data = type { [4 x i8] }

;;; Check basic usage of rri format load instructions.
;;; Our target is DAG selection mechanism for LD1BSXrri.
;;; We prepared following three styles.
;;;   1. LD1BSXrri with %reg1 + %reg2
;;;   2. LD1BSXrri with %frame-index + %reg
;;;   3. LD1BSXrri with %reg + %frame-index

; Function Attrs: norecurse nounwind readonly
define signext i8 @func_rr(%struct.data* nocapture readonly %0, i32 signext %1) {
; CHECK-LABEL: func_rr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sll %s1, %s1, 2
; CHECK-NEXT:    ld1b.sx %s0, (%s1, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = sext i32 %1 to i64
  %4 = getelementptr inbounds %struct.data, %struct.data* %0, i64 %3, i32 0, i64 0
  %5 = load i8, i8* %4, align 1
  ret i8 %5
}

; Function Attrs: nounwind
define signext i8 @func_fr(%struct.data* readonly %0, i32 signext %1) {
; CHECK-LABEL: func_fr:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sll %s1, %s1, 2
; CHECK-NEXT:    ldl.sx %s0, (%s1, %s0)
; CHECK-NEXT:    stl %s0, 184(%s1, %s11)
; CHECK-NEXT:    ld1b.sx %s0, 184(%s1, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca [10 x %struct.data], align 1
  %4 = getelementptr inbounds [10 x %struct.data], [10 x %struct.data]* %3, i64 0, i64 0, i32 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %4)
  %5 = sext i32 %1 to i64
  %6 = getelementptr inbounds [10 x %struct.data], [10 x %struct.data]* %3, i64 0, i64 %5, i32 0, i64 0
  %7 = getelementptr inbounds %struct.data, %struct.data* %0, i64 %5, i32 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 %6, i8* align 1 %7, i64 4, i1 true)
  %8 = load volatile i8, i8* %6, align 1
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %4)
  ret i8 %8
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

%"basic_string" = type { %union.anon.3, [23 x i8] }
%union.anon.3 = type { i8 }

define signext i8 @func_rf(i8* readonly %0, i64 %1, i32 signext %2) {
; CHECK-LABEL: func_rf:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld1b.sx %s0, 184(%s1, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %buf = alloca %"basic_string", align 8

  %sub631 = add nsw i64 %1, -1
  %add.ptr.i = getelementptr inbounds %"basic_string", %"basic_string"* %buf, i64 0, i32 1, i64 %sub631
  %ret = load i8, i8* %add.ptr.i, align 1
  ret i8 %ret
}
