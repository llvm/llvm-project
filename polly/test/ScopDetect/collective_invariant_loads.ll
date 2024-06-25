; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting -disable-output< %s 2>&1 | FileCheck %s

;CHECK:     Function: test_init_chpl
;CHECK-NEXT:     Region: %bb1---%bb16
;CHECK-NEXT:     Max Loop Depth:  2
;CHECK-NEXT:     Invariant Accesses: {
;CHECK-NEXT:             ReadAccess := [Reduction Type: NONE] [Scalar: 0]
;CHECK-NEXT:                 [tmp5] -> { Stmt_bb2[i0, i1] -> MemRef_arg[1] };
;CHECK-NEXT:             Execution Context: [tmp5] -> {  :  }
;CHECK-NEXT:             ReadAccess := [Reduction Type: NONE] [Scalar: 0]
;CHECK-NEXT:                 [tmp5] -> { Stmt_bb2[i0, i1] -> MemRef_tmp3[9] };
;CHECK-NEXT:             Execution Context: [tmp5] -> {  :  }
;CHECK-NEXT:             ReadAccess := [Reduction Type: NONE] [Scalar: 0]
;CHECK-NEXT:                 [tmp5] -> { Stmt_bb2[i0, i1] -> MemRef_tmp3[2] };
;CHECK-NEXT:             Execution Context: [tmp5] -> {  :  }
;CHECK-NEXT:     }


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%array_ty = type { i64, ptr, i8 }
%array_ptr = type { [2 x i64], [2 x i64], [2 x i64], i64, i64, ptr, ptr, i8 }

; Function Attrs: noinline
define weak dso_local void @test_init_chpl(ptr nonnull %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb14, %bb
  %.0 = phi i64 [ 0, %bb ], [ %tmp15, %bb14 ]
  br label %bb2

bb2:                                              ; preds = %bb2, %bb1
  %.01 = phi i64 [ 0, %bb1 ], [ %tmp13, %bb2 ]
  %tmp = getelementptr inbounds %array_ty, ptr %arg, i64 0, i32 1
  %tmp3 = load ptr, ptr %tmp, align 8
  %tmp4 = getelementptr inbounds %array_ptr, ptr %tmp3, i64 0, i32 1, i64 0
  %tmp5 = load i64, ptr %tmp4, align 8
  %tmp6 = mul nsw i64 %tmp5, %.0
  %tmp7 = add nsw i64 %tmp6, %.01
  %tmp8 = getelementptr inbounds %array_ptr, ptr %tmp3, i64 0, i32 6
  %tmp9 = load ptr, ptr %tmp8, align 8
  %tmp10 = getelementptr inbounds double, ptr %tmp9, i64 %tmp7
  store double 13.0, ptr %tmp10, align 8
  %tmp13 = add nuw nsw i64 %.01, 1
  %exitcond = icmp ne i64 %tmp13, 1000
  br i1 %exitcond, label %bb2, label %bb14

bb14:                                             ; preds = %bb2
  %tmp15 = add nuw nsw i64 %.0, 1
  %exitcond8 = icmp ne i64 %tmp15, 1000
  br i1 %exitcond8, label %bb1, label %bb16

bb16:                                             ; preds = %bb14
  ret void
}
