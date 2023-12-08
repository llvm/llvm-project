; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

; CHECK-LABEL:   polly.stmt.bb48:
; CHECK-NEXT:   %[[offset:.*]] = shl i64 %polly.indvar, 3
; CHECK-NEXT:   %scevgep = getelementptr i8, ptr %A, i64 %[[offset]]
; CHECK-NEXT:   %tmp51_p_scalar_ = load i64, ptr %scevgep,
; CHECK-NEXT:   %p_tmp52 = and i64 %tmp51_p_scalar_, %tmp26
; CHECK-NEXT:   %p_tmp53 = icmp eq i64 %p_tmp52, %tmp26
; CHECK-NEXT:   store i64 42, ptr %scevgep, align 8
; CHECK-NEXT:   br i1 %p_tmp53, label %polly.stmt.bb54, label %polly.stmt.bb56.exit

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @quux(i32 %arg, i32 %arg1, ptr %A, i64 %tmp9, i64 %tmp24, i64 %tmp14, i64 %tmp22, i64 %tmp44) {
bb:
  %tmp26 = or i64 %tmp22, %tmp24
  br label %bb39

bb39:                                             ; preds = %bb39, %bb38
  %tmp45 = icmp eq i64 %tmp44, %tmp9
  br i1 %tmp45, label %bb46, label %bb81

bb46:                                             ; preds = %bb39
  %tmp47 = or i64 1, %tmp14
  br label %bb48

bb48:                                             ; preds = %bb56, %bb46
  %tmp49 = phi i64 [ 0, %bb46 ], [ %tmp57, %bb56 ]
  %tmp50 = getelementptr inbounds i64, ptr %A, i64 %tmp49
  %tmp51 = load i64, ptr %tmp50, align 8
  %tmp52 = and i64 %tmp51, %tmp26
  %tmp53 = icmp eq i64 %tmp52, %tmp26
  store i64 42, ptr %tmp50, align 8
  br i1 %tmp53, label %bb54, label %bb56

bb54:                                             ; preds = %bb48
  %tmp55 = xor i64 %tmp51, %tmp47
  store i64 %tmp55, ptr %tmp50, align 8
  br label %bb56

bb56:                                             ; preds = %bb54, %bb48
  %tmp57 = add nuw nsw i64 %tmp49, 1
  %tmp58 = icmp eq i64 %tmp57, %tmp9
  br i1 %tmp58, label %bb81, label %bb48

bb81:                                             ; preds = %bb74, %bb56
  ret void
}
