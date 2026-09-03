; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare i16 @llvm.nvvm.neg.bf16(i16)
declare i32 @llvm.nvvm.neg.bf16x2(i32)

define i16 @upgrade_neg_bf16(i16 %x) {
; CHECK-LABEL: define i16 @upgrade_neg_bf16(
; CHECK: [[ARG:%.*]] = bitcast i16 %x to bfloat
; CHECK: [[CALL:%.*]] = call bfloat @llvm.nvvm.neg.bf16(bfloat [[ARG]])
; CHECK: [[RET:%.*]] = bitcast bfloat [[CALL]] to i16
; CHECK: ret i16 [[RET]]
  %t = call i16 @llvm.nvvm.neg.bf16(i16 %x)
  ret i16 %t
}

define i32 @upgrade_neg_bf16x2(i32 %x) {
; CHECK-LABEL: define i32 @upgrade_neg_bf16x2(
; CHECK: [[ARG:%.*]] = bitcast i32 %x to <2 x bfloat>
; CHECK: [[CALL:%.*]] = call <2 x bfloat> @llvm.nvvm.neg.bf16x2(<2 x bfloat> [[ARG]])
; CHECK: [[RET:%.*]] = bitcast <2 x bfloat> [[CALL]] to i32
; CHECK: ret i32 [[RET]]
  %t = call i32 @llvm.nvvm.neg.bf16x2(i32 %x)
  ret i32 %t
}
