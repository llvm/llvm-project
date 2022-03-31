; RUN: llc < %s -march=nvptx | FileCheck %s


declare i64 @llvm.nvvm.rotate.b64(i64, i32)
declare i64 @llvm.nvvm.rotate.right.b64(i64, i32)

; CHECK: rotate64
define i64 @rotate64(i64 %a, i32 %b) {
; CHECK: shl.b64         [[LHS:%.*]], [[RD1:%.*]], 3;
; CHECK: shr.b64         [[RHS:%.*]], [[RD1]], 61;
; CHECK: add.u64         [[RD2:%.*]], [[LHS]], [[RHS]];
; CHECK: ret
  %val = tail call i64 @llvm.nvvm.rotate.b64(i64 %a, i32 3)
  ret i64 %val
}

; CHECK: rotateright64
define i64 @rotateright64(i64 %a, i32 %b) {
; CHECK: shl.b64         [[LHS:%.*]], [[RD1:%.*]], 61;
; CHECK: shr.b64         [[RHS:%.*]], [[RD1]], 3;
; CHECK: add.u64         [[RD2:%.*]], [[LHS]], [[RHS]];
; CHECK: ret
  %val = tail call i64 @llvm.nvvm.rotate.right.b64(i64 %a, i32 3)
  ret i64 %val
}
