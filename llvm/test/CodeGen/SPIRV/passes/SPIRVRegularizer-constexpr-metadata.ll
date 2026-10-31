; RUN: opt -S -passes=spirv-regularizer -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; Verify that a ConstantExpr wrapped in metadata (a call operand) is lowered
; to an instruction, matching the legacy pass behavior in runLowerConstExpr.

@g = addrspace(1) global i32 0

declare void @llvm.use_md(metadata)

define void @constexpr_in_metadata() {
; CHECK-LABEL: define void @constexpr_in_metadata(
; CHECK:    [[V:%.*]] = ptrtoint ptr addrspace(1) @g to i64
; CHECK:    call void @llvm.use_md(metadata i64 [[V]])
  call void @llvm.use_md(metadata i64 ptrtoint (ptr addrspace(1) @g to i64))
  ret void
}
