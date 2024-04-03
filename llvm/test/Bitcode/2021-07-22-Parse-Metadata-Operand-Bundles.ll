; This test ensures that we parse metadata operand bundle values.
; RUN: llvm-as < %s

declare void @callee()

define void @call_with_operand_bundle() {
  call void @callee() [ "op_type"(metadata !"metadata_string") ]
  ret void
}
