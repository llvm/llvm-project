; This test ensures that we get a metadata operand bundle value in and out.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare void @callee()

; CHECK-LABEL: call_with_operand_bundle(
define void @call_with_operand_bundle() {
  ; CHECK: call void @op_bundle_callee() [ "op_type"(metadata !"metadata_string") ]
  call void @callee() [ "op_type"(metadata !"metadata_string") ]

  ret void
}
