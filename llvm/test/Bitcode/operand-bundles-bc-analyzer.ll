; RUN: llvm-as < %s | llvm-bcanalyzer -dump -disable-histogram | FileCheck %s

; CHECK:  <OPERAND_BUNDLE_TAGS_BLOCK
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; CHECK-NEXT:  </OPERAND_BUNDLE_TAGS_BLOCK

; CHECK:   <FUNCTION_BLOCK
; CHECK:    <OPERAND_BUNDLE
; CHECK:    <OPERAND_BUNDLE
; CHECK-NOT: <OPERAND_BUNDLE
; CHECK:  </FUNCTION_BLOCK

; CHECK: Block ID #{{[0-9]+}} (OPERAND_BUNDLE_TAGS_BLOCK)

declare void @callee0()

define void @f0(i32* %ptr) {
 entry:
  %l = load i32, i32* %ptr
  %x = add i32 42, 1
  call void @callee0() [ "foo"(i32 42, i64 100, i32 %x), "bar"(float  0.000000e+00, i64 100, i32 %l) ]
  ret void
}
