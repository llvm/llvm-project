; RUN: llvm-as < %s | llvm-bcanalyzer -dump -disable-histogram | FileCheck %s

; COM: Check that all built-in and user-defined bundle tags are serialized.
; CHECK:  <OPERAND_BUNDLE_TAGS_BLOCK
; COM: "deopt"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "funclet"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "gc-transition"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "cfguardtarget"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "preallocated"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "gc-live"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "clang.arc.attachedcall"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "ptrauth"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "kcfi"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "convergencectrl"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "align"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "deactivation-symbol"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "fp.control"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "fp.except"
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "foo" (user-defined, from call below)
; CHECK-NEXT:    <OPERAND_BUNDLE_TAG
; COM: "bar" (user-defined, from call below)
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
