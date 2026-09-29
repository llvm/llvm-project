; RUN: opt -S -passes='cgscc(inline)' < %s | FileCheck %s

; Check that the inliner does not inline through arbitrary unknown
; operand bundles.

define i32 @callee() {
 entry:
  ret i32 2
}

define i32 @caller() {
; CHECK: @caller(
 entry:
; CHECK: call i32 @callee() [ "unknown"() ]
  %x = call i32 @callee() [ "unknown"() ]
  ret i32 %x
}
