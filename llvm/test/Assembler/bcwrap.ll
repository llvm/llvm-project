; RUN: llvm-as < %s > %t
; RUN: llvm-nm %t | FileCheck %s
; RUN: verify-uselistorder %s
; Test for isBitcodeFile, llvm-nm must read from a file for this test.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.2.2"

; CHECK: foo

define i32 @foo() {
  ret i32 0
}
