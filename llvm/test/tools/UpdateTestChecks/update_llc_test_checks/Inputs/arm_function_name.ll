; Check that we accept functions with '$' in the name.

; RUN: llc -mtriple=armv7-unknown-linux < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-darwin < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios < %s | FileCheck %s

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}
