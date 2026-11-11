; Check that we accept functions with '$' in the name.
; Also check that we accept functions with multiple leading underscores.

; RUN: llc -mtriple=aarch64-unknown-linux < %s | FileCheck --check-prefix=LINUX %s
; RUN: llc -mtriple=aarch64-apple-darwin < %s | FileCheck --check-prefix=DARWIN %s
; RUN: llc -mtriple=arm64-apple-macosx < %s | FileCheck --check-prefix=DARWIN %s

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}

define i32 @__muloti4() {
entry:
  ret i32 3
}
