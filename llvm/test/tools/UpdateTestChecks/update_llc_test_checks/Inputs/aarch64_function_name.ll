; Check that we accept functions with '$' in the name.

; RUN: llc -mtriple=aarch64-unknown-linux < %s | FileCheck --check-prefix=LINUX %s
; RUN: llc -mtriple=aarch64-apple-darwin < %s | FileCheck --check-prefix=DARWIN %s

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}
