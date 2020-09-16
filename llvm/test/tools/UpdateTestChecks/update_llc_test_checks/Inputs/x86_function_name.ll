; Check that we accept functions with '$' in the name.
;
; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s
;
define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}
