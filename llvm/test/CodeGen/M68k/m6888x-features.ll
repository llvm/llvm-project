; RUN: llc -mtriple=m68k -mattr="+isa-68881" %s -o - 2>&1 | FileCheck %s
; RUN: llc -mtriple=m68k -mattr="+isa-68882" %s -o - 2>&1 | FileCheck %s

define dso_local i32 @f() {
entry:
  ret i32 0
}

; Make sure that all of the features listed are recognized.
; CHECK-NOT:    is not a recognized feature for this target
