; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=UNKNOWN_EXTENSION %s -o %t.spvt 2>&1 | FileCheck %s
; CHECK: Invalid extension list format UNKNOWN_EXTENSION

define i8 @foo() {
  ret i8 2
}
