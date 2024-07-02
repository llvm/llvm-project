; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+UNKNOWN_EXTENSION %s -o %t.spvt 2>&1 | FileCheck %s
; CHECK: Unknown SPIR-V extension: +UNKNOWN_EXTENSION

define i8 @foo() {
  ret i8 2
}
