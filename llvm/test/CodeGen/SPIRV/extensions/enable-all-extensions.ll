; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=all %s -o - | FileCheck %s

define i6 @getConstantI6() {
  ret i6 2
}

; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_integers"
