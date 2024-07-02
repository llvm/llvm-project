; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_integers,-SPV_INTEL_arbitrary_precision_integers %s -o %t.spvt 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=-SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_arbitrary_precision_integers %s -o %t.spvt 2>&1 | FileCheck %s
; CHECK: Extension cannot be allowed and disallowed at the same time: SPV_INTEL_arbitrary_precision_integers

define i8 @foo() {
  ret i8 2
}
