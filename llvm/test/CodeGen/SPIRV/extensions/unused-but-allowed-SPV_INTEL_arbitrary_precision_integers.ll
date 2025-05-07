; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s

define i8 @getConstantI8() {
  ret i8 2
}
define i16 @getConstantI16() {
  ret i16 2
}
define i32 @getConstantI32() {
  ret i32 2
}

define i64 @getConstantI64() {
  ret i64 42
}

;; Capabilities:
; CHECK-NOT: OpExtension "SPV_INTEL_arbitrary_precision_integers"
; CHECK-NOT: OpCapability ArbitraryPrecisionIntegersINTEL
