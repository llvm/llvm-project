; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=SPV_INTEL_optnone %s -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXTENSION

; CHECK-EXTENSION: OpCapability OptNoneINTEL
; CHECK-EXTENSION: OpExtension "SPV_INTEL_optnone"
; CHECK-NO-EXTENSION-NOT: OpCapability OptNoneINTEL
; CHECK-NO-EXTENSION-NOT: OpExtension "SPV_INTEL_optnone"

;; Per SPIR-V spec:
;; FunctionControlDontInlineMask = 0x2 (2)
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] DontInline

; Function Attrs: nounwind optnone noinline
define spir_func void @_Z3foov() #0 {
entry:
  ret void
}

attributes #0 = { nounwind optnone noinline }
