;; Check that optnone is correctly ignored when extension is not enabled
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; Per SPIR-V spec:
;; FunctionControlDontInlineMask = 0x2 (2)
; CHECK-SPIRV: %[[#]] = OpFunction %[[#]] DontInline

; Function Attrs: nounwind optnone noinline
define spir_func void @_Z3foov() #0 {
entry:
  ret void
}

attributes #0 = { nounwind optnone noinline }
