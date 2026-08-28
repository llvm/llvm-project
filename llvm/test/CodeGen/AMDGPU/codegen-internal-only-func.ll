; REQUIRES: asserts
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s

; Don't longer silently skips internal Functions that are unreachable from
; ExternalCallingNode. This aligns with non-SCC-ordered codegen and with
; NPM-codegen behaviour: an internal function with no IR-level uses is still
; emitted, and the linker drops it at final-link time (--gc-sections) if
; truly unused. That also correctly preserves the symbol for
; inline-asm/linker-script references that IR-level use-tracking doesn't see.

; CHECK-LABEL: func:

define internal i32 @func() {
  ret i32 0
}
