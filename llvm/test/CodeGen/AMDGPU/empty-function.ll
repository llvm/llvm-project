; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=SI %s
; RUN: llc -mtriple=amdgpu8.02 < %s | FileCheck -check-prefix=SI %s

; Make sure we don't assert on empty functions

; SI: .text
; SI-LABEL: {{^}}empty_function_ret:
; SI: s_endpgm
; SI: codeLenInByte = 4
define amdgpu_kernel void @empty_function_ret() #0 {
  ret void
}

; SI: .text
; SI-LABEL: {{^}}empty_function_unreachable:
; SI: codeLenInByte = 0
define amdgpu_kernel void @empty_function_unreachable() #0 {
  unreachable
}

attributes #0 = { nounwind }
