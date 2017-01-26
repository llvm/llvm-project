; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}bfm_pattern:
; SI: s_bfm_b32 {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
define void @bfm_pattern(i32 addrspace(1)* %out, i32 %x, i32 %y) #0 {
  %a = shl i32 1, %x
  %b = sub i32 %a, 1
  %c = shl i32 %b, %y
  store i32 %c, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}bfm_pattern_simple:
; SI: s_bfm_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0
define void @bfm_pattern_simple(i32 addrspace(1)* %out, i32 %x) #0 {
  %a = shl i32 1, %x
  %b = sub i32 %a, 1
  store i32 %b, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
