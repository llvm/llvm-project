; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu7.00 < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu8.02 < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.00 < %s | FileCheck %s

; This checks for a bug where uniform control flow can result in multiple
; v_cmp results being combined together with s_and_b64, s_or_b64 and s_xor_b64,
; using the resulting mask in s_cbranch_vccnz
; without ensuring that the resulting mask has bits clear for inactive lanes.
; The problematic case is s_xor_b64, as, unlike the other ops, it can actually
; set bits for inactive lanes.
;
; The check for an s_xor_b64 is just to check that this test tests what it is
; supposed to test. If the s_xor_b64 disappears due to some other case, it does
; not necessarily mean that the bug has reappeared.
;
; The check for "s_and_b64 vcc, exec, something" checks that the bug is fixed.

; CHECK: {{^}}main:
; CHECK: s_xor_b64
; CHECK: s_and_b64 vcc, exec,

define amdgpu_cs void @main(i32 inreg %arg) {
.entry:
  %tmp44 = load volatile <2 x float>, ptr addrspace(1) poison
  %tmp16 = load volatile float, ptr addrspace(1) poison
  %tmp22 = load volatile float, ptr addrspace(1) poison
  %tmp25 = load volatile float, ptr addrspace(1) poison
  %tmp31 = fcmp olt float %tmp16, 0x3FA99999A0000000
  %tmp42 = fcmp olt float %tmp25, 0x3FA99999A0000000
  %tmp43 = and i1 %tmp31, %tmp42
  %tmp46 = fcmp olt <2 x float> %tmp44, <float 0x3FA99999A0000000, float 0x3FA99999A0000000>
  %tmp47 = extractelement <2 x i1> %tmp46, i32 0
  %tmp48 = extractelement <2 x i1> %tmp46, i32 1
  %tmp49 = and i1 %tmp47, %tmp48
  %tmp50 = and i1 %tmp43, %tmp49
  %tmp53 = fcmp olt float %tmp22, 0x3FA99999A0000000
  %tmp54 = and i1 %tmp50, %tmp53
  br i1 %tmp54, label %.exit3.i, label %.exit.thread

.exit3.i:
  store volatile i32 0, ptr addrspace(1) poison
  br label %.exit.thread

.exit.thread:
  ret void
}

