; RUN: llc -march=amdgcn -O3 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_i128_sdiv:
; GCN: s_getpc_b64 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN-NEXT: s_add_u32 s[[LO]], s[[LO]], __divti3@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[HI]], s[[HI]], __divti3@rel32@hi+12
; GCN: s_swappc_b64 s[[[RET_LO:[0-9]+]]:[[RET_HI:[0-9]+]]], s[[[LO]]:[[HI]]]

; GCN-LABEL: {{^}}__divti3:
; GCN: s_getpc_b64 s[[[LO2:[0-9]+]]:[[HI2:[0-9]+]]]
; GCN-NEXT: s_add_u32 s[[LO2]], s[[LO2]], __divti3_impl@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[HI2]], s[[HI2]], __divti3_impl@rel32@hi+12
; GCN: s_swappc_b64 s[[[RET_LO2:[0-9]+]]:[[RET_HI2:[0-9]+]]], s[[[LO2]]:[[HI2]]]
; GCN: s_setpc_b64 s[[[RET_LO]]:[[RET_HI]]]

; GCN-LABEL: {{^}}__divti3_impl:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN: v_add_i32_e32 v0, vcc, v0, v4
; GCN: v_addc_u32_e32 v1, vcc, v1, v5, vcc
; GCN: v_addc_u32_e32 v2, vcc, v2, v6, vcc
; GCN: v_addc_u32_e32 v3, vcc, v3, v7, vcc
; GCN: s_setpc_b64 s[[[RET_LO2]]:[[RET_HI2]]]

define amdgpu_kernel void @test_i128_sdiv(ptr addrspace(1) %x, i128 %y, i128 %z) {
entry:
  %div = sdiv i128 %y, %z
  store i128 %div, ptr addrspace(1) %x, align 16
  ret void
}

; compiler-rt lib function for 128 bit signed division
define hidden i128 @__divti3(i128 %a, i128 %b) #0 {
entry:
  %call = call i128 @__divti3_impl(i128 %a, i128 %b)
  ret i128 %call
}

define hidden i128 @__divti3_impl(i128 %a, i128 %b) #0 {
entry:
  %add = add i128 %a, %b
  ret i128 %add
}

attributes #0 = { "amdgpu-lib-fun" }
