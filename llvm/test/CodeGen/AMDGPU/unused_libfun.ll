; RUN: llc -march=amdgcn -O3 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_i128_add:
; GCN-NOT: s_swappc_b64

; GCN-NOT: {{^}}__divti3:

define amdgpu_kernel void @test_i128_add(ptr addrspace(1) %x, i128 %y, i128 %z) {
entry:
  %add = add i128 %y, %z
  store i128 %add, ptr addrspace(1) %x, align 16
  ret void
}

; unused lib function should be removed
define hidden i128 @__divti3(i128 %a, i128 %b) #0 {
entry:
  %add = add i128 %a, %b
  ret i128 %add
}

attributes #0 = { "amdgpu-lib-fun" }
