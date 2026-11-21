; RUN: llc -mtriple=amdgpu8.02-amd-amdhsa < %s | FileCheck --check-prefixes=CHECK,TONGA %s
; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa < %s | FileCheck --check-prefixes=CHECK,FIJI %s

; With the SGPR init bug, the total number of SGPRs of a kernel is limited to
; 96. It is computed from the highest SGPR used by the kernel and its callees,
; plus the FLAT_SCRATCH, XNACK_MASK and VCC SGPRs of the kernel. A callable
; function does not set up flat scratch itself, but it must leave room for the
; FLAT_SCRATCH SGPRs of the kernels calling it, so it may only use SGPRs up to
; s89. Otherwise, a kernel using flat scratch fails with
; "scalar registers (97) exceeds limit (96)", which also makes this test fail.
;
; The callee needs more than 90 SGPRs without that limit, as shown by Fiji,
; which does not have the SGPR init bug.

; CHECK-LABEL: {{^}}callee:
; TONGA: .set .Lcallee.numbered_sgpr, {{([0-9]|[1-8][0-9]|90)}}{{$}}
; FIJI: .set .Lcallee.numbered_sgpr, {{(9[1-9]|10[0-2])}}{{$}}
define hidden <26 x i32> @callee(<52 x i16> inreg %a, i32 inreg %b) #0 {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <52 x i16> %a, splat (i16 3)
  %a2 = bitcast <52 x i16> %a1 to <26 x i32>
  br label %end

cmp.false:
  %a3 = bitcast <52 x i16> %a to <26 x i32>
  br label %end

end:
  %phi = phi <26 x i32> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <26 x i32> %phi
}

; CHECK-LABEL: {{^}}kernel:
; CHECK: .amdhsa_user_sgpr_flat_scratch_init 1
; CHECK: .set .Lkernel.uses_flat_scratch, or(1, .Lcallee.uses_flat_scratch)
define amdgpu_kernel void @kernel(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %b) {
  %a = load <52 x i16>, ptr addrspace(1) %in
  %r = call <26 x i32> @callee(<52 x i16> inreg %a, i32 inreg %b)
  store <26 x i32> %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { noinline nounwind }
