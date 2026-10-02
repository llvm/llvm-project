; RUN: opt -passes=inline -mtriple=amdgpu-amd-amdhsa -inline-instr-cost=50 \
; RUN:     -pass-remarks=inline -pass-remarks-missed=inline < %s 2>&1 | FileCheck %s

; A call left out of line in a kernel is register allocated against the worst
; case for the whole kernel, so it costs the hot path too and the reduced
; cold-callsite threshold does not apply. It still applies in a callable
; caller. Both callsites are equally cold and the callee has two uses, so the
; only difference is the caller's calling convention.

; CHECK-DAG: 'callee' inlined into 'kernel'
; CHECK-DAG: 'callee' not inlined into 'func' because too costly to inline

define internal void @callee(ptr addrspace(1) %p, i32 %x) {
entry:
  %v0 = mul i32 %x, 3
  %v1 = mul i32 %v0, 4
  %v2 = mul i32 %v1, 5
  %v3 = mul i32 %v2, 6
  %v4 = mul i32 %v3, 7
  %v5 = mul i32 %v4, 8
  %v6 = mul i32 %v5, 9
  %v7 = mul i32 %v6, 10
  %v8 = mul i32 %v7, 11
  %v9 = mul i32 %v8, 12
  %v10 = mul i32 %v9, 13
  %v11 = mul i32 %v10, 14
  %v12 = mul i32 %v11, 15
  %v13 = mul i32 %v12, 16
  %v14 = mul i32 %v13, 17
  %v15 = mul i32 %v14, 18
  store i32 %v15, ptr addrspace(1) %p, align 4
  ret void
}

define amdgpu_kernel void @kernel(ptr addrspace(1) %p, i32 %x, i1 %c) {
entry:
  br i1 %c, label %cold, label %exit, !prof !0

cold:
  call void @callee(ptr addrspace(1) %p, i32 %x)
  br label %exit

exit:
  ret void
}

define void @func(ptr addrspace(1) %p, i32 %x, i1 %c) {
entry:
  br i1 %c, label %cold, label %exit, !prof !0

cold:
  call void @callee(ptr addrspace(1) %p, i32 %x)
  br label %exit

exit:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 4000}
