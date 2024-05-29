; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG --check-prefix=FUNC %s
; RUN: llc -mtriple=r600 -mcpu=cayman  < %s | FileCheck --check-prefix=CM --check-prefix=FUNC %s

; store to rat 0
; FUNC-LABEL: {{^}}store_typed_rat0:
; EG: MEM_RAT STORE_TYPED RAT(0) {{T[0-9]+, T[0-9]+}}, 1
; CM: MEM_RAT STORE_TYPED RAT(0) {{T[0-9]+, T[0-9]+}}

define amdgpu_kernel void @store_typed_rat0(<4 x i32> %data, <4 x i32> %index) {
  call void @llvm.r600.rat.store.typed(<4 x i32> %data, <4 x i32> %index, i32 0)
  ret void
}

; store to rat 11
; FUNC-LABEL: {{^}}store_typed_rat11:
; EG: MEM_RAT STORE_TYPED RAT(11) {{T[0-9]+, T[0-9]+}}, 1
; CM: MEM_RAT STORE_TYPED RAT(11) {{T[0-9]+, T[0-9]+}}

define amdgpu_kernel void @store_typed_rat11(<4 x i32> %data, <4 x i32> %index) {
  call void @llvm.r600.rat.store.typed(<4 x i32> %data, <4 x i32> %index, i32 11)
  ret void
}

declare void @llvm.r600.rat.store.typed(<4 x i32>, <4 x i32>, i32)
