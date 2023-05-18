; RUN: llc -amdgpu-scalar-ir-passes=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}select_undef_lhs:
; GCN: s_waitcnt
; GCN-NOT: v_cmp
; GCN-NOT: v_cndmask
; GCN-NEXT: s_setpc_b64
define float @select_undef_lhs(float %val, i1 %cond) {
  %sel = select i1 %cond, float undef, float %val
  ret float %sel
}

; GCN-LABEL: {{^}}select_undef_rhs:
; GCN: s_waitcnt
; GCN-NOT: v_cmp
; GCN-NOT: v_cndmask
; GCN-NEXT: s_setpc_b64
define float @select_undef_rhs(float %val, i1 %cond) {
  %sel = select i1 %cond, float %val, float undef
  ret float %sel
}

; GCN-LABEL: {{^}}select_undef_n1:
; GCN: v_mov_b32_e32 [[RES:v[0-9]+]], 1.0
; GCN: store_dword {{[^,]+}}, [[RES]]
define void @select_undef_n1(ptr addrspace(1) %a, i32 %c) {
  %cc = icmp eq i32 %c, 0
  %sel = select i1 %cc, float 1.000000e+00, float undef
  store float %sel, ptr addrspace(1) %a
  ret void
}

; GCN-LABEL: {{^}}select_undef_n2:
; GCN: v_mov_b32_e32 [[RES:v[0-9]+]], 1.0
; GCN: store_dword {{[^,]+}}, [[RES]]
define void @select_undef_n2(ptr addrspace(1) %a, i32 %c) {
  %cc = icmp eq i32 %c, 0
  %sel = select i1 %cc, float undef, float 1.000000e+00
  store float %sel, ptr addrspace(1) %a
  ret void
}

declare float @llvm.amdgcn.rcp.f32(float)


; Make sure the vector undef isn't lowered into 0s.
; GCN-LABEL: {{^}}undef_v6f32:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v6f32(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <6 x float> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <6 x float>, ptr addrspace(3) undef
  %add = fadd <6 x float> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <6 x float> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v6i32:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v6i32(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <6 x i32> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <6 x i32>, ptr addrspace(3) undef
  %add = add <6 x i32> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <6 x i32> %add, ptr addrspace(3) undef
  ret void
}

; Make sure the vector undef isn't lowered into 0s.
; GCN-LABEL: {{^}}undef_v5f32:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v5f32(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <5 x float> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <5 x float>, ptr addrspace(3) undef
  %add = fadd <5 x float> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <5 x float> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v5i32:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v5i32(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <5 x i32> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <5 x i32>, ptr addrspace(3) undef
  %add = add <5 x i32> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <5 x i32> %add, ptr addrspace(3) undef
  ret void
}

; Make sure the vector undef isn't lowered into 0s.
; GCN-LABEL: {{^}}undef_v3f64:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v3f64(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <3 x double> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <3 x double>, ptr addrspace(3) %ptr
  %add = fadd <3 x double> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <3 x double> %add, ptr addrspace(3) %ptr
  ret void
}

; GCN-LABEL: {{^}}undef_v3i64:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v3i64(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <3 x i64> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <3 x i64>, ptr addrspace(3) %ptr
  %add = add <3 x i64> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <3 x i64> %add, ptr addrspace(3) %ptr
  ret void
}

; Make sure the vector undef isn't lowered into 0s.
; GCN-LABEL: {{^}}undef_v4f16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v4f16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <4 x half> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <4 x half>, ptr addrspace(3) %ptr
  %add = fadd <4 x half> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <4 x half> %add, ptr addrspace(3) %ptr
  ret void
}

; GCN-LABEL: {{^}}undef_v4i16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v4i16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <4 x i16> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <4 x i16>, ptr addrspace(3) %ptr
  %add = add <4 x i16> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <4 x i16> %add, ptr addrspace(3) %ptr
  ret void
}

; Make sure the vector undef isn't lowered into 0s.
; GCN-LABEL: {{^}}undef_v2f16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v2f16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <2 x half> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <2 x half>, ptr addrspace(3) %ptr
  %add = fadd <2 x half> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <2 x half> %add, ptr addrspace(3) %ptr
  ret void
}

; GCN-LABEL: {{^}}undef_v2i16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v2i16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <2 x i16> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <2 x i16>, ptr addrspace(3) %ptr
  %add = add <2 x i16> %load, %phi
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <2 x i16> %add, ptr addrspace(3) %ptr
  ret void
}

; We were expanding undef vectors into zero vectors. Optimizations
; would then see we used no elements of the vector, and reform the
; undef vector resulting in a combiner loop.
; GCN-LABEL: {{^}}inf_loop_undef_vector:
; GCN: s_waitcnt
; GCN-NEXT: v_mad_u64_u32
; GCN-NEXT: v_mul_lo_u32
; GCN-NEXT: v_mul_lo_u32
; GCN-NEXT: v_add3_u32
; GCN-NEXT: global_store_dwordx2
define void @inf_loop_undef_vector(<6 x float> %arg, float %arg1, i64 %arg2) {
  %i = insertelement <6 x float> %arg, float %arg1, i64 2
  %i3 = bitcast <6 x float> %i to <3 x i64>
  %i4 = extractelement <3 x i64> %i3, i64 0
  %i5 = extractelement <3 x i64> %i3, i64 1
  %i6 = mul i64 %i5, %arg2
  %i7 = add i64 %i6, %i4
  store volatile i64 %i7, ptr addrspace(1) undef, align 4
  ret void
}

; GCN-LABEL: {{^}}undef_bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi bfloat [ undef, %entry ], [ %add, %loop ]
  %load = load volatile bfloat, ptr addrspace(3) undef
  %bc.0 = bitcast bfloat %load to i16
  %bc.1 = bitcast bfloat %phi to i16
  %add.i = add i16 %bc.0, %bc.1
  %add = bitcast i16 %add.i to bfloat
  br i1 %cond, label %loop, label %ret

ret:
  store volatile bfloat %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v2bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v2bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <2 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <2 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <2 x bfloat> %load to <2 x i16>
  %bc.1 = bitcast <2 x bfloat> %phi to <2 x i16>
  %add.i = add <2 x i16> %bc.0, %bc.1
  %add = bitcast <2 x i16> %add.i to <2 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <2 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v3bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v3bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <3 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <3 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <3 x bfloat> %load to <3 x i16>
  %bc.1 = bitcast <3 x bfloat> %phi to <3 x i16>
  %add.i = add <3 x i16> %bc.0, %bc.1
  %add = bitcast <3 x i16> %add.i to <3 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <3 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v4bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v4bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <4 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <4 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <4 x bfloat> %load to <4 x i16>
  %bc.1 = bitcast <4 x bfloat> %phi to <4 x i16>
  %add.i = add <4 x i16> %bc.0, %bc.1
  %add = bitcast <4 x i16> %add.i to <4 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <4 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v6bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v6bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <6 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <6 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <6 x bfloat> %load to <6 x i16>
  %bc.1 = bitcast <6 x bfloat> %phi to <6 x i16>
  %add.i = add <6 x i16> %bc.0, %bc.1
  %add = bitcast <6 x i16> %add.i to <6 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <6 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v8bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v8bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <8 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <8 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <8 x bfloat> %load to <8 x i16>
  %bc.1 = bitcast <8 x bfloat> %phi to <8 x i16>
  %add.i = add <8 x i16> %bc.0, %bc.1
  %add = bitcast <8 x i16> %add.i to <8 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <8 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v16bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v16bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <16 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <16 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <16 x bfloat> %load to <16 x i16>
  %bc.1 = bitcast <16 x bfloat> %phi to <16 x i16>
  %add.i = add <16 x i16> %bc.0, %bc.1
  %add = bitcast <16 x i16> %add.i to <16 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <16 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}undef_v32bf16:
; GCN-NOT: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, 0{{$}}
; GCN: s_cbranch_vccnz
define amdgpu_kernel void @undef_v32bf16(ptr addrspace(3) %ptr, i1 %cond) {
entry:
  br label %loop

loop:
  %phi = phi <32 x bfloat> [ undef, %entry ], [ %add, %loop ]
  %load = load volatile <32 x bfloat>, ptr addrspace(3) undef
  %bc.0 = bitcast <32 x bfloat> %load to <32 x i16>
  %bc.1 = bitcast <32 x bfloat> %phi to <32 x i16>
  %add.i = add <32 x i16> %bc.0, %bc.1
  %add = bitcast <32 x i16> %add.i to <32 x bfloat>
  br i1 %cond, label %loop, label %ret

ret:
  store volatile <32 x bfloat> %add, ptr addrspace(3) undef
  ret void
}

