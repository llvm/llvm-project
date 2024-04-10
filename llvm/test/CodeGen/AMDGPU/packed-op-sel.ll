; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}fma_vector_vector_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2

  %scalar0.vec = insertelement <2 x half> undef, half %scalar0, i32 0
  %scalar0.broadcast = shufflevector <2 x half> %scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %scalar0.broadcast)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; Apply fneg to broadcasted vector
; GCN-LABEL: {{^}}fma_vector_vector_neg_broadcast_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_broadcast_scalar_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2

  %scalar0.vec = insertelement <2 x half> undef, half %scalar0, i32 0
  %scalar0.broadcast = shufflevector <2 x half> %scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer
  %neg.scalar0.broadcast = fsub <2 x half> <half -0.0, half -0.0>, %scalar0.broadcast

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.scalar0.broadcast)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; Apply fneg before broadcast
; GCN-LABEL: {{^}}fma_vector_vector_neg_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_scalar_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %neg.scalar0, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x half> %neg.scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.scalar0.broadcast)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; Apply fneg before and after broadcast, and should cancel out.
; GCN-LABEL: {{^}}fma_vector_vector_neg_broadcast_neg_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_broadcast_neg_scalar_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %neg.scalar0, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x half> %neg.scalar0.vec, <2 x half> undef, <2 x i32> zeroinitializer
  %neg.neg.scalar0.broadcast = fsub <2 x half> <half -0.0, half -0.0>, %neg.scalar0.broadcast

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.neg.scalar0.broadcast)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; Add scalar, but negate low component
; GCN-LABEL: {{^}}fma_vector_vector_scalar_neg_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_lo:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_neg_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %neg.scalar0, i32 0
  %neg.scalar0.scalar0 = insertelement <2 x half> %neg.scalar0.vec, half %scalar0, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.scalar0.scalar0)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; Add scalar, but negate high component
; GCN-LABEL: {{^}}fma_vector_vector_scalar_neg_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[SCALAR0]] op_sel_hi:[1,1,0] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_neg_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2

  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.vec = insertelement <2 x half> undef, half %scalar0, i32 0
  %scalar0.neg.scalar0 = insertelement <2 x half> %neg.scalar0.vec, half %neg.scalar0, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %scalar0.neg.scalar0)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; Apply fneg before broadcast with bitcast
; GCN-LABEL: {{^}}add_vector_neg_bitcast_scalar_lo:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_add_u16 v{{[0-9]+}}, [[VEC0]], [[SCALAR0]] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]{{$}}
define amdgpu_kernel void @add_vector_neg_bitcast_scalar_lo(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %vec0 = load volatile <2 x i16>, ptr addrspace(3) %lds, align 4
  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2
  %neg.scalar0 = fsub half -0.0, %scalar0
  %neg.scalar0.bc = bitcast half %neg.scalar0 to i16

  %neg.scalar0.vec = insertelement <2 x i16> undef, i16 %neg.scalar0.bc, i32 0
  %neg.scalar0.broadcast = shufflevector <2 x i16> %neg.scalar0.vec, <2 x i16> undef, <2 x i32> zeroinitializer

  %result = add <2 x i16> %vec0, %neg.scalar0.broadcast
  store <2 x i16> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_scalar_lo_neg_scalar_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR0:v[0-9]+]]
; GCN: ds_read_u16 [[SCALAR1:v[0-9]+]]

; FIXME: Remove and
; GCN-DAG: v_and_b32_e32 [[SCALAR0]], 0xffff, [[SCALAR0]]
; GCN-DAG: v_xor_b32_e32 [[SCALAR1]], 0x8000, [[SCALAR1]]
; GCN: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[SCALAR1]], 16, [[SCALAR0]]

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[PACKED]]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_lo_neg_scalar_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %arg2.gep = getelementptr inbounds half, ptr addrspace(3) %arg2, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4

  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2
  %scalar1 = load volatile half, ptr addrspace(3) %arg2.gep, align 2

  %neg.scalar1 = fsub half -0.0, %scalar1
  %vec.ins0 = insertelement <2 x half> undef, half %scalar0, i32 0
  %vec2 = insertelement <2 x half> %vec.ins0, half %neg.scalar1, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %vec2)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; FIXME: Can we avoid waitcnt between the two halves?
; GCN-LABEL: {{^}}fma_vector_vector_neg_scalar_lo_scalar_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_u16 [[PACKED:v[0-9]+]]
; GCN: s_waitcnt
; GCN: ds_read_u16_d16_hi [[PACKED]]

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[PACKED]] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_scalar_lo_scalar_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds, ptr addrspace(3) %arg2) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %arg2.gep = getelementptr inbounds half, ptr addrspace(3) %arg2, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4

  %scalar0 = load volatile half, ptr addrspace(3) %arg2, align 2
  %scalar1 = load volatile half, ptr addrspace(3) %arg2.gep, align 2

  %vec.ins0 = insertelement <2 x half> undef, half %scalar0, i32 0
  %vec2 = insertelement <2 x half> %vec.ins0, half %scalar1, i32 1
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.vec2)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_neg_vector_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] op_sel:[0,0,1] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_vector_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %vec2.fneg = fsub <2 x half> <half -0.0, half -0.0>, %vec2
  %vec2.fneg.elt1.broadcast = shufflevector <2 x half> %vec2.fneg, <2 x half> undef, <2 x i32> <i32 1, i32 1>

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %vec2.fneg.elt1.broadcast)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_vector_neg_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_vector_neg_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %vec2.elt1 = extractelement <2 x half> %vec2, i32 1
  %neg.vec2.elt1 = fsub half -0.0, %vec2.elt1

  %neg.vec2.elt1.insert = insertelement <2 x half> %vec2, half %neg.vec2.elt1, i32 1
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.vec2.elt1.insert)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}add_vector_scalar_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_add_u16 v{{[0-9]+}}, [[VEC0]], [[VEC1]] op_sel:[0,1]{{$}}
define amdgpu_kernel void @add_vector_scalar_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x i16>, ptr addrspace(3) %lds, i32 1

  %vec0 = load volatile <2 x i16>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x i16>, ptr addrspace(3) %lds.gep1, align 4

  %vec1.elt1.broadcast = shufflevector <2 x i16> %vec1, <2 x i16> undef, <2 x i32> <i32 1, i32 1>
  %result = add <2 x i16> %vec0, %vec1.elt1.broadcast

  store <2 x i16> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_scalar_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] op_sel:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_scalar_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %vec2.elt1.broadcast = shufflevector <2 x half> %vec2, <2 x half> undef, <2 x i32> <i32 1, i32 1>

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %vec2.elt1.broadcast)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_neg_vector_lo_neg_hi:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]]{{$}}
define amdgpu_kernel void @fma_vector_vector_neg_vector_lo_neg_hi(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2
  %neg.vec2.elt1 = extractelement <2 x half> %neg.vec2, i32 1
  %neg.neg.vec2.elt1 = fsub half -0.0, %neg.vec2.elt1
  %neg.neg.vec2.elt1.insert = insertelement <2 x half> %vec2, half %neg.neg.vec2.elt1, i32 1

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.neg.vec2.elt1.insert)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_swap_vector:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] op_sel:[0,0,1] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @fma_vector_vector_swap_vector(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %vec2.swap = shufflevector <2 x half> %vec2, <2 x half> undef, <2 x i32> <i32 1, i32 0>
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %vec2.swap)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_swap_neg_vector:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or
; GCN-NOT: xor

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] op_sel:[0,0,1] op_sel_hi:[1,1,0] neg_lo:[0,0,1] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_swap_neg_vector(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2

  %neg.vec2.swap = shufflevector <2 x half> %neg.vec2, <2 x half> undef, <2 x i32> <i32 1, i32 0>
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %neg.vec2.swap)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_blend_vector_neg_vector_0:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or
; GCN-NOT: xor

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] op_sel:[0,0,1] op_sel_hi:[1,1,0] neg_lo:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_blend_vector_neg_vector_0(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2
  %combined = shufflevector <2 x half> %vec2, <2 x half> %neg.vec2, <2 x i32> <i32 3, i32 0>
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %combined)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_blend_vector_neg_vector_1:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or
; GCN-NOT: xor

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] neg_lo:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_blend_vector_neg_vector_1(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2
  %combined = shufflevector <2 x half> %vec2, <2 x half> %neg.vec2, <2 x i32> <i32 2, i32 1>
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %combined)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_blend_vector_neg_vector_2:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or
; GCN-NOT: xor

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] neg_hi:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_blend_vector_neg_vector_2(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2
  %combined = shufflevector <2 x half> %vec2, <2 x half> %neg.vec2, <2 x i32> <i32 0, i32 3>
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %combined)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}fma_vector_vector_blend_vector_neg_vector_3:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: or
; GCN-NOT: xor

; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[VEC2]] op_sel:[0,0,1] neg_lo:[0,0,1]{{$}}
define amdgpu_kernel void @fma_vector_vector_blend_vector_neg_vector_3(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4
  %neg.vec2 = fsub <2 x half> <half -0.0, half -0.0>, %vec2
  %combined = shufflevector <2 x half> %vec2, <2 x half> %neg.vec2, <2 x i32> <i32 3, i32 1>
  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %combined)

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}bitcast_fneg_f32:
; GCN: v_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+$}}
define amdgpu_kernel void @bitcast_fneg_f32(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %f32 = load volatile float, ptr addrspace(3) undef, align 4
  %neg.f32 = fsub float -0.0, %f32
  %bc = bitcast float %neg.f32 to <2 x half>
  %result = fadd <2 x half> %vec0, %bc

  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}shuffle_bitcast_fneg_f32:
; GCN: v_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} op_sel:[0,1] op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @shuffle_bitcast_fneg_f32(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4

  %f32 = load volatile float, ptr addrspace(3) undef, align 4
  %neg.f32 = fsub float -0.0, %f32
  %bc = bitcast float %neg.f32 to <2 x half>
  %shuf = shufflevector <2 x half> %bc, <2 x half> undef, <2 x i32> <i32 1, i32 0>
  %result = fadd <2 x half> %vec0, %shuf
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}extract_from_i64:
; GCN: v_lshl_or_b32
; GCN: v_pk_add_u16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+$}}
define amdgpu_kernel void @extract_from_i64(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %vec0 = load volatile <2 x i16>, ptr addrspace(3) %lds, align 4
  %i64 = load volatile i64, ptr addrspace(1) undef

  %elt0 = trunc i64 %i64 to i16
  %hi = lshr i64 %i64, 16
  %elt1 = trunc i64 %hi to i16

  %ins0 = insertelement <2 x i16> undef, i16 %elt1, i32 0
  %ins1 = insertelement <2 x i16> %ins0, i16 %elt0, i32 1
  %result = add <2 x i16> %vec0, %ins1
  store <2 x i16> %result, ptr addrspace(1) %out, align 4
  ret void
}


; Bitcast is final obstacle to identifying same source register
; GCN-LABEL: {{^}}bitcast_lo_elt_op_sel:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: _or

; GCN: v_pk_add_f16 [[FADD:v[0-9]+]]
; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[FADD]] op_sel:[0,0,1] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @bitcast_lo_elt_op_sel(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %scalar0 = load volatile i16, ptr addrspace(1) undef
  %shl = shl i16 %scalar0, 1
  %shl.bc = bitcast i16 %shl to half

  %fadd = fadd <2 x half> %vec2, <half 2.0, half 2.0>
  %shuffle = shufflevector <2 x half> %fadd, <2 x half> %vec2, <2 x i32> <i32 1, i32 0>

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %shuffle)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}


; Bitcast is final obstacle to identifying same source register
; GCN-LABEL: {{^}}mix_elt_types_op_sel:
; GCN: ds_read_b32 [[VEC0:v[0-9]+]]
; GCN: ds_read_b32 [[VEC1:v[0-9]+]]
; GCN: ds_read_b32 [[VEC2:v[0-9]+]]

; GCN-NOT: pack
; GCN-NOT: and
; GCN-NOT: shl
; GCN-NOT: _or

; GCN: v_pk_add_f16 [[FADD:v[0-9]+]]
; GCN: v_pk_fma_f16 v{{[0-9]+}}, [[VEC0]], [[VEC1]], [[FADD]] op_sel:[0,0,1] op_sel_hi:[1,1,0]{{$}}
define amdgpu_kernel void @mix_elt_types_op_sel(ptr addrspace(1) %out, ptr addrspace(3) %lds) #0 {
bb:
  %lds.gep1 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 1
  %lds.gep2 = getelementptr inbounds <2 x half>, ptr addrspace(3) %lds, i32 2

  %vec0 = load volatile <2 x half>, ptr addrspace(3) %lds, align 4
  %vec1 = load volatile <2 x half>, ptr addrspace(3) %lds.gep1, align 4
  %vec2 = load volatile <2 x half>, ptr addrspace(3) %lds.gep2, align 4

  %scalar0 = load volatile i16, ptr addrspace(1) undef
  %scalar1 = load volatile half, ptr addrspace(1) undef
  %shl = shl i16 %scalar0, 1
  %shl.bc = bitcast i16 %shl to half

  %insert0 = insertelement <2 x half> undef, half %shl.bc, i32 0

  %fadd = fadd <2 x half> %vec2, <half 2.0, half 2.0>
  %insert1 = shufflevector <2 x half> %fadd, <2 x half> %insert0, <2 x i32> <i32 1, i32 0>

  %result = tail call <2 x half> @llvm.fma.v2f16(<2 x half> %vec0, <2 x half> %vec1, <2 x half> %insert1)
  store <2 x half> %result, ptr addrspace(1) %out, align 4
  ret void
}

declare <2 x half> @llvm.fma.v2f16(<2 x half>, <2 x half>, <2 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
