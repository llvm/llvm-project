; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-ALLOCA %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GCN-PROMOTE %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-promote-alloca -sroa -instcombine < %s | FileCheck -check-prefix=OPT %s

target datalayout = "A5"

; OPT-LABEL: @vector_read_alloca_bitcast(
; OPT-NOT:   alloca
; OPT:       %0 = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
; OPT-NEXT:  store i32 %0, i32 addrspace(1)* %out, align 4

; GCN-LABEL: {{^}}vector_read_alloca_bitcast:
; GCN-ALLOCA-COUNT-4: buffer_store_dword
; GCN-ALLOCA:         buffer_load_dword

; GCN_PROMOTE: s_cmp_lg_u32 s{{[0-9]+}}, 2
; GCN-PROMOTE: s_cmp_eq_u32 s{{[0-9]+}}, 1
; GCN-PROMOTE: s_cselect_b64 [[CC1:[^,]+]], -1, 0
; GCN-PROMOTE: v_cndmask_b32_e{{32|64}} [[IND1:v[0-9]+]], 0, 1, [[CC1]]
; GCN-PROMOTE: s_cselect_b64 vcc, -1, 0
; GCN_PROMOTE: s_cmp_lg_u32 s{{[0-9]+}}, 3
; GCN-PROMOTE: v_cndmask_b32_e{{32|64}} [[IND2:v[0-9]+]], 2, [[IND1]], vcc
; GCN-PROMOTE: s_cselect_b64 vcc, -1, 0
; GCN-PROMOTE: v_cndmask_b32_e{{32|64}} [[IND3:v[0-9]+]], 3, [[IND2]], vcc
; GCN-PROMOTE: ScratchSize: 0

define amdgpu_kernel void @vector_read_alloca_bitcast(i32 addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %x = bitcast [4 x i32] addrspace(5)* %tmp to i32 addrspace(5)*
  %y = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 3
  store i32 0, i32 addrspace(5)* %x
  store i32 1, i32 addrspace(5)* %y
  store i32 2, i32 addrspace(5)* %z
  store i32 3, i32 addrspace(5)* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i32, i32 addrspace(5)* %tmp1
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_write_alloca_bitcast(
; OPT-NOT:   alloca
; OPT:       %0 = insertelement <4 x i32> zeroinitializer, i32 1, i32 %w_index
; OPT-NEXT:  %1 = extractelement <4 x i32> %0, i32 %r_index
; OPT-NEXT:  store i32 %1, i32 addrspace(1)* %out, align 

; GCN-LABEL: {{^}}vector_write_alloca_bitcast:
; GCN-ALLOCA-COUNT-5: buffer_store_dword
; GCN-ALLOCA:         buffer_load_dword

; GCN-PROMOTE-COUNT-7: v_cndmask

; GCN-PROMOTE: ScratchSize: 0

define amdgpu_kernel void @vector_write_alloca_bitcast(i32 addrspace(1)* %out, i32 %w_index, i32 %r_index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %x = bitcast [4 x i32] addrspace(5)* %tmp to i32 addrspace(5)*
  %y = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 3
  store i32 0, i32 addrspace(5)* %x
  store i32 0, i32 addrspace(5)* %y
  store i32 0, i32 addrspace(5)* %z
  store i32 0, i32 addrspace(5)* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 %w_index
  store i32 1, i32 addrspace(5)* %tmp1
  %tmp2 = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 %r_index
  %tmp3 = load i32, i32 addrspace(5)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_write_read_bitcast_to_float(
; OPT-NOT:   alloca
; OPT: bb2:
; OPT:  %tmp.sroa.0.0 = phi <6 x float> [ undef, %bb ], [ %0, %bb2 ]
; OPT:  %0 = insertelement <6 x float> %tmp.sroa.0.0, float %tmp73, i32 %tmp10
; OPT: .preheader:
; OPT:  %bc = bitcast <6 x float> %0 to <6 x i32>
; OPT:  %1 = extractelement <6 x i32> %bc, i32 %tmp20

; GCN-LABEL: {{^}}vector_write_read_bitcast_to_float:
; GCN-ALLOCA: buffer_store_dword

; GCN-PROMOTE-COUNT-6: v_cmp_eq_u16
; GCN-PROMOTE-COUNT-6: v_cndmask

; GCN: s_cbranch

; GCN-ALLOCA: buffer_load_dword

; GCN-PROMOTE: v_cmp_eq_u16
; GCN-PROMOTE: v_cndmask
; GCN-PROMOTE: v_cmp_eq_u16
; GCN-PROMOTE: v_cndmask
; GCN-PROMOTE: v_cmp_eq_u16
; GCN-PROMOTE: v_cndmask
; GCN-PROMOTE: v_cmp_eq_u16
; GCN-PROMOTE: v_cndmask
; GCN-PROMOTE: v_cmp_eq_u16
; GCN-PROMOTE: v_cndmask

; GCN-PROMOTE: ScratchSize: 0

define amdgpu_kernel void @vector_write_read_bitcast_to_float(float addrspace(1)* %arg) {
bb:
  %tmp = alloca [6 x float], align 4, addrspace(5)
  %tmp1 = bitcast [6 x float] addrspace(5)* %tmp to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 24, i8 addrspace(5)* %tmp1) #2
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i32 [ 0, %bb ], [ %tmp13, %bb2 ]
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp4
  %tmp6 = bitcast float addrspace(1)* %tmp5 to i32 addrspace(1)*
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = trunc i32 %tmp3 to i16
  %tmp9 = urem i16 %tmp8, 6
  %tmp10 = zext i16 %tmp9 to i32
  %tmp11 = getelementptr inbounds [6 x float], [6 x float] addrspace(5)* %tmp, i32 0, i32 %tmp10
  %tmp12 = bitcast float addrspace(5)* %tmp11 to i32 addrspace(5)*
  store i32 %tmp7, i32 addrspace(5)* %tmp12, align 4
  %tmp13 = add nuw nsw i32 %tmp3, 1
  %tmp14 = icmp eq i32 %tmp13, 1000
  br i1 %tmp14, label %.preheader, label %bb2

bb15:                                             ; preds = %.preheader
  call void @llvm.lifetime.end.p5i8(i64 24, i8 addrspace(5)* %tmp1) #2
  ret void

.preheader:                                       ; preds = %.preheader, %bb2
  %tmp16 = phi i32 [ %tmp27, %.preheader ], [ 0, %bb2 ]
  %tmp17 = trunc i32 %tmp16 to i16
  %tmp18 = urem i16 %tmp17, 6
  %tmp19 = sub nuw nsw i16 5, %tmp18
  %tmp20 = zext i16 %tmp19 to i32
  %tmp21 = getelementptr inbounds [6 x float], [6 x float] addrspace(5)* %tmp, i32 0, i32 %tmp20
  %tmp22 = bitcast float addrspace(5)* %tmp21 to i32 addrspace(5)*
  %tmp23 = load i32, i32 addrspace(5)* %tmp22, align 4
  %tmp24 = zext i32 %tmp16 to i64
  %tmp25 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp24
  %tmp26 = bitcast float addrspace(1)* %tmp25 to i32 addrspace(1)*
  store i32 %tmp23, i32 addrspace(1)* %tmp26, align 4
  %tmp27 = add nuw nsw i32 %tmp16, 1
  %tmp28 = icmp eq i32 %tmp27, 1000
  br i1 %tmp28, label %bb15, label %.preheader
}

; OPT-LABEL: @vector_write_read_bitcast_to_double(
; OPT-NOT:   alloca
; OPT: bb2:
; OPT:  %tmp.sroa.0.0 = phi <6 x double> [ undef, %bb ], [ %0, %bb2 ]
; OPT:  %0 = insertelement <6 x double> %tmp.sroa.0.0, double %tmp73, i32 %tmp10
; OPT: .preheader:
; OPT:  %bc = bitcast <6 x double> %0 to <6 x i64>
; OPT:  %1 = extractelement <6 x i64> %bc, i32 %tmp20

; GCN-LABEL: {{^}}vector_write_read_bitcast_to_double:

; GCN-ALLOCA-COUNT-2: buffer_store_dword
; GCN-PROMOTE-COUNT-2: v_movreld_b32_e32

; GCN: s_cbranch

; GCN-ALLOCA-COUNT-2: buffer_load_dword
; GCN-PROMOTE-COUNT-2: v_movrels_b32_e32

; GCN-PROMOTE: ScratchSize: 0

define amdgpu_kernel void @vector_write_read_bitcast_to_double(double addrspace(1)* %arg) {
bb:
  %tmp = alloca [6 x double], align 8, addrspace(5)
  %tmp1 = bitcast [6 x double] addrspace(5)* %tmp to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 48, i8 addrspace(5)* %tmp1) #2
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i32 [ 0, %bb ], [ %tmp13, %bb2 ]
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = getelementptr inbounds double, double addrspace(1)* %arg, i64 %tmp4
  %tmp6 = bitcast double addrspace(1)* %tmp5 to i64 addrspace(1)*
  %tmp7 = load i64, i64 addrspace(1)* %tmp6, align 8
  %tmp8 = trunc i32 %tmp3 to i16
  %tmp9 = urem i16 %tmp8, 6
  %tmp10 = zext i16 %tmp9 to i32
  %tmp11 = getelementptr inbounds [6 x double], [6 x double] addrspace(5)* %tmp, i32 0, i32 %tmp10
  %tmp12 = bitcast double addrspace(5)* %tmp11 to i64 addrspace(5)*
  store i64 %tmp7, i64 addrspace(5)* %tmp12, align 8
  %tmp13 = add nuw nsw i32 %tmp3, 1
  %tmp14 = icmp eq i32 %tmp13, 1000
  br i1 %tmp14, label %.preheader, label %bb2

bb15:                                             ; preds = %.preheader
  call void @llvm.lifetime.end.p5i8(i64 48, i8 addrspace(5)* %tmp1) #2
  ret void

.preheader:                                       ; preds = %.preheader, %bb2
  %tmp16 = phi i32 [ %tmp27, %.preheader ], [ 0, %bb2 ]
  %tmp17 = trunc i32 %tmp16 to i16
  %tmp18 = urem i16 %tmp17, 6
  %tmp19 = sub nuw nsw i16 5, %tmp18
  %tmp20 = zext i16 %tmp19 to i32
  %tmp21 = getelementptr inbounds [6 x double], [6 x double] addrspace(5)* %tmp, i32 0, i32 %tmp20
  %tmp22 = bitcast double addrspace(5)* %tmp21 to i64 addrspace(5)*
  %tmp23 = load i64, i64 addrspace(5)* %tmp22, align 8
  %tmp24 = zext i32 %tmp16 to i64
  %tmp25 = getelementptr inbounds double, double addrspace(1)* %arg, i64 %tmp24
  %tmp26 = bitcast double addrspace(1)* %tmp25 to i64 addrspace(1)*
  store i64 %tmp23, i64 addrspace(1)* %tmp26, align 8
  %tmp27 = add nuw nsw i32 %tmp16, 1
  %tmp28 = icmp eq i32 %tmp27, 1000
  br i1 %tmp28, label %bb15, label %.preheader
}

; OPT-LABEL: @vector_write_read_bitcast_to_i64(
; OPT-NOT:   alloca
; OPT: bb2:
; OPT:  %tmp.sroa.0.0 = phi <6 x i64> [ undef, %bb ], [ %0, %bb2 ]
; OPT:  %0 = insertelement <6 x i64> %tmp.sroa.0.0, i64 %tmp6, i32 %tmp9
; OPT: .preheader:
; OPT:  %1 = extractelement <6 x i64> %0, i32 %tmp18

; GCN-LABEL: {{^}}vector_write_read_bitcast_to_i64:

; GCN-ALLOCA-COUNT-2: buffer_store_dword
; GCN-PROMOTE-COUNT-2: v_movreld_b32_e32

; GCN: s_cbranch

; GCN-ALLOCA-COUNT-2: buffer_load_dword
; GCN-PROMOTE-COUNT-2: v_movrels_b32_e32

; GCN-PROMOTE: ScratchSize: 0

define amdgpu_kernel void @vector_write_read_bitcast_to_i64(i64 addrspace(1)* %arg) {
bb:
  %tmp = alloca [6 x i64], align 8, addrspace(5)
  %tmp1 = bitcast [6 x i64] addrspace(5)* %tmp to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 48, i8 addrspace(5)* %tmp1) #2
  br label %bb2

bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i32 [ 0, %bb ], [ %tmp11, %bb2 ]
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i64 %tmp4
  %tmp6 = load i64, i64 addrspace(1)* %tmp5, align 8
  %tmp7 = trunc i32 %tmp3 to i16
  %tmp8 = urem i16 %tmp7, 6
  %tmp9 = zext i16 %tmp8 to i32
  %tmp10 = getelementptr inbounds [6 x i64], [6 x i64] addrspace(5)* %tmp, i32 0, i32 %tmp9
  store i64 %tmp6, i64 addrspace(5)* %tmp10, align 8
  %tmp11 = add nuw nsw i32 %tmp3, 1
  %tmp12 = icmp eq i32 %tmp11, 1000
  br i1 %tmp12, label %.preheader, label %bb2

bb13:                                             ; preds = %.preheader
  call void @llvm.lifetime.end.p5i8(i64 48, i8 addrspace(5)* %tmp1) #2
  ret void

.preheader:                                       ; preds = %.preheader, %bb2
  %tmp14 = phi i32 [ %tmp23, %.preheader ], [ 0, %bb2 ]
  %tmp15 = trunc i32 %tmp14 to i16
  %tmp16 = urem i16 %tmp15, 6
  %tmp17 = sub nuw nsw i16 5, %tmp16
  %tmp18 = zext i16 %tmp17 to i32
  %tmp19 = getelementptr inbounds [6 x i64], [6 x i64] addrspace(5)* %tmp, i32 0, i32 %tmp18
  %tmp20 = load i64, i64 addrspace(5)* %tmp19, align 8
  %tmp21 = zext i32 %tmp14 to i64
  %tmp22 = getelementptr inbounds i64, i64 addrspace(1)* %arg, i64 %tmp21
  store i64 %tmp20, i64 addrspace(1)* %tmp22, align 8
  %tmp23 = add nuw nsw i32 %tmp14, 1
  %tmp24 = icmp eq i32 %tmp23, 1000
  br i1 %tmp24, label %bb13, label %.preheader
}

; TODO: llvm.assume can be ingored

; OPT-LABEL: @vector_read_alloca_bitcast_assume(
; OPT:      %tmp = alloca <4 x i32>, align 16, addrspace(5)
; OPT-NEXT: %x = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(5)* %tmp, i64 0, i64 0
; OPT-NEXT: store i32 0, i32 addrspace(5)* %x, align 16
; OPT-NEXT: %0 = load <4 x i32>, <4 x i32> addrspace(5)* %tmp, align 16
; OPT-NEXT: %1 = shufflevector <4 x i32> %0, <4 x i32> <i32 poison, i32 1, i32 2, i32 3>, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
; OPT-NEXT: store <4 x i32> %1, <4 x i32> addrspace(5)* %tmp, align 16
; OPT-NEXT: %2 = extractelement <4 x i32> %1, i32 %index
; OPT-NEXT: store i32 %2, i32 addrspace(1)* %out, align 4

; GCN-LABEL: {{^}}vector_read_alloca_bitcast_assume:
; GCN-COUNT-4: buffer_store_dword

define amdgpu_kernel void @vector_read_alloca_bitcast_assume(i32 addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %x = bitcast [4 x i32] addrspace(5)* %tmp to i32 addrspace(5)*
  %cmp = icmp ne i32 addrspace(5)* %x, null
  call void @llvm.assume(i1 %cmp)
  %y = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 3
  store i32 0, i32 addrspace(5)* %x
  store i32 1, i32 addrspace(5)* %y
  store i32 2, i32 addrspace(5)* %z
  store i32 3, i32 addrspace(5)* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i32, i32 addrspace(5)* %tmp1
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_read_alloca_multiuse(
; OPT-NOT:   alloca
; OPT:       %0 = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 3>, i32 %index
; OPT-NEXT:  %add2 = add nuw nsw i32 %0, 1
; OPT-NEXT:  store i32 %add2, i32 addrspace(1)* %out, align 4

; GCN-LABEL: {{^}}vector_read_alloca_multiuse:
; GCN-ALLOCA-COUNT-4: buffer_store_dword
; GCN-ALLOCA:         buffer_load_dword

; GCN-PROMOTE: s_cmp_eq_u32 s{{[0-9]+}}, 1
; GCN-PROMOTE: s_cselect_b64 [[CC1:[^,]+]], -1, 0
; GCN_PROMOTE: s_cmp_lg_u32 s{{[0-9]+}}, 2
; GCN-PROMOTE: v_cndmask_b32_e{{32|64}} [[IND1:v[0-9]+]], 0, 1, [[CC1]]
; GCN-PROMOTE: s_cselect_b64 vcc, -1, 0
; GCN_PROMOTE: s_cmp_lg_u32 s{{[0-9]+}}, 3
; GCN-PROMOTE: v_cndmask_b32_e{{32|64}} [[IND2:v[0-9]+]], 2, [[IND1]], vcc
; GCN-PROMOTE: s_cselect_b64 vcc, -1, 0
; GCN-PROMOTE: v_cndmask_b32_e{{32|64}} [[IND3:v[0-9]+]], 3, [[IND2]], vcc

; GCN-PROMOTE: ScratchSize: 0

define amdgpu_kernel void @vector_read_alloca_multiuse(i32 addrspace(1)* %out, i32 %index) {
entry:
  %tmp = alloca [4 x i32], addrspace(5)
  %b = bitcast [4 x i32] addrspace(5)* %tmp to float addrspace(5)*
  %x = bitcast float addrspace(5)* %b to i32 addrspace(5)*
  %y = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 3
  store i32 0, i32 addrspace(5)* %x
  store i32 1, i32 addrspace(5)* %y
  store i32 2, i32 addrspace(5)* %z
  store i32 3, i32 addrspace(5)* %w
  %tmp1 = getelementptr [4 x i32], [4 x i32] addrspace(5)* %tmp, i32 0, i32 %index
  %tmp2 = load i32, i32 addrspace(5)* %tmp1
  %tmp3 = load i32, i32 addrspace(5)* %x
  %tmp4 = load i32, i32 addrspace(5)* %y
  %add1 = add i32 %tmp2, %tmp3
  %add2 = add i32 %add1, %tmp4
  store i32 %add2, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @bitcast_vector_to_vector(
; OPT-NOT:   alloca
; OPT:       store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(1)* %out, align 16

; GCN-LABEL: {{^}}bitcast_vector_to_vector:
; GCN: v_mov_b32_e32 v0, 1
; GCN: v_mov_b32_e32 v1, 2
; GCN: v_mov_b32_e32 v2, 3
; GCN: v_mov_b32_e32 v3, 4

; GCN: ScratchSize: 0

define amdgpu_kernel void @bitcast_vector_to_vector(<4 x i32> addrspace(1)* %out)  {
.entry:
  %alloca = alloca <4 x float>, align 16, addrspace(5)
  %cast = bitcast <4 x float> addrspace(5)* %alloca to <4 x i32> addrspace(5)*
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(5)* %cast
  %load = load <4 x i32>, <4 x i32> addrspace(5)* %cast, align 16
  store <4 x i32> %load, <4 x i32> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_bitcast_from_alloca_array(
; OPT-NOT:   alloca
; OPT:       store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(1)* %out, align 16

; GCN-LABEL: {{^}}vector_bitcast_from_alloca_array:
; GCN: v_mov_b32_e32 v0, 1
; GCN: v_mov_b32_e32 v1, 2
; GCN: v_mov_b32_e32 v2, 3
; GCN: v_mov_b32_e32 v3, 4

; GCN: ScratchSize: 0

define amdgpu_kernel void @vector_bitcast_from_alloca_array(<4 x i32> addrspace(1)* %out)  {
.entry:
  %alloca = alloca [4 x float], align 16, addrspace(5)
  %cast = bitcast [4 x float] addrspace(5)* %alloca to <4 x i32> addrspace(5)*
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> addrspace(5)* %cast
  %load = load <4 x i32>, <4 x i32> addrspace(5)* %cast, align 16
  store <4 x i32> %load, <4 x i32> addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_bitcast_to_array_from_alloca_array(
; OPT-NOT:   alloca
; OPT:      %out.repack = getelementptr inbounds [4 x i32], [4 x i32] addrspace(1)* %out, i64 0, i64 0
; OPT-NEXT: store i32 1, i32 addrspace(1)* %out.repack, align 4
; OPT-NEXT: %out.repack1 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(1)* %out, i64 0, i64 1
; OPT-NEXT: store i32 2, i32 addrspace(1)* %out.repack1, align 4
; OPT-NEXT: %out.repack2 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(1)* %out, i64 0, i64 2
; OPT-NEXT: store i32 3, i32 addrspace(1)* %out.repack2, align 4
; OPT-NEXT: %out.repack3 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(1)* %out, i64 0, i64 3
; OPT-NEXT: store i32 4, i32 addrspace(1)* %out.repack3, align 4

; GCN-LABEL: {{^}}vector_bitcast_to_array_from_alloca_array:
; GCN: v_mov_b32_e32 v0, 1
; GCN: v_mov_b32_e32 v1, 2
; GCN: v_mov_b32_e32 v2, 3
; GCN: v_mov_b32_e32 v3, 4

; GCN: ScratchSize: 0

define amdgpu_kernel void @vector_bitcast_to_array_from_alloca_array([4 x i32] addrspace(1)* %out)  {
.entry:
  %alloca = alloca [4 x float], align 16, addrspace(5)
  %cast = bitcast [4 x float] addrspace(5)* %alloca to [4 x i32] addrspace(5)*
  store [4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] addrspace(5)* %cast
  %load = load [4 x i32], [4 x i32] addrspace(5)* %cast, align 16
  store [4 x i32] %load, [4 x i32] addrspace(1)* %out
  ret void
}

; OPT-LABEL: @vector_bitcast_to_struct_from_alloca_array(
; OPT-NOT:   alloca
; OPT:      %out.repack = getelementptr inbounds %struct.v4, %struct.v4 addrspace(1)* %out, i64 0, i32 0
; OPT-NEXT: store i32 1, i32 addrspace(1)* %out.repack, align 4
; OPT-NEXT: %out.repack1 = getelementptr inbounds %struct.v4, %struct.v4 addrspace(1)* %out, i64 0, i32 1
; OPT-NEXT: store i32 2, i32 addrspace(1)* %out.repack1, align 4
; OPT-NEXT: %out.repack2 = getelementptr inbounds %struct.v4, %struct.v4 addrspace(1)* %out, i64 0, i32 2
; OPT-NEXT: store i32 3, i32 addrspace(1)* %out.repack2, align 4
; OPT-NEXT: %out.repack3 = getelementptr inbounds %struct.v4, %struct.v4 addrspace(1)* %out, i64 0, i32 3
; OPT-NEXT: store i32 4, i32 addrspace(1)* %out.repack3, align 4

; GCN-LABEL: {{^}}vector_bitcast_to_struct_from_alloca_array:
; GCN: v_mov_b32_e32 v0, 1
; GCN: v_mov_b32_e32 v1, 2
; GCN: v_mov_b32_e32 v2, 3
; GCN: v_mov_b32_e32 v3, 4

; GCN: ScratchSize: 0

%struct.v4 = type { i32, i32, i32, i32 }

define amdgpu_kernel void @vector_bitcast_to_struct_from_alloca_array(%struct.v4 addrspace(1)* %out)  {
.entry:
  %alloca = alloca [4 x float], align 16, addrspace(5)
  %cast = bitcast [4 x float] addrspace(5)* %alloca to %struct.v4 addrspace(5)*
  store %struct.v4 { i32 1, i32 2, i32 3, i32 4 }, %struct.v4 addrspace(5)* %cast
  %load = load %struct.v4, %struct.v4 addrspace(5)* %cast, align 16
  store %struct.v4 %load, %struct.v4 addrspace(1)* %out
  ret void
}

declare void @llvm.lifetime.start.p5i8(i64 immarg, i8 addrspace(5)* nocapture)

declare void @llvm.lifetime.end.p5i8(i64 immarg, i8 addrspace(5)* nocapture)

declare void @llvm.assume(i1)
