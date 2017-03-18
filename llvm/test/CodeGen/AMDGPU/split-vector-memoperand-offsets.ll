; RUN: llc -march=amdgcn -mcpu=hawaii -enable-amdgpu-aa=0 -verify-machineinstrs -mattr=-promote-alloca,-load-store-opt < %s | FileCheck -check-prefix=GCN %s

@sPrivateStorage = internal addrspace(3) global [256 x [8 x <4 x i64>]] undef

; GCN-LABEL: {{^}}ds_reorder_vector_split:

; Write zeroinitializer
; GCN-DAG: ds_write_b64 [[PTR:v[0-9]+]], [[VAL:v\[[0-9]+:[0-9]+\]]] offset:24
; GCN-DAG: ds_write_b64 [[PTR]], [[VAL]] offset:16
; GCN-DAG: ds_write_b64 [[PTR]], [[VAL]] offset:8
; GCN-DAG: ds_write_b64 [[PTR]], [[VAL]]{{$}}

; GCN: s_waitcnt vmcnt

; GCN-DAG: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}} offset:24
; GCN-DAG: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}} offset:16
; GCN-DAG: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}} offset:8
; Appears to be dead store of vector component.
; GCN-DAG: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]$}}


; GCN-DAG: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:8
; GCN-DAG: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:16
; GCN-DAG: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:24

; GCN-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8
; GCN-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16
; GCN-DAG: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:24

; GCN: s_endpgm
define void @ds_reorder_vector_split(<4 x i64> addrspace(1)* nocapture readonly %srcValues, i32 addrspace(1)* nocapture readonly %offsets, <4 x i64> addrspace(1)* nocapture %destBuffer, i32 %alignmentOffset) #0 {
entry:
  %tmp = tail call i32 @llvm.r600.read.local.size.y()
  %tmp1 = tail call i32 @llvm.r600.read.local.size.z()
  %tmp2 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp3 = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp4 = tail call i32 @llvm.amdgcn.workitem.id.z()
  %tmp6 = mul i32 %tmp2, %tmp
  %tmp10 = add i32 %tmp3, %tmp6
  %tmp11 = mul i32 %tmp10, %tmp1
  %tmp9 = add i32 %tmp11, %tmp4
  %x.i.i = tail call i32 @llvm.amdgcn.workgroup.id.x() #1
  %x.i.12.i = tail call i32 @llvm.r600.read.local.size.x() #1
  %mul.26.i = mul i32 %x.i.12.i, %x.i.i
  %add.i = add i32 %tmp2, %mul.26.i
  %arrayidx = getelementptr [256 x [8 x <4 x i64>]], [256 x [8 x <4 x i64>]] addrspace(3)* @sPrivateStorage, i32 0, i32 %tmp9, i32 %add.i
  store <4 x i64> zeroinitializer, <4 x i64> addrspace(3)* %arrayidx
  %tmp12 = sext i32 %add.i to i64
  %arrayidx1 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %srcValues, i64 %tmp12
  %tmp13 = load <4 x i64>, <4 x i64> addrspace(1)* %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %offsets, i64 %tmp12
  %tmp14 = load i32, i32 addrspace(1)* %arrayidx2
  %add.ptr = getelementptr [256 x [8 x <4 x i64>]], [256 x [8 x <4 x i64>]] addrspace(3)* @sPrivateStorage, i32 0, i32 %tmp9, i32 0, i32 %alignmentOffset
  %mul.i = shl i32 %tmp14, 2
  %arrayidx.i = getelementptr inbounds i64, i64 addrspace(3)* %add.ptr, i32 %mul.i
  %tmp15 = bitcast i64 addrspace(3)* %arrayidx.i to <4 x i64> addrspace(3)*
  store <4 x i64> %tmp13, <4 x i64> addrspace(3)* %tmp15
  %add.ptr6 = getelementptr [256 x [8 x <4 x i64>]], [256 x [8 x <4 x i64>]] addrspace(3)* @sPrivateStorage, i32 0, i32 %tmp9, i32 %tmp14, i32 %alignmentOffset
  %tmp16 = sext i32 %tmp14 to i64
  %tmp17 = sext i32 %alignmentOffset to i64
  %add.ptr9 = getelementptr inbounds <4 x i64>, <4 x i64> addrspace(1)* %destBuffer, i64 %tmp16, i64 %tmp17
  %tmp18 = bitcast <4 x i64> %tmp13 to i256
  %trunc = trunc i256 %tmp18 to i64
  store i64 %trunc, i64 addrspace(1)* %add.ptr9
  %arrayidx10.1 = getelementptr inbounds i64, i64 addrspace(3)* %add.ptr6, i32 1
  %tmp19 = load i64, i64 addrspace(3)* %arrayidx10.1
  %arrayidx11.1 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr9, i64 1
  store i64 %tmp19, i64 addrspace(1)* %arrayidx11.1
  %arrayidx10.2 = getelementptr inbounds i64, i64 addrspace(3)* %add.ptr6, i32 2
  %tmp20 = load i64, i64 addrspace(3)* %arrayidx10.2
  %arrayidx11.2 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr9, i64 2
  store i64 %tmp20, i64 addrspace(1)* %arrayidx11.2
  %arrayidx10.3 = getelementptr inbounds i64, i64 addrspace(3)* %add.ptr6, i32 3
  %tmp21 = load i64, i64 addrspace(3)* %arrayidx10.3
  %arrayidx11.3 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr9, i64 3
  store i64 %tmp21, i64 addrspace(1)* %arrayidx11.3
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.z() #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind readnone }
