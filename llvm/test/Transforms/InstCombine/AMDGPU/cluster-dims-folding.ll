; RUN: opt -mtriple=amdgpu12.50-amd-amdhsa -S -passes=instcombine %s | FileCheck %s

declare i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
declare i32 @llvm.amdgcn.cluster.workgroup.max.id.y()
declare i32 @llvm.amdgcn.cluster.workgroup.max.id.z()
declare i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()
declare i32 @llvm.amdgcn.cluster.workgroup.id.x()

; CHECK-LABEL: @fixed_dims(
; CHECK: store <4 x i32> <i32 3, i32 3, i32 0, i32 15>, ptr addrspace(1) %out
define amdgpu_kernel void @fixed_dims(ptr addrspace(1) %out) #0 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  %y = call i32 @llvm.amdgcn.cluster.workgroup.max.id.y()
  %z = call i32 @llvm.amdgcn.cluster.workgroup.max.id.z()
  %flat = call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()
  %v0 = insertelement <4 x i32> poison, i32 %x, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %flat, i32 3
  store <4 x i32> %v3, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @fixed_dims_3d(
; CHECK: store <4 x i32> <i32 1, i32 2, i32 1, i32 11>, ptr addrspace(1) %out
define amdgpu_kernel void @fixed_dims_3d(ptr addrspace(1) %out) #1 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  %y = call i32 @llvm.amdgcn.cluster.workgroup.max.id.y()
  %z = call i32 @llvm.amdgcn.cluster.workgroup.max.id.z()
  %flat = call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()
  %v0 = insertelement <4 x i32> poison, i32 %x, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %flat, i32 3
  store <4 x i32> %v3, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @unit_dims(
; CHECK: store <4 x i32> zeroinitializer, ptr addrspace(1) %out
define amdgpu_kernel void @unit_dims(ptr addrspace(1) %out) #2 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  %y = call i32 @llvm.amdgcn.cluster.workgroup.max.id.y()
  %z = call i32 @llvm.amdgcn.cluster.workgroup.max.id.z()
  %flat = call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()
  %v0 = insertelement <4 x i32> poison, i32 %x, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %flat, i32 3
  store <4 x i32> %v3, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @dynamic_dims(
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.id.y()
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.id.z()
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()
define amdgpu_kernel void @dynamic_dims(ptr addrspace(1) %out) #3 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  %y = call i32 @llvm.amdgcn.cluster.workgroup.max.id.y()
  %z = call i32 @llvm.amdgcn.cluster.workgroup.max.id.z()
  %flat = call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()
  %v0 = insertelement <4 x i32> poison, i32 %x, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %y, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %z, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %flat, i32 3
  store <4 x i32> %v3, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @disabled_cluster(
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
define amdgpu_kernel void @disabled_cluster(ptr addrspace(1) %out) #4 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  store i32 %x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @unknown_dims(
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
define amdgpu_kernel void @unknown_dims(ptr addrspace(1) %out) {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  store i32 %x, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @non_kernel(
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
define i32 @non_kernel() #0 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  ret i32 %x
}

; CHECK-LABEL: @block_id_stays_dynamic(
; CHECK: call i32 @llvm.amdgcn.cluster.workgroup.id.x()
; CHECK: store i32 3, ptr addrspace(1) %out
define amdgpu_kernel void @block_id_stays_dynamic(ptr addrspace(1) %out, ptr addrspace(1) %id_out) #0 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  %id = call i32 @llvm.amdgcn.cluster.workgroup.id.x()
  store i32 %x, ptr addrspace(1) %out
  store i32 %id, ptr addrspace(1) %id_out
  ret void
}

define internal i32 @cluster_size_helper() #5 {
  %x = call i32 @llvm.amdgcn.cluster.workgroup.max.id.x()
  ret i32 %x
}

; INLINE-LABEL: @fold_after_inlining(
; INLINE: store i32 3, ptr addrspace(1) %out
; INLINE-NEXT: ret void
define amdgpu_kernel void @fold_after_inlining(ptr addrspace(1) %out) #0 {
  %x = call i32 @cluster_size_helper()
  store i32 %x, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-cluster-dims"="4,4,1" }
attributes #1 = { "amdgpu-cluster-dims"="2,3,2" }
attributes #2 = { "amdgpu-cluster-dims"="1,1,1" }
attributes #3 = { "amdgpu-cluster-dims"="1024,1024,1024" }
attributes #4 = { "amdgpu-cluster-dims"="0,0,0" }
attributes #5 = { alwaysinline }
