; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_subgroup_rotate %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_subgroup_rotate %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpGroupNonUniformRotateKHR instruction requires the following SPIR-V extension: SPV_KHR_subgroup_rotate

; CHECK: OpCapability GroupNonUniformRotateKHR
; CHECK: OpExtension "SPV_KHR_subgroup_rotate"

; CHECK-DAG: %[[TyInt8:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[TyInt16:.*]] = OpTypeInt 16 0
; CHECK-DAG: %[[TyInt32:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[TyInt64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[TyFloat:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[TyHalf:.*]] = OpTypeFloat 16
; CHECK-DAG: %[[TyDouble:.*]] = OpTypeFloat 64
; CHECK-DAG: %[[ScopeSubgroup:.*]] = OpConstant %[[TyInt32]] 3
; CHECK-DAG: %[[ConstInt2:.*]] = OpConstant %[[TyInt32]] 2
; CHECK-DAG: %[[ConstInt4:.*]] = OpConstant %[[TyInt32]] 4

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateChar(ptr addrspace(1) noundef align 1 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i8, align 1
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i8 0, ptr %v, align 1
  %value = load i8, ptr %v, align 1
; CHECK: OpGroupNonUniformRotateKHR %[[TyInt8]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func signext i8 @_Z16sub_group_rotateci(i8 noundef signext %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %data, i32 0
  store i8 %call, ptr addrspace(1) %arrayidx, align 1
  %value_clustered = load i8, ptr %v, align 1
; CHECK: OpGroupNonUniformRotateKHR %[[TyInt8]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func signext i8 @_Z26sub_group_clustered_rotatecij(i8 noundef signext %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr addrspace(1) %data2, i32 1
  store i8 %call1, ptr addrspace(1) %arrayidx2, align 1
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func signext i8 @_Z16sub_group_rotateci(i8 noundef signext, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func signext i8 @_Z26sub_group_clustered_rotatecij(i8 noundef signext, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateUChar(ptr addrspace(1) noundef align 1 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i8, align 1
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i8 0, ptr %v, align 1
  %value = load i8, ptr %v, align 1
; CHECK: OpGroupNonUniformRotateKHR %[[TyInt8]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func zeroext i8 @_Z16sub_group_rotatehi(i8 noundef zeroext %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %data, i32 0
  store i8 %call, ptr addrspace(1) %arrayidx, align 1
  %value_clustered = load i8, ptr %v, align 1
; CHECK: OpGroupNonUniformRotateKHR %[[TyInt8]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func zeroext i8 @_Z26sub_group_clustered_rotatehij(i8 noundef zeroext %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i8, ptr addrspace(1) %data2, i32 1
  store i8 %call1, ptr addrspace(1) %arrayidx2, align 1
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func zeroext i8 @_Z16sub_group_rotatehi(i8 noundef zeroext, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func zeroext i8 @_Z26sub_group_clustered_rotatehij(i8 noundef zeroext, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateShort(ptr addrspace(1) noundef align 2 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i16, align 2
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i16 0, ptr %v, align 2
  %value = load i16, ptr %v, align 2
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt16]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func signext i16 @_Z16sub_group_rotatesi(i16 noundef signext %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i16, ptr addrspace(1) %data, i32 0
  store i16 %call, ptr addrspace(1) %arrayidx, align 2
  %value_clustered = load i16, ptr %v, align 2
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt16]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func signext i16 @_Z26sub_group_clustered_rotatesij(i16 noundef signext %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i16, ptr addrspace(1) %data2, i32 1
  store i16 %call1, ptr addrspace(1) %arrayidx2, align 2
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func signext i16 @_Z16sub_group_rotatesi(i16 noundef signext, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func signext i16 @_Z26sub_group_clustered_rotatesij(i16 noundef signext, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateUShort(ptr addrspace(1) noundef align 2 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !9 !kernel_arg_base_type !9 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i16, align 2
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i16 0, ptr %v, align 2
  %value = load i16, ptr %v, align 2
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt16]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func zeroext i16 @_Z16sub_group_rotateti(i16 noundef zeroext %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i16, ptr addrspace(1) %data, i32 0
  store i16 %call, ptr addrspace(1) %arrayidx, align 2
  %value_clustered = load i16, ptr %v, align 2
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt16]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func zeroext i16 @_Z26sub_group_clustered_rotatetij(i16 noundef zeroext %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i16, ptr addrspace(1) %data2, i32 1
  store i16 %call1, ptr addrspace(1) %arrayidx2, align 2
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func zeroext i16 @_Z16sub_group_rotateti(i16 noundef zeroext, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func zeroext i16 @_Z26sub_group_clustered_rotatetij(i16 noundef zeroext, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateInt(ptr addrspace(1) noundef align 4 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i32, align 4
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i32 0, ptr %v, align 4
  %value = load i32, ptr %v, align 4
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt32]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func i32 @_Z16sub_group_rotateii(i32 noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %data, i32 0
  store i32 %call, ptr addrspace(1) %arrayidx, align 4
  %value_clustered = load i32, ptr %v, align 4
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt32]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func i32 @_Z26sub_group_clustered_rotateiij(i32 noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %data2, i32 1
  store i32 %call1, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z16sub_group_rotateii(i32 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z26sub_group_clustered_rotateiij(i32 noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateUInt(ptr addrspace(1) noundef align 4 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i32, align 4
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i32 0, ptr %v, align 4
  %value = load i32, ptr %v, align 4
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt32]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func i32 @_Z16sub_group_rotateji(i32 noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %data, i32 0
  store i32 %call, ptr addrspace(1) %arrayidx, align 4
  %value_clustered = load i32, ptr %v, align 4
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt32]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func i32 @_Z26sub_group_clustered_rotatejij(i32 noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %data2, i32 1
  store i32 %call1, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z16sub_group_rotateji(i32 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func i32 @_Z26sub_group_clustered_rotatejij(i32 noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateLong(ptr addrspace(1) noundef align 8 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !12 !kernel_arg_base_type !12 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i64, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i64 0, ptr %v, align 8
  %value = load i64, ptr %v, align 8
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt64]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func i64 @_Z16sub_group_rotateli(i64 noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i64, ptr addrspace(1) %data, i32 0
  store i64 %call, ptr addrspace(1) %arrayidx, align 8
  %value_clustered = load i64, ptr %v, align 8
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt64]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func i64 @_Z26sub_group_clustered_rotatelij(i64 noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i64, ptr addrspace(1) %data2, i32 1
  store i64 %call1, ptr addrspace(1) %arrayidx2, align 8
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z16sub_group_rotateli(i64 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z26sub_group_clustered_rotatelij(i64 noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateULong(ptr addrspace(1) noundef align 8 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !13 !kernel_arg_base_type !13 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca i64, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store i64 0, ptr %v, align 8
  %value = load i64, ptr %v, align 8
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt64]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func i64 @_Z16sub_group_rotatemi(i64 noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds i64, ptr addrspace(1) %data, i32 0
  store i64 %call, ptr addrspace(1) %arrayidx, align 8
  %value_clustered = load i64, ptr %v, align 8
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyInt64]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func i64 @_Z26sub_group_clustered_rotatemij(i64 noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds i64, ptr addrspace(1) %data2, i32 1
  store i64 %call1, ptr addrspace(1) %arrayidx2, align 8
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z16sub_group_rotatemi(i64 noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z26sub_group_clustered_rotatemij(i64 noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateFloat(ptr addrspace(1) noundef align 4 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca float, align 4
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store float 0.000000e+00, ptr %v, align 4
  %value = load float, ptr %v, align 4
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyFloat]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func float @_Z16sub_group_rotatefi(float noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %data, i32 0
  store float %call, ptr addrspace(1) %arrayidx, align 4
  %value_clustered = load float, ptr %v, align 4
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyFloat]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func float @_Z26sub_group_clustered_rotatefij(float noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds float, ptr addrspace(1) %data2, i32 1
  store float %call1, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func float @_Z16sub_group_rotatefi(float noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func float @_Z26sub_group_clustered_rotatefij(float noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateHalf(ptr addrspace(1) noundef align 2 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca half, align 2
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store half 0xH0000, ptr %v, align 2
  %value = load half, ptr %v, align 2
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyHalf]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func half @_Z16sub_group_rotateDhi(half noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds half, ptr addrspace(1) %data, i32 0
  store half %call, ptr addrspace(1) %arrayidx, align 2
  %value_clustered = load half, ptr %v, align 2
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyHalf]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func half @_Z26sub_group_clustered_rotateDhij(half noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds half, ptr addrspace(1) %data2, i32 1
  store half %call1, ptr addrspace(1) %arrayidx2, align 2
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func half @_Z16sub_group_rotateDhi(half noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func half @_Z26sub_group_clustered_rotateDhij(half noundef, i32 noundef, i32 noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @testRotateDouble(ptr addrspace(1) noundef align 8 %dst) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !16 !kernel_arg_base_type !16 !kernel_arg_type_qual !6 {
entry:
  %dst.addr = alloca ptr addrspace(1), align 4
  %v = alloca double, align 8
  store ptr addrspace(1) %dst, ptr %dst.addr, align 4
  store double 0.000000e+00, ptr %v, align 8
  %value = load double, ptr %v, align 8
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyDouble]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]]
  %call = call spir_func double @_Z16sub_group_rotatedi(double noundef %value, i32 noundef 2) #2
  %data = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx = getelementptr inbounds double, ptr addrspace(1) %data, i32 0
  store double %call, ptr addrspace(1) %arrayidx, align 8
  %value_clustered = load double, ptr %v, align 8
  ; CHECK: OpGroupNonUniformRotateKHR %[[TyDouble]] %[[ScopeSubgroup]] %[[#]] %[[ConstInt2]] %[[ConstInt4]]
  %call1 = call spir_func double @_Z26sub_group_clustered_rotatedij(double noundef %value_clustered, i32 noundef 2, i32 noundef 4) #2
  %data2 = load ptr addrspace(1), ptr %dst.addr, align 4
  %arrayidx2 = getelementptr inbounds double, ptr addrspace(1) %data2, i32 1
  store double %call1, ptr addrspace(1) %arrayidx2, align 8
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func double @_Z16sub_group_rotatedi(double noundef, i32 noundef) #1

; Function Attrs: convergent nounwind
declare spir_func double @_Z26sub_group_clustered_rotatedij(double noundef, i32 noundef, i32 noundef) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 19.0.0"}
!3 = !{i32 1}
!4 = !{!"none"}
!5 = !{!"char*"}
!6 = !{!""}
!7 = !{!"uchar*"}
!8 = !{!"short*"}
!9 = !{!"ushort*"}
!10 = !{!"int*"}
!11 = !{!"uint*"}
!12 = !{!"long*"}
!13 = !{!"ulong*"}
!14 = !{!"float*"}
!15 = !{!"half*"}
!16 = !{!"double*"}
