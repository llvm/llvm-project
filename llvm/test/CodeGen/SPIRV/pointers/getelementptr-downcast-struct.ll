; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:       %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:     %[[#uint64:]] = OpTypeInt 64 0
; CHECK-DAG:    %[[#uint_pp:]] = OpTypePointer Private %[[#uint]]
; CHECK-DAG:     %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:     %[[#uint_1:]] = OpConstant %[[#uint]] 1
; CHECK-DAG:    %[[#uint_10:]] = OpConstant %[[#uint]] 10
; CHECK-DAG:    %[[#t_array:]] = OpTypeArray %[[#uint]] %[[#uint_10]]
; CHECK-DAG:       %[[#t_s1:]] = OpTypeStruct %[[#t_array]]
; CHECK-DAG:       %[[#t_s2_s_a_s:]] = OpTypeStruct %[[#uint]] %[[#uint]]
; CHECK-DAG:       %[[#t_s2_s_a:]] = OpTypeArray %[[#t_s2_s_a_s]] %[[#uint_10]]
; CHECK-DAG:       %[[#t_s2_s:]] = OpTypeStruct %[[#t_s2_s_a]]
; CHECK-DAG:       %[[#t_s2:]] = OpTypeStruct %[[#t_s2_s]] %[[#uint]]
; CHECK-DAG:     %[[#null_s1:]] = OpConstantNull %[[#t_s1]]
; CHECK-DAG:     %[[#null_s2:]] = OpConstantNull %[[#t_s2]]
; CHECK-DAG:  %[[#ptr_s1:]] = OpTypePointer Private %[[#t_s1]]
; CHECK-DAG:  %[[#ptr_s2:]] = OpTypePointer Private %[[#t_s2]]

%S1 = type { [10 x i32] }
%S2 = type { { [10 x { i32, i32 } ] }, i32 }

; CHECK-DAG: %[[#global1:]] = OpVariable %[[#ptr_s1]] Private %[[#null_s1]]
@global1 = internal addrspace(10) global %S1 zeroinitializer
; CHECK-DAG: %[[#global2:]] = OpVariable %[[#ptr_s2]] Private %[[#null_s2]]
@global2 = internal addrspace(10) global %S2 zeroinitializer

define spir_func noundef i32 @foo(i64 noundef %index) local_unnamed_addr {
; CHECK: %[[#index:]] = OpFunctionParameter %[[#uint64]]
entry:
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#uint_pp]] %[[#global1]] %[[#uint_0]] %[[#index]]
  %ptr = getelementptr inbounds %S1, ptr addrspace(10) @global1, i64 0, i32 0, i64 %index
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]] Aligned 4
  %val = load i32, ptr addrspace(10) %ptr
  ret i32 %val
}

define spir_func noundef i32 @bar(i64 noundef %index) local_unnamed_addr {
; CHECK: %[[#index:]] = OpFunctionParameter %[[#uint64]]
entry:
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#uint_pp]] %[[#global2]] %[[#uint_0]] %[[#uint_0]] %[[#index]] %[[#uint_1]]
  %ptr = getelementptr inbounds %S2, ptr addrspace(10) @global2, i64 0, i32 0, i32 0, i64 %index, i32 1
; CHECK: %[[#val:]] = OpLoad %[[#uint]] %[[#ptr]] Aligned 4
  %val = load i32, ptr addrspace(10) %ptr
  ret i32 %val
}

define spir_func void @foos(i64 noundef %index) local_unnamed_addr {
; CHECK: %[[#index:]] = OpFunctionParameter %[[#uint64]]
entry:
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#uint_pp]] %[[#global1]] %[[#uint_0]] %[[#index]]
  %ptr = getelementptr inbounds %S1, ptr addrspace(10) @global1, i64 0, i32 0, i64 %index
; CHECK: OpStore %[[#ptr]] %[[#uint_0]] Aligned 4
  store i32 0, ptr addrspace(10) %ptr
  ret void
}

define spir_func void @bars(i64 noundef %index) local_unnamed_addr {
; CHECK: %[[#index:]] = OpFunctionParameter %[[#uint64]]
entry:
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#uint_pp]] %[[#global2]] %[[#uint_0]] %[[#uint_0]] %[[#index]] %[[#uint_1]]
  %ptr = getelementptr inbounds %S2, ptr addrspace(10) @global2, i64 0, i32 0, i32 0, i64 %index, i32 1
; CHECK: OpStore %[[#ptr]] %[[#uint_0]] Aligned 4
  store i32 0, ptr addrspace(10) %ptr
  ret void
}
