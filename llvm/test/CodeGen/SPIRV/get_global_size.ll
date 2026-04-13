; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#int32:]] = OpTypeInt 32 0
; CHECK: %[[#int64:]] = OpTypeInt 64 0
; CHECK: %[[#vec3:]] = OpTypeVector %[[#int64]] 3
; CHECK: %[[#ptr_input_vec3:]] = OpTypePointer Input %[[#vec3]]
; CHECK: %[[#global_size_var:]] = OpVariable %[[#ptr_input_vec3]] Input

; CHECK: %[[#load_gs1:]] = OpLoad %[[#vec3]] %[[#global_size_var]] Aligned 1
; CHECK: %[[#extract3:]] = OpCompositeExtract %[[#int64]] %[[#load_gs1]] 0

; CHECK: %[[#bitcast1:]] = OpBitcast %[[#]] %[[#]]
; CHECK: %[[#load_out1:]] = OpLoad %[[#]] %[[#bitcast1]] Aligned 8
; CHECK: %[[#gep1:]] = OpInBoundsPtrAccessChain %[[#]] %[[#load_out1]] %[[#]]
; CHECK: OpStore %[[#gep1]] %[[#extract3]] Aligned 8

; CHECK: %[[#load_param_x:]] = OpLoad %[[#int32]] %[[#]]
; CHECK: %[[#load_gs2:]] = OpLoad %[[#vec3]] %[[#global_size_var]] Aligned 1
; CHECK: %[[#dyn_extract:]] = OpVectorExtractDynamic %[[#int64]] %[[#load_gs2]] %[[#load_param_x]]
; CHECK: %[[#cmp:]] = OpULessThan %[[#]] %[[#load_param_x]] %[[#]]
; CHECK: %[[#select2:]] = OpSelect %[[#int64]] %[[#cmp]] %[[#dyn_extract]] %[[#]]
; CHECK: %[[#bitcast2:]] = OpBitcast %[[#]] %[[#]]
; CHECK: %[[#load_out2:]] = OpLoad %[[#]] %[[#bitcast2]] Aligned 8
; CHECK: %[[#gep2:]] = OpInBoundsPtrAccessChain %[[#]] %[[#load_out2]] %[[#]]
; CHECK: OpStore %[[#gep2]] %[[#select2]] Aligned 8

define dso_local spir_kernel void @ggs(ptr noundef align 8 %out, i32 noundef %x) {
entry:
  %out.addr = alloca ptr, align 8
  %x.addr = alloca i32, align 4
  store ptr %out, ptr %out.addr, align 8
  store i32 %x, ptr %x.addr, align 4
  %call = call i64 @_Z15get_global_sizej(i32 noundef 0)
  %0 = load ptr, ptr %out.addr, align 8
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 0
  store i64 %call, ptr %arrayidx, align 8
  %call1 = call i64 @_Z15get_global_sizej(i32 noundef 3)
  %1 = load ptr, ptr %out.addr, align 8
  %arrayidx2 = getelementptr inbounds i64, ptr %1, i64 1
  store i64 %call1, ptr %arrayidx2, align 8
  %2 = load i32, ptr %x.addr, align 4
  %call3 = call i64 @_Z15get_global_sizej(i32 noundef %2)
  %3 = load ptr, ptr %out.addr, align 8
  %arrayidx4 = getelementptr inbounds i64, ptr %3, i64 2
  store i64 %call3, ptr %arrayidx4, align 8
  ret void
}

declare i64 @_Z15get_global_sizej(i32 noundef)
