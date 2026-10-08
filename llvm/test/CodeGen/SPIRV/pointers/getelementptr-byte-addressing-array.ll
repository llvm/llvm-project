; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#int64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec2:]] = OpTypeVector %[[#float]] 2
; CHECK-DAG: %[[#arr3:]] = OpTypeArray %[[#vec2]] %[[#]]
; CHECK-DAG: %[[#ptr_vec:]] = OpTypePointer Private %[[#vec2]]
; CHECK-DAG: %[[#int_16:]] = OpConstant %[[#int32]] 16
; CHECK-DAG: %[[#int_8:]] = OpConstant %[[#int32]] 8

; CHECK-DAG: %[[#short:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#arr5:]] = OpTypeArray %[[#short]] %[[#]]
; CHECK-DAG: %[[#ptr_arr5:]] = OpTypePointer Private %[[#arr5]]

@global_arr = internal addrspace(10) constant [3 x <2 x float>] [<2 x float> zeroinitializer, <2 x float> zeroinitializer, <2 x float> zeroinitializer]
@global_arr_10 = internal addrspace(10) constant [3 x [5 x i16]] [[5 x i16] zeroinitializer, [5 x i16] zeroinitializer, [5 x i16] zeroinitializer]

@in_idx = internal addrspace(10) global i32 zeroinitializer
@out_dyn = internal addrspace(10) global <2 x float> zeroinitializer
@out_const = internal addrspace(10) global <2 x float> zeroinitializer
@out_dyn10 = internal addrspace(10) global [5 x i16] zeroinitializer

define void @main() #0 {
entry:
  %idx = load i32, ptr addrspace(10) @in_idx
  
  ; Dynamic index (stride 8)
  ; CHECK: %[[#idx_dyn:]] = OpLoad %[[#int32]]
  ; CHECK: %[[#access_dyn10:]] = OpInBoundsAccessChain %[[#ptr_arr5]] %[[#]] %[[#idx_dyn]]
  ; CHECK: %[[#access_dyn:]] = OpInBoundsAccessChain %[[#ptr_vec]] %[[#]] %[[#idx_dyn]]
  ; CHECK: %[[#val_dyn:]] = OpLoad %[[#vec2]] %[[#access_dyn]]
  %gep_dyn = getelementptr inbounds nuw [8 x i8], ptr addrspace(10) @global_arr, i32 %idx
  %val_dyn = load <2 x float>, ptr addrspace(10) %gep_dyn
  store <2 x float> %val_dyn, ptr addrspace(10) @out_dyn

  ; Constant index (index 1 -> offset 8 bytes)
  ; CHECK: %[[#access_const:]] = OpInBoundsAccessChain %[[#ptr_vec]] %[[#]] %[[#]]
  ; CHECK: %[[#val_const:]] = OpLoad %[[#vec2]] %[[#access_const]]
  %gep_const = getelementptr inbounds nuw [8 x i8], ptr addrspace(10) @global_arr, i32 1
  %val_const = load <2 x float>, ptr addrspace(10) %gep_const
  store <2 x float> %val_const, ptr addrspace(10) @out_const

  ; Dynamic index (stride 10)
  ; CHECK: %[[#val_dyn10:]] = OpLoad %[[#arr5]] %[[#access_dyn10]]
  %gep_dyn10 = getelementptr inbounds nuw [10 x i8], ptr addrspace(10) @global_arr_10, i32 %idx
  %val_dyn10 = load [5 x i16], ptr addrspace(10) %gep_dyn10
  store [5 x i16] %val_dyn10, ptr addrspace(10) @out_dyn10

  ; Dynamic index (stride 16 on size 8 array)
  ; CHECK: %[[#mul:]] = OpIMul %[[#int32]] %[[#idx_dyn]] %[[#int_16]]
  ; CHECK: %[[#div:]] = OpUDiv %[[#int32]] %[[#mul]] %[[#int_8]]
  ; CHECK: %[[#access_dyn:]] = OpInBoundsAccessChain %[[#ptr_vec]] %[[#]] %[[#div]]
  ; CHECK: %[[#val_dyn10:]] = OpLoad %[[#vec2]] %[[#access_dyn]]
  %gep_dyn16 = getelementptr inbounds nuw [16 x i8], ptr addrspace(10) @global_arr, i32 %idx
  %val_dyn16 = load <2 x float>, ptr addrspace(10) %gep_dyn16
  store <2 x float> %val_dyn16, ptr addrspace(10) @out_dyn


  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
