; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpName %[[#idx:]] "idx"
; CHECK-DAG: OpName %[[#idx2:]] "idx2"
; CHECK-DAG: OpName %[[#val:]] "val"

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#c6:]] = OpConstant %[[#int]] 6
; CHECK-DAG: %[[#arr6int:]] = OpTypeArray %[[#int]] %[[#c6]]
; CHECK-DAG: %[[#ptr_arr6int:]] = OpTypePointer Function %[[#arr6int]]
; CHECK-DAG: %[[#ptr_f_int:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG: %[[#ptr_p_int:]] = OpTypePointer Private %[[#int]]
; CHECK-DAG: %[[#c0:]] = OpConstant %[[#int]] 0
; CHECK-DAG: %[[#c1:]] = OpConstant %[[#int]] 1
; CHECK-DAG: %[[#c2:]] = OpConstant %[[#int]] 2
; CHECK-DAG: %[[#c3:]] = OpConstant %[[#int]] 3
; CHECK-DAG: %[[#c4:]] = OpConstant %[[#int]] 4
; CHECK-DAG: %[[#c5:]] = OpConstant %[[#int]] 5
; CHECK-DAG: %[[#undef:]] = OpUndef %[[#int]]

; CHECK-DAG: %[[#out:]] = OpVariable %[[#ptr_p_int]] Private


@out = internal addrspace(10) global i32 0
@idx = internal addrspace(10) global i32 0
@idx2 = internal addrspace(10) global i32 0
@val = internal addrspace(10) global i32 0

; CHECK: %[[#test_full:]] = OpFunction %[[#]] None %[[#]]
define void @test_full() #0 {
  ; CHECK:      %[[#label:]] = OpLabel
  ; CHECK-DAG:  %[[#v_ins:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#v_ext:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#i0:]] = OpLoad %[[#int]] %[[#idx]]
  ; CHECK-DAG:  %[[#i1:]] = OpLoad %[[#int]] %[[#idx2]]
  ; CHECK-DAG:  %[[#val_val:]] = OpLoad %[[#int]] %[[#val]]
  ; CHECK-DAG:  %[[#idx64:]] = OpUConvert %[[#long]] %[[#i0]]
  ; CHECK-DAG:  %[[#idx2_64:]] = OpUConvert %[[#long]] %[[#i1]]

  %idx = load i32, ptr addrspace(10) @idx
  %idx2 = load i32, ptr addrspace(10) @idx2
  %val = load i32, ptr addrspace(10) @val

  %ptr = alloca <6 x i32>
  %loaded = load <6 x i32>, ptr %ptr
  %idx64 = zext i32 %idx to i64
  %idx2_64 = zext i32 %idx2 to i64

  ; Insertelement with dynamic index spills to stack
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c0]]
; CHECK:  OpStore %[[#ac]] %[[#undef]] Aligned 32
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c1]]
; CHECK:  OpStore %[[#ac]] %[[#undef]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c2]]
; CHECK:  OpStore %[[#ac]] %[[#undef]] Aligned 8
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c3]]
; CHECK:  OpStore %[[#ac]] %[[#undef]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c4]]
; CHECK:  OpStore %[[#ac]] %[[#undef]] Aligned 16
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c5]]
; CHECK:  OpStore %[[#ac]] %[[#undef]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#idx64]]
; CHECK:  OpStore %[[#ac]] %[[#val_val]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c0]]
; CHECK:  %[[#v0:]] = OpLoad %[[#int]] %[[#ac]] Aligned 32
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c1]]
; CHECK:  %[[#v1:]] = OpLoad %[[#int]] %[[#ac]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c2]]
; CHECK:  %[[#v2:]] = OpLoad %[[#int]] %[[#ac]] Aligned 8
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c3]]
; CHECK:  %[[#v3:]] = OpLoad %[[#int]] %[[#ac]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c4]]
; CHECK:  %[[#v4:]] = OpLoad %[[#int]] %[[#ac]] Aligned 16
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ins]] %[[#c5]]
; CHECK:  %[[#v5:]] = OpLoad %[[#int]] %[[#ac]] Aligned 4
  %inserted = insertelement <6 x i32> %loaded, i32 %val, i64 %idx64

  ; Extractelement with dynamic index spills to stack


; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#c0]]
; CHECK:  OpStore %[[#ac]] %[[#v0]] Aligned 32
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#c1]]
; CHECK:  OpStore %[[#ac]] %[[#v1]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#c2]]
; CHECK:  OpStore %[[#ac]] %[[#v2]] Aligned 8
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#c3]]
; CHECK:  OpStore %[[#ac]] %[[#v3]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#c4]]
; CHECK:  OpStore %[[#ac]] %[[#v4]] Aligned 16
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#c5]]
; CHECK:  OpStore %[[#ac]] %[[#v5]] Aligned 4
; CHECK:  %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_ext]] %[[#idx2_64]]
; CHECK:  %[[#extracted:]] = OpLoad %[[#int]] %[[#ac]] Aligned 4
  %extracted = extractelement <6 x i32> %inserted, i64 %idx2_64

  ; CHECK: OpStore %[[#out]] %[[#extracted]]
  store i32 %extracted, ptr addrspace(10) @out
  ret void
}

; CHECK: %[[#test_undef:]] = OpFunction %[[#]] None %[[#]]
define void @test_undef() #0 {
  ; CHECK:      %[[#label:]] = OpLabel
  ; CHECK:      %[[#v_undef:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#idx_val:]] = OpLoad %[[#int]] %[[#idx]]
  ; CHECK-DAG:  %[[#val_val:]] = OpLoad %[[#int]] %[[#val]]
  ; CHECK:      %[[#idx64:]] = OpUConvert %[[#long]] %[[#idx_val]]
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c0]]
  ; CHECK:      OpStore %[[#ac]] %[[#undef]] Aligned 32
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c1]]
  ; CHECK:      OpStore %[[#ac]] %[[#undef]] Aligned 4
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c2]]
  ; CHECK:      OpStore %[[#ac]] %[[#undef]] Aligned 8
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c3]]
  ; CHECK:      OpStore %[[#ac]] %[[#undef]] Aligned 4
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c4]]
  ; CHECK:      OpStore %[[#ac]] %[[#undef]] Aligned 16
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c5]]
  ; CHECK:      OpStore %[[#ac]] %[[#undef]] Aligned 4
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#idx64]]
  ; CHECK:      OpStore %[[#ac]] %[[#val_val]] Aligned 4
  ; CHECK:      %[[#ptr0:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_undef]] %[[#c0]]
  ; CHECK:      %[[#res:]] = OpLoad %[[#int]] %[[#ptr0]] Aligned 32
  ; CHECK:      OpStore %[[#out]] %[[#res]]
  %idx = load i32, ptr addrspace(10) @idx
  %val = load i32, ptr addrspace(10) @val
  %idx64 = zext i32 %idx to i64
  %inserted = insertelement <6 x i32> poison, i32 %val, i64 %idx64
  %extracted = extractelement <6 x i32> %inserted, i64 0
  store i32 %extracted, ptr addrspace(10) @out
  ret void
}

; CHECK: %[[#test_zero:]] = OpFunction %[[#]] None %[[#]]
define void @test_zero() #0 {
  ; CHECK:      %[[#label:]] = OpLabel
  ; CHECK:      %[[#v_zero:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#idx_val:]] = OpLoad %[[#int]] %[[#idx]]
  ; CHECK-DAG:  %[[#val_val:]] = OpLoad %[[#int]] %[[#val]]
  ; CHECK:      %[[#idx64:]] = OpUConvert %[[#long]] %[[#idx_val]]
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c0]]
  ; CHECK:      OpStore %[[#ac]] %[[#c0]] Aligned 32
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c1]]
  ; CHECK:      OpStore %[[#ac]] %[[#c0]] Aligned 4
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c2]]
  ; CHECK:      OpStore %[[#ac]] %[[#c0]] Aligned 8
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c3]]
  ; CHECK:      OpStore %[[#ac]] %[[#c0]] Aligned 4
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c4]]
  ; CHECK:      OpStore %[[#ac]] %[[#c0]] Aligned 16
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c5]]
  ; CHECK:      OpStore %[[#ac]] %[[#c0]] Aligned 4
  ; CHECK:      %[[#ac:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#idx64]]
  ; CHECK:      OpStore %[[#ac]] %[[#val_val]] Aligned 4
  ; CHECK:      %[[#ptr0:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#v_zero]] %[[#c0]]
  ; CHECK:      %[[#res:]] = OpLoad %[[#int]] %[[#ptr0]] Aligned 32
  ; CHECK:      OpStore %[[#out]] %[[#res]]
  %idx = load i32, ptr addrspace(10) @idx
  %val = load i32, ptr addrspace(10) @val
  %idx64 = zext i32 %idx to i64
  %inserted = insertelement <6 x i32> zeroinitializer, i32 %val, i64 %idx64
  %extracted = extractelement <6 x i32> %inserted, i64 0
  store i32 %extracted, ptr addrspace(10) @out
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
