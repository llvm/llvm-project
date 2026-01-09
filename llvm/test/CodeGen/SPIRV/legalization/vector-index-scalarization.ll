; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
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
  ; CHECK-DAG:  %[[#V_INS:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#V_EXT:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#]] = OpLoad %[[#int]] %[[#]]
  ; CHECK-DAG:  %[[#]] = OpLoad %[[#int]] %[[#]]
  ; CHECK-DAG:  %[[#VAL:]] = OpLoad %[[#int]] %[[#]]
  ; CHECK-DAG:  %[[#IDX64:]] = OpUConvert %[[#]] %[[#]]
  ; CHECK-DAG:  %[[#IDX2_64:]] = OpUConvert %[[#]] %[[#]]


  %idx = load i32, ptr addrspace(10) @idx
  %idx2 = load i32, ptr addrspace(10) @idx2
  %val = load i32, ptr addrspace(10) @val

  %ptr = alloca <6 x i32>
  %loaded = load <6 x i32>, ptr %ptr
  %idx64 = zext i32 %idx to i64
  %idx2_64 = zext i32 %idx2 to i64

  ; Insertelement with dynamic index spills to stack
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c0]]
; CHECK:  OpStore %[[#]] %[[#undef]] Aligned 32
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c1]]
; CHECK:  OpStore %[[#]] %[[#undef]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c2]]
; CHECK:  OpStore %[[#]] %[[#undef]] Aligned 8
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c3]]
; CHECK:  OpStore %[[#]] %[[#undef]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c4]]
; CHECK:  OpStore %[[#]] %[[#undef]] Aligned 16
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c5]]
; CHECK:  OpStore %[[#]] %[[#undef]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#IDX64]]
; CHECK:  OpStore %[[#]] %[[#VAL]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c0]]
; CHECK:  %[[#V0:]] = OpLoad %[[#int]] %[[#]] Aligned 32
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c1]]
; CHECK:  %[[#V1:]] = OpLoad %[[#int]] %[[#]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c2]]
; CHECK:  %[[#V2:]] = OpLoad %[[#int]] %[[#]] Aligned 8
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c3]]
; CHECK:  %[[#V3:]] = OpLoad %[[#int]] %[[#]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c4]]
; CHECK:  %[[#V4:]] = OpLoad %[[#int]] %[[#]] Aligned 16
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_INS]] %[[#c5]]
; CHECK:  %[[#V5:]] = OpLoad %[[#int]] %[[#]] Aligned 4
  %inserted = insertelement <6 x i32> %loaded, i32 %val, i64 %idx64

  ; Extractelement with dynamic index spills to stack


; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#c0]]
; CHECK:  OpStore %[[#]] %[[#V0]] Aligned 32
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#c1]]
; CHECK:  OpStore %[[#]] %[[#V1]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#c2]]
; CHECK:  OpStore %[[#]] %[[#V2]] Aligned 8
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#c3]]
; CHECK:  OpStore %[[#]] %[[#V3]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#c4]]
; CHECK:  OpStore %[[#]] %[[#V4]] Aligned 16
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#c5]]
; CHECK:  OpStore %[[#]] %[[#V5]] Aligned 4
; CHECK:  %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_EXT]] %[[#IDX2_64]]
; CHECK:  %[[#EXTRACTED:]] = OpLoad %[[#int]] %[[#]] Aligned 4
  %extracted = extractelement <6 x i32> %inserted, i64 %idx2_64

  ; CHECK: OpStore %[[#out]] %[[#EXTRACTED]]
  store i32 %extracted, ptr addrspace(10) @out
  ret void
}

; CHECK: %[[#test_undef:]] = OpFunction %[[#]] None %[[#]]
define void @test_undef() #0 {
  ; CHECK:      %[[#label:]] = OpLabel
  ; CHECK:      %[[#V_UNDEF:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#IDX:]] = OpLoad %[[#int]] %[[#]]
  ; CHECK-DAG:  %[[#VAL:]] = OpLoad %[[#int]] %[[#]]
  ; CHECK:      %[[#IDX64:]] = OpUConvert %[[#]] %[[#IDX]]
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c0]]
  ; CHECK:      OpStore %[[#]] %[[#undef]] Aligned 32
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c1]]
  ; CHECK:      OpStore %[[#]] %[[#undef]] Aligned 4
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c2]]
  ; CHECK:      OpStore %[[#]] %[[#undef]] Aligned 8
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c3]]
  ; CHECK:      OpStore %[[#]] %[[#undef]] Aligned 4
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c4]]
  ; CHECK:      OpStore %[[#]] %[[#undef]] Aligned 16
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c5]]
  ; CHECK:      OpStore %[[#]] %[[#undef]] Aligned 4
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#IDX64]]
  ; CHECK:      OpStore %[[#]] %[[#VAL]] Aligned 4
  ; CHECK:      %[[#PTR0:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_UNDEF]] %[[#c0]]
  ; CHECK:      %[[#RES:]] = OpLoad %[[#int]] %[[#PTR0]] Aligned 32
  ; CHECK:      OpStore %[[#out]] %[[#RES]]
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
  ; CHECK:      %[[#V_ZERO:]] = OpVariable %[[#ptr_arr6int]] Function
  ; CHECK-DAG:  %[[#IDX:]] = OpLoad %[[#int]] %[[#]]
  ; CHECK-DAG:  %[[#VAL:]] = OpLoad %[[#int]] %[[#]]
  ; CHECK:      %[[#IDX64:]] = OpUConvert %[[#]] %[[#IDX]]
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c0]]
  ; CHECK:      OpStore %[[#]] %[[#c0]] Aligned 32
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c1]]
  ; CHECK:      OpStore %[[#]] %[[#c0]] Aligned 4
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c2]]
  ; CHECK:      OpStore %[[#]] %[[#c0]] Aligned 8
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c3]]
  ; CHECK:      OpStore %[[#]] %[[#c0]] Aligned 4
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c4]]
  ; CHECK:      OpStore %[[#]] %[[#c0]] Aligned 16
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c5]]
  ; CHECK:      OpStore %[[#]] %[[#c0]] Aligned 4
  ; CHECK:      %[[#]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#IDX64]]
  ; CHECK:      OpStore %[[#]] %[[#VAL]] Aligned 4
  ; CHECK:      %[[#PTR0:]] = OpInBoundsAccessChain %[[#ptr_f_int]] %[[#V_ZERO]] %[[#c0]]
  ; CHECK:      %[[#RES:]] = OpLoad %[[#int]] %[[#PTR0]] Aligned 32
  ; CHECK:      OpStore %[[#out]] %[[#RES]]
  %idx = load i32, ptr addrspace(10) @idx
  %val = load i32, ptr addrspace(10) @val
  %idx64 = zext i32 %idx to i64
  %inserted = insertelement <6 x i32> zeroinitializer, i32 %val, i64 %idx64
  %extracted = extractelement <6 x i32> %inserted, i64 0
  store i32 %extracted, ptr addrspace(10) @out
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
