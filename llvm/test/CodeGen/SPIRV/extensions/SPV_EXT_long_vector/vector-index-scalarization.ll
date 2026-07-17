; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; spirv-val seems to have problems reading OpTypeVectorIdEXT correctly, enable once fixed
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#idx:]] "idx"
; CHECK-DAG: OpName %[[#idx2:]] "idx2"
; CHECK-DAG: OpName %[[#val:]] "val"

; CHECK-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#vec17:]] = OpTypeVectorIdEXT %[[#int]] 17
; CHECK-DAG: %[[#ptr_vec17:]] = OpTypePointer Function %[[#vec17]]
; CHECK-DAG: %[[#undef:]] = OpUndef %[[#vec17]]
; CHECK-DAG: %[[#null:]] = OpConstantNull %[[#vec17]]

; CHECK-DAG: %[[#out:]] = OpVariable %[[#ptr_p_int:]] CrossWorkgroup

@out = internal addrspace(1) global i32 0
@idx = internal addrspace(1) global i32 0
@idx2 = internal addrspace(1) global i32 0
@val = internal addrspace(1) global i32 0

; CHECK: %[[#test_full:]] = OpFunction %[[#]] None %[[#]]
define void @test_full() {
  ; CHECK-DAG:  %[[#i0:]] = OpLoad %[[#int]] %[[#idx]]
  ; CHECK-DAG:  %[[#i1:]] = OpLoad %[[#int]] %[[#idx2]]
  ; CHECK-DAG:  %[[#val_val:]] = OpLoad %[[#int]] %[[#val]]
  ; CHECK-DAG:  %[[#idx64:]] = OpUConvert %[[#long]] %[[#i0]]
  ; CHECK-DAG:  %[[#idx2_64:]] = OpUConvert %[[#long]] %[[#i1]]

  %idx = load i32, ptr addrspace(1) @idx
  %idx2 = load i32, ptr addrspace(1) @idx2
  %val = load i32, ptr addrspace(1) @val

  ; CHECK-DAG:  %[[#ptr:]] = OpVariable %[[#ptr_vec17]] Function
  ; CHECK-DAG:  %[[#loaded:]] = OpLoad %[[#vec17]] %[[#ptr]]
  %ptr = alloca <17 x i32>
  %loaded = load <17 x i32>, ptr %ptr
  %idx64 = zext i32 %idx to i64
  %idx2_64 = zext i32 %idx2 to i64

  ; Insertelement with dynamic index spills to stack
  ; CHECK: %[[#inserted:]] = OpVectorInsertDynamic %[[#vec17]] %[[#loaded]] %[[#val_val]] %[[#idx64]]
  %inserted = insertelement <17 x i32> %loaded, i32 %val, i64 %idx64

  ; Extractelement with dynamic index spills to stack
  ; CHECK: %[[#extracted:]] = OpVectorExtractDynamic %[[#int]] %[[#inserted]] %[[#idx2_64]]
  %extracted = extractelement <17 x i32> %inserted, i64 %idx2_64

  ; CHECK: OpStore %[[#out]] %[[#extracted]]
  store i32 %extracted, ptr addrspace(1) @out
  ret void
}

; CHECK: %[[#test_undef:]] = OpFunction %[[#]] None %[[#]]
define void @test_undef() {
  ; CHECK:      %[[#label:]] = OpLabel
  ; CHECK-DAG:  %[[#idx_val:]] = OpLoad %[[#int]] %[[#idx]]
  ; CHECK-DAG:  %[[#val_val:]] = OpLoad %[[#int]] %[[#val]]
  ; CHECK:      %[[#idx64:]] = OpUConvert %[[#long]] %[[#idx_val]]
  ; CHECK:      %[[#inserted1:]] = OpVectorInsertDynamic %[[#vec17]] %[[#undef]] %[[#val_val]] %[[#idx64]]
  ; CHECK:      %[[#extracted1:]] = OpCompositeExtract %[[#int]] %[[#inserted1]]
  ; CHECK:      OpStore %[[#out]] %[[#extracted1]]
  %idx = load i32, ptr addrspace(1) @idx
  %val = load i32, ptr addrspace(1) @val
  %idx64 = zext i32 %idx to i64
  %inserted = insertelement <17 x i32> poison, i32 %val, i64 %idx64
  %extracted = extractelement <17 x i32> %inserted, i64 0
  store i32 %extracted, ptr addrspace(1) @out
  ret void
}

; CHECK: %[[#test_zero:]] = OpFunction %[[#]] None %[[#]]
define void @test_zero() {
  ; CHECK:      %[[#label:]] = OpLabel
  ; CHECK-DAG:  %[[#idx_val:]] = OpLoad %[[#int]] %[[#idx]]
  ; CHECK-DAG:  %[[#val_val:]] = OpLoad %[[#int]] %[[#val]]
  ; CHECK:      %[[#idx64:]] = OpUConvert %[[#long]] %[[#idx_val]]
  ; CHECK:      %[[#inserted2:]] = OpVectorInsertDynamic %[[#vec17]] %[[#null]] %[[#val_val]] %[[#idx64]]
  ; CHECK:      %[[#extracted2:]] = OpCompositeExtract %[[#int]] %[[#inserted2]]
  ; CHECK:      OpStore %[[#out]] %[[#extracted2]]
  %idx = load i32, ptr addrspace(1) @idx
  %val = load i32, ptr addrspace(1) @val
  %idx64 = zext i32 %idx to i64
  %inserted = insertelement <17 x i32> zeroinitializer, i32 %val, i64 %idx64
  %extracted = extractelement <17 x i32> %inserted, i64 0
  store i32 %extracted, ptr addrspace(1) @out
  ret void
}
