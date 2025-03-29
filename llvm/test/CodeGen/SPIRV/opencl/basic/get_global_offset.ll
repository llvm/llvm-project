; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpEntryPoint Kernel %[[#test_func:]] "test"
; CHECK-DAG: OpDecorate %[[#f2_decl:]] LinkageAttributes "BuiltInGlobalOffset" Import
; CHECK-DAG: %[[#int32_ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#i32ptr_ty:]] = OpTypePointer CrossWorkgroup  %[[#int32_ty]]
; CHECK-DAG: %[[#void_ty:]] = OpTypeVoid
; CHECK-DAG: %[[#func_ty:]] = OpTypeFunction %[[#void_ty]] %[[#i32ptr_ty]]
; CHECK-DAG: %[[#int64_ty:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#vec_ty:]] = OpTypeVector %[[#int64_ty]] 3
; CHECK-DAG: %[[#func2_ty:]] = OpTypeFunction %[[#vec_ty]]
;; TODO: add 64-bit constant defs
; CHECK-DAG: %[[#f2_decl]] = OpFunction %[[#vec_ty]] Pure %[[#func2_ty]]
; CHECK: OpFunctionEnd
;; Check that the function register name does not match other registers
; CHECK-NOT: %[[#void_ty]] = OpFunction
; CHECK-NOT: %[[#func_ty]] = OpFunction
; CHECK-NOT: %[[#int64_ty]] = OpFunction
; CHECK-NOT: %[[#vec_ty]] = OpFunction
; CHECK-NOT: %[[#func2_ty]] = OpFunction
; CHECK-NOT: %[[#f2_decl]] = OpFunction

define spir_kernel void @test(i32 addrspace(1)* %outOffsets) {
entry:
  %0 = call spir_func <3 x i64> @BuiltInGlobalOffset() #1
  %call = extractelement <3 x i64> %0, i32 0
  %conv = trunc i64 %call to i32
; CHECK: %[[#i1:]] = OpInBoundsPtrAccessChain %[[#i32ptr_ty]] %[[#outOffsets:]]
; CHECK: OpStore %[[#i1:]] %[[#]] Aligned 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %outOffsets, i64 0
  store i32 %conv, i32 addrspace(1)* %arrayidx, align 4
  %1 = call spir_func <3 x i64> @BuiltInGlobalOffset() #1
  %call1 = extractelement <3 x i64> %1, i32 1
  %conv2 = trunc i64 %call1 to i32
; CHECK: %[[#i2:]] = OpInBoundsPtrAccessChain %[[#i32ptr_ty]] %[[#outOffsets]]
; CHECK: OpStore %[[#i2:]] %[[#]] Aligned 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %outOffsets, i64 1
  store i32 %conv2, i32 addrspace(1)* %arrayidx3, align 4
  %2 = call spir_func <3 x i64> @BuiltInGlobalOffset() #1
  %call4 = extractelement <3 x i64> %2, i32 2
  %conv5 = trunc i64 %call4 to i32
; CHECK: %[[#i3:]] = OpInBoundsPtrAccessChain %[[#i32ptr_ty]] %[[#outOffsets]]
; CHECK: OpStore %[[#i3:]] %[[#]] Aligned 4
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(1)* %outOffsets, i64 2
  store i32 %conv5, i32 addrspace(1)* %arrayidx6, align 4
  ret void
}

declare spir_func <3 x i64> @BuiltInGlobalOffset() #1

attributes #1 = { nounwind readnone }
