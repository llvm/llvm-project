; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeCooperativeMatrixKHR type requires the following SPIR-V extension: SPV_KHR_cooperative_matrix

; CHECK: OpCapability CooperativeMatrixKHR
; CHECK: OpExtension "SPV_KHR_cooperative_matrix"

; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const12:]] = OpConstant %[[#Int32Ty]] 12
; CHECK-DAG: %[[#Const48:]] = OpConstant %[[#Int32Ty]] 48
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Int32Ty]] 1
; CHECK-DAG: %[[#Const0:]] = OpConstant %[[#Int32Ty]] 0
; CHECK-DAG: %[[#MatTy1:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const12]] %[[#Const12]] %[[#Const2]]
; CHECK-DAG: %[[#MatTy2:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const12]] %[[#Const48]] %[[#Const0]]
; CHECK-DAG: %[[#MatTy3:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const48]] %[[#Const12]] %[[#Const1]]
; CHECK: OpCompositeConstruct %[[#MatTy1]]
; CHECK: %[[#Load1:]] = OpCooperativeMatrixLoadKHR %[[#MatTy2]]
; CHECK: OpCooperativeMatrixLengthKHR %[[#Int32Ty]] %[[#MatTy2:]]
; CHECK: OpCooperativeMatrixLoadKHR %[[#MatTy3]]
; CHECK: OpCooperativeMatrixMulAddKHR %[[#MatTy1]]
; CHECK: OpCooperativeMatrixStoreKHR

define spir_kernel void @matr_mult(ptr addrspace(1) align 1 %_arg_accA, ptr addrspace(1) align 1 %_arg_accB, ptr addrspace(1) align 4 %_arg_accC, i64 %_arg_N, i64 %_arg_K) {
entry:
  %addr1 = alloca target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), align 8
  %res = alloca target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), align 8
  %m1 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32 0)
  store target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m1, ptr %addr1, align 8
  %accA3 = addrspacecast ptr addrspace(1) %_arg_accA to ptr addrspace(3)
  %m2 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3) %accA3, i32 0, i64 %_arg_K, i32 1)
  %len = tail call spir_func i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) %m2)
  %accB3 = addrspacecast ptr addrspace(1) %_arg_accB to ptr addrspace(3)
  %m3 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3) %accB3, i32 0, i64 0)
  %m4 = load target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), ptr %addr1, align 8
  %m5 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) %m2, target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) %m3, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m4, i32 12)
  store target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m5, ptr %res, align 8
  %r = load i64, ptr %res, align 8
  store i64 %r, ptr %addr1, align 8
  %accC3 = addrspacecast ptr addrspace(1) %_arg_accC to ptr addrspace(3)
  %m6 = load target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), ptr %addr1, align 8
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3) %accC3, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m6, i32 0, i64 %_arg_N, i32 1)
  ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32)
declare dso_local spir_func i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0))
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0), target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32)
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)
