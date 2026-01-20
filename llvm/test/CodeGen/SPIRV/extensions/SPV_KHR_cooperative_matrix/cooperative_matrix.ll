; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - -filetype=obj | spirv-val %}

; TODO: This test currently fails with LLVM_ENABLE_EXPENSIVE_CHECKS enabled
; XFAIL: expensive_checks

; CHECK-ERROR: LLVM ERROR: OpTypeCooperativeMatrixKHR type requires the following SPIR-V extension: SPV_KHR_cooperative_matrix

; CHECK: OpCapability CooperativeMatrixKHR
; CHECK: OpExtension "SPV_KHR_cooperative_matrix"

; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Int16Ty:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#Const12:]] = OpConstant %[[#Int32Ty]] 12{{$}}
; CHECK-DAG: %[[#Const48:]] = OpConstant %[[#Int32Ty]] 48{{$}}
; CHECK-DAG: %[[#Const16:]] = OpConstant %[[#Int32Ty]] 16{{$}}
; CHECK-DAG: %[[#Const8:]] = OpConstant %[[#Int32Ty]] 8{{$}}
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Int32Ty]] 3{{$}}
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int32Ty]] 2{{$}}
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Int32Ty]] 1{{$}}
; CHECK-DAG: %[[#Const0:]] = OpConstantNull %[[#Int32Ty]]
; CHECK-DAG: %[[#MatTy1:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const12]] %[[#Const12]] %[[#Const2]]
; CHECK-DAG: %[[#MatTy2:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const12]] %[[#Const48]] %[[#Const0]]
; CHECK-DAG: %[[#MatTy3:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const48]] %[[#Const12]] %[[#Const1]]
; CHECK-DAG: %[[#MatTy4:]] = OpTypeCooperativeMatrixKHR %[[#Int16Ty]] %[[#Const3]] %[[#Const8]] %[[#Const16]] %[[#Const0]]
; CHECK-DAG: %[[#MatTy5:]] = OpTypeCooperativeMatrixKHR %[[#Int16Ty]] %[[#Const3]] %[[#Const16]] %[[#Const16]] %[[#Const1]]
; CHECK-DAG: %[[#MatTy6:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const8]] %[[#Const16]] %[[#Const2]]

; CHECK: OpCompositeConstruct %[[#MatTy1]]
; CHECK: %[[#Load1:]] = OpCooperativeMatrixLoadKHR %[[#MatTy2]]
; CHECK: OpCooperativeMatrixLengthKHR %[[#Int32Ty]] %[[#MatTy2:]]
; CHECK: OpCooperativeMatrixLoadKHR %[[#MatTy3]]
; CHECK: OpCooperativeMatrixMulAddKHR %[[#MatTy1]] %[[#]] %[[#]] %[[#]] MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR
; CHECK: OpCooperativeMatrixStoreKHR

; CHECK: %[[#MatA:]] = OpCooperativeMatrixLoadKHR %[[#MatTy4]] %[[#]] %[[#Const1]] %[[#]] 0{{$}}
; CHECK: %[[#MatB:]] = OpCooperativeMatrixLoadKHR %[[#MatTy5]] %[[#]] %[[#Const0]] %[[#]] 0{{$}}
; CHECK: %[[#MatC:]] = OpCompositeConstruct %[[#MatTy6]] %[[#Const1]]
; CHECK: OpCooperativeMatrixMulAddKHR %[[#MatTy6]] %[[#MatA]] %[[#MatB]] %[[#MatC]] MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR
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

define spir_kernel void @matr_mult_with_complex_mangling(ptr addrspace(1) %_arg_A, ptr addrspace(1) %_arg_B, i64 %_arg_M, i64 %_arg_N) {
entry:
  ; __spv::__spirv_CooperativeMatrixKHR<short, (__spv::Scope::Flag)3, 8ul, 16ul, (__spv::MatrixUse)0>* __spirv_CooperativeMatrixLoadKHR<short AS1, short, 8ul, 16ul, (__spv::MatrixUse)0, (__spv::MatrixLayout)1, (__spv::Scope::Flag)3>(short AS1*, __spv::MatrixLayout, unsigned long, int)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ssLm8ELm16ELN5__spv9MatrixUseE0ELNS1_12MatrixLayoutE1ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) %_arg_A, i32 1, i64 %_arg_M, i32 0)
  ; __spv::__spirv_CooperativeMatrixKHR<short, (__spv::Scope::Flag)3, 16ul, 16ul, (__spv::MatrixUse)1>* __spirv_CooperativeMatrixLoadKHR<short AS1, short, 16ul, 16ul, (__spv::MatrixUse)1, (__spv::MatrixLayout)0, (__spv::Scope::Flag)3>(short AS1*, __spv::MatrixLayout, unsigned long, int)
  %call1 = call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ssLm16ELm16ELN5__spv9MatrixUseE1ELNS1_12MatrixLayoutE0ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1) %_arg_B, i32 0, i64 %_arg_N, i32 0)
  ; __spv::__spirv_CooperativeMatrixKHR<int, (__spv::Scope::Flag)3, 8ul, 16ul, (__spv::MatrixUse)2>* __spirv_CompositeConstruct<int, int, 8ul, 16ul, (__spv::MatrixUse)2, (__spv::MatrixLayout)3, (__spv::Scope::Flag)3>(int)
  %call2 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2) @_Z26__spirv_CompositeConstructIiiLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEET_(i32 1)
  ; __spv::__spirv_CooperativeMatrixKHR<int, (__spv::Scope::Flag)3, 8ul, 16ul, (__spv::MatrixUse)2>* __spirv_CooperativeMatrixMulAddKHR<short, short, int, int, 8ul, 16ul, 16ul, (__spv::MatrixUse)0, (__spv::MatrixUse)1, (__spv::MatrixUse)2, (__spv::MatrixLayout)0, (__spv::MatrixLayout)0, (__spv::MatrixLayout)0, (__spv::Scope::Flag)3>(__spv::__spirv_CooperativeMatrixKHR<short, (__spv::Scope::Flag)3, 8ul, 16ul, (__spv::MatrixUse)0>*, __spv::__spirv_CooperativeMatrixKHR<short, (__spv::Scope::Flag)3, 16ul, 16ul, (__spv::MatrixUse)1>*, __spv::__spirv_CooperativeMatrixKHR<int, (__spv::Scope::Flag)3, 8ul, 16ul, (__spv::MatrixUse)2>*, unsigned long)
  %call3 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHRIssiiLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS1_1ELS1_2ELNS0_12MatrixLayoutE0ELS2_0ELS2_0ELNS0_5Scope4FlagE3EEPNS0_28__spirv_CooperativeMatrixKHRIT2_XT12_EXT3_EXT5_EXT8_EEEPNS5_IT_XT12_EXT3_EXT4_EXT6_EEEPNS5_IT0_XT12_EXT4_EXT5_EXT7_EEEPNS5_IT1_XT12_EXT3_EXT5_EXT8_EEEm(target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) %call, target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) %call1, target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2) %call2, i64 3)
  ; void __spirv_CooperativeMatrixStoreKHR<int AS1, int, 8ul, 16ul, (__spv::MatrixUse)2, (__spv::MatrixLayout)3, (__spv::Scope::Flag)3>(int AS1*, __spv::__spirv_CooperativeMatrixKHR<int, (__spv::Scope::Flag)3, 8ul, 16ul, (__spv::MatrixUse)2>*, __spv::MatrixLayout, unsigned long, int)
  call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHRIU3AS1iiLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEES3_mi(ptr addrspace(1) %_arg_A, target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2) %call2, i32 1, i64 %_arg_M, i32 0)
  ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32)
declare dso_local spir_func i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0))
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0), target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32)
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2) @_Z26__spirv_CompositeConstructIiiLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEET_(i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ssLm8ELm16ELN5__spv9MatrixUseE0ELNS1_12MatrixLayoutE1ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHRIU3AS1ssLm16ELm16ELN5__spv9MatrixUseE1ELNS1_12MatrixLayoutE0ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_S3_mi(ptr addrspace(1), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHRIssiiLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS1_1ELS1_2ELNS0_12MatrixLayoutE0ELS2_0ELS2_0ELNS0_5Scope4FlagE3EEPNS0_28__spirv_CooperativeMatrixKHRIT2_XT12_EXT3_EXT5_EXT8_EEEPNS5_IT_XT12_EXT3_EXT4_EXT6_EEEPNS5_IT0_XT12_EXT4_EXT5_EXT7_EEEPNS5_IT1_XT12_EXT3_EXT5_EXT8_EEEm(target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0), target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1), target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2), i64)
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHRIU3AS1iiLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEES3_mi(ptr addrspace(1), target("spirv.CooperativeMatrixKHR", i32, 3, 8, 16, 2), i32, i64, i32)
