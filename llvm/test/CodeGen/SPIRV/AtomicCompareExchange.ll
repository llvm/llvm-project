; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG:  %[[#Int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG:  %[[#MemScope_CrossDevice:]] = OpConstantNull %[[#Int]]
; CHECK-SPIRV-DAG:  %[[#MemSemEqual_SeqCst:]] = OpConstant %[[#Int]] 16{{$}}
; CHECK-SPIRV-DAG:  %[[#MemSemUnequal_Acquire:]] = OpConstant %[[#Int]] 2{{$}}
; CHECK-SPIRV-DAG:  %[[#Constant_456:]] = OpConstant %[[#Int]] 456{{$}}
; CHECK-SPIRV-DAG:  %[[#Constant_128:]] = OpConstant %[[#Int]] 128{{$}}
; CHECK-SPIRV-DAG:  %[[#Bool:]] = OpTypeBool
; CHECK-SPIRV-DAG:  %[[#Struct:]] = OpTypeStruct %[[#Int]] %[[#Bool]]
; CHECK-SPIRV-DAG:  %[[#UndefStruct:]] = OpUndef %[[#Struct]]
; CHECK-SPIRV-DAG:  %[[#Int8:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG:  %[[#Constant_45:]] = OpConstant %[[#Int8]] 45{{$}}
; CHECK-SPIRV-DAG:  %[[#Constant_12:]] = OpConstant %[[#Int8]] 12{{$}}
; CHECK-SPIRV-DAG:  %[[#Struct8:]] = OpTypeStruct %[[#Int8]] %[[#Bool]]
; CHECK-SPIRV-DAG:  %[[#UndefStruct8:]] = OpUndef %[[#Struct8]]

; CHECK-SPIRV:      %[[#Value:]] = OpLoad %[[#Int]] %[[#Value_ptr:]]
; CHECK-SPIRV:      %[[#Res:]] = OpAtomicCompareExchange %[[#Int]] %[[#Pointer:]] %[[#MemScope_CrossDevice]]
; CHECK-SPIRV-SAME: %[[#MemSemEqual_SeqCst]] %[[#MemSemUnequal_Acquire]] %[[#Value]] %[[#Comparator:]]
; CHECK-SPIRV:      %[[#Success:]] = OpIEqual %[[#]] %[[#Res]] %[[#Comparator]]
; CHECK-SPIRV:      %[[#Composite_0:]] = OpCompositeInsert %[[#Struct]] %[[#Res]] %[[#UndefStruct]] 0
; CHECK-SPIRV:      %[[#Composite_1:]] = OpCompositeInsert %[[#Struct]] %[[#Success]] %[[#Composite_0]] 1
; CHECK-SPIRV:      %[[#]] = OpCompositeExtract %[[#Bool]] %[[#Composite_1]] 1

define dso_local spir_func void @test(ptr %ptr, ptr %value_ptr, i32 %comparator) local_unnamed_addr {
entry:
  %0 = load i32, ptr %value_ptr, align 4
  %1 = cmpxchg ptr %ptr, i32 %comparator, i32 %0 seq_cst acquire
  %2 = extractvalue { i32, i1 } %1, 1
  br i1 %2, label %cmpxchg.continue, label %cmpxchg.store_expected

cmpxchg.store_expected:                           ; preds = %entry
  %3 = extractvalue { i32, i1 } %1, 0
  store i32 %3, ptr %value_ptr, align 4
  br label %cmpxchg.continue

cmpxchg.continue:                                 ; preds = %cmpxchg.store_expected, %entry
  ret void
}

; CHECK-SPIRV:      %[[#Res_1:]] = OpAtomicCompareExchange %[[#Int]] %[[#Ptr:]] %[[#MemScope_CrossDevice]]
; CHECK-SPIRV-SAME: %[[#MemSemEqual_SeqCst]] %[[#MemSemUnequal_Acquire]] %[[#Constant_456]] %[[#Constant_128]]
; CHECK-SPIRV:      %[[#Success_1:]] = OpIEqual %[[#]] %[[#Res_1]] %[[#Constant_128]]
; CHECK-SPIRV:      %[[#Composite:]] = OpCompositeInsert %[[#Struct]] %[[#Res_1]] %[[#UndefStruct]] 0
; CHECK-SPIRV:      %[[#Composite_1:]] = OpCompositeInsert %[[#Struct]] %[[#Success_1]] %[[#Composite]] 1
; CHECK-SPIRV:      OpStore %[[#Store_ptr:]] %[[#Composite_1]]

define dso_local spir_func void @test2(ptr %ptr, ptr %store_ptr) local_unnamed_addr {
entry:
  %0 = cmpxchg ptr %ptr, i32 128, i32 456 seq_cst acquire
  store { i32, i1 } %0, ptr %store_ptr, align 4
  ret void
}

; CHECK-SPIRV:      %[[#Res_2:]] = OpAtomicCompareExchange %[[#Int8]] %[[#Ptr:]] %[[#MemScope_CrossDevice]]
; CHECK-SPIRV-SAME: %[[#MemSemEqual_SeqCst]] %[[#MemSemUnequal_Acquire]] %[[#Constant_45]] %[[#Constant_12]]
; CHECK-SPIRV:      %[[#Success_2:]] = OpIEqual %[[#]] %[[#Res_2]] %[[#Constant_12]]
; CHECK-SPIRV:      %[[#Composite_2:]] = OpCompositeInsert %[[#Struct8]] %[[#Res_2]] %[[#UndefStruct8]] 0
; CHECK-SPIRV:      %[[#Composite_3:]] = OpCompositeInsert %[[#Struct8]] %[[#Success_2]] %[[#Composite_2]] 1
; CHECK-SPIRV:      OpStore %[[#Store_ptr:]] %[[#Composite_3]]

define dso_local spir_func void @test3(ptr %ptr, ptr %store_ptr) local_unnamed_addr {
entry:
  %0 = cmpxchg ptr %ptr, i8 12, i8 45 seq_cst acquire
  store { i8, i1 } %0, ptr %store_ptr, align 1
  ret void
}
