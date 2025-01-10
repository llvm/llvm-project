; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[Void:.*]] = OpTypeVoid
; CHECK-DAG: %[[PtrChar:.*]] = OpTypePointer Function %[[Char]]
; CHECK-DAG: %[[StructChar:.*]] = OpTypeStruct %[[Char]] %[[Char]]
; CHECK-DAG: %[[ZeroChar:.*]] = OpConstant %[[Char]] 0
; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[PtrInt:.*]] = OpTypePointer Function %[[Int]]
; CHECK-DAG: %[[StructInt:.*]] = OpTypeStruct %[[Int]] %[[Int]]
; CHECK-DAG: %[[ZeroInt:.*]] = OpConstant %[[Int]] 0
; CHECK-DAG: %[[Bool:.*]] = OpTypeBool
; CHECK-DAG: %[[V2Bool:.*]] = OpTypeVector %[[Bool]] 2
; CHECK-DAG: %[[Long:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[V2Long:.*]] = OpTypeVector %[[Long]] 2
; CHECK-DAG: %[[PtrV2Long:.*]] = OpTypePointer Function %[[V2Long]]
; CHECK-DAG: %[[StructV2Long:.*]] = OpTypeStruct %[[V2Long]] %[[V2Long]]
; CHECK-DAG: %[[ZeroV2Long:.*]] = OpConstantNull %[[V2Long]]

; CHECK: OpFunction
; CHECK: %[[A:.*]] = OpFunctionParameter %[[Char]]
; CHECK: %[[B:.*]] = OpFunctionParameter %[[Char]]
; CHECK: %[[Ptr:.*]] = OpFunctionParameter %[[PtrChar]]
; CHECK: %[[Struct:.*]] = OpUMulExtended %[[StructChar]] %[[A]] %[[B]]
; CHECK: %[[Val:.*]] = OpCompositeExtract %[[Char]] %[[Struct]] 0
; CHECK: %[[Over:.*]] = OpCompositeExtract %[[Char]] %[[Struct]] 1
; CHECK: %[[IsOver:.*]] = OpINotEqual %[[Bool]] %[[Over]] %[[ZeroChar]]
; CHECK: %[[Res:.*]] = OpSelect %[[Char]] %[[IsOver]] %[[ZeroChar]] %[[Val]]
; CHECK: OpStore %[[Ptr]] %[[Res]] Aligned 1
; CHECK: OpReturn
define dso_local spir_func void @umulo_i8(i8 zeroext %a, i8 zeroext %b, ptr nocapture %c) local_unnamed_addr {
entry:
  %umul = tail call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %a, i8 %b)
  %cmp = extractvalue { i8, i1 } %umul, 1
  %umul.value = extractvalue { i8, i1 } %umul, 0
  %storemerge = select i1 %cmp, i8 0, i8 %umul.value
  store i8 %storemerge, ptr %c, align 1
  ret void
}

; CHECK: OpFunction
; CHECK: %[[A2:.*]] = OpFunctionParameter %[[Int]]
; CHECK: %[[B2:.*]] = OpFunctionParameter %[[Int]]
; CHECK: %[[Ptr2:.*]] = OpFunctionParameter %[[PtrInt]]
; CHECK: %[[Struct2:.*]] = OpUMulExtended %[[StructInt]] %[[B2]] %[[A2]]
; CHECK: %[[Val2:.*]] = OpCompositeExtract %[[Int]] %[[Struct2]] 0
; CHECK: %[[Over2:.*]] = OpCompositeExtract %[[Int]] %[[Struct2]] 1
; CHECK: %[[IsOver2:.*]] = OpINotEqual %[[Bool]] %[[Over2]] %[[ZeroInt]]
; CHECK: %[[Res2:.*]] = OpSelect %[[Int]] %[[IsOver2]] %[[ZeroInt]] %[[Val2]]
; CHECK: OpStore %[[Ptr2]] %[[Res2]] Aligned 4
; CHECK: OpReturn
define dso_local spir_func void @umulo_i32(i32 %a, i32 %b, ptr nocapture %c) local_unnamed_addr {
entry:
  %umul = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %b, i32 %a)
  %umul.val = extractvalue { i32, i1 } %umul, 0
  %umul.ov = extractvalue { i32, i1 } %umul, 1
  %spec.select = select i1 %umul.ov, i32 0, i32 %umul.val
  store i32 %spec.select, ptr %c, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[A3:.*]] = OpFunctionParameter %[[V2Long]]
; CHECK: %[[B3:.*]] = OpFunctionParameter %[[V2Long]]
; CHECK: %[[Ptr3:.*]] = OpFunctionParameter %[[PtrV2Long]]
; CHECK: %[[Struct3:.*]] = OpUMulExtended %[[StructV2Long]] %[[A3]] %[[B3]]
; CHECK: %[[Val3:.*]] = OpCompositeExtract %[[V2Long]] %[[Struct3]] 0
; CHECK: %[[Over3:.*]] = OpCompositeExtract %[[V2Long]] %[[Struct3]] 1
; CHECK: %[[IsOver3:.*]] = OpINotEqual %[[V2Bool]] %[[Over3]] %[[ZeroV2Long]]
; CHECK: %[[Res3:.*]] = OpSelect %[[V2Long]] %[[IsOver3]] %[[ZeroV2Long]] %[[Val3]]
; CHECK: OpStore %[[Ptr3]] %[[Res3]] Aligned 16
; CHECK: OpReturn
define dso_local spir_func void @umulo_v2i64(<2 x i64> %a, <2 x i64> %b, ptr %p) nounwind {
  %umul = call {<2 x i64>, <2 x i1>} @llvm.umul.with.overflow.v2i64(<2 x i64> %a, <2 x i64> %b)
  %umul.val = extractvalue {<2 x i64>, <2 x i1>} %umul, 0
  %umul.ov = extractvalue {<2 x i64>, <2 x i1>} %umul, 1
  %zero = alloca <2 x i64>, align 16
  %spec.select = select <2 x i1> %umul.ov, <2 x i64> <i64 0, i64 0>, <2 x i64> %umul.val
  store <2 x i64> %spec.select, ptr %p
  ret void
}

declare {i8, i1} @llvm.umul.with.overflow.i8(i8, i8)
declare {i32, i1} @llvm.umul.with.overflow.i32(i32, i32)
declare {<2 x i64>, <2 x i1>} @llvm.umul.with.overflow.v2i64(<2 x i64>, <2 x i64>)
