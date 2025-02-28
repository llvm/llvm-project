; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;===---------------------------------------------------------------------===//
; Type definitions.
; CHECK-DAG: %[[I16:.*]] = OpTypeInt 16 0
; CHECK-DAG: %[[Bool:.*]] = OpTypeBool
; CHECK-DAG: %[[I32:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[I64:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[PtrI16:.*]] = OpTypePointer Function %[[I16]]
; CHECK-DAG: %[[PtrI32:.*]] = OpTypePointer Function %[[I32]]
; CHECK-DAG: %[[PtrI64:.*]] = OpTypePointer Function %[[I64]]
; CHECK-DAG: %[[StructI16:.*]] = OpTypeStruct %[[I16]] %[[Bool]]
; CHECK-DAG: %[[StructI32:.*]] = OpTypeStruct %[[I32]] %[[Bool]]
; CHECK-DAG: %[[StructI64:.*]] = OpTypeStruct %[[I64]] %[[Bool]]
; CHECK-DAG: %[[ZeroI16:.*]] = OpConstant %[[I16]] 0
; CHECK-DAG: %[[ZeroI32:.*]] = OpConstant %[[I32]] 0
; CHECK-DAG: %[[ZeroI64:.*]] = OpConstantNull %[[I64]]
; CHECK-DAG: %[[V4I32:.*]] = OpTypeVector %[[I32]] 4
; CHECK-DAG: %[[V4Bool:.*]] = OpTypeVector %[[Bool]] 4
; CHECK-DAG: %[[PtrV4I32:.*]] = OpTypePointer Function %[[V4I32]]
; CHECK-DAG: %[[StructV4I32:.*]] = OpTypeStruct %[[V4I32]] %[[V4Bool]]
; CHECK-DAG: %[[ZeroV4I32:.*]] = OpConstantNull %[[V4I32]]
;===---------------------------------------------------------------------===//
; Function for i16 sadd.with.overflow.
; CHECK: OpFunction
; CHECK: %[[A16:.*]] = OpFunctionParameter %[[I16]]
; CHECK: %[[B16:.*]] = OpFunctionParameter %[[I16]]
; CHECK: %[[Ptr16:.*]] = OpFunctionParameter %[[PtrI16]]
; CHECK: %[[Sum16:.*]] = OpIAdd %[[I16]] %[[A16]] %[[B16]]
; CHECK: %[[PosCmp16_1:.*]] = OpSGreaterThan %[[Bool]] %[[A16]] %[[ZeroI16]]
; CHECK: %[[PosCmp16_2:.*]] = OpSGreaterThan %[[Bool]] %[[B16]] %[[ZeroI16]]
; CHECK: %[[PosCmp16_3:.*]] = OpSLessThan %[[Bool]] %[[Sum16]] %[[B16]]
; CHECK: %[[PosCond16:.*]] = OpLogicalAnd %[[Bool]] %[[PosCmp16_1]] %[[PosCmp16_2]]
; CHECK: %[[PosOverflow16:.*]] = OpLogicalAnd %[[Bool]] %[[PosCond16]] %[[PosCmp16_3]]
; CHECK: %[[NegCmp16_1:.*]] = OpSLessThan %[[Bool]] %[[A16]] %[[ZeroI16]]
; CHECK: %[[NegCmp16_2:.*]] = OpSLessThan %[[Bool]] %[[B16]] %[[ZeroI16]]
; CHECK: %[[NegCmp16_3:.*]] = OpSGreaterThan %[[Bool]] %[[Sum16]] %[[B16]]
; CHECK: %[[NegCond16:.*]] = OpLogicalAnd %[[Bool]] %[[NegCmp16_1]] %[[NegCmp16_2]]
; CHECK: %[[NegOverflow16:.*]] = OpLogicalAnd %[[Bool]] %[[NegCond16]] %[[NegCmp16_3]]
; CHECK: %[[Overflow16:.*]] = OpLogicalOr %[[Bool]] %[[NegOverflow16]] %[[PosOverflow16]]
; CHECK: %[[Comp16:.*]] = OpCompositeConstruct %[[StructI16]] %[[Sum16]] %[[Overflow16]]
; CHECK: %[[ExtOver16:.*]] = OpCompositeExtract %[[Bool]] %[[Comp16]] 1
; CHECK: %[[Final16:.*]] = OpLogicalNotEqual %[[Bool]] %[[ExtOver16]] %[[#]]
; CHECK: OpReturn
define spir_func void @smulo_i16(i16 %a, i16 %b, ptr nocapture %c) {
entry:
  %umul = tail call { i16, i1 } @llvm.sadd.with.overflow.i16(i16 %a, i16 %b)
  %cmp = extractvalue { i16, i1 } %umul, 1
  %umul.value = extractvalue { i16, i1 } %umul, 0
  %storemerge = select i1 %cmp, i16 0, i16 %umul.value
  store i16 %storemerge, ptr %c, align 1
  ret void
}

;===---------------------------------------------------------------------===//
; Function for i32 sadd.with.overflow.
; CHECK: OpFunction
; CHECK: %[[A32:.*]] = OpFunctionParameter %[[I32]]
; CHECK: %[[B32:.*]] = OpFunctionParameter %[[I32]]
; CHECK: %[[Ptr32:.*]] = OpFunctionParameter %[[PtrI32]]
; CHECK: %[[Sum32:.*]] = OpIAdd %[[I32]] %[[A32]] %[[B32]]
; CHECK: %[[PosCmp32_1:.*]] = OpSGreaterThan %[[Bool]] %[[A32]] %[[ZeroI32]]
; CHECK: %[[PosCmp32_2:.*]] = OpSGreaterThan %[[Bool]] %[[B32]] %[[ZeroI32]]
; CHECK: %[[PosCmp32_3:.*]] = OpSLessThan %[[Bool]] %[[Sum32]] %[[B32]]
; CHECK: %[[PosCond32:.*]] = OpLogicalAnd %[[Bool]] %[[PosCmp32_1]] %[[PosCmp32_2]]
; CHECK: %[[PosOverflow32:.*]] = OpLogicalAnd %[[Bool]] %[[PosCond32]] %[[PosCmp32_3]]
; CHECK: %[[NegCmp32_1:.*]] = OpSLessThan %[[Bool]] %[[A32]] %[[ZeroI32]]
; CHECK: %[[NegCmp32_2:.*]] = OpSLessThan %[[Bool]] %[[B32]] %[[ZeroI32]]
; CHECK: %[[NegCmp32_3:.*]] = OpSGreaterThan %[[Bool]] %[[Sum32]] %[[B32]]
; CHECK: %[[NegCond32:.*]] = OpLogicalAnd %[[Bool]] %[[NegCmp32_1]] %[[NegCmp32_2]]
; CHECK: %[[NegOverflow32:.*]] = OpLogicalAnd %[[Bool]] %[[NegCond32]] %[[NegCmp32_3]]
; CHECK: %[[Overflow32:.*]] = OpLogicalOr %[[Bool]] %[[NegOverflow32]] %[[PosOverflow32]]
; CHECK: %[[Comp32:.*]] = OpCompositeConstruct %[[StructI32]] %[[Sum32]] %[[Overflow32]]
; CHECK: %[[ExtOver32:.*]] = OpCompositeExtract %[[Bool]] %[[Comp32]] 1
; CHECK: %[[Final32:.*]] = OpLogicalNotEqual %[[Bool]] %[[ExtOver32]] %[[#]]
; CHECK: OpReturn
define spir_func void @smulo_i32(i32 %a, i32 %b, ptr nocapture %c) {
entry:
  %umul = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %cmp = extractvalue { i32, i1 } %umul, 1
  %umul.value = extractvalue { i32, i1 } %umul, 0
  %storemerge = select i1 %cmp, i32 0, i32 %umul.value
  store i32 %storemerge, ptr %c, align 4
  ret void
}

;===---------------------------------------------------------------------===//
; Function for i64 sadd.with.overflow.
; CHECK: OpFunction
; CHECK: %[[A64:.*]] = OpFunctionParameter %[[I64]]
; CHECK: %[[B64:.*]] = OpFunctionParameter %[[I64]]
; CHECK: %[[Ptr64:.*]] = OpFunctionParameter %[[PtrI64]]
; CHECK: %[[Sum64:.*]] = OpIAdd %[[I64]] %[[A64]] %[[B64]]
; CHECK: %[[PosCmp64_1:.*]] = OpSGreaterThan %[[Bool]] %[[A64]] %[[ZeroI64]]
; CHECK: %[[PosCmp64_2:.*]] = OpSGreaterThan %[[Bool]] %[[B64]] %[[ZeroI64]]
; CHECK: %[[PosCmp64_3:.*]] = OpSLessThan %[[Bool]] %[[Sum64]] %[[B64]]
; CHECK: %[[PosCond64:.*]] = OpLogicalAnd %[[Bool]] %[[PosCmp64_1]] %[[PosCmp64_2]]
; CHECK: %[[PosOverflow64:.*]] = OpLogicalAnd %[[Bool]] %[[PosCond64]] %[[PosCmp64_3]]
; CHECK: %[[NegCmp64_1:.*]] = OpSLessThan %[[Bool]] %[[A64]] %[[ZeroI64]]
; CHECK: %[[NegCmp64_2:.*]] = OpSLessThan %[[Bool]] %[[B64]] %[[ZeroI64]]
; CHECK: %[[NegCmp64_3:.*]] = OpSGreaterThan %[[Bool]] %[[Sum64]] %[[B64]]
; CHECK: %[[NegCond64:.*]] = OpLogicalAnd %[[Bool]] %[[NegCmp64_1]] %[[NegCmp64_2]]
; CHECK: %[[NegOverflow64:.*]] = OpLogicalAnd %[[Bool]] %[[NegCond64]] %[[NegCmp64_3]]
; CHECK: %[[Overflow64:.*]] = OpLogicalOr %[[Bool]] %[[NegOverflow64]] %[[PosOverflow64]]
; CHECK: %[[Comp64:.*]] = OpCompositeConstruct %[[StructI64]] %[[Sum64]] %[[Overflow64]]
; CHECK: %[[ExtOver64:.*]] = OpCompositeExtract %[[Bool]] %[[Comp64]] 1
; CHECK: %[[Final64:.*]] = OpLogicalNotEqual %[[Bool]] %[[ExtOver64]] %[[#]]
; CHECK: OpReturn
define spir_func void @smulo_i64(i64 %a, i64 %b, ptr nocapture %c) {
entry:
  %umul = tail call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b)
  %cmp = extractvalue { i64, i1 } %umul, 1
  %umul.value = extractvalue { i64, i1 } %umul, 0
  %storemerge = select i1 %cmp, i64 0, i64 %umul.value
  store i64 %storemerge, ptr %c, align 8
  ret void
}

;===---------------------------------------------------------------------===//
; Function for vector (4 x i32) sadd.with.overflow.

; CHECK: OpFunction
; CHECK: %[[A4:.*]] = OpFunctionParameter %[[V4I32]]
; CHECK: %[[B4:.*]] = OpFunctionParameter %[[V4I32]]
; CHECK: %[[Ptr4:.*]] = OpFunctionParameter %[[PtrV4I32]]
; CHECK: %[[Sum4:.*]] = OpIAdd %[[V4I32]] %[[A4]] %[[B4]]
; CHECK: %[[PosCmp4_1:.*]] = OpSGreaterThan %[[V4Bool]] %[[A4]] %[[ZeroV4I32]]
; CHECK: %[[PosCmp4_2:.*]] = OpSGreaterThan %[[V4Bool]] %[[B4]] %[[ZeroV4I32]]
; CHECK: %[[PosCmp4_3:.*]] = OpSLessThan %[[V4Bool]] %[[Sum4]] %[[B4]]
; CHECK: %[[PosCond4:.*]] = OpLogicalAnd %[[V4Bool]] %[[PosCmp4_1]] %[[PosCmp4_2]]
; CHECK: %[[PosOverflow4:.*]] = OpLogicalAnd %[[V4Bool]] %[[PosCond4]] %[[PosCmp4_3]]
; CHECK: %[[NegCmp4_1:.*]] = OpSLessThan %[[V4Bool]] %[[A4]] %[[ZeroV4I32]]
; CHECK: %[[NegCmp4_2:.*]] = OpSLessThan %[[V4Bool]] %[[B4]] %[[ZeroV4I32]]
; CHECK: %[[NegCmp4_3:.*]] = OpSGreaterThan %[[V4Bool]] %[[Sum4]] %[[B4]]
; CHECK: %[[NegCond4:.*]] = OpLogicalAnd %[[V4Bool]] %[[NegCmp4_1]] %[[NegCmp4_2]]
; CHECK: %[[NegOverflow4:.*]] = OpLogicalAnd %[[V4Bool]] %[[NegCond4]] %[[NegCmp4_3]]
; CHECK: %[[Overflow4:.*]] = OpLogicalOr %[[V4Bool]] %[[NegOverflow4]] %[[PosOverflow4]]
; CHECK: %[[Comp4:.*]] = OpCompositeConstruct %[[StructV4I32]] %[[Sum4]] %[[Overflow4]]
; CHECK: %[[ExtOver4:.*]] = OpCompositeExtract %[[V4Bool]] %[[Comp4]] 1
; CHECK: %[[Final4:.*]] = OpLogicalNotEqual %[[V4Bool]] %[[ExtOver4]] %[[#]]
; CHECK: OpReturn
define spir_func void @smulo_v4i32(<4 x i32> %a, <4 x i32> %b, ptr nocapture %c) {
entry:
  %umul = tail call { <4 x i32>, <4 x i1> } @llvm.sadd.with.overflow.v4i32(<4 x i32> %a, <4 x i32> %b)
  %cmp = extractvalue { <4 x i32>, <4 x i1> } %umul, 1
  %umul.value = extractvalue { <4 x i32>, <4 x i1> } %umul, 0
  %storemerge = select <4 x i1> %cmp, <4 x i32> zeroinitializer, <4 x i32> %umul.value
  store <4 x i32> %storemerge, ptr %c, align 16
  ret void
}

;===---------------------------------------------------------------------===//
; Declarations of the intrinsics.
declare { i16, i1 } @llvm.sadd.with.overflow.i16(i16, i16)
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)
declare { <4 x i32>, <4 x i1> } @llvm.sadd.with.overflow.v4i32(<4 x i32>, <4 x i32>)
