; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450
; CHECK-DAG: [[Z:%.*]] = OpConstant %[[#]] 0
; CHECK-DAG: [[X:%.*]] = OpConstant %[[#]] 1

define noundef i32 @firstbituhigh_i32(i32 noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindUMsb %[[#]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i32(i32 %a)
  ret i32 %elt.firstbituhigh
}

define noundef <2 x i32> @firstbituhigh_2xi32(<2 x i32> noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindUMsb %[[#]]
  %elt.firstbituhigh = call <2 x i32> @llvm.spv.firstbituhigh.v2i32(<2 x i32> %a)
  ret <2 x i32> %elt.firstbituhigh
}

define noundef i32 @firstbituhigh_i16(i16 noundef %a) {
entry:
; CHECK: [[A:%.*]] = OpUConvert %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindUMsb [[A]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i16(i16 %a)
  ret i32 %elt.firstbituhigh
}

define noundef <2 x i32> @firstbituhigh_v2i16(<2 x i16> noundef %a) {
entry:
; CHECK: [[A:%.*]] = OpUConvert %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindUMsb [[A]]
  %elt.firstbituhigh = call <2 x i32> @llvm.spv.firstbituhigh.v2i16(<2 x i16> %a)
  ret <2 x i32> %elt.firstbituhigh
}

define noundef i32 @firstbituhigh_i64(i64 noundef %a) {
entry:
; CHECK: [[O:%.*]] = OpBitcast %[[#]] %[[#]]
; CHECK: [[N:%.*]] = OpExtInst %[[#]] %[[#]] FindUMsb [[O]]
; CHECK: [[M:%.*]] = OpVectorExtractDynamic %[[#]] [[N]] [[Z]]
; CHECK: [[L:%.*]] = OpVectorExtractDynamic %[[#]] [[N]] [[X]]
; CHECK: [[I:%.*]] = OpIEqual %[[#]] [[M]] %[[#]]
; CHECK: [[H:%.*]] = OpSelect %[[#]] [[I]] [[L]] [[M]]
; CHECK: [[C:%.*]] = OpSelect %[[#]] [[I]] %[[#]] %[[#]]
; CHECK: [[B:%.*]] = OpIAdd %[[#]] [[C]] [[H]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i64(i64 %a)
  ret i32 %elt.firstbituhigh
}

define noundef <2 x i32> @firstbituhigh_v2i64(<2 x i64> noundef %a) {
entry:
; CHECK: [[O:%.*]] = OpBitcast %[[#]] %[[#]]
; CHECK: [[N:%.*]] = OpExtInst %[[#]] %[[#]] FindUMsb [[O]]
; CHECK: [[M:%.*]] = OpVectorShuffle %[[#]] [[N]] [[N]] 0
; CHECK: [[L:%.*]] = OpVectorShuffle %[[#]] [[N]] [[N]] 1
; CHECK: [[I:%.*]] = OpIEqual %[[#]] [[M]] %[[#]]
; CHECK: [[H:%.*]] = OpSelect %[[#]] [[I]] [[L]] [[M]]
; CHECK: [[C:%.*]] = OpSelect %[[#]] [[I]] %[[#]] %[[#]]
; CHECK: [[B:%.*]] = OpIAdd %[[#]] [[C]] [[H]]
; CHECK: OpReturnValue [[B]]
  %elt.firstbituhigh = call <2 x i32> @llvm.spv.firstbituhigh.v2i64(<2 x i64> %a)
  ret <2 x i32> %elt.firstbituhigh
}

define noundef i32 @firstbitshigh_i32(i32 noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindSMsb %[[#]]
  %elt.firstbitshigh = call i32 @llvm.spv.firstbitshigh.i32(i32 %a)
  ret i32 %elt.firstbitshigh
}

define noundef i32 @firstbitshigh_i16(i16 noundef %a) {
entry:
; CHECK: [[A:%.*]] = OpSConvert %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FindSMsb %[[#]]
  %elt.firstbitshigh = call i32 @llvm.spv.firstbitshigh.i16(i16 %a)
  ret i32 %elt.firstbitshigh
}

define noundef i32 @firstbitshigh_i64(i64 noundef %a) {
entry:
; CHECK: [[O:%.*]] = OpBitcast %[[#]] %[[#]]
; CHECK: [[N:%.*]] = OpExtInst %[[#]] %[[#]] FindSMsb [[O]]
; CHECK: [[M:%.*]] = OpVectorExtractDynamic %[[#]] [[N]] [[Z]]
; CHECK: [[L:%.*]] = OpVectorExtractDynamic %[[#]] [[N]] [[X]]
; CHECK: [[I:%.*]] = OpIEqual %[[#]] [[M]] %[[#]]
; CHECK: [[H:%.*]] = OpSelect %[[#]] [[I]] [[L]] [[M]]
; CHECK: [[C:%.*]] = OpSelect %[[#]] [[I]] %[[#]] %[[#]]
; CHECK: [[B:%.*]] = OpIAdd %[[#]] [[C]] [[H]]
  %elt.firstbitshigh = call i32 @llvm.spv.firstbitshigh.i64(i64 %a)
  ret i32 %elt.firstbitshigh
}

;declare i16 @llvm.spv.firstbituhigh.i16(i16)
;declare i32 @llvm.spv.firstbituhigh.i32(i32)
;declare i64 @llvm.spv.firstbituhigh.i64(i64)
;declare i16 @llvm.spv.firstbituhigh.v2i16(<2 x i16>)
;declare i32 @llvm.spv.firstbituhigh.v2i32(<2 x i32>)
;declare i64 @llvm.spv.firstbituhigh.v2i64(<2 x i64>)
;declare i16 @llvm.spv.firstbitshigh.i16(i16)
;declare i32 @llvm.spv.firstbitshigh.i32(i32)
;declare i64 @llvm.spv.firstbitshigh.i64(i64)
;declare i16 @llvm.spv.firstbitshigh.v2i16(<2 x i16>)
;declare i32 @llvm.spv.firstbitshigh.v2i32(<2 x i32>)
;declare i64 @llvm.spv.firstbitshigh.v2i64(<2 x i64>)
