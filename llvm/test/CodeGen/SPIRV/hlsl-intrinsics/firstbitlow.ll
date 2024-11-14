; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[glsl_450_ext:%.+]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: OpMemoryModel Logical GLSL450
; CHECK-DAG: [[u32_t:%.+]] = OpTypeInt 32 0
; CHECK-DAG: [[u32x2_t:%.+]] = OpTypeVector [[u32_t]] 2
; CHECK-DAG: [[u32x4_t:%.+]] = OpTypeVector [[u32_t]] 4
; CHECK-DAG: [[const_zero:%.*]] = OpConstant [[u32_t]] 0
; CHECK-DAG: [[const_zerox2:%.*]] = OpConstantComposite [[u32x2_t]] [[const_zero]] [[const_zero]]
; CHECK-DAG: [[const_one:%.*]] = OpConstant [[u32_t]] 1
; CHECK-DAG: [[const_thirty_two:%.*]] = OpConstant [[u32_t]] 32
; CHECK-DAG: [[const_thirty_twox2:%.*]] = OpConstantComposite [[u32x2_t]] [[const_thirty_two]] [[const_thirty_two]]
; CHECK-DAG: [[const_neg_one:%.*]] = OpConstant [[u32_t]] 4294967295
; CHECK-DAG: [[const_neg_onex2:%.*]] = OpConstantComposite [[u32x2_t]] [[const_neg_one]] [[const_neg_one]]
; CHECK-DAG: [[u16_t:%.+]] = OpTypeInt 16 0
; CHECK-DAG: [[u16x2_t:%.+]] = OpTypeVector [[u16_t]] 2
; CHECK-DAG: [[u64_t:%.+]] = OpTypeInt 64 0
; CHECK-DAG: [[u64x2_t:%.+]] = OpTypeVector [[u64_t]] 2
; CHECK-DAG: [[bool_t:%.+]] = OpTypeBool
; CHECK-DAG: [[boolx2_t:%.+]] = OpTypeVector [[bool_t]] 2

; CHECK-LABEL: Begin function firstbitlow_i32
define noundef i32 @firstbitlow_i32(i32 noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32_t]] [[glsl_450_ext]] FindILsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call i32 @llvm.spv.firstbitlow.i32(i32 %a)
  ret i32 %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_2xi32
define noundef <2 x i32> @firstbitlow_2xi32(<2 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x2_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindILsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <2 x i32> @llvm.spv.firstbitlow.v2i32(<2 x i32> %a)
  ret <2 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_i16
define noundef i32 @firstbitlow_i16(i16 noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32_t]] [[glsl_450_ext]] FindILsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call i32 @llvm.spv.firstbitlow.i16(i16 %a)
  ret i32 %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v2i16
define noundef <2 x i32> @firstbitlow_v2i16(<2 x i16> noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16x2_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32x2_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindILsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <2 x i32> @llvm.spv.firstbitlow.v2i16(<2 x i16> %a)
  ret <2 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_i64
define noundef i32 @firstbitlow_i64(i64 noundef %a) {
entry:
; CHECK: [[a64:%.+]] = OpFunctionParameter [[u64_t]]
; CHECK: [[a32x2:%.+]] = OpBitcast [[u32x2_t]] [[a64]]
; CHECK: [[lsb_bits:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindILsb [[a32x2]]
; CHECK: [[high_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_zero]]
; CHECK: [[low_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_one]]
; CHECK: [[should_use_high:%.+]] = OpIEqual [[bool_t]] [[low_bits]] [[const_neg_one]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32_t]] [[should_use_high]] [[high_bits]] [[low_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32_t]] [[should_use_high]] [[const_thirty_two]] [[const_zero]]
; CHECK: [[ret:%.+]] = OpIAdd [[u32_t]] [[ans_offset]] [[ans_bits]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call i32 @llvm.spv.firstbitlow.i64(i64 %a)
  ret i32 %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v2i64
define noundef <2 x i32> @firstbitlow_v2i64(<2 x i64> noundef %a) {
entry:
; CHECK: [[a64x2:%.+]] = OpFunctionParameter [[u64x2_t]]
; CHECK: [[a32x4:%.+]] = OpBitcast [[u32x4_t]] [[a64x2]]
; CHECK: [[lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindILsb [[a32x4]]
; CHECK: [[high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[lsb_bits]] [[lsb_bits]] 0 2
; CHECK: [[low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[lsb_bits]] [[lsb_bits]] 1 3
; CHECK: [[should_use_high:%.+]] = OpIEqual [[boolx2_t]] [[low_bits]] [[const_neg_onex2]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32x2_t]] [[should_use_high]] [[high_bits]] [[low_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32x2_t]] [[should_use_high]] [[const_thirty_twox2]] [[const_zerox2]]
; CHECK: [[ret:%.+]] = OpIAdd [[u32x2_t]] [[ans_offset]] [[ans_bits]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <2 x i32> @llvm.spv.firstbitlow.v2i64(<2 x i64> %a)
  ret <2 x i32> %elt.firstbitlow
}

;declare i16 @llvm.spv.firstbitlow.i16(i16)
;declare i32 @llvm.spv.firstbitlow.i32(i32)
;declare i64 @llvm.spv.firstbitlow.i64(i64)
;declare i16 @llvm.spv.firstbitlow.v2i16(<2 x i16>)
;declare i32 @llvm.spv.firstbitlow.v2i32(<2 x i32>)
;declare i64 @llvm.spv.firstbitlow.v2i64(<2 x i64>)
