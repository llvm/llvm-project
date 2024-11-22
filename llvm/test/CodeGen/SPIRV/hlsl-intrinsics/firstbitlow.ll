; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[glsl_450_ext:%.+]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: OpMemoryModel Logical GLSL450
; CHECK-DAG: [[u32_t:%.+]] = OpTypeInt 32 0
; CHECK-DAG: [[u32x2_t:%.+]] = OpTypeVector [[u32_t]] 2
; CHECK-DAG: [[u32x3_t:%.+]] = OpTypeVector [[u32_t]] 3
; CHECK-DAG: [[u32x4_t:%.+]] = OpTypeVector [[u32_t]] 4
; CHECK-DAG: [[const_0:%.*]] = OpConstant [[u32_t]] 0
; CHECK-DAG: [[const_0x2:%.*]] = OpConstantComposite [[u32x2_t]] [[const_0]] [[const_0]]
; CHECK-DAG: [[const_1:%.*]] = OpConstant [[u32_t]] 1
; CHECK-DAG: [[const_32:%.*]] = OpConstant [[u32_t]] 32
; CHECK-DAG: [[const_32x2:%.*]] = OpConstantComposite [[u32x2_t]] [[const_32]] [[const_32]]
; CHECK-DAG: [[const_neg1:%.*]] = OpConstant [[u32_t]] 4294967295
; CHECK-DAG: [[const_neg1x2:%.*]] = OpConstantComposite [[u32x2_t]] [[const_neg1]] [[const_neg1]]
; CHECK-DAG: [[u16_t:%.+]] = OpTypeInt 16 0
; CHECK-DAG: [[u16x2_t:%.+]] = OpTypeVector [[u16_t]] 2
; CHECK-DAG: [[u16x3_t:%.+]] = OpTypeVector [[u16_t]] 3
; CHECK-DAG: [[u16x4_t:%.+]] = OpTypeVector [[u16_t]] 4
; CHECK-DAG: [[u64_t:%.+]] = OpTypeInt 64 0
; CHECK-DAG: [[u64x2_t:%.+]] = OpTypeVector [[u64_t]] 2
; CHECK-DAG: [[u64x3_t:%.+]] = OpTypeVector [[u64_t]] 3
; CHECK-DAG: [[u64x4_t:%.+]] = OpTypeVector [[u64_t]] 4
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

; CHECK-LABEL: Begin function firstbitlow_v2xi32
define noundef <2 x i32> @firstbitlow_v2xi32(<2 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x2_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindILsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <2 x i32> @llvm.spv.firstbitlow.v2i32(<2 x i32> %a)
  ret <2 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v3xi32
define noundef <3 x i32> @firstbitlow_v3xi32(<3 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x3_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x3_t]] [[glsl_450_ext]] FindILsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <3 x i32> @llvm.spv.firstbitlow.v3i32(<3 x i32> %a)
  ret <3 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v4xi32
define noundef <4 x i32> @firstbitlow_v4xi32(<4 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x4_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindILsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <4 x i32> @llvm.spv.firstbitlow.v4i32(<4 x i32> %a)
  ret <4 x i32> %elt.firstbitlow
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

; CHECK-LABEL: Begin function firstbitlow_v3xi16
define noundef <3 x i32> @firstbitlow_v3xi16(<3 x i16> noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16x3_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32x3_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x3_t]] [[glsl_450_ext]] FindILsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <3 x i32> @llvm.spv.firstbitlow.v3i16(<3 x i16> %a)
  ret <3 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v4xi16
define noundef <4 x i32> @firstbitlow_v4xi16(<4 x i16> noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16x4_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32x4_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindILsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <4 x i32> @llvm.spv.firstbitlow.v4i16(<4 x i16> %a)
  ret <4 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_i64
define noundef i32 @firstbitlow_i64(i64 noundef %a) {
entry:
; CHECK: [[a64:%.+]] = OpFunctionParameter [[u64_t]]
; CHECK: [[a32x2:%.+]] = OpBitcast [[u32x2_t]] [[a64]]
; CHECK: [[lsb_bits:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindILsb [[a32x2]]
; CHECK: [[high_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_0]]
; CHECK: [[low_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_1]]
; CHECK: [[should_use_high:%.+]] = OpIEqual [[bool_t]] [[low_bits]] [[const_neg1]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32_t]] [[should_use_high]] [[high_bits]] [[low_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32_t]] [[should_use_high]] [[const_32]] [[const_0]]
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
; CHECK: [[should_use_high:%.+]] = OpIEqual [[boolx2_t]] [[low_bits]] [[const_neg1x2]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32x2_t]] [[should_use_high]] [[high_bits]] [[low_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32x2_t]] [[should_use_high]] [[const_32x2]] [[const_0x2]]
; CHECK: [[ret:%.+]] = OpIAdd [[u32x2_t]] [[ans_offset]] [[ans_bits]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <2 x i32> @llvm.spv.firstbitlow.v2i64(<2 x i64> %a)
  ret <2 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v3i64
define noundef <3 x i32> @firstbitlow_v3i64(<3 x i64> noundef %a) {
entry:
; Split the i64x3 into i64, i64x2
; CHECK: [[a:%.+]] = OpFunctionParameter [[u64x3_t]]
; CHECK: [[left:%.+]] = OpVectorExtractDynamic [[u64_t]] [[a]] [[const_0]]
; CHECK: [[right:%.+]] = OpVectorShuffle [[u64x2_t]] [[a]] [[a]] 1 2

; Do firstbitlow on i64, i64x2
; CHECK: [[left_cast:%.+]] = OpBitcast [[u32x2_t]] [[left]]
; CHECK: [[left_lsb_bits:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindILsb [[left_cast]]
; CHECK: [[left_high_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[left_lsb_bits]] [[const_0]]
; CHECK: [[left_low_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[left_lsb_bits]] [[const_1]]
; CHECK: [[left_should_use_high:%.+]] = OpIEqual [[bool_t]] [[left_low_bits]] [[const_neg1]]
; CHECK: [[left_ans_bits:%.+]] = OpSelect [[u32_t]] [[left_should_use_high]] [[left_high_bits]] [[left_low_bits]]
; CHECK: [[left_ans_offset:%.+]] = OpSelect [[u32_t]] [[left_should_use_high]] [[const_32]] [[const_0]]
; CHECK: [[left_res:%.+]] = OpIAdd [[u32_t]] [[left_ans_offset]] [[left_ans_bits]]

; CHECK: [[right_cast:%.+]] = OpBitcast [[u32x4_t]] [[right]]
; CHECK: [[right_lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindILsb [[right_cast]]
; CHECK: [[right_high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[right_lsb_bits]] [[right_lsb_bits]] 0 2
; CHECK: [[right_low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[right_lsb_bits]] [[right_lsb_bits]] 1 3
; CHECK: [[right_should_use_high:%.+]] = OpIEqual [[boolx2_t]] [[right_low_bits]] [[const_neg1x2]]
; CHECK: [[right_ans_bits:%.+]] = OpSelect [[u32x2_t]] [[right_should_use_high]] [[right_high_bits]] [[right_low_bits]]
; CHECK: [[right_ans_offset:%.+]] = OpSelect [[u32x2_t]] [[right_should_use_high]] [[const_32x2]] [[const_0x2]]
; CHECK: [[right_res:%.+]] = OpIAdd [[u32x2_t]] [[right_ans_offset]] [[right_ans_bits]]

; Merge the resulting i32, i32x2 into the final i32x3 and return it
; CHECK: [[ret:%.+]] = OpCompositeConstruct [[u32x3_t]] [[left_res]] [[right_res]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <3 x i32> @llvm.spv.firstbitlow.v3i64(<3 x i64> %a)
  ret <3 x i32> %elt.firstbitlow
}

; CHECK-LABEL: Begin function firstbitlow_v4i64
define noundef <4 x i32> @firstbitlow_v4i64(<4 x i64> noundef %a) {
entry:
; Split the i64x4 into 2 i64x2
; CHECK: [[a:%.+]] = OpFunctionParameter [[u64x4_t]]
; CHECK: [[left:%.+]] = OpVectorShuffle [[u64x2_t]] [[a]] [[a]] 0 1
; CHECK: [[right:%.+]] = OpVectorShuffle [[u64x2_t]] [[a]] [[a]] 2 3

; Do firstbitlow on the 2 i64x2
; CHECK: [[left_cast:%.+]] = OpBitcast [[u32x4_t]] [[left]]
; CHECK: [[left_lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindILsb [[left_cast]]
; CHECK: [[left_high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[left_lsb_bits]] [[left_lsb_bits]] 0 2
; CHECK: [[left_low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[left_lsb_bits]] [[left_lsb_bits]] 1 3
; CHECK: [[left_should_use_high:%.+]] = OpIEqual [[boolx2_t]] [[left_low_bits]] [[const_neg1x2]]
; CHECK: [[left_ans_bits:%.+]] = OpSelect [[u32x2_t]] [[left_should_use_high]] [[left_high_bits]] [[left_low_bits]]
; CHECK: [[left_ans_offset:%.+]] = OpSelect [[u32x2_t]] [[left_should_use_high]] [[const_32x2]] [[const_0x2]]
; CHECK: [[left_res:%.+]] = OpIAdd [[u32x2_t]] [[left_ans_offset]] [[left_ans_bits]]

; CHECK: [[right_cast:%.+]] = OpBitcast [[u32x4_t]] [[right]]
; CHECK: [[right_lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindILsb [[right_cast]]
; CHECK: [[right_high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[right_lsb_bits]] [[right_lsb_bits]] 0 2
; CHECK: [[right_low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[right_lsb_bits]] [[right_lsb_bits]] 1 3
; CHECK: [[right_should_use_high:%.+]] = OpIEqual [[boolx2_t]] [[right_low_bits]] [[const_neg1x2]]
; CHECK: [[right_ans_bits:%.+]] = OpSelect [[u32x2_t]] [[right_should_use_high]] [[right_high_bits]] [[right_low_bits]]
; CHECK: [[right_ans_offset:%.+]] = OpSelect [[u32x2_t]] [[right_should_use_high]] [[const_32x2]] [[const_0x2]]
; CHECK: [[right_res:%.+]] = OpIAdd [[u32x2_t]] [[right_ans_offset]] [[right_ans_bits]]

; Merge the resulting 2 i32x2 into the final i32x4 and return it
; CHECK: [[ret:%.+]] = OpCompositeConstruct [[u32x4_t]] [[left_res]] [[right_res]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitlow = call <4 x i32> @llvm.spv.firstbitlow.v4i64(<4 x i64> %a)
  ret <4 x i32> %elt.firstbitlow
}

;declare i16 @llvm.spv.firstbitlow.i16(i16)
;declare i32 @llvm.spv.firstbitlow.i32(i32)
;declare i64 @llvm.spv.firstbitlow.i64(i64)
;declare i16 @llvm.spv.firstbitlow.v2i16(<2 x i16>)
;declare i32 @llvm.spv.firstbitlow.v2i32(<2 x i32>)
;declare i64 @llvm.spv.firstbitlow.v2i64(<2 x i64>)
