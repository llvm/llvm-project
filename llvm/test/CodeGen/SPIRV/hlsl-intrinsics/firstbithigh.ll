; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[glsl_450_ext:%.+]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: OpMemoryModel Logical GLSL450
; CHECK-DAG: [[u32_t:%.+]] = OpTypeInt 32 0
; CHECK-DAG: [[u32x2_t:%.+]] = OpTypeVector [[u32_t]] 2
; CHECK-DAG: [[u32x3_t:%.+]] = OpTypeVector [[u32_t]] 3
; CHECK-DAG: [[u32x4_t:%.+]] = OpTypeVector [[u32_t]] 4
; CHECK-DAG: [[const_0:%.*]] = OpConstant [[u32_t]] 0
; CHECK-DAG: [[const_2:%.*]] = OpConstant [[u32_t]] 2
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

; CHECK-LABEL: Begin function firstbituhigh_i32
define noundef i32 @firstbituhigh_i32(i32 noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32_t]] [[glsl_450_ext]] FindUMsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i32(i32 %a)
  ret i32 %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v2xi32
define noundef <2 x i32> @firstbituhigh_v2xi32(<2 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x2_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindUMsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <2 x i32> @llvm.spv.firstbituhigh.v2i32(<2 x i32> %a)
  ret <2 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v3xi32
define noundef <3 x i32> @firstbituhigh_v3xi32(<3 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x3_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x3_t]] [[glsl_450_ext]] FindUMsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <3 x i32> @llvm.spv.firstbituhigh.v3i32(<3 x i32> %a)
  ret <3 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v4xi32
define noundef <4 x i32> @firstbituhigh_v4xi32(<4 x i32> noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32x4_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindUMsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <4 x i32> @llvm.spv.firstbituhigh.v4i32(<4 x i32> %a)
  ret <4 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_i16
define noundef i32 @firstbituhigh_i16(i16 noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32_t]] [[glsl_450_ext]] FindUMsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i16(i16 %a)
  ret i32 %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v2xi16
define noundef <2 x i32> @firstbituhigh_v2xi16(<2 x i16> noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16x2_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32x2_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindUMsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <2 x i32> @llvm.spv.firstbituhigh.v2i16(<2 x i16> %a)
  ret <2 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v3xi16
define noundef <3 x i32> @firstbituhigh_v3xi16(<3 x i16> noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16x3_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32x3_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x3_t]] [[glsl_450_ext]] FindUMsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <3 x i32> @llvm.spv.firstbituhigh.v3i16(<3 x i16> %a)
  ret <3 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v4xi16
define noundef <4 x i32> @firstbituhigh_v4xi16(<4 x i16> noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16x4_t]]
; CHECK: [[a32:%.+]] = OpUConvert [[u32x4_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindUMsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <4 x i32> @llvm.spv.firstbituhigh.v4i16(<4 x i16> %a)
  ret <4 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_i64
define noundef i32 @firstbituhigh_i64(i64 noundef %a) {
entry:
; CHECK: [[a64:%.+]] = OpFunctionParameter [[u64_t]]
; CHECK: [[a32x2:%.+]] = OpBitcast [[u32x2_t]] [[a64]]
; CHECK: [[lsb_bits:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindUMsb [[a32x2]]
; CHECK: [[high_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_0]]
; CHECK: [[low_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_1]]
; CHECK: [[should_use_low:%.+]] = OpIEqual [[bool_t]] [[high_bits]] [[const_neg1]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32_t]] [[should_use_low]] [[low_bits]] [[high_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32_t]] [[should_use_low]] [[const_0]] [[const_32]]
; CHECK: [[ret:%.+]] = OpIAdd [[u32_t]] [[ans_offset]] [[ans_bits]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call i32 @llvm.spv.firstbituhigh.i64(i64 %a)
  ret i32 %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v2xi64
define noundef <2 x i32> @firstbituhigh_v2xi64(<2 x i64> noundef %a) {
entry:
; CHECK: [[a64x2:%.+]] = OpFunctionParameter [[u64x2_t]]
; CHECK: [[a32x4:%.+]] = OpBitcast [[u32x4_t]] [[a64x2]]
; CHECK: [[lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindUMsb [[a32x4]]
; CHECK: [[high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[lsb_bits]] [[lsb_bits]] 0 2
; CHECK: [[low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[lsb_bits]] [[lsb_bits]] 1 3
; CHECK: [[should_use_low:%.+]] = OpIEqual [[boolx2_t]] [[high_bits]] [[const_neg1x2]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32x2_t]] [[should_use_low]] [[low_bits]] [[high_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32x2_t]] [[should_use_low]] [[const_0x2]] [[const_32x2]]
; CHECK: [[ret:%.+]] = OpIAdd [[u32x2_t]] [[ans_offset]] [[ans_bits]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <2 x i32> @llvm.spv.firstbituhigh.v2i64(<2 x i64> %a)
  ret <2 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v3xi64
define noundef <3 x i32> @firstbituhigh_v3xi64(<3 x i64> noundef %a) {
entry:
; Preamble
; CHECK: [[a:%.+]] = OpFunctionParameter [[u64x3_t]]

; Extract first 2 components from %a
; CHECK: [[pt1:%.+]] = OpVectorShuffle [[u64x2_t]] [[a]] [[a]] 0 1

; Do firstbituhigh on the first 2 components
; CHECK: [[pt1_cast:%.+]] = OpBitcast [[u32x4_t]] [[pt1]]
; CHECK: [[pt1_lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindUMsb [[pt1_cast]]
; CHECK: [[pt1_high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[pt1_lsb_bits]] [[pt1_lsb_bits]] 0 2
; CHECK: [[pt1_low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[pt1_lsb_bits]] [[pt1_lsb_bits]] 1 3
; CHECK: [[pt1_should_use_low:%.+]] = OpIEqual [[boolx2_t]] [[pt1_high_bits]] [[const_neg1x2]]
; CHECK: [[pt1_ans_bits:%.+]] = OpSelect [[u32x2_t]] [[pt1_should_use_low]] [[pt1_low_bits]] [[pt1_high_bits]]
; CHECK: [[pt1_ans_offset:%.+]] = OpSelect [[u32x2_t]] [[pt1_should_use_low]] [[const_0x2]] [[const_32x2]]
; CHECK: [[pt1_res:%.+]] = OpIAdd [[u32x2_t]] [[pt1_ans_offset]] [[pt1_ans_bits]]

; Extract the last component from %a
; CHECK: [[pt2:%.+]] = OpVectorExtractDynamic [[u64_t]] [[a]] [[const_2]]

; Do firstbituhigh on the last component
; CHECK: [[pt2_cast:%.+]] = OpBitcast [[u32x2_t]] [[pt2]]
; CHECK: [[pt2_lsb_bits:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindUMsb [[pt2_cast]]
; CHECK: [[pt2_high_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[pt2_lsb_bits]] [[const_0]]
; CHECK: [[pt2_low_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[pt2_lsb_bits]] [[const_1]]
; CHECK: [[pt2_should_use_low:%.+]] = OpIEqual [[bool_t]] [[pt2_high_bits]] [[const_neg1]]
; CHECK: [[pt2_ans_bits:%.+]] = OpSelect [[u32_t]] [[pt2_should_use_low]] [[pt2_low_bits]] [[pt2_high_bits]]
; CHECK: [[pt2_ans_offset:%.+]] = OpSelect [[u32_t]] [[pt2_should_use_low]] [[const_0]] [[const_32]]
; CHECK: [[pt2_res:%.+]] = OpIAdd [[u32_t]] [[pt2_ans_offset]] [[pt2_ans_bits]]

; Merge the parts into the final i32x3 and return it
; CHECK: [[ret:%.+]] = OpCompositeConstruct [[u32x3_t]] [[pt1_res]] [[pt2_res]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <3 x i32> @llvm.spv.firstbituhigh.v3i64(<3 x i64> %a)
  ret <3 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbituhigh_v4xi64
define noundef <4 x i32> @firstbituhigh_v4xi64(<4 x i64> noundef %a) {
entry:
; Preamble
; CHECK: [[a:%.+]] = OpFunctionParameter [[u64x4_t]]

; Extract first 2 components from %a
; CHECK: [[pt1:%.+]] = OpVectorShuffle [[u64x2_t]] [[a]] [[a]] 0 1

; Do firstbituhigh on the first 2 components
; CHECK: [[pt1_cast:%.+]] = OpBitcast [[u32x4_t]] [[pt1]]
; CHECK: [[pt1_lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindUMsb [[pt1_cast]]
; CHECK: [[pt1_high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[pt1_lsb_bits]] [[pt1_lsb_bits]] 0 2
; CHECK: [[pt1_low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[pt1_lsb_bits]] [[pt1_lsb_bits]] 1 3
; CHECK: [[pt1_should_use_low:%.+]] = OpIEqual [[boolx2_t]] [[pt1_high_bits]] [[const_neg1x2]]
; CHECK: [[pt1_ans_bits:%.+]] = OpSelect [[u32x2_t]] [[pt1_should_use_low]] [[pt1_low_bits]] [[pt1_high_bits]]
; CHECK: [[pt1_ans_offset:%.+]] = OpSelect [[u32x2_t]] [[pt1_should_use_low]] [[const_0x2]] [[const_32x2]]
; CHECK: [[pt1_res:%.+]] = OpIAdd [[u32x2_t]] [[pt1_ans_offset]] [[pt1_ans_bits]]

; Extract last 2 components from %a
; CHECK: [[pt2:%.+]] = OpVectorShuffle [[u64x2_t]] [[a]] [[a]] 2 3

; Do firstbituhigh on the last 2 components
; CHECK: [[pt2_cast:%.+]] = OpBitcast [[u32x4_t]] [[pt2]]
; CHECK: [[pt2_lsb_bits:%.+]] = OpExtInst [[u32x4_t]] [[glsl_450_ext]] FindUMsb [[pt2_cast]]
; CHECK: [[pt2_high_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[pt2_lsb_bits]] [[pt2_lsb_bits]] 0 2
; CHECK: [[pt2_low_bits:%.+]] = OpVectorShuffle [[u32x2_t]] [[pt2_lsb_bits]] [[pt2_lsb_bits]] 1 3
; CHECK: [[pt2_should_use_low:%.+]] = OpIEqual [[boolx2_t]] [[pt2_high_bits]] [[const_neg1x2]]
; CHECK: [[pt2_ans_bits:%.+]] = OpSelect [[u32x2_t]] [[pt2_should_use_low]] [[pt2_low_bits]] [[pt2_high_bits]]
; CHECK: [[pt2_ans_offset:%.+]] = OpSelect [[u32x2_t]] [[pt2_should_use_low]] [[const_0x2]] [[const_32x2]]
; CHECK: [[pt2_res:%.+]] = OpIAdd [[u32x2_t]] [[pt2_ans_offset]] [[pt2_ans_bits]]

; Merge the parts into the final i32x4 and return it
; CHECK: [[ret:%.+]] = OpCompositeConstruct [[u32x4_t]] [[pt1_res]] [[pt2_res]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbituhigh = call <4 x i32> @llvm.spv.firstbituhigh.v4i64(<4 x i64> %a)
  ret <4 x i32> %elt.firstbituhigh
}

; CHECK-LABEL: Begin function firstbitshigh_i32
define noundef i32 @firstbitshigh_i32(i32 noundef %a) {
entry:
; CHECK: [[a:%.+]] = OpFunctionParameter [[u32_t]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32_t]] [[glsl_450_ext]] FindSMsb [[a]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitshigh = call i32 @llvm.spv.firstbitshigh.i32(i32 %a)
  ret i32 %elt.firstbitshigh
}

; CHECK-LABEL: Begin function firstbitshigh_i16
define noundef i32 @firstbitshigh_i16(i16 noundef %a) {
entry:
; CHECK: [[a16:%.+]] = OpFunctionParameter [[u16_t]]
; CHECK: [[a32:%.+]] = OpSConvert [[u32_t]] [[a16]]
; CHECK: [[ret:%.+]] = OpExtInst [[u32_t]] [[glsl_450_ext]] FindSMsb [[a32]]
; CHECK: OpReturnValue [[ret]]
  %elt.firstbitshigh = call i32 @llvm.spv.firstbitshigh.i16(i16 %a)
  ret i32 %elt.firstbitshigh
}

; CHECK-LABEL: Begin function firstbitshigh_i64
define noundef i32 @firstbitshigh_i64(i64 noundef %a) {
entry:
; CHECK: [[a64:%.+]] = OpFunctionParameter [[u64_t]]
; CHECK: [[a32x2:%.+]] = OpBitcast [[u32x2_t]] [[a64]]
; CHECK: [[lsb_bits:%.+]] = OpExtInst [[u32x2_t]] [[glsl_450_ext]] FindSMsb [[a32x2]]
; CHECK: [[high_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_0]]
; CHECK: [[low_bits:%.+]] = OpVectorExtractDynamic [[u32_t]] [[lsb_bits]] [[const_1]]
; CHECK: [[should_use_low:%.+]] = OpIEqual [[bool_t]] [[high_bits]] [[const_neg1]]
; CHECK: [[ans_bits:%.+]] = OpSelect [[u32_t]] [[should_use_low]] [[low_bits]] [[high_bits]]
; CHECK: [[ans_offset:%.+]] = OpSelect [[u32_t]] [[should_use_low]] [[const_0]] [[const_32]]
; CHECK: [[ret:%.+]] = OpIAdd [[u32_t]] [[ans_offset]] [[ans_bits]]
; CHECK: OpReturnValue [[ret]]
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
