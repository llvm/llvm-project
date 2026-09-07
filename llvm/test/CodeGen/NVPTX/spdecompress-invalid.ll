; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/bad-elem-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-ELEM
; RUN: not llvm-as -disable-output < %t/bad-idx-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX
; RUN: not llvm-as -disable-output < %t/bad-repeat-factor.ll 2>&1 | FileCheck %s --check-prefix=BAD-RPF
; RUN: not --crash llc < %t/bad-idx-size-for-target.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-IDX-TARGET
; RUN: not --crash llc < %t/bad-input-span.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-INPUT-SPAN
; RUN: not --crash llc < %t/bad-data-too-small.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-DATA-SMALL
; RUN: not --crash llc < %t/bad-mdata-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-MDATA
; RUN: not --crash llc < %t/bad-cdata-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-CDATA
; RUN: not --crash llc < %t/bad-data-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-DATA
; RUN: not --crash llc < %t/bad-result-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-RESULT-TY
; RUN: not --crash llc < %t/bad-mdata-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-MDATA-TY

;--- bad-elem-size.ll

define i32 @bad_spdecompress_elem_size(i32 %metadata, i32 %compdata) {
; BAD-ELEM: immarg value 9 for arg 2 out of range set
  %res = call i32 @llvm.nvvm.spdecompress.sp2to4.i32.i32.i32(i32 %metadata, i32 %compdata, i32 9, i32 2, i32 0)
  ret i32 %res
}

;--- bad-idx-size.ll

define i32 @bad_spdecompress_idx_size(i32 %metadata, i32 %compdata) {
; BAD-IDX: immarg value 3 for arg 3 out of range set
  %res = call i32 @llvm.nvvm.spdecompress.sp2to4.i32.i32.i32(i32 %metadata, i32 %compdata, i32 8, i32 3, i32 0)
  ret i32 %res
}

;--- bad-repeat-factor.ll

define i32 @bad_spdecompress_repeat_factor(i32 %metadata, i32 %compdata) {
; BAD-RPF: immarg value 7 for arg 4 out of range [0,7)
  %res = call i32 @llvm.nvvm.spdecompress.sp2to4.i32.i32.i32(i32 %metadata, i32 %compdata, i32 8, i32 2, i32 7)
  ret i32 %res
}

;--- bad-idx-size-for-target.ll

define <8 x i32> @bad_spdecompress_idx_size_for_target(
    i32 %metadata, i32 %compdata) {
; BAD-IDX-TARGET: LLVM ERROR: Invalid spdecompress flags: num_src=1, num_tgt=8, elem_size=8, idx_size=2, repeat_factor=2
  %res = call <8 x i32> @llvm.nvvm.spdecompress.sp1to8.v8i32.i32.i32(i32 %metadata, i32 %compdata, i32 8, i32 2, i32 2)
  ret <8 x i32> %res
}

;--- bad-input-span.ll

define <4 x i32> @bad_spdecompress_input_span(<2 x i32> %metadata,
                                              <2 x i32> %compdata) {
; BAD-INPUT-SPAN: LLVM ERROR: Invalid spdecompress flags: num_src=4, num_tgt=8, elem_size=16, idx_size=4, repeat_factor=0
  %res = call <4 x i32> @llvm.nvvm.spdecompress.sp4to8.v4i32.v2i32.v2i32(<2 x i32> %metadata, <2 x i32> %compdata, i32 16, i32 4, i32 0)
  ret <4 x i32> %res
}

;--- bad-data-too-small.ll

define i32 @bad_spdecompress_data_too_small(i32 %metadata, i32 %compdata) {
; BAD-DATA-SMALL: LLVM ERROR: Invalid spdecompress flags: num_src=1, num_tgt=2, elem_size=8, idx_size=2, repeat_factor=0
  %res = call i32 @llvm.nvvm.spdecompress.sp1to2.i32.i32.i32(i32 %metadata, i32 %compdata, i32 8, i32 2, i32 0)
  ret i32 %res
}

;--- bad-mdata-size.ll

define <4 x i32> @bad_spdecompress_metadata_register_count(
    <2 x i32> %metadata, <2 x i32> %compdata) {
; BAD-MDATA: LLVM ERROR: spdecompress operand/result types do not match the SP intrinsic flags
  %res = call <4 x i32> @llvm.nvvm.spdecompress.sp2to4.v4i32.v2i32.v2i32(<2 x i32> %metadata, <2 x i32> %compdata, i32 8, i32 4, i32 2)
  ret <4 x i32> %res
}

;--- bad-cdata-size.ll

define <4 x i32> @bad_spdecompress_compressed_data_register_count(
    i32 %metadata, i32 %compdata) {
; BAD-CDATA: LLVM ERROR: spdecompress operand/result types do not match the SP intrinsic flags
  %res = call <4 x i32> @llvm.nvvm.spdecompress.sp2to4.v4i32.i32.i32(i32 %metadata, i32 %compdata, i32 16, i32 4, i32 1)
  ret <4 x i32> %res
}

;--- bad-data-size.ll

define <2 x i32> @bad_spdecompress_data_register_count(
    i32 %metadata, <2 x i32> %compdata) {
; BAD-DATA: LLVM ERROR: spdecompress operand/result types do not match the SP intrinsic flags
  %res = call <2 x i32> @llvm.nvvm.spdecompress.sp2to4.v2i32.i32.v2i32(i32 %metadata, <2 x i32> %compdata, i32 8, i32 4, i32 2)
  ret <2 x i32> %res
}

;--- bad-result-type.ll

define i64 @bad_spdecompress_result_type(i32 %metadata, i32 %compdata) {
; BAD-RESULT-TY: LLVM ERROR: spdecompress expects scalarized i32 results
  %res = call i64 @llvm.nvvm.spdecompress.sp2to4.i64.i32.i32(i32 %metadata, i32 %compdata, i32 8, i32 2, i32 0)
  ret i64 %res
}

;--- bad-mdata-type.ll

define i32 @bad_spdecompress_metadata_type(i64 %metadata, i32 %compdata) {
; BAD-MDATA-TY: LLVM ERROR: spdecompress expects scalarized i32 operands
  %res = call i32 @llvm.nvvm.spdecompress.sp2to4.i32.i64.i32(i64 %metadata, i32 %compdata, i32 8, i32 2, i32 0)
  ret i32 %res
}
