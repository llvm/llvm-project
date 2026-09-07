; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/bad-elem-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-ELEM
; RUN: not llvm-as -disable-output < %t/bad-idx-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX
; RUN: not llvm-as -disable-output < %t/bad-repeat-factor.ll 2>&1 | FileCheck %s --check-prefix=BAD-RPF
; RUN: not --crash llc < %t/bad-mdata-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-MDATA
; RUN: not --crash llc < %t/bad-cdata-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-CDATA
; RUN: not --crash llc < %t/bad-data-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-DATA
; RUN: not --crash llc < %t/bad-result-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-RESULT-TY
; RUN: not --crash llc < %t/bad-data-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-DATA-TY

;--- bad-elem-size.ll

define { i32, i32 } @bad_spcompress_elem_size(i32 %spdesc,
                                               <2 x i32> %data) {
; BAD-ELEM: immarg value 9 for arg 2 out of range set
  %res = call { i32, i32 } @llvm.nvvm.spcompress.sp2to4.i32.i32.v2i32(<2 x i32> %data, i32 %spdesc, i32 9, i32 2, i32 0)
  ret { i32, i32 } %res
}

;--- bad-idx-size.ll

define { i32, i32 } @bad_spcompress_idx_size(i32 %spdesc,
                                              <2 x i32> %data) {
; BAD-IDX: immarg value 3 for arg 3 out of range set
  %res = call { i32, i32 } @llvm.nvvm.spcompress.sp2to4.i32.i32.v2i32(<2 x i32> %data, i32 %spdesc, i32 8, i32 3, i32 0)
  ret { i32, i32 } %res
}

;--- bad-repeat-factor.ll

define { i32, i32 } @bad_spcompress_repeat_factor(i32 %spdesc,
                                                  <2 x i32> %data) {
; BAD-RPF: immarg value 7 for arg 4 out of range [0,7)
  %res = call { i32, i32 } @llvm.nvvm.spcompress.sp2to4.i32.i32.v2i32(<2 x i32> %data, i32 %spdesc, i32 8, i32 2, i32 7)
  ret { i32, i32 } %res
}

;--- bad-mdata-size.ll

define i32 @bad_spcompress_metadata_register_count(i32 %spdesc,
                                                   <8 x i32> %data) {
; BAD-MDATA: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { i32, <4 x i32> } @llvm.nvvm.spcompress.sp2to4.i32.v4i32.v8i32(<8 x i32> %data, i32 %spdesc, i32 8, i32 4, i32 2)
  %mdata = extractvalue { i32, <4 x i32> } %res, 0
  ret i32 %mdata
}

;--- bad-cdata-size.ll

define <2 x i32> @bad_spcompress_compressed_data_register_count(
    i32 %spdesc, <8 x i32> %data) {
; BAD-CDATA: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { <2 x i32>, <2 x i32> } @llvm.nvvm.spcompress.sp2to4.v2i32.v2i32.v8i32(<8 x i32> %data, i32 %spdesc, i32 8, i32 4, i32 2)
  %cdata = extractvalue { <2 x i32>, <2 x i32> } %res, 1
  ret <2 x i32> %cdata
}

;--- bad-data-size.ll

define <4 x i32> @bad_spcompress_data_register_count(i32 %spdesc,
                                                     <4 x i32> %data) {
; BAD-DATA: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { <2 x i32>, <4 x i32> } @llvm.nvvm.spcompress.sp2to4.v2i32.v4i32.v4i32(<4 x i32> %data, i32 %spdesc, i32 8, i32 4, i32 2)
  %cdata = extractvalue { <2 x i32>, <4 x i32> } %res, 1
  ret <4 x i32> %cdata
}

;--- bad-result-type.ll

define i64 @bad_spcompress_result_type(i32 %spdesc, <2 x i32> %data) {
; BAD-RESULT-TY: LLVM ERROR: spcompress expects scalarized i32 results
  %res = call { i64, i32 } @llvm.nvvm.spcompress.sp2to4.i64.i32.v2i32(<2 x i32> %data, i32 %spdesc, i32 8, i32 2, i32 0)
  %mdata = extractvalue { i64, i32 } %res, 0
  ret i64 %mdata
}

;--- bad-data-type.ll

define { i32, i32 } @bad_spcompress_data_type(i32 %spdesc, i64 %data) {
; BAD-DATA-TY: LLVM ERROR: spcompress expects scalarized i32 operands
  %res = call { i32, i32 } @llvm.nvvm.spcompress.sp2to4.i32.i32.i64(i64 %data, i32 %spdesc, i32 8, i32 2, i32 0)
  ret { i32, i32 } %res
}
