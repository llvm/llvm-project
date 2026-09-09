; RUN: split-file %s %t
; RUN: not --crash llc < %t/bad-elem-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-ELEM
; RUN: not llvm-as -disable-output < %t/bad-idx-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX
; RUN: not --crash llc < %t/bad-mdata-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-MDATA
; RUN: not --crash llc < %t/bad-cdata-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-CDATA
; RUN: not --crash llc < %t/bad-data-size.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-DATA
; RUN: not --crash llc < %t/bad-result-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-RESULT-TY
; RUN: not --crash llc < %t/bad-data-type.ll -march=nvptx64 -mcpu=sm_107a 2>&1 | FileCheck %s --check-prefix=BAD-DATA-TY

;--- bad-elem-type.ll

define { i32, <2 x i32> } @bad_spcompress_elem_type(i32 %spdesc,
                                                     <2 x i32> %data) {
; BAD-ELEM: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { i32, <2 x i32> } @llvm.nvvm.spcompress.sp2to4.i32.v2i32.v2i32(<2 x i32> %data, i32 %spdesc, i32 2)
  ret { i32, <2 x i32> } %res
}

;--- bad-idx-size.ll

define { i32, <4 x i8> } @bad_spcompress_idx_size(i32 %spdesc,
                                                  <8 x i8> %data) {
; BAD-IDX: immarg value 3 for arg 2 out of range set
  %res = call { i32, <4 x i8> } @llvm.nvvm.spcompress.sp2to4.i32.v4i8.v8i8(<8 x i8> %data, i32 %spdesc, i32 3)
  ret { i32, <4 x i8> } %res
}

;--- bad-mdata-size.ll

define i32 @bad_spcompress_metadata_register_count(i32 %spdesc,
                                                   <32 x i8> %data) {
; BAD-MDATA: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { i32, <16 x i8> } @llvm.nvvm.spcompress.sp2to4.i32.v16i8.v32i8(<32 x i8> %data, i32 %spdesc, i32 4)
  %mdata = extractvalue { i32, <16 x i8> } %res, 0
  ret i32 %mdata
}

;--- bad-cdata-size.ll

define <8 x i8> @bad_spcompress_compressed_data_register_count(
    i32 %spdesc, <32 x i8> %data) {
; BAD-CDATA: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { <2 x i32>, <8 x i8> } @llvm.nvvm.spcompress.sp2to4.v2i32.v8i8.v32i8(<32 x i8> %data, i32 %spdesc, i32 4)
  %cdata = extractvalue { <2 x i32>, <8 x i8> } %res, 1
  ret <8 x i8> %cdata
}

;--- bad-data-size.ll

define <16 x i8> @bad_spcompress_data_register_count(i32 %spdesc,
                                                     <24 x i8> %data) {
; BAD-DATA: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { <2 x i32>, <16 x i8> } @llvm.nvvm.spcompress.sp2to4.v2i32.v16i8.v24i8(<24 x i8> %data, i32 %spdesc, i32 4)
  %cdata = extractvalue { <2 x i32>, <16 x i8> } %res, 1
  ret <16 x i8> %cdata
}

;--- bad-result-type.ll

define i64 @bad_spcompress_result_type(i32 %spdesc, <8 x i8> %data) {
; BAD-RESULT-TY: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { i64, <4 x i8> } @llvm.nvvm.spcompress.sp2to4.i64.v4i8.v8i8(<8 x i8> %data, i32 %spdesc, i32 2)
  %mdata = extractvalue { i64, <4 x i8> } %res, 0
  ret i64 %mdata
}

;--- bad-data-type.ll

define { i32, <2 x i16> } @bad_spcompress_data_type(i32 %spdesc,
                                                    <8 x i8> %data) {
; BAD-DATA-TY: LLVM ERROR: spcompress operand/result types do not match the SP intrinsic flags
  %res = call { i32, <2 x i16> } @llvm.nvvm.spcompress.sp2to4.i32.v2i16.v8i8(<8 x i8> %data, i32 %spdesc, i32 2)
  ret { i32, <2 x i16> } %res
}
