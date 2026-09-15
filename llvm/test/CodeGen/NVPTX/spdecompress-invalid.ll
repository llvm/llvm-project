; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/bad-elem-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-ELEM
; RUN: not llvm-as -disable-output < %t/bad-idx-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX
; RUN: not llvm-as -disable-output < %t/bad-num-tgt.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-TGT
; RUN: not llvm-as -disable-output < %t/bad-idx-size-for-target.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX-TARGET
; RUN: not llvm-as -disable-output < %t/bad-input-span.ll 2>&1 | FileCheck %s --check-prefix=BAD-INPUT-SPAN
; RUN: not llvm-as -disable-output < %t/bad-mdata-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA
; RUN: not llvm-as -disable-output < %t/bad-cdata-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-CDATA
; RUN: not llvm-as -disable-output < %t/bad-data-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-DATA
; RUN: not llvm-as -disable-output < %t/bad-result-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-RESULT-TY
; RUN: not llvm-as -disable-output < %t/bad-mdata-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA-TY

;--- bad-elem-type.ll

define <2 x i32> @bad_spdecompress_elem_type(i32 %metadata,
                                             <2 x i32> %compdata) {
; BAD-ELEM: invalid llvm.nvvm.spdecompress operand or result type
  %res = call <2 x i32> @llvm.nvvm.spdecompress.v2i32.i32.v2i32(i32 %metadata, <2 x i32> %compdata, i32 2, i32 4)
  ret <2 x i32> %res
}

;--- bad-idx-size.ll

define <4 x i8> @bad_spdecompress_idx_size(i32 %metadata,
                                           <4 x i8> %compdata) {
; BAD-IDX: immarg value 3 for arg 2 out of range set
  %res = call <4 x i8> @llvm.nvvm.spdecompress.v4i8.i32.v4i8(i32 %metadata, <4 x i8> %compdata, i32 3, i32 4)
  ret <4 x i8> %res
}

;--- bad-num-tgt.ll

define <4 x i8> @bad_spdecompress_num_tgt(i32 %metadata,
                                          <2 x i8> %compdata) {
; BAD-NUM-TGT: immarg value 3 for arg 3 out of range set
  %res = call <4 x i8> @llvm.nvvm.spdecompress.v4i8.i32.v2i8(i32 %metadata, <2 x i8> %compdata, i32 2, i32 3)
  ret <4 x i8> %res
}

;--- bad-idx-size-for-target.ll

define <32 x i8> @bad_spdecompress_idx_size_for_target(
    i32 %metadata, <4 x i8> %compdata) {
; BAD-IDX-TARGET: invalid llvm.nvvm.spdecompress layout
  %res = call <32 x i8> @llvm.nvvm.spdecompress.v32i8.i32.v4i8(i32 %metadata, <4 x i8> %compdata, i32 2, i32 8)
  ret <32 x i8> %res
}

;--- bad-input-span.ll

define <8 x i16> @bad_spdecompress_input_span(<2 x i32> %metadata,
                                              <4 x i16> %compdata) {
; BAD-INPUT-SPAN: invalid llvm.nvvm.spdecompress layout
  %res = call <8 x i16> @llvm.nvvm.spdecompress.v8i16.v2i32.v4i16(<2 x i32> %metadata, <4 x i16> %compdata, i32 4, i32 8)
  ret <8 x i16> %res
}

;--- bad-mdata-size.ll

define <16 x i8> @bad_spdecompress_metadata_register_count(
    <2 x i32> %metadata, <8 x i8> %compdata) {
; BAD-MDATA: invalid llvm.nvvm.spdecompress layout
  %res = call <16 x i8> @llvm.nvvm.spdecompress.v16i8.v2i32.v8i8(<2 x i32> %metadata, <8 x i8> %compdata, i32 4, i32 4)
  ret <16 x i8> %res
}

;--- bad-cdata-size.ll

define <8 x i16> @bad_spdecompress_compressed_data_register_count(
    i32 %metadata, <6 x i16> %compdata) {
; BAD-CDATA: invalid llvm.nvvm.spdecompress layout
  %res = call <8 x i16> @llvm.nvvm.spdecompress.v8i16.i32.v6i16(i32 %metadata, <6 x i16> %compdata, i32 4, i32 4)
  ret <8 x i16> %res
}

;--- bad-data-size.ll

define <12 x i8> @bad_spdecompress_data_register_count(
    i32 %metadata, <8 x i8> %compdata) {
; BAD-DATA: invalid llvm.nvvm.spdecompress layout
  %res = call <12 x i8> @llvm.nvvm.spdecompress.v12i8.i32.v8i8(i32 %metadata, <8 x i8> %compdata, i32 4, i32 4)
  ret <12 x i8> %res
}

;--- bad-result-type.ll

define <4 x i16> @bad_spdecompress_result_type(i32 %metadata,
                                               <2 x i8> %compdata) {
; BAD-RESULT-TY: invalid llvm.nvvm.spdecompress operand or result type
  %res = call <4 x i16> @llvm.nvvm.spdecompress.v4i16.i32.v2i8(i32 %metadata, <2 x i8> %compdata, i32 2, i32 4)
  ret <4 x i16> %res
}

;--- bad-mdata-type.ll

define <4 x i8> @bad_spdecompress_metadata_type(i64 %metadata,
                                                <2 x i8> %compdata) {
; BAD-MDATA-TY: invalid llvm.nvvm.spdecompress operand or result type
  %res = call <4 x i8> @llvm.nvvm.spdecompress.v4i8.i64.v2i8(i64 %metadata, <2 x i8> %compdata, i32 2, i32 4)
  ret <4 x i8> %res
}
