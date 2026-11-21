; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/bad-rowop.ll 2>&1 | FileCheck %s --check-prefix=BAD-ROWOP
; RUN: not llvm-as -disable-output < %t/bad-idx-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX
; RUN: not llvm-as -disable-output < %t/bad-mdata-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA
; RUN: not llvm-as -disable-output < %t/bad-num-small.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-SMALL
; RUN: not llvm-as -disable-output < %t/bad-num-large.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-LARGE
; RUN: not llvm-as -disable-output < %t/bad-num-pow2.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-POW2
; RUN: not llvm-as -disable-output < %t/bad-cdata-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-CDATA-TY
; RUN: not llvm-as -disable-output < %t/bad-mdata-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA-TY
; RUN: not llvm-as -disable-output < %t/bad-mdata-v1.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA-V1

;--- bad-rowop.ll

define { i32, <2 x float> } @bad_rowop(ptr addrspace(6) %taddr) {
; BAD-ROWOP: immarg value 2 for arg 1 out of range
  %res = call { i32, <2 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.i32.v2f32(ptr addrspace(6) %taddr, i32 2, i1 0, i32 2)
  ret { i32, <2 x float> } %res
}

;--- bad-idx-size.ll

define { i32, <2 x float> } @bad_idx_size(ptr addrspace(6) %taddr) {
; BAD-IDX: immarg value 4 for arg 3 out of range set
  %res = call { i32, <2 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.i32.v2f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 4)
  ret { i32, <2 x float> } %res
}

;--- bad-mdata-size.ll

define { <2 x i32>, <2 x float> } @bad_metadata_register_count(
    ptr addrspace(6) %taddr) {
; BAD-MDATA: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b layout
  %res = call { <2 x i32>, <2 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.v2i32.v2f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { <2 x i32>, <2 x float> } %res
}

;--- bad-num-small.ll

define { i32, <1 x float> } @bad_repeat_factor_x2(ptr addrspace(6) %taddr) {
; BAD-NUM-SMALL: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b layout
  %res = call { i32, <1 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.i32.v1f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { i32, <1 x float> } %res
}

;--- bad-num-large.ll

define { <8 x i32>, <128 x float> } @bad_repeat_factor_x256(
    ptr addrspace(6) %taddr) {
; BAD-NUM-LARGE: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b layout
  %res = call { <8 x i32>, <128 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.v8i32.v128f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { <8 x i32>, <128 x float> } %res
}

;--- bad-num-pow2.ll

define { i32, <3 x float> } @bad_repeat_factor_x6(ptr addrspace(6) %taddr) {
; BAD-NUM-POW2: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b layout
  %res = call { i32, <3 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.i32.v3f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { i32, <3 x float> } %res
}

;--- bad-cdata-type.ll

define { i32, <2 x double> } @bad_compressed_data_type(ptr addrspace(6) %taddr) {
; BAD-CDATA-TY: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b result type
  %res = call { i32, <2 x double> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.i32.v2f64(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { i32, <2 x double> } %res
}

;--- bad-mdata-type.ll

define { i64, <2 x float> } @bad_metadata_type(ptr addrspace(6) %taddr) {
; BAD-MDATA-TY: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b result type
  %res = call { i64, <2 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.i64.v2f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { i64, <2 x float> } %res
}

;--- bad-mdata-v1.ll

define { <1 x i32>, <2 x float> } @bad_metadata_v1i32(ptr addrspace(6) %taddr) {
; BAD-MDATA-V1: invalid llvm.nvvm.tcgen05.ld.spcompress.32x32b result type
  %res = call { <1 x i32>, <2 x float> } @llvm.nvvm.tcgen05.ld.spcompress.32x32b.v1i32.v2f32(ptr addrspace(6) %taddr, i32 0, i1 0, i32 2)
  ret { <1 x i32>, <2 x float> } %res
}
