; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/bad-rowop.ll 2>&1 | FileCheck %s --check-prefix=BAD-ROWOP
; RUN: not llvm-as -disable-output < %t/bad-idx-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-IDX
; RUN: not llvm-as -disable-output < %t/bad-mdata-size.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA
; RUN: not llvm-as -disable-output < %t/bad-num-small.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-SMALL
; RUN: not llvm-as -disable-output < %t/bad-num-large.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-LARGE
; RUN: not llvm-as -disable-output < %t/bad-num-pow2.ll 2>&1 | FileCheck %s --check-prefix=BAD-NUM-POW2
; RUN: not llvm-as -disable-output < %t/bad-cdata-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-CDATA-TY
; RUN: not llvm-as -disable-output < %t/bad-mdata-type.ll 2>&1 | FileCheck %s --check-prefix=BAD-MDATA-TY

;--- bad-rowop.ll

define { i32, <2 x float>, float } @bad_rowop(ptr addrspace(6) %taddr) {
; BAD-ROWOP: immarg value 2 for arg 1 out of range
  %res = call { i32, <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i32.v2f32(ptr addrspace(6) %taddr, i32 2, i1 0, i1 0, i32 2)
  ret { i32, <2 x float>, float } %res
}

;--- bad-idx-size.ll

define { i32, <2 x float>, float } @bad_idx_size(ptr addrspace(6) %taddr) {
; BAD-IDX: immarg value 4 for arg 4 out of range set
  %res = call { i32, <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i32.v2f32(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 4)
  ret { i32, <2 x float>, float } %res
}

;--- bad-mdata-size.ll

define { i32, <32 x float>, float } @bad_metadata_register_count(
    ptr addrspace(6) %taddr) {
; BAD-MDATA: invalid llvm.nvvm.tcgen05.ld.red.spcompress.32x32b layout
  %res = call { i32, <32 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i32.v32f32(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 2)
  ret { i32, <32 x float>, float } %res
}

;--- bad-num-small.ll

define { i32, <1 x float>, float } @bad_repeat_factor_x2(
    ptr addrspace(6) %taddr) {
; BAD-NUM-SMALL: invalid llvm.nvvm.tcgen05.ld.red.spcompress.32x32b layout
  %res = call { i32, <1 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i32.v1f32(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 2)
  ret { i32, <1 x float>, float } %res
}

;--- bad-num-large.ll

define { <8 x i32>, <128 x float>, float } @bad_repeat_factor_x256(
    ptr addrspace(6) %taddr) {
; BAD-NUM-LARGE: invalid llvm.nvvm.tcgen05.ld.red.spcompress.32x32b layout
  %res = call { <8 x i32>, <128 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.v8i32.v128f32(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 2)
  ret { <8 x i32>, <128 x float>, float } %res
}

;--- bad-num-pow2.ll

define { i32, <3 x float>, float } @bad_repeat_factor_x6(
    ptr addrspace(6) %taddr) {
; BAD-NUM-POW2: invalid llvm.nvvm.tcgen05.ld.red.spcompress.32x32b layout
  %res = call { i32, <3 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i32.v3f32(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 2)
  ret { i32, <3 x float>, float } %res
}

;--- bad-cdata-type.ll

define { i32, <2 x double>, float } @bad_compressed_data_type(
    ptr addrspace(6) %taddr) {
; BAD-CDATA-TY: invalid llvm.nvvm.tcgen05.ld.red.spcompress.32x32b result type
  %res = call { i32, <2 x double>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i32.v2f64(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 2)
  ret { i32, <2 x double>, float } %res
}

;--- bad-mdata-type.ll

define { i64, <2 x float>, float } @bad_metadata_type(ptr addrspace(6) %taddr) {
; BAD-MDATA-TY: invalid llvm.nvvm.tcgen05.ld.red.spcompress.32x32b result type
  %res = call { i64, <2 x float>, float } @llvm.nvvm.tcgen05.ld.red.spcompress.32x32b.i64.v2f32(ptr addrspace(6) %taddr, i32 0, i1 0, i1 0, i32 2)
  ret { i64, <2 x float>, float } %res
}
