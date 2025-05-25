// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @kernel_func(%numberOfThreads : i32) {
  // expected-error @below {{'nvvm.barrier' op barrier id is missing, it should be set between 0 to 15}}
  nvvm.barrier number_of_threads = %numberOfThreads
}

// -----

// expected-error @below {{'"nvvm.minctasm"' attribute must be integer constant}}
llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.minctasm = "foo"} {
  llvm.return
}

// -----

// expected-error @below {{'"nvvm.maxnreg"' attribute must be integer constant}}
llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.maxnreg = "boo"} {
  llvm.return
}

// -----

// expected-error @below {{'"nvvm.reqntid"' attribute must be integer array with maximum 3 index}}
llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.reqntid = array<i32: 3, 4, 5, 6>} {
  llvm.return
}

// -----

// expected-error @below {{'"nvvm.maxntid"' attribute must be integer array with maximum 3 index}}
llvm.func @kernel_func() attributes {nvvm.kernel, nvvm.maxntid = array<i32: 3, 4, 5, 6>} {
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_acquire(%addr : !llvm.ptr, %size : i32) {
  // expected-error @below {{'nvvm.fence.proxy.acquire' op uni-directional proxies only support generic for from_proxy attribute}}
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cta> %addr, %size from_proxy=#nvvm.proxy_kind<tensormap> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_release() {
  // expected-error @below {{'nvvm.fence.proxy.release' op uni-directional proxies only support generic for from_proxy attribute}}
  nvvm.fence.proxy.release #nvvm.mem_scope<cta> from_proxy=#nvvm.proxy_kind<tensormap> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_acquire(%addr : !llvm.ptr, %size : i32) {
  // expected-error @below {{'nvvm.fence.proxy.acquire' op uni-directional proxies only support tensormap for to_proxy attribute}}
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cta> %addr, %size  from_proxy=#nvvm.proxy_kind<generic> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @nvvm_fence_proxy_release() {
  // expected-error @below {{'nvvm.fence.proxy.release' op uni-directional proxies only support tensormap for to_proxy attribute}}
  nvvm.fence.proxy.release #nvvm.mem_scope<cta> from_proxy=#nvvm.proxy_kind<generic> to_proxy=#nvvm.proxy_kind<generic>
  llvm.return
}

// -----

llvm.func @tma_prefetch_0d(%tma_desc : !llvm.ptr, %d0 : i32, %ch : i64) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[] : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_2d_im2col(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %off0 : i16, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1] im2col[%off0] l2_cache_hint = %ch : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_prefetch_5d_im2col(%tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %off0 : i16, %off1 : i16, %off2 : i16, %ch : i64) {
  // expected-error @below {{im2col offsets must be 2 less than number of coordinates}}
  nvvm.cp.async.bulk.tensor.prefetch %tma_desc, box[%d0, %d1, %d2, %d3, %d4] im2col[%off0, %off1] : !llvm.ptr
  llvm.return
}

// -----

llvm.func @tma_reduce_0d(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %ch : i64) {
  // expected-error @below {{expects coordinates between 1 to 5 dimension}}
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[] {redKind = #nvvm.tma_redux_kind<add>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

llvm.func @tma_reduce_2d_im2col(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %d0 : i32, %d1 : i32, %ch : i64) {
  // expected-error @below {{to use im2col mode, the tensor has to be at least 3-dimensional}}
  nvvm.cp.async.bulk.tensor.reduce %tma_desc, %src, box[%d0, %d1] {redKind = #nvvm.tma_redux_kind<and>, mode = #nvvm.tma_store_mode<im2col>}: !llvm.ptr, !llvm.ptr<3>
  llvm.return
}

// -----

llvm.func @convert_float_to_tf32_rna_relu(%src : f32) -> i32 {
  // expected-error @below {{Relu not supported with rna rounding mode.}}
  %res = nvvm.cvt.float.to.tf32 %src {rnd = #nvvm.fp_rnd_mode<rna>, relu=true}
  llvm.return %res : i32
}

// -----

llvm.func @convert_float_to_tf32_no_rnd_mode(%src : f32) -> i32 {
  // expected-error @below {{Only {rn,rz,rna} rounding modes supported for CvtFloatToTF32Op.}}
  %res = nvvm.cvt.float.to.tf32 %src
  llvm.return %res : i32
}

// -----

llvm.func @nvvm_st_bulk_initval_nonzero(%addr : !llvm.ptr, %size : i64) {
  // expected-error @below {{only 0 is supported for initVal, got 1}}
  nvvm.st.bulk %addr, size =  %size, init =  1 : !llvm.ptr
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_cp_128x256b_mc(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // expected-error @below {{Invalid multicast type for tcgen05.cp Op}}
  nvvm.tcgen05.cp %taddr, %smem_desc {shape = #nvvm.tcgen05_cp_shape<shape_128x256b>, multicast = #nvvm.tcgen05_cp_multicast<warpx2_02_13>}
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_cp_32x128b_wx2(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // expected-error @below {{Shape 32x128b requires multicast warpx4 for tcgen05.cp Op}}
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_32x128b>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx2_01_23>
  }
  llvm.return
}

// -----

llvm.func @nvvm_tcgen05_cp_64x128b(%taddr : !llvm.ptr<6>, %smem_desc : i64) {
  // expected-error @below {{Shape 64x128b requires multicast warpx2_01_23 or warpx2_02_13 for tcgen05.cp Op}}
  nvvm.tcgen05.cp %taddr, %smem_desc {
    shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
    multicast = #nvvm.tcgen05_cp_multicast<warpx4>
  }
  llvm.return
}

// -----

llvm.func @nvvm_match_sync_all(%val32: i32, %thread_mask: i32) {
  // expected-error @below {{match.sync 'all' returns a two element struct with first element as i32 and second element as i1}}
  %0 = nvvm.match.sync all %thread_mask, %val32 : i32 -> !llvm.struct<(i32, i8)>
  llvm.return
}

// -----

llvm.func @nvvm_match_sync_any(%val32: i32, %thread_mask: i32) {
  // expected-error @below {{match.sync 'any' returns an i32}}
  %0 = nvvm.match.sync any %thread_mask, %val32 : i32 -> !llvm.struct<(i32, i1)>
  llvm.return
}

// -----

llvm.func @nvvm_cvt_float_to_f8x2_invalid_rounding_e4m3(%a : f32, %b : f32) {
  // expected-error @below {{Only RN rounding mode is supported for conversions from f32x2 to .e4m3x2 or .e5m2x2 types}}
  %res = nvvm.cvt.f32x2.to.f8x2 <e4m3> %a, %b {rnd = #nvvm.fp_rnd_mode<rz>, sat = #nvvm.sat_mode<satfinite>} : i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_float_to_f8x2_invalid_rounding_e5m2(%a : f32, %b : f32) {
  // expected-error @below {{Only RN rounding mode is supported for conversions from f32x2 to .e4m3x2 or .e5m2x2 types}}
  %res = nvvm.cvt.f32x2.to.f8x2 <e5m2> %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, sat = #nvvm.sat_mode<satfinite>} : i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_float_to_f8x2_invalid_rounding_ue8m0(%a : f32, %b : f32) {
  // expected-error @below {{Only RZ or RP rounding modes are supported for conversions from f32x2 to .ue8m0x2 type}}
  %res = nvvm.cvt.f32x2.to.f8x2 <ue8m0> %a, %b {rnd = #nvvm.fp_rnd_mode<rn>} : i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_float_to_f8x2_invalid_saturation_e4m3(%a : f32, %b : f32) {
  // expected-error @below {{Only SATFINITE saturation mode is supported for conversions from f32x2 to .e4m3x2 or .e5m2x2 types}}
  %res = nvvm.cvt.f32x2.to.f8x2 <e4m3> %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<none>} : i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_float_to_f8x2_invalid_saturation_e5m2(%a : f32, %b : f32) {
  // expected-error @below {{Only SATFINITE saturation mode is supported for conversions from f32x2 to .e4m3x2 or .e5m2x2 types}}
  %res = nvvm.cvt.f32x2.to.f8x2 <e5m2> %a, %b {rnd = #nvvm.fp_rnd_mode<rn>, sat = #nvvm.sat_mode<none>} : i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_float_to_f8x2_relu_not_supported_ue8m0(%a : f32, %b : f32) {
  // expected-error @below {{relu not supported for conversions to .ue8m0x2 type}}
  %res = nvvm.cvt.f32x2.to.f8x2 <ue8m0> %a, %b {rnd = #nvvm.fp_rnd_mode<rp>, relu = true} : i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_f16x2_to_f8x2_invalid_type(%src : vector<2xf16>) {
  // expected-error @below {{Only .e4m3 or .e5m2 types are supported for conversions from f16x2 to f8x2.}}
  %res = nvvm.cvt.f16x2.to.f8x2 <ue8m0> %src : vector<2xf16> -> i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_bf16x2_to_f8x2_invalid_type(%src : vector<2xbf16>) {
  // expected-error @below {{Only .ue8m0 type is supported for conversions from bf16x2 to f8x2.}}
  %res = nvvm.cvt.bf16x2.to.f8x2 <e4m3> %src {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16> -> i16
  llvm.return
}

// -----

llvm.func @nvvm_cvt_bf16x2_to_f8x2_invalid_rounding(%src : vector<2xbf16>) {
  // expected-error @below {{Only RZ and RP rounding modes are supported for conversions from bf16x2 to f8x2.}}
  %res = nvvm.cvt.bf16x2.to.f8x2 <ue8m0> %src {rnd = #nvvm.fp_rnd_mode<rn>} : vector<2xbf16> -> i16
  llvm.return
}
