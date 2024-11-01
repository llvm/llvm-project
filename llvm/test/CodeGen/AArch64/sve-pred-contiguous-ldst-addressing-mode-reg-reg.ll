; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s | FileCheck %s

; 2-lane contiguous load/stores

define void @test_masked_ldst_sv2i8(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i8:
; CHECK-NEXT: ld1b { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1]
; CHECK-NEXT: st1b { z[[DATA]].d }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %data = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 2 x i1> %mask,
                                                          <vscale x 2 x i8> undef)
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %data,
                                      <vscale x 2 x i8>* %base_i8,
                                      i32 1,
                                      <vscale x 2 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv2i16(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].d }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %data = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i16> undef)
  call void @llvm.masked.store.nxv2i16(<vscale x 2 x i16> %data,
                                       <vscale x 2 x i16>* %base_i16,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv2i32(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i32:
; CHECK-NEXT: ld1w  { z0.d }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: st1w  { z0.d }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_i32 = getelementptr i32, ptr %base, i64 %offset
  %data = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %base_i32,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i32> undef)
  call void @llvm.masked.store.nxv2i32(<vscale x 2 x i32> %data,
                                       <vscale x 2 x i32>* %base_i32,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i64:
; CHECK-NEXT: ld1d  { z0.d }, p0/z, [x0, x1, lsl #3]
; CHECK-NEXT: st1d  { z0.d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %base_i64 = getelementptr i64, ptr %base, i64 %offset
  %data = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64(<vscale x 2 x i64>* %base_i64,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i64> undef)
  call void @llvm.masked.store.nxv2i64(<vscale x 2 x i64> %data,
                                       <vscale x 2 x i64>* %base_i64,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv2f16(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].d }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_half = getelementptr half, ptr %base, i64 %offset
  %data = call <vscale x 2 x half> @llvm.masked.load.nxv2f16(<vscale x 2 x half>* %base_half,
                                                             i32 1,
                                                             <vscale x 2 x i1> %mask,
                                                             <vscale x 2 x half> undef)
  call void @llvm.masked.store.nxv2f16(<vscale x 2 x half> %data,
                                       <vscale x 2 x half>* %base_half,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
} 

define void @test_masked_ldst_sv2f32(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f32:
; CHECK-NEXT: ld1w { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: st1w { z[[DATA]].d }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_float = getelementptr float, ptr %base, i64 %offset
  %data = call <vscale x 2 x float> @llvm.masked.load.nxv2f32(<vscale x 2 x float>* %base_float,
                                                              i32 1,
                                                              <vscale x 2 x i1> %mask,
                                                              <vscale x 2 x float> undef)
  call void @llvm.masked.store.nxv2f32(<vscale x 2 x float> %data,
                                       <vscale x 2 x float>* %base_float,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv2f64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f64:
; CHECK-NEXT: ld1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #3]
; CHECK-NEXT: st1d { z[[DATA]].d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %base_double = getelementptr double, ptr %base, i64 %offset
  %data = call <vscale x 2 x double> @llvm.masked.load.nxv2f64(<vscale x 2 x double>* %base_double,
                                                               i32 1,
                                                               <vscale x 2 x i1> %mask,
                                                               <vscale x 2 x double> undef)
  call void @llvm.masked.store.nxv2f64(<vscale x 2 x double> %data,
                                       <vscale x 2 x double>* %base_double,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

; 2-lane zero/sign extended contiguous loads.

define <vscale x 2 x i64> @masked_zload_sv2i8_to_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_zload_sv2i8_to_sv2i64:
; CHECK-NEXT: ld1b { z0.d }, p0/z, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %load = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 2 x i1> %mask,
                                                          <vscale x 2 x i8> undef)
  %ext = zext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_sv2i8_to_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_sload_sv2i8_to_sv2i64:
; CHECK-NEXT: ld1sb { z0.d }, p0/z, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %load = call <vscale x 2 x i8> @llvm.masked.load.nxv2i8(<vscale x 2 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 2 x i1> %mask,
                                                          <vscale x 2 x i8> undef)
  %ext = sext <vscale x 2 x i8> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_zload_sv2i16_to_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_zload_sv2i16_to_sv2i64:
; CHECK-NEXT: ld1h { z0.d }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i16> undef)
  %ext = zext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_sv2i16_to_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_sload_sv2i16_to_sv2i64:
; CHECK-NEXT: ld1sh { z0.d }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %load = call <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i16> undef)
  %ext = sext <vscale x 2 x i16> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}


define <vscale x 2 x i64> @masked_zload_sv2i32_to_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_zload_sv2i32_to_sv2i64:
; CHECK-NEXT: ld1w { z0.d }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_i32 = getelementptr i32, ptr %base, i64 %offset
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %base_i32,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i32> undef)
  %ext = zext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

define <vscale x 2 x i64> @masked_sload_sv2i32_to_sv2i64(ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_sload_sv2i32_to_sv2i64:
; CHECK-NEXT: ld1sw { z0.d }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_i32 = getelementptr i32, ptr %base, i64 %offset
  %load = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>* %base_i32,
                                                            i32 1,
                                                            <vscale x 2 x i1> %mask,
                                                            <vscale x 2 x i32> undef)
  %ext = sext <vscale x 2 x i32> %load to <vscale x 2 x i64>
  ret <vscale x 2 x i64> %ext
}

; 2-lane truncating contiguous stores.

define void @masked_trunc_store_sv2i64_to_sv2i8(<vscale x 2 x i64> %val, ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_trunc_store_sv2i64_to_sv2i8:
; CHECK-NEXT: st1b { z0.d }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i8>
  call void @llvm.masked.store.nxv2i8(<vscale x 2 x i8> %trunc,
                                      <vscale x 2 x i8> *%base_i8,
                                      i32 1,
                                      <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_sv2i64_to_sv2i16(<vscale x 2 x i64> %val, ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_trunc_store_sv2i64_to_sv2i16:
; CHECK-NEXT: st1h { z0.d }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i16>
  call void @llvm.masked.store.nxv2i16(<vscale x 2 x i16> %trunc,
                                       <vscale x 2 x i16> *%base_i16,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

define void @masked_trunc_store_sv2i64_to_sv2i32(<vscale x 2 x i64> %val, ptr %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_trunc_store_sv2i64_to_sv2i32:
; CHECK-NEXT: st1w { z0.d }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_i32 = getelementptr i32, ptr %base, i64 %offset
  %trunc = trunc <vscale x 2 x i64> %val to <vscale x 2 x i32>
  call void @llvm.masked.store.nxv2i32(<vscale x 2 x i32> %trunc,
                                       <vscale x 2 x i32> *%base_i32,
                                       i32 1,
                                       <vscale x 2 x i1> %mask)
  ret void
}

; 4-lane contiguous load/stores.

define void @test_masked_ldst_sv4i8(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i8:
; CHECK-NEXT: ld1b { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1]
; CHECK-NEXT: st1b { z[[DATA]].s }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %data = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 4 x i1> %mask,
                                                          <vscale x 4 x i8> undef)
  call void @llvm.masked.store.nxv4i8(<vscale x 4 x i8> %data,
                                      <vscale x 4 x i8>* %base_i8,
                                      i32 1,
                                      <vscale x 4 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv4i16(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].s }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %data = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 4 x i1> %mask,
                                                            <vscale x 4 x i16> undef)
  call void @llvm.masked.store.nxv4i16(<vscale x 4 x i16> %data,
                                       <vscale x 4 x i16>* %base_i16,
                                       i32 1,
                                       <vscale x 4 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv4i32(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i32:
; CHECK-NEXT: ld1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: st1w { z[[DATA]].s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_i32 = getelementptr i32, ptr %base, i64 %offset
  %data = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32(<vscale x 4 x i32>* %base_i32,
                                                            i32 1,
                                                            <vscale x 4 x i1> %mask,
                                                            <vscale x 4 x i32> undef)
  call void @llvm.masked.store.nxv4i32(<vscale x 4 x i32> %data,
                                       <vscale x 4 x i32>* %base_i32,
                                       i32 1,
                                       <vscale x 4 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv4f16(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4f16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].s }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_f16 = getelementptr half, ptr %base, i64 %offset
  %data = call <vscale x 4 x half> @llvm.masked.load.nxv4f16(<vscale x 4 x half>* %base_f16,
                                                             i32 1,
                                                             <vscale x 4 x i1> %mask,
                                                             <vscale x 4 x half> undef)
  call void @llvm.masked.store.nxv4f16(<vscale x 4 x half> %data,
                                       <vscale x 4 x half>* %base_f16,
                                       i32 1,
                                       <vscale x 4 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv4f32(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4f32:
; CHECK-NEXT: ld1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: st1w { z[[DATA]].s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_f32 = getelementptr float, ptr %base, i64 %offset
  %data = call <vscale x 4 x float> @llvm.masked.load.nxv4f32(<vscale x 4 x float>* %base_f32,
                                                              i32 1,
                                                              <vscale x 4 x i1> %mask,
                                                              <vscale x 4 x float> undef)
  call void @llvm.masked.store.nxv4f32(<vscale x 4 x float> %data,
                                       <vscale x 4 x float>* %base_f32,
                                       i32 1,
                                       <vscale x 4 x i1> %mask)
  ret void
}

; 4-lane zero/sign extended contiguous loads.

define <vscale x 4 x i32> @masked_zload_sv4i8_to_sv4i32(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_zload_sv4i8_to_sv4i32:
; CHECK-NEXT: ld1b { z0.s }, p0/z, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 4 x i1> %mask,
                                                          <vscale x 4 x i8> undef)
  %ext = zext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_sload_sv4i8_to_sv4i32(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_sload_sv4i8_to_sv4i32:
; CHECK-NEXT: ld1sb { z0.s }, p0/z, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %load = call <vscale x 4 x i8> @llvm.masked.load.nxv4i8(<vscale x 4 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 4 x i1> %mask,
                                                          <vscale x 4 x i8> undef)
  %ext = sext <vscale x 4 x i8> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_zload_sv4i16_to_sv4i32(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_zload_sv4i16_to_sv4i32:
; CHECK-NEXT: ld1h { z0.s }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 4 x i1> %mask,
                                                            <vscale x 4 x i16> undef)
  %ext = zext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

define <vscale x 4 x i32> @masked_sload_sv4i16_to_sv4i32(ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_sload_sv4i16_to_sv4i32:
; CHECK-NEXT: ld1sh { z0.s }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %load = call <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 4 x i1> %mask,
                                                            <vscale x 4 x i16> undef)
  %ext = sext <vscale x 4 x i16> %load to <vscale x 4 x i32>
  ret <vscale x 4 x i32> %ext
}

; 4-lane truncating contiguous stores.

define void @masked_trunc_store_sv4i32_to_sv4i8(<vscale x 4 x i32> %val, ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_trunc_store_sv4i32_to_sv4i8:
; CHECK-NEXT: st1b { z0.s }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %trunc = trunc <vscale x 4 x i32> %val to <vscale x 4 x i8>
  call void @llvm.masked.store.nxv4i8(<vscale x 4 x i8> %trunc,
                                      <vscale x 4 x i8> *%base_i8,
                                      i32 1,
                                      <vscale x 4 x i1> %mask)
  ret void
}

define void @masked_trunc_store_sv4i32_to_sv4i16(<vscale x 4 x i32> %val, ptr %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_trunc_store_sv4i32_to_sv4i16:
; CHECK-NEXT: st1h { z0.s }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %trunc = trunc <vscale x 4 x i32> %val to <vscale x 4 x i16>
  call void @llvm.masked.store.nxv4i16(<vscale x 4 x i16> %trunc,
                                       <vscale x 4 x i16> *%base_i16,
                                       i32 1,
                                       <vscale x 4 x i1> %mask)
  ret void
}

; 8-lane contiguous load/stores.

define void @test_masked_ldst_sv8i8(ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8i8:
; CHECK-NEXT: ld1b { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1]
; CHECK-NEXT: st1b { z[[DATA]].h }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %data = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 8 x i1> %mask,
                                                          <vscale x 8 x i8> undef)
  call void @llvm.masked.store.nxv8i8(<vscale x 8 x i8> %data,
                                      <vscale x 8 x i8>* %base_i8,
                                      i32 1,
                                      <vscale x 8 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv8i16(ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8i16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, ptr %base, i64 %offset
  %data = call <vscale x 8 x i16> @llvm.masked.load.nxv8i16(<vscale x 8 x i16>* %base_i16,
                                                            i32 1,
                                                            <vscale x 8 x i1> %mask,
                                                            <vscale x 8 x i16> undef)
  call void @llvm.masked.store.nxv8i16(<vscale x 8 x i16> %data,
                                       <vscale x 8 x i16>* %base_i16,
                                       i32 1,
                                       <vscale x 8 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv8f16(ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8f16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_f16 = getelementptr half, ptr %base, i64 %offset
  %data = call <vscale x 8 x half> @llvm.masked.load.nxv8f16(<vscale x 8 x half>* %base_f16,
                                                             i32 1,
                                                             <vscale x 8 x i1> %mask,
                                                             <vscale x 8 x half> undef)
  call void @llvm.masked.store.nxv8f16(<vscale x 8 x half> %data,
                                       <vscale x 8 x half>* %base_f16,
                                       i32 1,
                                       <vscale x 8 x i1> %mask)
  ret void
}

define void @test_masked_ldst_sv8bf16(ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind #0 {
; CHECK-LABEL: test_masked_ldst_sv8bf16:
; CHECK-NEXT: ld1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: st1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_f16 = getelementptr bfloat, ptr %base, i64 %offset
  %data = call <vscale x 8 x bfloat> @llvm.masked.load.nxv8bf16(<vscale x 8 x bfloat>* %base_f16,
                                                               i32 1,
                                                               <vscale x 8 x i1> %mask,
                                                               <vscale x 8 x bfloat> undef)
  call void @llvm.masked.store.nxv8bf16(<vscale x 8 x bfloat> %data,
                                        <vscale x 8 x bfloat>* %base_f16,
                                        i32 1,
                                        <vscale x 8 x i1> %mask)
  ret void
}

; 8-lane zero/sign extended contiguous loads.

define <vscale x 8 x i16> @masked_zload_sv8i8_to_sv8i16(ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_zload_sv8i8_to_sv8i16:
; CHECK-NEXT: ld1b { z0.h }, p0/z, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %load = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 8 x i1> %mask,
                                                          <vscale x 8 x i8> undef)
  %ext = zext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %ext
}

define <vscale x 8 x i16> @masked_sload_sv8i8_to_sv8i16(ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_sload_sv8i8_to_sv8i16:
; CHECK-NEXT: ld1sb { z0.h }, p0/z, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %load = call <vscale x 8 x i8> @llvm.masked.load.nxv8i8(<vscale x 8 x i8>* %base_i8,
                                                          i32 1,
                                                          <vscale x 8 x i1> %mask,
                                                          <vscale x 8 x i8> undef)
  %ext = sext <vscale x 8 x i8> %load to <vscale x 8 x i16>
  ret <vscale x 8 x i16> %ext
}

; 8-lane truncating contiguous stores.

define void @masked_trunc_store_sv8i16_to_sv8i8(<vscale x 8 x i16> %val, ptr %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: masked_trunc_store_sv8i16_to_sv8i8:
; CHECK-NEXT: st1b { z0.h }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %trunc = trunc <vscale x 8 x i16> %val to <vscale x 8 x i8>
  call void @llvm.masked.store.nxv8i8(<vscale x 8 x i8> %trunc,
                                      <vscale x 8 x i8> *%base_i8,
                                      i32 1,
                                      <vscale x 8 x i1> %mask)
  ret void
}

; 16-lane contiguous load/stores.

define void @test_masked_ldst_sv16i8(ptr %base, <vscale x 16 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv16i8:
; CHECK-NEXT: ld1b { z[[DATA:[0-9]+]].b }, p0/z, [x0, x1]
; CHECK-NEXT: st1b { z[[DATA]].b }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, ptr %base, i64 %offset
  %data = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8(<vscale x 16 x i8>* %base_i8,
                                                            i32 1,
                                                            <vscale x 16 x i1> %mask,
                                                            <vscale x 16 x i8> undef)
  call void @llvm.masked.store.nxv16i8(<vscale x 16 x i8> %data,
                                       <vscale x 16 x i8>* %base_i8,
                                       i32 1,
                                       <vscale x 16 x i1> %mask)
  ret void
}

; 2-element contiguous loads.
declare <vscale x 2 x i8>  @llvm.masked.load.nxv2i8 (<vscale x 2 x i8>* , i32, <vscale x 2 x i1>, <vscale x 2 x i8> )
declare <vscale x 2 x i16> @llvm.masked.load.nxv2i16(<vscale x 2 x i16>*, i32, <vscale x 2 x i1>, <vscale x 2 x i16>)
declare <vscale x 2 x i32> @llvm.masked.load.nxv2i32(<vscale x 2 x i32>*, i32, <vscale x 2 x i1>, <vscale x 2 x i32>)
declare <vscale x 2 x i64> @llvm.masked.load.nxv2i64(<vscale x 2 x i64>*, i32, <vscale x 2 x i1>, <vscale x 2 x i64>)
declare <vscale x 2 x half> @llvm.masked.load.nxv2f16(<vscale x 2 x half>*, i32, <vscale x 2 x i1>, <vscale x 2 x half>)
declare <vscale x 2 x float> @llvm.masked.load.nxv2f32(<vscale x 2 x float>*, i32, <vscale x 2 x i1>, <vscale x 2 x float>)
declare <vscale x 2 x double> @llvm.masked.load.nxv2f64(<vscale x 2 x double>*, i32, <vscale x 2 x i1>, <vscale x 2 x double>)

; 4-element contiguous loads.
declare <vscale x 4 x i8>  @llvm.masked.load.nxv4i8 (<vscale x 4 x i8>* , i32, <vscale x 4 x i1>, <vscale x 4 x i8> )
declare <vscale x 4 x i16> @llvm.masked.load.nxv4i16(<vscale x 4 x i16>*, i32, <vscale x 4 x i1>, <vscale x 4 x i16>)
declare <vscale x 4 x i32> @llvm.masked.load.nxv4i32(<vscale x 4 x i32>*, i32, <vscale x 4 x i1>, <vscale x 4 x i32>)
declare <vscale x 4 x half> @llvm.masked.load.nxv4f16(<vscale x 4 x half>*, i32, <vscale x 4 x i1>, <vscale x 4 x half>)
declare <vscale x 4 x float> @llvm.masked.load.nxv4f32(<vscale x 4 x float>*, i32, <vscale x 4 x i1>, <vscale x 4 x float>)

; 8-element contiguous loads.
declare <vscale x 8 x i8>  @llvm.masked.load.nxv8i8 (<vscale x 8 x i8>* , i32, <vscale x 8 x i1>, <vscale x 8 x i8> )
declare <vscale x 8 x i16> @llvm.masked.load.nxv8i16(<vscale x 8 x i16>*, i32, <vscale x 8 x i1>, <vscale x 8 x i16>)
declare <vscale x 8 x half> @llvm.masked.load.nxv8f16(<vscale x 8 x half>*, i32, <vscale x 8 x i1>, <vscale x 8 x half>)
declare <vscale x 8 x bfloat> @llvm.masked.load.nxv8bf16(<vscale x 8 x bfloat>*, i32, <vscale x 8 x i1>, <vscale x 8 x bfloat>)

; 16-element contiguous loads.
declare <vscale x 16 x i8> @llvm.masked.load.nxv16i8(<vscale x 16 x i8>*, i32, <vscale x 16 x i1>, <vscale x 16 x i8>)

; 2-element contiguous stores.
declare void @llvm.masked.store.nxv2i8 (<vscale x 2 x i8> , <vscale x 2 x i8>* , i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i16>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f16(<vscale x 2 x half>, <vscale x 2 x half>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f32(<vscale x 2 x float>, <vscale x 2 x float>*, i32, <vscale x 2 x i1>)
declare void @llvm.masked.store.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>*, i32, <vscale x 2 x i1>)

; 4-element contiguous stores.
declare void @llvm.masked.store.nxv4i8 (<vscale x 4 x i8> , <vscale x 4 x i8>* , i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i16>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4f16(<vscale x 4 x half>, <vscale x 4 x half>*, i32, <vscale x 4 x i1>)
declare void @llvm.masked.store.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>*, i32, <vscale x 4 x i1>)

; 8-element contiguous stores.
declare void @llvm.masked.store.nxv8i8 (<vscale x 8 x i8> , <vscale x 8 x i8>* , i32, <vscale x 8 x i1>)
declare void @llvm.masked.store.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>*, i32, <vscale x 8 x i1>)
declare void @llvm.masked.store.nxv8f16(<vscale x 8 x half>, <vscale x 8 x half>*, i32, <vscale x 8 x i1>)
declare void @llvm.masked.store.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>*, i32, <vscale x 8 x i1>)

; 16-element contiguous stores.
declare void @llvm.masked.store.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>*, i32, <vscale x 16 x i1>)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
