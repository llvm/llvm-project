; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s | FileCheck %s

; 2-lane non-temporal load/stores

define void @test_masked_ldst_sv2i64(i64* %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i64:
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #3]
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %base_i64 = getelementptr i64, i64* %base, i64 %offset
  %base_addr = bitcast i64* %base_i64 to <vscale x 2 x i64>*
  %data = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %mask,
                                                                  <vscale x 2 x i64>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %mask,
                                            <vscale x 2 x i64>* %base_addr)
  ret void
}

define void @test_masked_ldst_sv2f64(double* %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f64:
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #3]
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %base_double = getelementptr double, double* %base, i64 %offset
  %base_addr = bitcast double* %base_double to <vscale x 2 x double>*
  %data = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1> %mask,
                                                                    <vscale x 2 x double>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data,
                                            <vscale x 2 x i1> %mask,
                                            <vscale x 2 x double>* %base_addr)
  ret void
}

; 4-lane non-temporal load/stores.

define void @test_masked_ldst_sv4i32(i32* %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i32:
; CHECK-NEXT: ldnt1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: stnt1w { z[[DATA]].s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_i32 = getelementptr i32, i32* %base, i64 %offset
  %base_addr = bitcast i32* %base_i32 to <vscale x 4 x i32>*
  %data = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1> %mask,
                                                                  <vscale x 4 x i32>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data,
                                            <vscale x 4 x i1> %mask,
                                            <vscale x 4 x i32>* %base_addr)
  ret void
}

define void @test_masked_ldst_sv4f32(float* %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4f32:
; CHECK-NEXT: ldnt1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: stnt1w { z[[DATA]].s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %base_float = getelementptr float, float* %base, i64 %offset
  %base_addr = bitcast float* %base_float to <vscale x 4 x float>*
  %data = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1> %mask,
                                                                    <vscale x 4 x float>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data,
                                            <vscale x 4 x i1> %mask,
                                            <vscale x 4 x float>* %base_addr)
  ret void
}


; 8-lane non-temporal load/stores.

define void @test_masked_ldst_sv8i16(i16* %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8i16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_i16 = getelementptr i16, i16* %base, i64 %offset
  %base_addr = bitcast i16* %base_i16 to <vscale x 8 x i16>*
  %data = call <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1> %mask,
                                                                  <vscale x 8 x i16>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data,
                                            <vscale x 8 x i1> %mask,
                                            <vscale x 8 x i16>* %base_addr)
  ret void
}

define void @test_masked_ldst_sv8f16(half* %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8f16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %base_half = getelementptr half, half* %base, i64 %offset
  %base_addr = bitcast half* %base_half to <vscale x 8 x half>*
  %data = call <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1> %mask,
                                                                   <vscale x 8 x half>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data,
                                            <vscale x 8 x i1> %mask,
                                            <vscale x 8 x half>* %base_addr)
  ret void
}

; 16-lane non-temporal load/stores.

define void @test_masked_ldst_sv16i8(i8* %base, <vscale x 16 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv16i8:
; CHECK-NEXT: ldnt1b { z[[DATA:[0-9]+]].b }, p0/z, [x0, x1]
; CHECK-NEXT: stnt1b { z[[DATA]].b }, p0, [x0, x1]
; CHECK-NEXT: ret
  %base_i8 = getelementptr i8, i8* %base, i64 %offset
  %base_addr = bitcast i8* %base_i8 to <vscale x 16 x i8>*
  %data = call <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1> %mask,
                                                                  <vscale x 16 x i8>* %base_addr)
  call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data,
                                            <vscale x 16 x i1> %mask,
                                            <vscale x 16 x i8>* %base_addr)
  ret void
}

; 2-element non-temporal loads.
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x i64>*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>*)

; 4-element non-temporal loads.
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x float>*)

; 8-element non-temporal loads.
declare <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x half>*)

; 16-element non-temporal loads.
declare <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>*)

; 2-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, <vscale x 2 x i64>*)
declare void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, <vscale x 2 x double>*)
                                                                      
; 4-element non-temporal stores.                                        
declare void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, <vscale x 4 x i32>*)
declare void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, <vscale x 4 x float>*)
                                                                      
; 8-element non-temporal stores.                                        
declare void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, <vscale x 8 x i16>*)
declare void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, <vscale x 8 x half>*)

; 16-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, <vscale x 16 x i8>*)
