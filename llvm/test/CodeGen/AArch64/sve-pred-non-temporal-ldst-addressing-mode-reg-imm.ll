; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s | FileCheck %s

; Range checks: for all the instruction tested in this file, the
; immediate must be within the range [-8, 7] (4-bit immediate). Out of
; range values are tested only in one case (following). Valid values
; are tested all through the rest of the file.

define void @imm_out_of_range(<vscale x 2 x i64> * %base, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: imm_out_of_range:
; CHECK-NEXT: rdvl x8, #8
; CHECK-NEXT: add x8, x0, x8
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x{{[0-9]+}}]
; CHECK-NEXT: rdvl x8, #-9
; CHECK-NEXT: add x8, x0, x8
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x{{[0-9]+}}]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %base, i64 8
  %data = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %mask,
                                                                  <vscale x 2 x i64>* %base_load)
  %base_store = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64> * %base, i64 -9
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %mask,
                                            <vscale x 2 x i64>* %base_store)
  ret void
}

; 2-lane non-temporal load/stores


define void @test_masked_ldst_sv2i64(<vscale x 2 x i64> * %base, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i64:
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, #-8, mul vl]
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x0, #-7, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %base, i64 -8
  %data = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %mask,
                                                                  <vscale x 2 x i64>* %base_load)
  %base_store = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64> * %base, i64 -7
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %mask,
                                            <vscale x 2 x i64>* %base_store)
  ret void
}

define void @test_masked_ldst_sv2f64(<vscale x 2 x double> * %base, <vscale x 2 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f64:
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, #-6, mul vl]
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x0, #-5, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %base, i64 -6
  %data = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1> %mask,
                                                                    <vscale x 2 x double>* %base_load)
  %base_store = getelementptr <vscale x 2 x double>, <vscale x 2 x double> * %base, i64 -5
  call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data,
                                            <vscale x 2 x i1> %mask,
                                            <vscale x 2 x double>* %base_store)
  ret void
}

; 4-lane non-temporal load/stores.

define void @test_masked_ldst_sv4i32(<vscale x 4 x i32> * %base, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i32:
; CHECK-NEXT: ldnt1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT: stnt1w { z[[DATA]].s }, p0, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %base, i64 6
  %data = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1> %mask,
                                                                  <vscale x 4 x i32>* %base_load)
  %base_store = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32> * %base, i64 7
  call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data,
                                            <vscale x 4 x i1> %mask,
                                            <vscale x 4 x i32>* %base_store)
  ret void
}

define void @test_masked_ldst_sv4f32(<vscale x 4 x float> * %base, <vscale x 4 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4f32:
; CHECK-NEXT: ldnt1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, #-1, mul vl]
; CHECK-NEXT: stnt1w { z[[DATA]].s }, p0, [x0, #2, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %base, i64 -1
  %data = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1> %mask,
                                                                    <vscale x 4 x float>* %base_load)
  %base_store = getelementptr <vscale x 4 x float>, <vscale x 4 x float> * %base, i64 2
  call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data,
                                            <vscale x 4 x i1> %mask,
                                            <vscale x 4 x float>* %base_store)
  ret void
}


; 8-lane non-temporal load/stores.

define void @test_masked_ldst_sv8i16(<vscale x 8 x i16> * %base, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8i16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %base, i64 6
  %data = call <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1> %mask,
                                                                  <vscale x 8 x i16>* %base_load)
  %base_store = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16> * %base, i64 7
  call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data,
                                            <vscale x 8 x i1> %mask,
                                            <vscale x 8 x i16>* %base_store)
  ret void
}

define void @test_masked_ldst_sv8f16(<vscale x 8 x half> * %base, <vscale x 8 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8f16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, #-1, mul vl]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, #2, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %base, i64 -1
  %data = call <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1> %mask,
                                                                   <vscale x 8 x half>* %base_load)
  %base_store = getelementptr <vscale x 8 x half>, <vscale x 8 x half> * %base, i64 2
  call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data,
                                            <vscale x 8 x i1> %mask,
                                            <vscale x 8 x half>* %base_store)
  ret void
}

; 16-lane non-temporal load/stores.

define void @test_masked_ldst_sv16i8(<vscale x 16 x i8> * %base, <vscale x 16 x i1> %mask) nounwind {
; CHECK-LABEL: test_masked_ldst_sv16i8:
; CHECK-NEXT: ldnt1b { z[[DATA:[0-9]+]].b }, p0/z, [x0, #6, mul vl]
; CHECK-NEXT: stnt1b { z[[DATA]].b }, p0, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_load = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base, i64 6
  %data = call <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1> %mask,
                                                                  <vscale x 16 x i8>* %base_load)
  %base_store = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8> * %base, i64 7
  call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data,
                                            <vscale x 16 x i1> %mask,
                                            <vscale x 16 x i8>* %base_store)
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

