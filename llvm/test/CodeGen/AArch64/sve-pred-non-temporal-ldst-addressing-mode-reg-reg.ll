; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve --asm-verbose=false < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; 2-lane non-temporal load/stores

define void @test_masked_ldst_sv2i64(i64* %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2i64:
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #3]
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %gep = getelementptr i64, i64* %base, i64 %offset
  %data = call <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1> %mask,
                                                                  i64* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64> %data,
                                            <vscale x 2 x i1> %mask,
                                            i64* %gep)
  ret void
}

define void @test_masked_ldst_sv2f64(double* %base, <vscale x 2 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv2f64:
; CHECK-NEXT: ldnt1d { z[[DATA:[0-9]+]].d }, p0/z, [x0, x1, lsl #3]
; CHECK-NEXT: stnt1d { z[[DATA]].d }, p0, [x0, x1, lsl #3]
; CHECK-NEXT: ret
  %gep = getelementptr double, double* %base, i64 %offset
  %data = call <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1> %mask,
                                                                    double* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double> %data,
                                            <vscale x 2 x i1> %mask,
                                            double* %gep)
  ret void
}

; 4-lane non-temporal load/stores.

define void @test_masked_ldst_sv4i32(i32* %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4i32:
; CHECK-NEXT: ldnt1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: stnt1w { z[[DATA]].s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %gep = getelementptr i32, i32* %base, i64 %offset
  %data = call <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1> %mask,
                                                                  i32* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32> %data,
                                            <vscale x 4 x i1> %mask,
                                            i32* %gep)
  ret void
}

define void @test_masked_ldst_sv4f32(float* %base, <vscale x 4 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv4f32:
; CHECK-NEXT: ldnt1w { z[[DATA:[0-9]+]].s }, p0/z, [x0, x1, lsl #2]
; CHECK-NEXT: stnt1w { z[[DATA]].s }, p0, [x0, x1, lsl #2]
; CHECK-NEXT: ret
  %gep = getelementptr float, float* %base, i64 %offset
  %data = call <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1> %mask,
                                                                    float* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float> %data,
                                            <vscale x 4 x i1> %mask,
                                            float* %gep)
  ret void
}


; 8-lane non-temporal load/stores.

define void @test_masked_ldst_sv8i16(i16* %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8i16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %gep = getelementptr i16, i16* %base, i64 %offset
  %data = call <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1> %mask,
                                                                  i16* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16> %data,
                                            <vscale x 8 x i1> %mask,
                                            i16* %gep)
  ret void
}

define void @test_masked_ldst_sv8f16(half* %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv8f16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %gep = getelementptr half, half* %base, i64 %offset
  %data = call <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1> %mask,
                                                                   half* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half> %data,
                                            <vscale x 8 x i1> %mask,
                                            half* %gep)
  ret void
}

define void @test_masked_ldst_sv8bf16(bfloat* %base, <vscale x 8 x i1> %mask, i64 %offset) nounwind #0 {
; CHECK-LABEL: test_masked_ldst_sv8bf16:
; CHECK-NEXT: ldnt1h { z[[DATA:[0-9]+]].h }, p0/z, [x0, x1, lsl #1]
; CHECK-NEXT: stnt1h { z[[DATA]].h }, p0, [x0, x1, lsl #1]
; CHECK-NEXT: ret
  %gep = getelementptr bfloat, bfloat* %base, i64 %offset
  %data = call <vscale x 8 x bfloat> @llvm.aarch64.sve.ldnt1.nxv8bf16(<vscale x 8 x i1> %mask,
                                                                      bfloat* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv8bf16(<vscale x 8 x bfloat> %data,
                                             <vscale x 8 x i1> %mask,
                                             bfloat* %gep)
  ret void
}

; 16-lane non-temporal load/stores.

define void @test_masked_ldst_sv16i8(i8* %base, <vscale x 16 x i1> %mask, i64 %offset) nounwind {
; CHECK-LABEL: test_masked_ldst_sv16i8:
; CHECK-NEXT: ldnt1b { z[[DATA:[0-9]+]].b }, p0/z, [x0, x1]
; CHECK-NEXT: stnt1b { z[[DATA]].b }, p0, [x0, x1]
; CHECK-NEXT: ret
  %gep = getelementptr i8, i8* %base, i64 %offset
  %data = call <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1> %mask,
                                                                  i8* %gep)
  call void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8> %data,
                                            <vscale x 16 x i1> %mask,
                                            i8* %gep)
  ret void
}

; 2-element non-temporal loads.
declare <vscale x 2 x i64> @llvm.aarch64.sve.ldnt1.nxv2i64(<vscale x 2 x i1>, i64*)
declare <vscale x 2 x double> @llvm.aarch64.sve.ldnt1.nxv2f64(<vscale x 2 x i1>, double*)

; 4-element non-temporal loads.
declare <vscale x 4 x i32> @llvm.aarch64.sve.ldnt1.nxv4i32(<vscale x 4 x i1>, i32*)
declare <vscale x 4 x float> @llvm.aarch64.sve.ldnt1.nxv4f32(<vscale x 4 x i1>, float*)

; 8-element non-temporal loads.
declare <vscale x 8 x i16> @llvm.aarch64.sve.ldnt1.nxv8i16(<vscale x 8 x i1>, i16*)
declare <vscale x 8 x half> @llvm.aarch64.sve.ldnt1.nxv8f16(<vscale x 8 x i1>, half*)
declare <vscale x 8 x bfloat> @llvm.aarch64.sve.ldnt1.nxv8bf16(<vscale x 8 x i1>, bfloat*)

; 16-element non-temporal loads.
declare <vscale x 16 x i8> @llvm.aarch64.sve.ldnt1.nxv16i8(<vscale x 16 x i1>, i8*)

; 2-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*)
declare void @llvm.aarch64.sve.stnt1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*)

; 4-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*)
declare void @llvm.aarch64.sve.stnt1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*)

; 8-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16*)
declare void @llvm.aarch64.sve.stnt1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, half*)
declare void @llvm.aarch64.sve.stnt1.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x i1>, bfloat*)

; 16-element non-temporal stores.
declare void @llvm.aarch64.sve.stnt1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i8*)

; +bf16 is required for the bfloat version.
attributes #0 = { "target-features"="+sve,+bf16" }
