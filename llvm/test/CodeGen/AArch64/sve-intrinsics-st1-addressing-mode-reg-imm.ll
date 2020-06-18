; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

;
; ST1B
;

define void @st1b_upper_bound(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_upper_bound:
; CHECK: st1b { z0.b }, p0, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_inbound(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_inbound:
; CHECK: st1b { z0.b }, p0, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_lower_bound(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_lower_bound:
; CHECK: st1b { z0.b }, p0, [x0, #-8, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 -8
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_out_of_upper_bound(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_out_of_upper_bound:
; CHECK: rdvl x[[OFFSET:[0-9]+]], #8
; CHECK: st1b { z0.b }, p0, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 8
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_out_of_lower_bound(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_out_of_lower_bound:
; CHECK: rdvl x[[OFFSET:[0-9]+]], #-9
; CHECK: st1b { z0.b }, p0, [x0, x[[OFFSET]]]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 16 x i8>*
  %base = getelementptr <vscale x 16 x i8>, <vscale x 16 x i8>* %base_scalable, i64 -9
  %base_scalar = bitcast <vscale x 16 x i8>* %base to i8*
  call void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_s_inbound(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_s_inbound:
; CHECK: st1b { z0.s }, p0, [x0, #7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 4 x i8>*
  %base = getelementptr <vscale x 4 x i8>, <vscale x 4 x i8>* %base_scalable, i64 7
  %base_scalar = bitcast <vscale x 4 x i8>* %base to i8*
  %trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i8>
  call void @llvm.aarch64.sve.st1.nxv4i8(<vscale x 4 x i8> %trunc, <vscale x 4 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_h_inbound(<vscale x 8 x i16> %data, <vscale x 8 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_h_inbound:
; CHECK: st1b { z0.h }, p0, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 8 x i8>*
  %base = getelementptr <vscale x 8 x i8>, <vscale x 8 x i8>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 8 x i8>* %base to i8*
  %trunc = trunc <vscale x 8 x i16> %data to <vscale x 8 x i8>
  call void @llvm.aarch64.sve.st1.nxv8i8(<vscale x 8 x i8> %trunc, <vscale x 8 x i1> %pg, i8* %base_scalar)
  ret void
}

define void @st1b_d_inbound(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i8* %a) {
; CHECK-LABEL: st1b_d_inbound:
; CHECK: st1b { z0.d }, p0, [x0, #-7, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i8* %a to <vscale x 2 x i8>*
  %base = getelementptr <vscale x 2 x i8>, <vscale x 2 x i8>* %base_scalable, i64 -7
  %base_scalar = bitcast <vscale x 2 x i8>* %base to i8*
  %trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i8>
  call void @llvm.aarch64.sve.st1.nxv2i8(<vscale x 2 x i8> %trunc, <vscale x 2 x i1> %pg, i8* %base_scalar)
  ret void
}

;
; ST1H
;

define void @st1h_inbound(<vscale x 8 x i16> %data, <vscale x 8 x i1> %pg, i16* %a) {
; CHECK-LABEL: st1h_inbound:
; CHECK: st1h { z0.h }, p0, [x0, #-1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 8 x i16>*
  %base = getelementptr <vscale x 8 x i16>, <vscale x 8 x i16>* %base_scalable, i64 -1
  %base_scalar = bitcast <vscale x 8 x i16>* %base to i16*
  call void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i1> %pg, i16* %base_scalar)
  ret void
}

define void @st1h_f16_inbound(<vscale x 8 x half> %data, <vscale x 8 x i1> %pg, half* %a) {
; CHECK-LABEL: st1h_f16_inbound:
; CHECK: st1h { z0.h }, p0, [x0, #-5, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast half* %a to <vscale x 8 x half>*
  %base = getelementptr <vscale x 8 x half>, <vscale x 8 x half>* %base_scalable, i64 -5
  %base_scalar = bitcast <vscale x 8 x half>* %base to half*
  call void @llvm.aarch64.sve.st1.nxv8f16(<vscale x 8 x half> %data, <vscale x 8 x i1> %pg, half* %base_scalar)
  ret void
}

define void @st1h_s_inbound(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i16* %a) {
; CHECK-LABEL: st1h_s_inbound:
; CHECK: st1h { z0.s }, p0, [x0, #2, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 4 x i16>*
  %base = getelementptr <vscale x 4 x i16>, <vscale x 4 x i16>* %base_scalable, i64 2
  %base_scalar = bitcast <vscale x 4 x i16>* %base to i16*
  %trunc = trunc <vscale x 4 x i32> %data to <vscale x 4 x i16>
  call void @llvm.aarch64.sve.st1.nxv4i16(<vscale x 4 x i16> %trunc, <vscale x 4 x i1> %pg, i16* %base_scalar)
  ret void
}

define void @st1h_d_inbound(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i16* %a) {
; CHECK-LABEL: st1h_d_inbound:
; CHECK: st1h { z0.d }, p0, [x0, #-4, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i16* %a to <vscale x 2 x i16>*
  %base = getelementptr <vscale x 2 x i16>, <vscale x 2 x i16>* %base_scalable, i64 -4
  %base_scalar = bitcast <vscale x 2 x i16>* %base to i16*
  %trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i16>
  call void @llvm.aarch64.sve.st1.nxv2i16(<vscale x 2 x i16> %trunc, <vscale x 2 x i1> %pg, i16* %base_scalar)
  ret void
}

;
; ST1W
;

define void @st1w_inbound(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %a) {
; CHECK-LABEL: st1w_inbound:
; CHECK: st1w { z0.s }, p0, [x0, #6, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i32* %a to <vscale x 4 x i32>*
  %base = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %base_scalable, i64 6
  %base_scalar = bitcast <vscale x 4 x i32>* %base to i32*
  call void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32> %data, <vscale x 4 x i1> %pg, i32* %base_scalar)
  ret void
}

define void @st1w_f32_inbound(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %a) {
; CHECK-LABEL: st1w_f32_inbound:
; CHECK: st1w { z0.s }, p0, [x0, #-1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast float* %a to <vscale x 4 x float>*
  %base = getelementptr <vscale x 4 x float>, <vscale x 4 x float>* %base_scalable, i64 -1
  %base_scalar = bitcast <vscale x 4 x float>* %base to float*
  call void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float> %data, <vscale x 4 x i1> %pg, float* %base_scalar)
  ret void
}

define void @st1w_d_inbound(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i32* %a) {
; CHECK-LABEL: st1w_d_inbound:
; CHECK: st1w { z0.d }, p0, [x0, #1, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i32* %a to <vscale x 2 x i32>*
  %base = getelementptr <vscale x 2 x i32>, <vscale x 2 x i32>* %base_scalable, i64 1
  %base_scalar = bitcast <vscale x 2 x i32>* %base to i32*
  %trunc = trunc <vscale x 2 x i64> %data to <vscale x 2 x i32>
  call void @llvm.aarch64.sve.st1.nxv2i32(<vscale x 2 x i32> %trunc, <vscale x 2 x i1> %pg, i32* %base_scalar)
  ret void
}

;
; ST1D
;

define void @st1d_inbound(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %a) {
; CHECK-LABEL: st1d_inbound:
; CHECK: st1d { z0.d }, p0, [x0, #5, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast i64* %a to <vscale x 2 x i64>*
  %base = getelementptr <vscale x 2 x i64>, <vscale x 2 x i64>* %base_scalable, i64 5
  %base_scalar = bitcast <vscale x 2 x i64>* %base to i64*
  call void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64> %data, <vscale x 2 x i1> %pg, i64* %base_scalar)
  ret void
}

define void @st1d_f64_inbound(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %a) {
; CHECK-LABEL: st1d_f64_inbound:
; CHECK: st1d { z0.d }, p0, [x0, #-8, mul vl]
; CHECK-NEXT: ret
  %base_scalable = bitcast double* %a to <vscale x 2 x double>*
  %base = getelementptr <vscale x 2 x double>, <vscale x 2 x double>* %base_scalable, i64 -8
  %base_scalar = bitcast <vscale x 2 x double>* %base to double*
  call void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double> %data, <vscale x 2 x i1> %pg, double* %base_scalar)
  ret void
}

declare void @llvm.aarch64.sve.st1.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i1>, i8*)

declare void @llvm.aarch64.sve.st1.nxv8i8(<vscale x 8 x i8>, <vscale x 8 x i1>, i8*)
declare void @llvm.aarch64.sve.st1.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i1>, i16*)
declare void @llvm.aarch64.sve.st1.nxv8f16(<vscale x 8 x half>, <vscale x 8 x i1>, half*)

declare void @llvm.aarch64.sve.st1.nxv4i8(<vscale x 4 x i8>, <vscale x 4 x i1>, i8*)
declare void @llvm.aarch64.sve.st1.nxv4i16(<vscale x 4 x i16>, <vscale x 4 x i1>, i16*)
declare void @llvm.aarch64.sve.st1.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, i32*)
declare void @llvm.aarch64.sve.st1.nxv4f32(<vscale x 4 x float>, <vscale x 4 x i1>, float*)

declare void @llvm.aarch64.sve.st1.nxv2i8(<vscale x 2 x i8>, <vscale x 2 x i1>, i8*)
declare void @llvm.aarch64.sve.st1.nxv2i16(<vscale x 2 x i16>, <vscale x 2 x i1>, i16*)
declare void @llvm.aarch64.sve.st1.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i1>, i32*)
declare void @llvm.aarch64.sve.st1.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i1>, i64*)
declare void @llvm.aarch64.sve.st1.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, double*)
