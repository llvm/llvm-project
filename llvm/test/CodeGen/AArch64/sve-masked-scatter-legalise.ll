; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s | FileCheck %s

; Tests that exercise various type legalisation scenarios for ISD::MSCATTER.

; Code generate the scenario where the offset vector type is illegal.
define void @masked_scatter_nxv16i8(<vscale x 16 x i8> %data, i8* %base, <vscale x 16 x i8> %offsets, <vscale x 16 x i1> %mask) {
; CHECK-LABEL: masked_scatter_nxv16i8:
; CHECK-DAG: st1b { {{z[0-9]+}}.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw]
; CHECK-DAG: st1b { {{z[0-9]+}}.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw]
; CHECK-DAG: st1b { {{z[0-9]+}}.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw]
; CHECK-DAG: st1b { {{z[0-9]+}}.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw]
; CHECK: ret
  %ptrs = getelementptr i8, i8* %base, <vscale x 16 x i8> %offsets
  call void @llvm.masked.scatter.nxv16i8(<vscale x 16 x i8> %data, <vscale x 16 x i8*> %ptrs, i32 1, <vscale x 16 x i1> %mask)
  ret void
}

define void @masked_scatter_nxv8i16(<vscale x 8 x i16> %data, i16* %base, <vscale x 8 x i16> %offsets, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: masked_scatter_nxv8i16
; CHECK-DAG: st1h { {{z[0-9]+}}.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #1]
; CHECK-DAG: st1h { {{z[0-9]+}}.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #1]
; CHECK: ret
  %ptrs = getelementptr i16, i16* %base, <vscale x 8 x i16> %offsets
  call void @llvm.masked.scatter.nxv8i16(<vscale x 8 x i16> %data, <vscale x 8 x i16*> %ptrs, i32 1, <vscale x 8 x i1> %mask)
  ret void
}

define void @masked_scatter_nxv8f32(<vscale x 8 x float> %data, float* %base, <vscale x 8 x i32> %indexes, <vscale x 8 x i1> %masks) {
; CHECK-LABEL: masked_scatter_nxv8f32
; CHECK-DAG: st1w { z0.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, uxtw #2]
; CHECK-DAG: st1w { z1.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, uxtw #2]
  %ext = zext <vscale x 8 x i32> %indexes to <vscale x 8 x i64>
  %ptrs = getelementptr float, float* %base, <vscale x 8 x i64> %ext
  call void @llvm.masked.scatter.nxv8f32(<vscale x 8 x float> %data, <vscale x 8 x float*> %ptrs, i32 0, <vscale x 8 x i1> %masks)
  ret void
}

; Code generate the worst case scenario when all vector types are illegal.
define void @masked_scatter_nxv32i32(<vscale x 32 x i32> %data, i32* %base, <vscale x 32 x i32> %offsets, <vscale x 32 x i1> %mask) {
; CHECK-LABEL: masked_scatter_nxv32i32:
; CHECK-NOT: unpkhi
; CHECK-DAG: st1w { z0.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z1.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z2.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z3.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z4.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z5.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z6.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK-DAG: st1w { z7.s }, {{p[0-9]+}}, [x0, {{z[0-9]+}}.s, sxtw #2]
; CHECK: ret
  %ptrs = getelementptr i32, i32* %base, <vscale x 32 x i32> %offsets
  call void @llvm.masked.scatter.nxv32i32(<vscale x 32 x i32> %data, <vscale x 32 x i32*> %ptrs, i32 4, <vscale x 32 x i1> %mask)
  ret void
}

declare void @llvm.masked.scatter.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8*>,  i32, <vscale x 16 x i1>)
declare void @llvm.masked.scatter.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16*>,  i32, <vscale x 8 x i1>)
declare void @llvm.masked.scatter.nxv8f32(<vscale x 8 x float>, <vscale x 8 x float*>, i32, <vscale x 8 x i1>)
declare void @llvm.masked.scatter.nxv32i32(<vscale x 32 x i32>, <vscale x 32 x i32*>,  i32, <vscale x 32 x i1>)
