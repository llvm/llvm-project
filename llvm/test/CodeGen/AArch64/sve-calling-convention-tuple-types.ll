; RUN: llc -mtriple aarch64 -mattr=+sve -asm-verbose=0 < %s | FileCheck %s

;
; svint8x2_t
;

define <vscale x 32 x i8> @ret_svint8x2_t(<vscale x 16 x i8> %unused_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2) #0 {
; CHECK-LABEL: ret_svint8x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2)
  ret <vscale x 32 x i8> %tuple
}

define void @call_svint8x2_t(<vscale x 16 x i8> %dummy_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %dummy_z2, <vscale x 16 x i8> %z3) #0 {
; CHECK-LABEL: call_svint8x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z3.d
; CHECK-NEXT: bl callee_svint8x2_t
  %tuple = tail call <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z3)
  call void @callee_svint8x2_t(<vscale x 32 x i8> %tuple)
  ret void
}

;
; svint16x2_t
;

define <vscale x 16 x i16> @ret_svint16x2_t(<vscale x 8 x i16> %unused_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2) #0 {
; CHECK-LABEL: ret_svint16x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2)
  ret <vscale x 16 x i16> %tuple
}

define void @call_svint16x2_t(<vscale x 8 x i16> %dummy_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %dummy_z2, <vscale x 8 x i16> %z3) #0 {
; CHECK-LABEL: call_svint16x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z3.d
; CHECK-NEXT: bl callee_svint16x2_t
  %tuple = tail call <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z3)
  call void @callee_svint16x2_t(<vscale x 16 x i16> %tuple)
  ret void
}

;
; svint32x2_t
;

define <vscale x 8 x i32> @ret_svint32x2_t(<vscale x 4 x i32> %unused_z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2) #0 {
; CHECK-LABEL: ret_svint32x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2)
  ret <vscale x 8 x i32> %tuple
}

define void @call_svint32x2_t(<vscale x 4 x i32> %dummy_z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %dummy_z2, <vscale x 4 x i32> %z3) #0 {
; CHECK-LABEL: call_svint32x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z3.d
; CHECK-NEXT: bl callee_svint32x2_t
  %tuple = tail call <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32> %z1, <vscale x 4 x i32> %z3)
  call void @callee_svint32x2_t(<vscale x 8 x i32> %tuple)
  ret void
}

;
; svint64x2_t
;

define <vscale x 4 x i64> @ret_svint64x2_t(<vscale x 2 x i64> %unused_z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2) #0 {
; CHECK-LABEL: ret_svint64x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2)
  ret <vscale x 4 x i64> %tuple
}

define void @call_svint64x2_t(<vscale x 2 x i64> %dummy_z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %dummy_z2, <vscale x 2 x i64> %z3) #0 {
; CHECK-LABEL: call_svint64x2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z3.d
; CHECK-NEXT: bl callee_svint64x2_t
  %tuple = tail call <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64> %z1, <vscale x 2 x i64> %z3)
  call void @callee_svint64x2_t(<vscale x 4 x i64> %tuple)
  ret void
}

;
; svfloatx2_t
;

define <vscale x 8 x float> @ret_svfloatx2_t(<vscale x 4 x float> %unused_z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2) #0 {
; CHECK-LABEL: ret_svfloatx2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x float> @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float> %z1, <vscale x 4 x float> %z2)
  ret <vscale x 8 x float> %tuple
}

define void @call_svfloatx2_t(<vscale x 4 x float> %dummy_z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %dummy_z2, <vscale x 4 x float> %z3) #0 {
; CHECK-LABEL: call_svfloatx2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z3.d
; CHECK-NEXT: bl callee_svfloatx2_t
  %tuple = tail call <vscale x 8 x float> @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float> %z1, <vscale x 4 x float> %z3)
  call void @callee_svfloatx2_t(<vscale x 8 x float> %tuple)
  ret void
}

;
; svdoublex2_t
;

define <vscale x 4 x double> @ret_svdoublex2_t(<vscale x 2 x double> %unused_z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2) #0 {
; CHECK-LABEL: ret_svdoublex2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 4 x double> @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double> %z1, <vscale x 2 x double> %z2)
  ret <vscale x 4 x double> %tuple
}

define void @call_svdoublex2_t(<vscale x 2 x double> %dummy_z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %dummy_z2, <vscale x 2 x double> %z3) #0 {
; CHECK-LABEL: call_svdoublex2_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z3.d
; CHECK-NEXT: bl callee_svdoublex2_t
  %tuple = tail call <vscale x 4 x double> @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double> %z1, <vscale x 2 x double> %z3)
  call void @callee_svdoublex2_t(<vscale x 4 x double> %tuple)
  ret void
}

;
; svint8x3_t
;

define <vscale x 48 x i8> @ret_svint8x3_t(<vscale x 16 x i8> %unused_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3) #0 {
; CHECK-LABEL: ret_svint8x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3)
  ret <vscale x 48 x i8> %tuple
}

define void @call_svint8x3_t(<vscale x 16 x i8> %dummy_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %dummy_z3, <vscale x 16 x i8> %z4) #0 {
; CHECK-LABEL: call_svint8x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint8x3_t
  %tuple = tail call <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z4)
  call void @callee_svint8x3_t(<vscale x 48 x i8> %tuple)
  ret void
}

;
; svint16x3_t
;

define <vscale x 24 x i16> @ret_svint16x3_t(<vscale x 8 x i16> %unused_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3) #0 {
; CHECK-LABEL: ret_svint16x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3)
  ret <vscale x 24 x i16> %tuple
}

define void @call_svint16x3_t(<vscale x 8 x i16> %dummy_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %dummy_z3, <vscale x 8 x i16> %z4) #0 {
; CHECK-LABEL: call_svint16x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint16x3_t
  %tuple = tail call <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z4)
  call void @callee_svint16x3_t(<vscale x 24 x i16> %tuple)
  ret void
}

;
; svint32x3_t
;

define <vscale x 12 x i32> @ret_svint32x3_t(<vscale x 4 x i32> %unused_z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3) #0 {
; CHECK-LABEL: ret_svint32x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3)
  ret <vscale x 12 x i32> %tuple
}

define void @call_svint32x3_t(<vscale x 4 x i32> %dummy_z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %dummy_z3, <vscale x 4 x i32> %z4) #0 {
; CHECK-LABEL: call_svint32x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint32x3_t
  %tuple = tail call <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z4)
  call void @callee_svint32x3_t(<vscale x 12 x i32> %tuple)
  ret void
}

;
; svint64x3_t
;

define <vscale x 6 x i64> @ret_svint64x3_t(<vscale x 2 x i64> %unused_z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3) #0 {
; CHECK-LABEL: ret_svint64x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3)
  ret <vscale x 6 x i64> %tuple
}

define void @call_svint64x3_t(<vscale x 2 x i64> %dummy_z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %dummy_z3, <vscale x 2 x i64> %z4) #0 {
; CHECK-LABEL: call_svint64x3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint64x3_t
  %tuple = tail call <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z4)
  call void @callee_svint64x3_t(<vscale x 6 x i64> %tuple)
  ret void
}

;
; svfloatx3_t
;

define <vscale x 12 x float> @ret_svfloatx3_t(<vscale x 4 x float> %unused_z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3) #0 {
; CHECK-LABEL: ret_svfloatx3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3)
  ret <vscale x 12 x float> %tuple
}

define void @call_svfloatx3_t(<vscale x 4 x float> %dummy_z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %dummy_z3, <vscale x 4 x float> %z4) #0 {
; CHECK-LABEL: call_svfloatx3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svfloatx3_t
  %tuple = tail call <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z4)
  call void @callee_svfloatx3_t(<vscale x 12 x float> %tuple)
  ret void
}

;
; svdoublex3_t
;

define <vscale x 6 x double> @ret_svdoublex3_t(<vscale x 2 x double> %unused_z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3) #0 {
; CHECK-LABEL: ret_svdoublex3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 6 x double> @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3)
  ret <vscale x 6 x double> %tuple
}

define void @call_svdoublex3_t(<vscale x 2 x double> %dummy_z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %dummy_z3, <vscale x 2 x double> %z4) #0 {
; CHECK-LABEL: call_svdoublex3_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svdoublex3_t
  %tuple = tail call <vscale x 6 x double> @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z4)
  call void @callee_svdoublex3_t(<vscale x 6 x double> %tuple)
  ret void
}

;
; svint8x4_t
;

define <vscale x 64 x i8> @ret_svint8x4_t(<vscale x 16 x i8> %unused_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3, <vscale x 16 x i8> %z4) #0 {
; CHECK-LABEL: ret_svint8x4_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: mov z3.d, z4.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z3, <vscale x 16 x i8> %z4)
  ret <vscale x 64 x i8> %tuple
}

define void @call_svint8x4_t(<vscale x 16 x i8> %dummy_z0, <vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %dummy_z3, <vscale x 16 x i8> %z4, <vscale x 16 x i8> %z5) #0 {
; CHECK-LABEL: call_svint8x4_t
; CHECK:      mov z3.d, z5.d
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint8x4_t
  %tuple = tail call <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8> %z1, <vscale x 16 x i8> %z2, <vscale x 16 x i8> %z4, <vscale x 16 x i8> %z5)
  call void @callee_svint8x4_t(<vscale x 64 x i8> %tuple)
  ret void
}

;
; svint16x4_t
;

define <vscale x 32 x i16> @ret_svint16x4_t(<vscale x 8 x i16> %unused_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3, <vscale x 8 x i16> %z4) #0 {
; CHECK-LABEL: ret_svint16x4_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: mov z3.d, z4.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 32 x i16> @llvm.aarch64.sve.tuple.create4.nxv32i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z3, <vscale x 8 x i16> %z4)
  ret <vscale x 32 x i16> %tuple
}

define void @call_svint16x4_t(<vscale x 8 x i16> %dummy_z0, <vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %dummy_z3, <vscale x 8 x i16> %z4, <vscale x 8 x i16> %z5) #0 {
; CHECK-LABEL: call_svint16x4_t
; CHECK:      mov z3.d, z5.d
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint16x4_t
  %tuple = tail call <vscale x 32 x i16> @llvm.aarch64.sve.tuple.create4.nxv32i16.nxv8i16(<vscale x 8 x i16> %z1, <vscale x 8 x i16> %z2, <vscale x 8 x i16> %z4, <vscale x 8 x i16> %z5)
  call void @callee_svint16x4_t(<vscale x 32 x i16> %tuple)
  ret void
}

;
; svint32x4_t
;

define <vscale x 16 x i32> @ret_svint32x4_t(<vscale x 4 x i32> %unused_z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3, <vscale x 4 x i32> %z4) #0 {
; CHECK-LABEL: ret_svint32x4_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: mov z3.d, z4.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3, <vscale x 4 x i32> %z4)
  ret <vscale x 16 x i32> %tuple
}

define void @call_svint32x4_t(<vscale x 4 x i32> %dummy_z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %dummy_z3, <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5) #0 {
; CHECK-LABEL: call_svint32x4_t
; CHECK:      mov z3.d, z5.d
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint32x4_t
  %tuple = tail call <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5)
  call void @callee_svint32x4_t(<vscale x 16 x i32> %tuple)
  ret void
}

;
; svint64x4_t
;

define <vscale x 8 x i64> @ret_svint64x4_t(<vscale x 2 x i64> %unused_z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3, <vscale x 2 x i64> %z4) #0 {
; CHECK-LABEL: ret_svint64x4_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: mov z3.d, z4.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x i64> @llvm.aarch64.sve.tuple.create4.nxv8i64.nxv2i64(<vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z3, <vscale x 2 x i64> %z4)
  ret <vscale x 8 x i64> %tuple
}

define void @call_svint64x4_t(<vscale x 2 x i64> %dummy_z0, <vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %dummy_z3, <vscale x 2 x i64> %z4, <vscale x 2 x i64> %z5) #0 {
; CHECK-LABEL: call_svint64x4_t
; CHECK:      mov z3.d, z5.d
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svint64x4_t
  %tuple = tail call <vscale x 8 x i64> @llvm.aarch64.sve.tuple.create4.nxv8i64.nxv2i64(<vscale x 2 x i64> %z1, <vscale x 2 x i64> %z2, <vscale x 2 x i64> %z4, <vscale x 2 x i64> %z5)
  call void @callee_svint64x4_t(<vscale x 8 x i64> %tuple)
  ret void
}

;
; svfloatx4_t
;

define <vscale x 16 x float> @ret_svfloatx4_t(<vscale x 4 x float> %unused_z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3, <vscale x 4 x float> %z4) #0 {
; CHECK-LABEL: ret_svfloatx4_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: mov z3.d, z4.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 16 x float> @llvm.aarch64.sve.tuple.create4.nxv16f32.nxv4f32(<vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z3, <vscale x 4 x float> %z4)
  ret <vscale x 16 x float> %tuple
}

define void @call_svfloatx4_t(<vscale x 4 x float> %dummy_z0, <vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %dummy_z3, <vscale x 4 x float> %z4, <vscale x 4 x float> %z5) #0 {
; CHECK-LABEL: call_svfloatx4_t
; CHECK:      mov z3.d, z5.d
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svfloatx4_t
  %tuple = tail call <vscale x 16 x float> @llvm.aarch64.sve.tuple.create4.nxv16f32.nxv4f32(<vscale x 4 x float> %z1, <vscale x 4 x float> %z2, <vscale x 4 x float> %z4, <vscale x 4 x float> %z5)
  call void @callee_svfloatx4_t(<vscale x 16 x float> %tuple)
  ret void
}

;
; svdoublex4_t
;

define <vscale x 8 x double> @ret_svdoublex4_t(<vscale x 2 x double> %unused_z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3, <vscale x 2 x double> %z4) #0 {
; CHECK-LABEL: ret_svdoublex4_t
; CHECK:      mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z3.d
; CHECK-NEXT: mov z3.d, z4.d
; CHECK-NEXT: ret
  %tuple = tail call <vscale x 8 x double> @llvm.aarch64.sve.tuple.create4.nxv8f64.nxv2f64(<vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z3, <vscale x 2 x double> %z4)
  ret <vscale x 8 x double> %tuple
}

define void @call_svdoublex4_t(<vscale x 2 x double> %dummy_z0, <vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %dummy_z3, <vscale x 2 x double> %z4, <vscale x 2 x double> %z5) #0 {
; CHECK-LABEL: call_svdoublex4_t
; CHECK:      mov z3.d, z5.d
; CHECK-NEXT: mov z0.d, z1.d
; CHECK-NEXT: mov z1.d, z2.d
; CHECK-NEXT: mov z2.d, z4.d
; CHECK-NEXT: bl callee_svdoublex4_t
  %tuple = tail call <vscale x 8 x double> @llvm.aarch64.sve.tuple.create4.nxv8f64.nxv2f64(<vscale x 2 x double> %z1, <vscale x 2 x double> %z2, <vscale x 2 x double> %z4, <vscale x 2 x double> %z5)
  call void @callee_svdoublex4_t(<vscale x 8 x double> %tuple)
  ret void
}

attributes #0 = { nounwind "target-features"="+sve" }

declare void @callee_svint8x2_t(<vscale x 32 x i8>)
declare void @callee_svint16x2_t(<vscale x 16 x i16>)
declare void @callee_svint32x2_t(<vscale x 8 x i32>)
declare void @callee_svint64x2_t(<vscale x 4 x i64>)
declare void @callee_svfloatx2_t(<vscale x 8 x float>)
declare void @callee_svdoublex2_t(<vscale x 4 x double>)

declare void @callee_svint8x3_t(<vscale x 48 x i8>)
declare void @callee_svint16x3_t(<vscale x 24 x i16>)
declare void @callee_svint32x3_t(<vscale x 12 x i32>)
declare void @callee_svint64x3_t(<vscale x 6 x i64>)
declare void @callee_svfloatx3_t(<vscale x 12 x float>)
declare void @callee_svdoublex3_t(<vscale x 6 x double>)

declare void @callee_svint8x4_t(<vscale x 64 x i8>)
declare void @callee_svint16x4_t(<vscale x 32 x i16>)
declare void @callee_svint32x4_t(<vscale x 16 x i32>)
declare void @callee_svint64x4_t(<vscale x 8 x i64>)
declare void @callee_svfloatx4_t(<vscale x 16 x float>)
declare void @callee_svdoublex4_t(<vscale x 8 x double>)


; x2
declare <vscale x 32 x i8> @llvm.aarch64.sve.tuple.create2.nxv32i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 16 x i16> @llvm.aarch64.sve.tuple.create2.nxv16i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 8 x i32> @llvm.aarch64.sve.tuple.create2.nxv8i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 4 x i64> @llvm.aarch64.sve.tuple.create2.nxv4i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 8 x float> @llvm.aarch64.sve.tuple.create2.nxv8f32.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 4 x double> @llvm.aarch64.sve.tuple.create2.nxv4f64.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>)

; x3
declare <vscale x 48 x i8> @llvm.aarch64.sve.tuple.create3.nxv48i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 24 x i16> @llvm.aarch64.sve.tuple.create3.nxv24i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 12 x i32> @llvm.aarch64.sve.tuple.create3.nxv12i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 6 x i64> @llvm.aarch64.sve.tuple.create3.nxv6i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 12 x float> @llvm.aarch64.sve.tuple.create3.nxv12f32.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 6 x double> @llvm.aarch64.sve.tuple.create3.nxv6f64.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)

; x4
declare <vscale x 64 x i8> @llvm.aarch64.sve.tuple.create4.nxv64i8.nxv16i8(<vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>)
declare <vscale x 32 x i16> @llvm.aarch64.sve.tuple.create4.nxv32i16.nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>, <vscale x 8 x i16>)
declare <vscale x 16 x i32> @llvm.aarch64.sve.tuple.create4.nxv16i32.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>)
declare <vscale x 8 x i64> @llvm.aarch64.sve.tuple.create4.nxv8i64.nxv2i64(<vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>, <vscale x 2 x i64>)
declare <vscale x 16 x float> @llvm.aarch64.sve.tuple.create4.nxv16f32.nxv4f32(<vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>)
declare <vscale x 8 x double> @llvm.aarch64.sve.tuple.create4.nxv8f64.nxv2f64(<vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>, <vscale x 2 x double>)
