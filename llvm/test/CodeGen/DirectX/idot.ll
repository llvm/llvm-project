; RUN: opt -S  -dxil-intrinsic-expansion -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for dot are generated for int/uint vectors.

; CHECK-LABEL: dot_int16_t2
define noundef i16 @dot_int16_t2(<2 x i16> noundef %a, <2 x i16> noundef %b) {
entry:
; CHECK: extractelement <2 x i16> %a, i64 0
; CHECK: extractelement <2 x i16> %b, i64 0
; CHECK: mul i16 %{{.*}}, %{{.*}}
; CHECK: extractelement <2 x i16> %a, i64 1
; CHECK: extractelement <2 x i16> %b, i64 1
; EXPCHECK: call i16 @llvm.dx.imad.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
; DOPCHECK: call i16 @dx.op.tertiary.i16(i32 48, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  %dot = call i16 @llvm.dx.sdot.v3i16(<2 x i16> %a, <2 x i16> %b)
  ret i16 %dot
}

; CHECK-LABEL: dot_int4
define noundef i32 @dot_int4(<4 x i32> noundef %a, <4 x i32> noundef %b) {
entry:
; CHECK: extractelement <4 x i32> %a, i64 0
; CHECK: extractelement <4 x i32> %b, i64 0
; CHECK: mul i32 %{{.*}}, %{{.*}}
; CHECK: extractelement <4 x i32> %a, i64 1
; CHECK: extractelement <4 x i32> %b, i64 1
; EXPCHECK: call i32 @llvm.dx.imad.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; CHECK: extractelement <4 x i32> %a, i64 2
; CHECK: extractelement <4 x i32> %b, i64 2
; EXPCHECK: call i32 @llvm.dx.imad.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; CHECK: extractelement <4 x i32> %a, i64 3
; CHECK: extractelement <4 x i32> %b, i64 3
; EXPCHECK: call i32 @llvm.dx.imad.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.tertiary.i32(i32 48, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %dot = call i32 @llvm.dx.sdot.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret i32 %dot
}

; CHECK-LABEL: dot_uint16_t3
define noundef i16 @dot_uint16_t3(<3 x i16> noundef %a, <3 x i16> noundef %b) {
entry:
; CHECK: extractelement <3 x i16> %a, i64 0
; CHECK: extractelement <3 x i16> %b, i64 0
; CHECK: mul i16 %{{.*}}, %{{.*}}
; CHECK: extractelement <3 x i16> %a, i64 1
; CHECK: extractelement <3 x i16> %b, i64 1
; EXPCHECK: call i16 @llvm.dx.umad.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
; DOPCHECK: call i16 @dx.op.tertiary.i16(i32 49, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
; CHECK: extractelement <3 x i16> %a, i64 2
; CHECK: extractelement <3 x i16> %b, i64 2
; EXPCHECK: call i16 @llvm.dx.umad.i16(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
; DOPCHECK: call i16 @dx.op.tertiary.i16(i32 49, i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}})
  %dot = call i16 @llvm.dx.udot.v3i16(<3 x i16> %a, <3 x i16> %b)
  ret i16 %dot
}

; CHECK-LABEL: dot_uint4
define noundef i32 @dot_uint4(<4 x i32> noundef %a, <4 x i32> noundef %b) {
entry:
; CHECK: extractelement <4 x i32> %a, i64 0
; CHECK: extractelement <4 x i32> %b, i64 0
; CHECK: mul i32 %{{.*}}, %{{.*}}
; CHECK: extractelement <4 x i32> %a, i64 1
; CHECK: extractelement <4 x i32> %b, i64 1
; EXPCHECK: call i32 @llvm.dx.umad.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.tertiary.i32(i32 49, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; CHECK: extractelement <4 x i32> %a, i64 2
; CHECK: extractelement <4 x i32> %b, i64 2
; EXPCHECK: call i32 @llvm.dx.umad.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.tertiary.i32(i32 49, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; CHECK: extractelement <4 x i32> %a, i64 3
; CHECK: extractelement <4 x i32> %b, i64 3
; EXPCHECK: call i32 @llvm.dx.umad.i32(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.tertiary.i32(i32 49, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %dot = call i32 @llvm.dx.udot.v4i32(<4 x i32> %a, <4 x i32> %b)
  ret i32 %dot
}

; CHECK-LABEL: dot_uint64_t4
define noundef i64 @dot_uint64_t4(<2 x i64> noundef %a, <2 x i64> noundef %b) {
entry:
; CHECK: extractelement <2 x i64> %a, i64 0
; CHECK: extractelement <2 x i64> %b, i64 0
; CHECK: mul i64 %{{.*}}, %{{.*}}
; CHECK: extractelement <2 x i64> %a, i64 1
; CHECK: extractelement <2 x i64> %b, i64 1
; EXPCHECK: call i64 @llvm.dx.umad.i64(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
; DOPCHECK: call i64 @dx.op.tertiary.i64(i32 49, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  %dot = call i64 @llvm.dx.udot.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret i64 %dot
}

declare i16 @llvm.dx.sdot.v2i16(<2 x i16>, <2 x i16>)
declare i32 @llvm.dx.sdot.v4i32(<4 x i32>, <4 x i32>)
declare i16 @llvm.dx.udot.v3i32(<3 x i16>, <3 x i16>)
declare i32 @llvm.dx.udot.v4i32(<4 x i32>, <4 x i32>)
declare i64 @llvm.dx.udot.v2i64(<2 x i64>, <2 x i64>)
