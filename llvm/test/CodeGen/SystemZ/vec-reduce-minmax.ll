; Test vector min/max reductions on SystemZ.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define i8 @f_smin_v16i8(<16 x i8> %v) {
; CHECK-LABEL: f_smin_v16i8:
; CHECK-NOT: vpk
; CHECK-NOT: vrep
; CHECK: vsldb
; CHECK: vmnb
; CHECK: vlgvb %r2, {{%v[0-9]+}}, 0
  %r = call i8 @llvm.vector.reduce.smin.v16i8(<16 x i8> %v)
  ret i8 %r
}

define i8 @f_umax_v16i8(<16 x i8> %v) {
; CHECK-LABEL: f_umax_v16i8:
; CHECK-NOT: vpk
; CHECK-NOT: vrep
; CHECK: vsldb
; CHECK: vmxlb
; CHECK: vlgvb %r2, {{%v[0-9]+}}, 0
  %r = call i8 @llvm.vector.reduce.umax.v16i8(<16 x i8> %v)
  ret i8 %r
}

define i16 @f_smax_v8i16(<8 x i16> %v) {
; CHECK-LABEL: f_smax_v8i16:
; CHECK-NOT: vpk
; CHECK-NOT: vrep
; CHECK: vsldb
; CHECK: vmxh
; CHECK: vlgvh %r2, {{%v[0-9]+}}, 0
  %r = call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %v)
  ret i16 %r
}

define i16 @f_umin_v8i16(<8 x i16> %v) {
; CHECK-LABEL: f_umin_v8i16:
; CHECK-NOT: vpk
; CHECK-NOT: vrep
; CHECK: vsldb
; CHECK: vmnlh
; CHECK: vlgvh %r2, {{%v[0-9]+}}, 0
  %r = call i16 @llvm.vector.reduce.umin.v8i16(<8 x i16> %v)
  ret i16 %r
}

define i32 @f_smin_v4i32(<4 x i32> %v) {
; CHECK-LABEL: f_smin_v4i32:
; CHECK-NOT: vpk
; CHECK-NOT: vrep
; CHECK: vsldb
; CHECK: vmnf
; CHECK: vlgvf %r2, {{%v[0-9]+}}, 0
  %r = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> %v)
  ret i32 %r
}

define i32 @f_umax_v4i32(<4 x i32> %v) {
; CHECK-LABEL: f_umax_v4i32:
; CHECK-NOT: vpk
; CHECK-NOT: vrep
; CHECK: vsldb
; CHECK: vmxlf
; CHECK: vlgvf %r2, {{%v[0-9]+}}, 0
  %r = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> %v)
  ret i32 %r
}

declare i8 @llvm.vector.reduce.smin.v16i8(<16 x i8>)
declare i8 @llvm.vector.reduce.umax.v16i8(<16 x i8>)
declare i16 @llvm.vector.reduce.smax.v8i16(<8 x i16>)
declare i16 @llvm.vector.reduce.umin.v8i16(<8 x i16>)
declare i32 @llvm.vector.reduce.smin.v4i32(<4 x i32>)
declare i32 @llvm.vector.reduce.umax.v4i32(<4 x i32>)
