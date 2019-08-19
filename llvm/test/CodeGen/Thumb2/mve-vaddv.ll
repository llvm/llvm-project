; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve.fp %s -o - | FileCheck %s

declare i32 @llvm.experimental.vector.reduce.add.i32.v4i32(<4 x i32>)
define arm_aapcs_vfpcc i32 @vaddv_v4i32_i32(<4 x i32> %s1) {
; CHECK-LABEL: vaddv_v4i32_i32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vaddv.u32 r0, q0
; CHECK-NEXT:    bx lr
entry:
  %r = call i32 @llvm.experimental.vector.reduce.add.i32.v4i32(<4 x i32> %s1)
  ret i32 %r
}

declare i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16>)
define arm_aapcs_vfpcc i16 @vaddv_v16i16_i16(<8 x i16> %s1) {
; CHECK-LABEL: vaddv_v16i16_i16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vaddv.u16 r0, q0
; CHECK-NEXT:    bx lr
entry:
  %r = call i16 @llvm.experimental.vector.reduce.add.i16.v8i16(<8 x i16> %s1)
  ret i16 %r
}

declare i8 @llvm.experimental.vector.reduce.add.i8.v16i8(<16 x i8>)
define arm_aapcs_vfpcc i8 @vaddv_v16i8_i8(<16 x i8> %s1) {
; CHECK-LABEL: vaddv_v16i8_i8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vaddv.u8 r0, q0
; CHECK-NEXT:    bx lr
entry:
  %r = call i8 @llvm.experimental.vector.reduce.add.i8.v16i8(<16 x i8> %s1)
  ret i8 %r
}
