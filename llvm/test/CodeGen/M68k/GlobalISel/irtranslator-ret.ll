; RUN: llc -mtriple=m68k -global-isel -stop-after=irtranslator < %s | FileCheck %s


; CHECK: name: noArgRetVoid
; CHECK: RTS
define void @noArgRetVoid() {
  ret void
}

%struct.A = type { i8, float, i32, i32, i32 }

define void @test_arg_lowering1(i8 %x, i8 %y) {
  ; CHECK-LABEL: name: test_arg_lowering1
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD1]](s32)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_TRUNC:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD2]](s32)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering2(i16 %x, i16 %y) {
  ; CHECK-LABEL: name: test_arg_lowering2
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[G_LOAD1]](s32)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_TRUNC:%[0-9]+]]:_(s16) = G_TRUNC [[G_LOAD2]](s32)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering3(i32 %x, i32 %y) {
  ; CHECK-LABEL: name: test_arg_lowering3
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   {{%.*}} G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   {{%.*}} G_LOAD [[G_F_I2]](p0)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_vector(<5 x i8> %x) {
  ; CHECK-LABEL: name: test_arg_lowering_vector
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_F_I3:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD3:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I3]](p0)
  ; CHECK:   [[G_F_I4:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD4:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I4]](p0)
  ; CHECK:   [[G_F_I5:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD5:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I5]](p0)
  ; CHECK:   [[BUILD_VECTOR:%[0-9]+]]:_(<5 x s32>) = G_BUILD_VECTOR [[G_LOAD1]](s32), [[G_LOAD2]](s32), [[G_LOAD3]](s32), [[G_LOAD4]](s32), [[G_LOAD5]](s32)
  ; CHECK:   [[G_TRUNC:%[0-9]+]]:_(<5 x s8>) = G_TRUNC [[BUILD_VECTOR]](<5 x s32>)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_array([5 x i8] %x) {
  ; CHECK-LABEL: name: test_arg_lowering_array
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD1]](s32)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_TRUNC2:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD2]](s32)
  ; CHECK:   [[G_F_I3:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD3:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I3]](p0)
  ; CHECK:   [[G_TRUNC3:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD3]](s32)
  ; CHECK:   [[G_F_I4:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD4:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I4]](p0)
  ; CHECK:   [[G_TRUNC4:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD4]](s32)
  ; CHECK:   [[G_F_I5:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD5:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I5]](p0)
  ; CHECK:   [[G_TRUNC5:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD5]](s32)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_double(double %x) {
  ; CHECK-LABEL: name: test_arg_lowering_double
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_MERGE_VAL:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[G_LOAD1]](s32), [[G_LOAD2]](s32)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_float(float %x) {
  ; CHECK-LABEL: name: test_arg_lowering_float
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_multiple(i1 %a, i8 %b, i16 %c, i32 %d, i64 %e, i128 %f){
  ; CHECK-LABEL: name: test_arg_lowering_multiple
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_TRUNC1:%[0-9]+]]:_(s1) = G_TRUNC [[G_LOAD1]](s32)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_TRUNC2:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD2]](s32)
  ; CHECK:   [[G_F_I3:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD3:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I3]](p0)
  ; CHECK:   [[G_TRUNC3:%[0-9]+]]:_(s16) = G_TRUNC [[G_LOAD3]](s32)
  ; CHECK:   [[G_F_I4:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD4:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I4]](p0)
  ; CHECK:   [[G_F_I5:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD5:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I5]](p0)
  ; CHECK:   [[G_F_I6:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD6:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I6]](p0)
  ; CHECK:   [[G_MERGE_VAL:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[G_LOAD5]](s32), [[G_LOAD6]](s32)
  ; CHECK:   [[G_F_I7:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD7:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I7]](p0)
  ; CHECK:   [[G_F_I8:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD8:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I8]](p0)
  ; CHECK:   [[G_F_I9:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD9:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I9]](p0)
  ; CHECK:   [[G_F_I10:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD10:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I10]](p0)
  ; CHECK:   [[G_MERGE_VAL:%[0-9]+]]:_(s128) = G_MERGE_VALUES [[G_LOAD7]](s32), [[G_LOAD8]](s32), [[G_LOAD9]](s32), [[G_LOAD10]](s32)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_ptr(i32* %x) {
  ; CHECK-LABEL: name: test_arg_lowering_ptr
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(p0) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_float_ptr(float* %x) {
  ; CHECK-LABEL: name: test_arg_lowering_float_ptr
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(p0) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   RTS
  ret void
}

define void @test_arg_lowering_struct(%struct.A %a) #0 {
  ; CHECK-LABEL: name: test_arg_lowering_struct
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_TRUNC1:%[0-9]+]]:_(s8) = G_TRUNC [[G_LOAD1]](s32)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_F_I3:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD3:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I3]](p0)
  ; CHECK:   [[G_F_I4:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD4:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I4]](p0)
  ; CHECK:   [[G_F_I5:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD5:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I5]](p0)
  ; CHECK:   RTS
  ret void
}
