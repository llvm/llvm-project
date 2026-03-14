; RUN: llc -mtriple=m68k -global-isel -stop-after=legalizer < %s | FileCheck %s

define i32 @test_add(i32 %x, i32 %y) {
  ; CHECK-LABEL: name: test_add
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_ADD1:%[0-9]+]]:_(s32) = G_ADD [[G_LOAD1]], [[G_LOAD2]]
  ; CHECK:   $d0 = COPY [[G_ADD1]](s32)
  ; CHECK:   RTS implicit $d0
  %sum = add i32 %x, %y
  ret i32 %sum
}

define i32 @test_sub(i32 %x, i32 %y) {
  ; CHECK-LABEL: name: test_sub
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_SUB1:%[0-9]+]]:_(s32) = G_SUB [[G_LOAD1]], [[G_LOAD2]]
  ; CHECK:   $d0 = COPY [[G_SUB1]](s32)
  ; CHECK:   RTS implicit $d0
  %diff = sub i32 %x, %y
  ret i32 %diff
}

define i32 @test_mul(i32 %x, i32 %y) {
  ; CHECK-LABEL: name: test_mul
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_MUL1:%[0-9]+]]:_(s32) = G_MUL [[G_LOAD1]], [[G_LOAD2]]
  ; CHECK:   $d0 = COPY [[G_MUL1]](s32)
  ; CHECK:   RTS implicit $d0
  %prod = mul i32 %x, %y
  ret i32 %prod
}

define i32 @test_udiv(i32 %x, i32 %y) {
  ; CHECK-LABEL: name: test_udiv
  ; CHECK: bb.1 (%ir-block.0):
  ; CHECK:   [[G_F_I1:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I1]](p0)
  ; CHECK:   [[G_F_I2:%[0-9]+]]:_(p0) = G_FRAME_INDEX
  ; CHECK:   [[G_LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[G_F_I2]](p0)
  ; CHECK:   [[G_DIV1:%[0-9]+]]:_(s32) = G_UDIV [[G_LOAD1]], [[G_LOAD2]]
  ; CHECK:   $d0 = COPY [[G_DIV1]](s32)
  ; CHECK:   RTS implicit $d0
  %div = udiv i32 %x, %y
  ret i32 %div
}
