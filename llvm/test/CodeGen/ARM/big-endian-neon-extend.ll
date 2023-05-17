; RUN: llc < %s -mtriple armeb-eabi -mattr v7,neon -o - | FileCheck %s

define void @vector_ext_2i8_to_2i64( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_2i8_to_2i64:
; CHECK:      vld1.16   {[[REG:d[0-9]+]][0]}, [r0:16]
; CHECK-NEXT: vrev16.8  [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u8  [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vmovl.u16 [[QREG]], [[REG]]
; CHECK-NEXT: vmovl.u32 [[QREG]], [[REG]]
; CHECK-NEXT: vst1.64   {[[REG]], {{d[0-9]+}}}, [r1]
; CHECK-NEXT: bx        lr
  %1 = load <2 x i8>, ptr %loadaddr
  %2 = zext <2 x i8> %1 to <2 x i64>
  store <2 x i64> %2, ptr %storeaddr
  ret void
}

define void @vector_ext_2i16_to_2i64( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_2i16_to_2i64:
; CHECK:      vld1.32   {[[REG:d[0-9]+]][0]}, [r0:32]
; CHECK-NEXT: vrev32.16 [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u16 [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vmovl.u32 [[QREG]], [[REG]]
; CHECK-NEXT: vst1.64   {[[REG]], {{d[0-9]+}}}, [r1]
; CHECK-NEXT: bx        lr
  %1 = load <2 x i16>, ptr %loadaddr
  %2 = zext <2 x i16> %1 to <2 x i64>
  store <2 x i64> %2, ptr %storeaddr
  ret void
}


define void @vector_ext_2i8_to_2i32( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_2i8_to_2i32:
; CHECK:      vld1.16   {[[REG:d[0-9]+]][0]}, [r0:16]
; CHECK-NEXT: vrev16.8  [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u8  [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vmovl.u16 [[QREG]], [[REG]]
; CHECK-NEXT: vrev64.32 [[REG]], [[REG]]
; CHECK-NEXT: vstr      [[REG]], [r1]
; CHECK-NEXT: bx        lr
  %1 = load <2 x i8>, ptr %loadaddr
  %2 = zext <2 x i8> %1 to <2 x i32>
  store <2 x i32> %2, ptr %storeaddr
  ret void
}

define void @vector_ext_2i16_to_2i32( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_2i16_to_2i32:
; CHECK:      vld1.32   {[[REG:d[0-9]+]][0]}, [r0:32]
; CHECK-NEXT: vrev32.16 [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u16 [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vrev64.32 [[REG]], [[REG]]
; CHECK-NEXT: vstr      [[REG]], [r1]
; CHECK-NEXT: bx        lr
  %1 = load <2 x i16>, ptr %loadaddr
  %2 = zext <2 x i16> %1 to <2 x i32>
  store <2 x i32> %2, ptr %storeaddr
  ret void
}

define void @vector_ext_2i8_to_2i16( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_2i8_to_2i16:
; CHECK:      vld1.16   {[[REG:d[0-9]+]][0]}, [r0:16]
; CHECK-NEXT: vrev16.8  [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u8  [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vmovl.u16 [[QREG]], [[REG]]
; CHECK-NEXT: vrev32.16 [[REG]], [[REG]]
; CHECK-NEXT: vuzp.16   [[REG]], {{d[0-9]+}}
; CHECK-NEXT: vrev32.16 [[REG]], {{d[0-9]+}}
; CHECK-NEXT: vst1.32   {[[REG]][0]}, [r1:32]
; CHECK-NEXT: bx        lr
  %1 = load <2 x i8>, ptr %loadaddr
  %2 = zext <2 x i8> %1 to <2 x i16>
  store <2 x i16> %2, ptr %storeaddr
  ret void
}

define void @vector_ext_4i8_to_4i32( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_4i8_to_4i32:
; CHECK:      vld1.32   {[[REG:d[0-9]+]][0]}, [r0:32]
; CHECK-NEXT: vrev32.8  [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u8  [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vmovl.u16 [[QREG]], [[REG]]
; CHECK-NEXT: vrev64.32 [[QREG]], [[QREG]]
; CHECK-NEXT: vst1.64   {[[REG]], {{d[0-9]+}}}, [r1]
; CHECK-NEXT: bx        lr
  %1 = load <4 x i8>, ptr %loadaddr
  %2 = zext <4 x i8> %1 to <4 x i32>
  store <4 x i32> %2, ptr %storeaddr
  ret void
}

define void @vector_ext_4i8_to_4i16( ptr %loadaddr, ptr %storeaddr ) {
; CHECK-LABEL: vector_ext_4i8_to_4i16:
; CHECK:      vld1.32   {[[REG:d[0-9]+]][0]}, [r0:32]
; CHECK-NEXT: vrev32.8  [[REG]], [[REG]]
; CHECK-NEXT: vmovl.u8  [[QREG:q[0-9]+]], [[REG]]
; CHECK-NEXT: vrev64.16 [[REG]], [[REG]]
; CHECK-NEXT: vstr      [[REG]], [r1]
; CHECK-NEXT: bx        lr
  %1 = load <4 x i8>, ptr %loadaddr
  %2 = zext <4 x i8> %1 to <4 x i16>
  store <4 x i16> %2, ptr %storeaddr
  ret void
}
