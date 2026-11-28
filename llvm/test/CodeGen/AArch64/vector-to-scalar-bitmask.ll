; RUN: llc -O3 -verify-machineinstrs -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LE
; RUN: llc -O3 -verify-machineinstrs -mtriple=aarch64_be-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-BE
; RUN: llc -O3 -verify-machineinstrs -mtriple=aarch64-linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=256 %s -o - | FileCheck %s
; RUN: llc -O3 -verify-machineinstrs -mtriple=aarch64-linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=512 %s -o - | FileCheck %s

define i16 @convert_to_bitmask16(<16 x i8> %vec) {
  %cmp_result = icmp ne <16 x i8> %vec, zeroinitializer
  %bitmask = bitcast <16 x i1> %cmp_result to i16
  ret i16 %bitmask
}

define i16 @convert_to_bitmask8(<8 x i16> %vec) {
  %cmp_result = icmp ne <8 x i16> %vec, zeroinitializer
  %bitmask = bitcast <8 x i1> %cmp_result to i8
  %extended_bitmask = zext i8 %bitmask to i16
  ret i16 %extended_bitmask
}

; Little endian

; CHECK-LE-LABEL: .LCPI0_0:
; CHECK-LE-NEXT: .byte 1
; CHECK-LE-NEXT: .byte 2
; CHECK-LE-NEXT: .byte 4
; CHECK-LE-NEXT: .byte 8
; CHECK-LE-NEXT: .byte 16
; CHECK-LE-NEXT: .byte 32
; CHECK-LE-NEXT: .byte 64
; CHECK-LE-NEXT: .byte 128
; CHECK-LE-NEXT: .byte 1
; CHECK-LE-NEXT: .byte 2
; CHECK-LE-NEXT: .byte 4
; CHECK-LE-NEXT: .byte 8
; CHECK-LE-NEXT: .byte 16
; CHECK-LE-NEXT: .byte 32
; CHECK-LE-NEXT: .byte 64
; CHECK-LE-NEXT: .byte 128

; CHECK-LE-LABEL: .LCPI1_0:
; CHECK-LE-NEXT: .hword 1
; CHECK-LE-NEXT: .hword 2
; CHECK-LE-NEXT: .hword 4
; CHECK-LE-NEXT: .hword 8
; CHECK-LE-NEXT: .hword 16
; CHECK-LE-NEXT: .hword 32
; CHECK-LE-NEXT: .hword 64
; CHECK-LE-NEXT: .hword 128

; Big endian

; CHECK-BE-LABEL: .LCPI0_0:
; CHECK-BE-NEXT: .byte 128
; CHECK-BE-NEXT: .byte 64
; CHECK-BE-NEXT: .byte 32
; CHECK-BE-NEXT: .byte 16
; CHECK-BE-NEXT: .byte 8
; CHECK-BE-NEXT: .byte 4
; CHECK-BE-NEXT: .byte 2
; CHECK-BE-NEXT: .byte 1
; CHECK-BE-NEXT: .byte 128
; CHECK-BE-NEXT: .byte 64
; CHECK-BE-NEXT: .byte 32
; CHECK-BE-NEXT: .byte 16
; CHECK-BE-NEXT: .byte 8
; CHECK-BE-NEXT: .byte 4
; CHECK-BE-NEXT: .byte 2
; CHECK-BE-NEXT: .byte 1

; CHECK-BE-LABEL: .LCPI1_0:
; CHECK-BE-NEXT: .hword 128
; CHECK-BE-NEXT: .hword 64
; CHECK-BE-NEXT: .hword 32
; CHECK-BE-NEXT: .hword 16
; CHECK-BE-NEXT: .hword 8
; CHECK-BE-NEXT: .hword 4
; CHECK-BE-NEXT: .hword 2
; CHECK-BE-NEXT: .hword 1

define i32 @byte_mask32(<32 x i8> %bytes) {
; CHECK-LABEL: byte_mask32:
; CHECK-COUNT-3: addp
; CHECK-NOT: addp
; CHECK: fmov w0, s{{[0-9]+}}
  %cmp = icmp eq <32 x i8> %bytes, zeroinitializer
  %mask = bitcast <32 x i1> %cmp to i32
  ret i32 %mask
}

define void @store_byte_mask32(<32 x i8> %bytes, ptr %dst) {
; CHECK-LABEL: store_byte_mask32:
; CHECK-COUNT-3: addp
; CHECK-NOT: addp
; CHECK: str s{{[0-9]+}}, [x0]
  %cmp = icmp eq <32 x i8> %bytes, zeroinitializer
  store <32 x i1> %cmp, ptr %dst, align 1
  ret void
}

define i64 @byte_mask64(<64 x i8> %bytes) {
; CHECK-LABEL: byte_mask64:
; CHECK-COUNT-4: addp
; CHECK-NOT: addp
; CHECK: fmov x0, d{{[0-9]+}}
  %cmp = icmp eq <64 x i8> %bytes, zeroinitializer
  %mask = bitcast <64 x i1> %cmp to i64
  ret i64 %mask
}

define void @store_byte_mask64(<64 x i8> %bytes, ptr %dst) {
; CHECK-LABEL: store_byte_mask64:
; CHECK-COUNT-4: addp
; CHECK-NOT: addp
; CHECK: str d{{[0-9]+}}, [x0]
  %cmp = icmp eq <64 x i8> %bytes, zeroinitializer
  store <64 x i1> %cmp, ptr %dst, align 1
  ret void
}

define i64 @mask_v32i16(<32 x i16> %values) {
; CHECK-LABEL: mask_v32i16:
; CHECK-COUNT-3: addp
; CHECK-NOT: addp
; CHECK: fmov w0, s{{[0-9]+}}
  %cmp = icmp ne <32 x i16> %values, zeroinitializer
  %bits = bitcast <32 x i1> %cmp to i32
  %ext = zext i32 %bits to i64
  ret i64 %ext
}

define void @store_mask_v64f32(ptr %src, ptr %dst) {
; CHECK-LABEL: store_mask_v64f32:
; CHECK-COUNT-4: addp
; CHECK-NOT: addp
; CHECK: str d{{[0-9]+}}, [x1]
  %values = load <64 x float>, ptr %src, align 1
  %cmp = fcmp ugt <64 x float> %values, zeroinitializer
  store <64 x i1> %cmp, ptr %dst, align 1
  ret void
}

define i64 @mask_v64i4(ptr %src) {
; CHECK-LABEL: mask_v64i4:
; CHECK-COUNT-4: addp
; CHECK-NOT: addp
; CHECK: fmov x0, d{{[0-9]+}}
  %values = load <64 x i4>, ptr %src, align 1
  %cmp = icmp ne <64 x i4> %values, zeroinitializer
  %bits = bitcast <64 x i1> %cmp to i64
  ret i64 %bits
}
