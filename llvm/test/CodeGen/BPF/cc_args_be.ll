; RUN: not llc < %s -march=bpfeb -show-mc-encoding | FileCheck %s
; test big endian

define void @test() #0 {
entry:
; CHECK: test:

; CHECK: r1 = 123 # encoding: [0xb7,0x10,0x00,0x00,0x00,0x00,0x00,0x7b]
; CHECK: call f_i16
  call void @f_i16(i16 123)

; CHECK: r1 = 12345678 # encoding: [0xb7,0x10,0x00,0x00,0x00,0xbc,0x61,0x4e]
; CHECK: call f_i32
  call void @f_i32(i32 12345678)

; CHECK: r1 = 72623859790382856 ll # encoding: [0x18,0x10,0x00,0x00,0x05,0x06,0x07,0x08,0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04]
; CHECK: call f_i64
  call void @f_i64(i64 72623859790382856)

; CHECK: r1 = 1234
; CHECK: r2 = 5678
; CHECK: call f_i32_i32
  call void @f_i32_i32(i32 1234, i32 5678)

; CHECK: r1 = 2
; CHECK: r2 = 3
; CHECK: r3 = 4
; CHECK: call f_i16_i32_i16
  call void @f_i16_i32_i16(i16 2, i32 3, i16 4)

; CHECK: r1 = 5
; CHECK: r2 = 7262385979038285 ll
; CHECK: r3 = 6
; CHECK: call f_i16_i64_i16
  call void @f_i16_i64_i16(i16 5, i64 7262385979038285, i16 6)

  ret void
}

@g_i16 = common global i16 0, align 2
@g_i32 = common global i32 0, align 2
@g_i64 = common global i64 0, align 4

define void @f_i16(i16 %a) #0 {
; CHECK: f_i16:
; CHECK: *(u16 *)(r2 + 0) = r1 # encoding: [0x6b,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  store volatile i16 %a, i16* @g_i16, align 2
  ret void
}

define void @f_i32(i32 %a) #0 {
; CHECK: f_i32:
; CHECK: *(u16 *)(r2 + 2) = r1 # encoding: [0x6b,0x21,0x00,0x02,0x00,0x00,0x00,0x00]
; CHECK: *(u16 *)(r2 + 0) = r1 # encoding: [0x6b,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
  store volatile i32 %a, i32* @g_i32, align 2
  ret void
}

define void @f_i64(i64 %a) #0 {
; CHECK: f_i64:
; CHECK: *(u32 *)(r2 + 4) = r1 # encoding: [0x63,0x21,0x00,0x04,0x00,0x00,0x00,0x00]
; CHECK: *(u32 *)(r2 + 0) = r1
  store volatile i64 %a, i64* @g_i64, align 2
  ret void
}

define void @f_i32_i32(i32 %a, i32 %b) #0 {
; CHECK: f_i32_i32:
; CHECK: *(u32 *)(r3 + 0) = r1
  store volatile i32 %a, i32* @g_i32, align 4
; CHECK: *(u32 *)(r3 + 0) = r2
  store volatile i32 %b, i32* @g_i32, align 4
  ret void
}

define void @f_i16_i32_i16(i16 %a, i32 %b, i16 %c) #0 {
; CHECK: f_i16_i32_i16:
; CHECK: *(u16 *)(r4 + 0) = r1
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: *(u32 *)(r1 + 0) = r2
  store volatile i32 %b, i32* @g_i32, align 4
; CHECK: *(u16 *)(r4 + 0) = r3
  store volatile i16 %c, i16* @g_i16, align 2
  ret void
}

define void @f_i16_i64_i16(i16 %a, i64 %b, i16 %c) #0 {
; CHECK: f_i16_i64_i16:
; CHECK: *(u16 *)(r4 + 0) = r1
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: *(u64 *)(r1 + 0) = r2 # encoding: [0x7b,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
  store volatile i64 %b, i64* @g_i64, align 8
; CHECK: *(u16 *)(r4 + 0) = r3
  store volatile i16 %c, i16* @g_i16, align 2
  ret void
}
