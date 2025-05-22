; RUN: llc < %s -mtriple=bpfel -mcpu=v1 | FileCheck %s

define void @test() #0 {
entry:
; CHECK: test:

; CHECK: call f_i16
; CHECK: *(u16 *)(r1 + 0) = r0
  %0 = call i16 @f_i16()
  store volatile i16 %0, ptr @g_i16

; CHECK: call f_i32
; CHECK: *(u32 *)(r1 + 0) = r0
  %1 = call i32 @f_i32()
  store volatile i32 %1, ptr @g_i32

; CHECK: call f_i64
; CHECK: *(u64 *)(r1 + 0) = r0
  %2 = call i64 @f_i64()
  store volatile i64 %2, ptr @g_i64

  ret void
}

@g_i16 = common global i16 0, align 2
@g_i32 = common global i32 0, align 2
@g_i64 = common global i64 0, align 2

define i16 @f_i16() #0 {
; CHECK: f_i16:
; CHECK: r0 = 1
; CHECK: exit
  ret i16 1
}

define i32 @f_i32() #0 {
; CHECK: f_i32:
; CHECK: r0 = 16909060
; CHECK: exit
  ret i32 16909060
}

define i64 @f_i64() #0 {
; CHECK: f_i64:
; CHECK: r0 = 72623859790382856 ll
; CHECK: exit
  ret i64 72623859790382856
}
