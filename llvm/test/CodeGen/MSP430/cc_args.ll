; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16-a0:16:16"
target triple = "msp430---elf"

define void @test() #0 {
entry:
; CHECK: test:

; CHECK: mov.w #1, r12
; CHECK: call #f_i16
  call void @f_i16(i16 1)

; CHECK: mov.w #772, r12
; CHECK: mov.w #258, r13
; CHECK: call #f_i32
  call void @f_i32(i32 16909060)

; CHECK: mov.w #1800, r12
; CHECK: mov.w #1286, r13
; CHECK: mov.w #772, r14
; CHECK: mov.w #258, r15
; CHECK: call #f_i64
  call void @f_i64(i64 72623859790382856)

; CHECK: mov.w #772, r12
; CHECK: mov.w #258, r13
; CHECK: mov.w #1800, r14
; CHECK: mov.w #1286, r15
; CHECK: call #f_i32_i32
  call void @f_i32_i32(i32 16909060, i32 84281096)

; CHECK: mov.w #1, r12
; CHECK: mov.w #772, r13
; CHECK: mov.w #258, r14
; CHECK: mov.w #2, r15
; CHECK: call #f_i16_i32_i16
  call void @f_i16_i32_i16(i16 1, i32 16909060, i16 2)

; CHECK: mov.w #1286, 0(r1)
; CHECK: mov.w #1, r12
; CHECK: mov.w #772, r13
; CHECK: mov.w #258, r14
; CHECK: mov.w #1800, r15
; CHECK: call #f_i16_i32_i32
  call void @f_i16_i32_i32(i16 1, i32 16909060, i32 84281096)

; CHECK: mov.w #258, 6(r1)
; CHECK: mov.w #772, 4(r1)
; CHECK: mov.w #1286, 2(r1)
; CHECK: mov.w #1800, 0(r1)
; CHECK: mov.w #1, r12
; CHECK: mov.w #2, r13
; CHECK: call #f_i16_i64_i16
  call void @f_i16_i64_i16(i16 1, i64 72623859790382856, i16 2)

  ret void
}

@g_i16 = common global i16 0, align 2
@g_i32 = common global i32 0, align 2
@g_i64 = common global i64 0, align 2

define void @f_i16(i16 %a) #0 {
; CHECK: f_i16:
; CHECK: mov.w r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
  ret void
}

define void @f_i32(i32 %a) #0 {
; CHECK: f_i32:
; CHECK: mov.w r13, &g_i32+2
; CHECK: mov.w r12, &g_i32
  store volatile i32 %a, i32* @g_i32, align 2
  ret void
}

define void @f_i64(i64 %a) #0 {
; CHECK: f_i64:
; CHECK: mov.w r15, &g_i64+6
; CHECK: mov.w r14, &g_i64+4
; CHECK: mov.w r13, &g_i64+2
; CHECK: mov.w r12, &g_i64
  store volatile i64 %a, i64* @g_i64, align 2
  ret void
}

define void @f_i32_i32(i32 %a, i32 %b) #0 {
; CHECK: f_i32_i32:
; CHECK: mov.w r13, &g_i32+2
; CHECK: mov.w r12, &g_i32
  store volatile i32 %a, i32* @g_i32, align 2
; CHECK: mov.w r15, &g_i32+2
; CHECK: mov.w r14, &g_i32
  store volatile i32 %b, i32* @g_i32, align 2
  ret void
}

define void @f_i16_i32_i32(i16 %a, i32 %b, i32 %c) #0 {
; CHECK: f_i16_i32_i32:
; CHECK: mov.w r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: mov.w r14, &g_i32+2
; CHECK: mov.w r13, &g_i32
  store volatile i32 %b, i32* @g_i32, align 2
; CHECK: mov.w r15, &g_i32
; CHECK: mov.w 4(r4), &g_i32+2
  store volatile i32 %c, i32* @g_i32, align 2
  ret void
}

define void @f_i16_i32_i16(i16 %a, i32 %b, i16 %c) #0 {
; CHECK: f_i16_i32_i16:
; CHECK: mov.w r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: mov.w r14, &g_i32+2
; CHECK: mov.w r13, &g_i32
  store volatile i32 %b, i32* @g_i32, align 2
; CHECK: mov.w r15, &g_i16
  store volatile i16 %c, i16* @g_i16, align 2
  ret void
}

define void @f_i16_i64_i16(i16 %a, i64 %b, i16 %c) #0 {
; CHECK: f_i16_i64_i16:
; CHECK: mov.w r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
;CHECK: mov.w 10(r4), &g_i64+6
;CHECK: mov.w 8(r4), &g_i64+4
;CHECK: mov.w 6(r4), &g_i64+2
;CHECK: mov.w 4(r4), &g_i64
  store volatile i64 %b, i64* @g_i64, align 2
;CHECK: mov.w r13, &g_i16
  store volatile i16 %c, i16* @g_i16, align 2
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
