; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16-a0:16:16"
target triple = "msp430---elf"

define void @test() #0 {
entry:
; CHECK: test:

; CHECK: call #f_i16
; CHECK: mov r12, &g_i16
  %0 = call i16 @f_i16()
  store volatile i16 %0, i16* @g_i16

; CHECK: call #f_i32
; CHECK: mov r13, &g_i32+2
; CHECK: mov r12, &g_i32
  %1 = call i32 @f_i32()
  store volatile i32 %1, i32* @g_i32

; CHECK: call #f_i64
; CHECK: mov r15, &g_i64+6
; CHECK: mov r14, &g_i64+4
; CHECK: mov r13, &g_i64+2
; CHECK: mov r12, &g_i64
  %2 = call i64 @f_i64()
  store volatile i64 %2, i64* @g_i64

  ret void
}

@g_i16 = common global i16 0, align 2
@g_i32 = common global i32 0, align 2
@g_i64 = common global i64 0, align 2

define i16 @f_i16() #0 {
; CHECK: f_i16:
; CHECK: mov #1, r12
; CHECK: ret
  ret i16 1
}

define i32 @f_i32() #0 {
; CHECK: f_i32:
; CHECK: mov #772, r12
; CHECK: mov #258, r13
; CHECK: ret
  ret i32 16909060
}

define i64 @f_i64() #0 {
; CHECK: f_i64:
; CHECK: mov #1800, r12
; CHECK: mov #1286, r13
; CHECK: mov #772, r14
; CHECK: mov #258, r15
; CHECK: ret
  ret i64 72623859790382856
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
