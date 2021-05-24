; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16-a0:16:16"
target triple = "msp430---elf"

define void @test() #0 {
entry:
; CHECK: test:

; CHECK: mov #1, r12
; CHECK: call #f_i16
  call void @f_i16(i16 1)

; CHECK: mov #772, r12
; CHECK: mov #258, r13
; CHECK: call #f_i32
  call void @f_i32(i32 16909060)

; CHECK: mov #1800, r12
; CHECK: mov #1286, r13
; CHECK: mov #772, r14
; CHECK: mov #258, r15
; CHECK: call #f_i64
  call void @f_i64(i64 72623859790382856)

; CHECK: mov #772, r12
; CHECK: mov #258, r13
; CHECK: mov #1800, r14
; CHECK: mov #1286, r15
; CHECK: call #f_i32_i32
  call void @f_i32_i32(i32 16909060, i32 84281096)

; CHECK: mov #1, r12
; CHECK: mov #772, r13
; CHECK: mov #258, r14
; CHECK: mov #2, r15
; CHECK: call #f_i16_i32_i16
  call void @f_i16_i32_i16(i16 1, i32 16909060, i16 2)

; CHECK: mov #1286, 0(r1)
; CHECK: mov #1, r12
; CHECK: mov #772, r13
; CHECK: mov #258, r14
; CHECK: mov #1800, r15
; CHECK: call #f_i16_i32_i32
  call void @f_i16_i32_i32(i16 1, i32 16909060, i32 84281096)

; CHECK: mov #258, 6(r1)
; CHECK: mov #772, 4(r1)
; CHECK: mov #1286, 2(r1)
; CHECK: mov #1800, 0(r1)
; CHECK: mov #1, r12
; CHECK: mov #2, r13
; CHECK: call #f_i16_i64_i16
  call void @f_i16_i64_i16(i16 1, i64 72623859790382856, i16 2)

; Check that r15 is not used and the last i32 argument passed through the stack.
; CHECK: mov	#258, 10(r1)
; CHECK: mov	#772, 8(r1)
; CHECK: mov	#258, 6(r1)
; CHECK: mov	#772, 4(r1)
; CHECK: mov	#1286, 2(r1)
; CHECK: mov	#1800, 0(r1)
; CHECK: mov	#1, r12
; CHECK: mov	#772, r13
; CHECK: mov	#258, r14
  call void @f_i16_i64_i32_i32(i16 1, i64 72623859790382856, i32 16909060, i32 16909060)

; CHECK: mov	#258, 6(r1)
; CHECK: mov	#772, 4(r1)
; CHECK: mov	#1286, 2(r1)
; CHECK: mov	#1800, 0(r1)
; CHECK: mov	#1800, r12
; CHECK: mov	#1286, r13
; CHECK: mov	#772, r14
; CHECK: mov	#258, r15
; CHECK: call	#f_i64_i64
  call void @f_i64_i64(i64 72623859790382856, i64 72623859790382856)

  ret void
}

@g_i16 = common global i16 0, align 2
@g_i32 = common global i32 0, align 2
@g_i64 = common global i64 0, align 2

define void @f_i16(i16 %a) #0 {
; CHECK: f_i16:
; CHECK: mov r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
  ret void
}

define void @f_i32(i32 %a) #0 {
; CHECK: f_i32:
; CHECK: mov r13, &g_i32+2
; CHECK: mov r12, &g_i32
  store volatile i32 %a, i32* @g_i32, align 2
  ret void
}

define void @f_i64(i64 %a) #0 {
; CHECK: f_i64:
; CHECK: mov r15, &g_i64+6
; CHECK: mov r14, &g_i64+4
; CHECK: mov r13, &g_i64+2
; CHECK: mov r12, &g_i64
  store volatile i64 %a, i64* @g_i64, align 2
  ret void
}

define void @f_i32_i32(i32 %a, i32 %b) #0 {
; CHECK: f_i32_i32:
; CHECK: mov r13, &g_i32+2
; CHECK: mov r12, &g_i32
  store volatile i32 %a, i32* @g_i32, align 2
; CHECK: mov r15, &g_i32+2
; CHECK: mov r14, &g_i32
  store volatile i32 %b, i32* @g_i32, align 2
  ret void
}

define void @f_i16_i32_i32(i16 %a, i32 %b, i32 %c) #0 {
; CHECK: f_i16_i32_i32:
; CHECK: mov r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: mov r14, &g_i32+2
; CHECK: mov r13, &g_i32
  store volatile i32 %b, i32* @g_i32, align 2
; CHECK: mov r15, &g_i32
; CHECK: mov 4(r4), &g_i32+2
  store volatile i32 %c, i32* @g_i32, align 2
  ret void
}

define void @f_i16_i32_i16(i16 %a, i32 %b, i16 %c) #0 {
; CHECK: f_i16_i32_i16:
; CHECK: mov r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: mov r14, &g_i32+2
; CHECK: mov r13, &g_i32
  store volatile i32 %b, i32* @g_i32, align 2
; CHECK: mov r15, &g_i16
  store volatile i16 %c, i16* @g_i16, align 2
  ret void
}

define void @f_i16_i64_i16(i16 %a, i64 %b, i16 %c) #0 {
; CHECK: f_i16_i64_i16:
; CHECK: mov r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
;CHECK: mov 10(r4), &g_i64+6
;CHECK: mov 8(r4), &g_i64+4
;CHECK: mov 6(r4), &g_i64+2
;CHECK: mov 4(r4), &g_i64
  store volatile i64 %b, i64* @g_i64, align 2
;CHECK: mov r13, &g_i16
  store volatile i16 %c, i16* @g_i16, align 2
  ret void
}

define void @f_i64_i64(i64 %a, i64 %b) #0 {
; CHECK: f_i64_i64:
; CHECK: mov	r15, &g_i64+6
; CHECK: mov	r14, &g_i64+4
; CHECK: mov	r13, &g_i64+2
; CHECK: mov	r12, &g_i64
  store volatile i64 %a, i64* @g_i64, align 2
; CHECK-DAG: mov	10(r4), &g_i64+6
; CHECK-DAG: mov	8(r4), &g_i64+4
; CHECK-DAG: mov	6(r4), &g_i64+2
; CHECK-DAG: mov	4(r4), &g_i64
  store volatile i64 %b, i64* @g_i64, align 2
  ret void
}

define void @f_i16_i64_i32_i32(i16 %a, i64 %b, i32 %c, i32 %d) #0 {
; CHECK-LABEL: f_i16_i64_i32_i32:
; CHECK: mov	r12, &g_i16
  store volatile i16 %a, i16* @g_i16, align 2
; CHECK: mov	10(r4), &g_i64+6
; CHECK: mov	8(r4), &g_i64+4
; CHECK: mov	6(r4), &g_i64+2
; CHECK: mov	4(r4), &g_i64
  store volatile i64 %b, i64* @g_i64, align 2
; CHECK: mov	r14, &g_i32+2
; CHECK: mov	r13, &g_i32
  store volatile i32 %c, i32* @g_i32, align 2
; CHECK: mov	14(r4), &g_i32+2
; CHECK: mov	12(r4), &g_i32
  store volatile i32 %d, i32* @g_i32, align 2
  ret void
}
; MSP430 EABI p. 6.3
; For helper functions which take two long long arguments
; the first argument is passed in R8::R11 and the second argument
; is in R12::R15.

@g_i64_2 = common global i64 0, align 2

define i64 @helper_call_i64() #0 {
  %1 = load i64, i64* @g_i64, align 2
  %2 = load i64, i64* @g_i64_2, align 2
; CHECK-LABEL: helper_call_i64:
; CHECK: mov	&g_i64, r8
; CHECK: mov	&g_i64+2, r9
; CHECK: mov	&g_i64+4, r10
; CHECK: mov	&g_i64+6, r11
; CHECK: mov	&g_i64_2, r12
; CHECK: mov	&g_i64_2+2, r13
; CHECK: mov	&g_i64_2+4, r14
; CHECK: mov	&g_i64_2+6, r15
; CHECK: call	#__mspabi_divlli
  %3 = sdiv i64 %1, %2
  ret i64 %3
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
