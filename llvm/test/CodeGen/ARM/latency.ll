; RUN: llc -mtriple=thumb-none-eabi %s -o - -mcpu=cortex-m0 -asm-print-latency=1 | FileCheck %s --match-full-lines --check-prefix=ON
; RUN: llc -mtriple=thumb-none-eabi %s -o - -mcpu=cortex-m0 -asm-print-latency=0 | FileCheck %s --match-full-lines --check-prefix=OFF
; RUN: llc -mtriple=thumb-none-eabi %s -o - -mcpu=cortex-m0                      | FileCheck %s --match-full-lines --check-prefix=OFF

define i64 @load_i64(ptr %ptr){
; ON:   ldr     r2, [r0]                        @  Latency: 4
; ON:   ldr     r1, [r0, #4]                    @  Latency: 4
; ON:   mov     r0, r2                          @  Latency: 2
; ON:   bx      lr
; OFF:  ldr     r2, [r0]
; OFF:  ldr     r1, [r0, #4]
; OFF:  mov     r0, r2
; OFf:  bx      lr
  %a = load i64, ptr %ptr
  ret i64 %a
}
