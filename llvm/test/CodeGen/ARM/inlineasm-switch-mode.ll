;RUN: llc -mtriple=thumbv7-linux-gnueabi < %s | llvm-mc -triple=thumbv7-linux-gnueabi -filetype=obj > %t
; Two pass decoding needed because llvm-objdump does not respect mapping symbols
;RUN: llvm-objdump -triple=armv7   -d %t | FileCheck %s --check-prefix=ARM
;RUN: llvm-objdump -triple=thumbv7 -d %t | FileCheck %s --check-prefix=THUMB

define hidden i32 @bah(i8* %start) #0 align 2 {
  %1 = ptrtoint i8* %start to i32
  %2 = tail call i32 asm sideeffect "@ Enter ARM Mode  \0A\09adr r3, 1f \0A\09bx  r3 \0A\09.align 2 \0A\09.code 32 \0A1:  push {r7} \0A\09mov r7, $4 \0A\09svc 0x0 \0A\09pop {r7} \0A\09@ Enter THUMB Mode\0A\09adr r3, 2f+1 \0A\09bx  r3 \0A\09.code 16 \0A2: \0A\09", "={r0},{r0},{r1},{r2},r,~{r3}"(i32 %1, i32 %1, i32 0, i32 983042) #3
  %3 = add i32 %1, 1
  ret i32 %3
}

; ARM: $a
; ARM-NEXT: 04 70 2d e5     str     r7, [sp, #-4]!
; ARM: $t
; ARM-NEXT: 48 1c

; THUMB: $a
; THUMB-NEXT: 04 70
; THUMB-NEXT: 2d e5
; THUMB: $t
; THUMB-NEXT: 48 1c   adds    r0, r1, #1
