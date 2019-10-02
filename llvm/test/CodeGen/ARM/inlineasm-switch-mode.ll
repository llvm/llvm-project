;RUN: llc -mtriple=thumbv7-linux-gnueabi < %s | llvm-mc -triple=thumbv7-linux-gnueabi -filetype=obj | llvm-objdump -d - | FileCheck %s

define hidden i32 @bah(i8* %start) #0 align 2 {
  %1 = ptrtoint i8* %start to i32
  %2 = tail call i32 asm sideeffect "@ Enter ARM Mode  \0A\09adr r3, 1f \0A\09bx  r3 \0A\09.align 2 \0A\09.code 32 \0A1:  push {r7} \0A\09mov r7, $4 \0A\09svc 0x0 \0A\09pop {r7} \0A\09@ Enter THUMB Mode\0A\09adr r3, 2f+1 \0A\09bx  r3 \0A\09.code 16 \0A2: \0A\09", "={r0},{r0},{r1},{r2},r,~{r3}"(i32 %1, i32 %1, i32 0, i32 983042) #3
  %3 = add i32 %1, 1
  ret i32 %3
}

; CHECK: $a{{.*}}:
; CHECK-NEXT: 04 70 2d e5     str     r7, [sp, #-4]!
; CHECK: $t{{.*}}:
; CHECK-NEXT: 48 1c   adds    r0, r1, #1
