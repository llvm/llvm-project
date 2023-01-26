;RUN:  llc -mtriple=armv7-linux-gnueabi < %s | llvm-mc -triple=armv7-linux-gnueabi -filetype=obj | llvm-objdump --triple=armv7 -d - | FileCheck %s
;RUN:  llc -mtriple=armv7-linux-gnueabi < %s | FileCheck %s -check-prefix=ASM
;RUN:  llc -mtriple=armv7-apple-darwin < %s | FileCheck %s -check-prefix=ASM

define hidden i32 @bah(ptr %start) #0 align 2 {
  %1 = ptrtoint ptr %start to i32
  %2 = tail call i32 asm sideeffect "@ Enter THUMB Mode\0A\09adr r3, 2f+1 \0A\09bx  r3 \0A\09.code 16 \0A2: push {r7} \0A\09mov r7, $4 \0A\09svc 0x0 \0A\09pop {r7} \0A\09", "={r0},{r0},{r1},{r2},r,~{r3}"(i32 %1, i32 %1, i32 0, i32 983042) #3
  %3 = add i32 %1, 1
  ret i32 %3
}
; CHECK: $t
; CHECK: $a
; CHECK: e2810001        add     r0, r1, #1

; .code 32 is implicit
; ASM-LABEL: bah:
; ASM: .code 16
; ASM: .code 32
