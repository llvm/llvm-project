; RUN: llc -mtriple=arm-linux-gnueabihf -filetype=obj <%s | llvm-objdump --triple=armv7 --no-show-raw-insn -d - | FileCheck %s

;; Expect architectural nop to be used between func2 and func3 but not func1
;; and func2 due to lack of subtarget support in func2.

define i32 @func1() #0 align 16 {
entry:
  ret i32 0
}

define i32 @func2() #1 align 16 {
entry:
  ret i32 0
}

define i32 @func3() #0 align 16 {
entry:
  ret i32 0
}

attributes #0 = { "target-cpu"="generic" "target-features"="+armv7-a,+dsp,+neon,+vfp3,-thumb-mode" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "target-cpu"="arm7tdmi" "target-features"="+armv4t" "use-soft-float"="true" }


; CHECK: 00000000 <func1>:
; CHECK-NEXT:  0: mov     r0, #0
; CHECK-NEXT:  4: bx      lr
; CHECK-NEXT:  8: mov     r0, r0
; CHECK-NEXT:  c: mov     r0, r0

; CHECK: 00000010 <func2>:
; CHECK-NEXT: 10: mov     r0, #0
; CHECK-NEXT: 14: bx      lr
; CHECK-NEXT: 18: nop
; CHECK-NEXT: 1c: nop

; CHECK: 00000020 <func3>:
; CHECK-NEXT: 20: mov     r0, #0
; CHECK-NEXT: 24: bx      lr
