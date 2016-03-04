; RUN: llc -mattr=+cx16 -x86-use-base-pointer=true -stackrealign -stack-alignment=32  %s -o - | FileCheck --check-prefix=CHECK --check-prefix=USE_BASE %s
; RUN: llc -mattr=+cx16 -x86-use-base-pointer=false -stackrealign -stack-alignment=32  %s -o - | FileCheck --check-prefix=CHECK --check-prefix=DONT_USE_BASE %s
target triple = "x86_64-apple-macosx"

; This function uses dynamic allocated stack to force the use
; of a frame pointer.
; The inline asm clobbers a bunch of registers to make sure
; the frame pointer will need to be used (for spilling in that case).
;
; Then, we check that when we use rbx as the base pointer,
; we do not use cmpxchg, since using that instruction requires
; to clobbers rbx to set the arguments of the instruction and when
; rbx is used as the base pointer, RA cannot fix the code for us.
;
; CHECK-LABEL: cmp_and_swap16:
; Check that we actually use rbx.
; USE_BASE: movq %rsp, %rbx
; USE_BASE-NOT: cmpxchg
;
; DONT_USE_BASE-NOT: movq %rsp, %rbx
; DONT_USE_BASE: cmpxchg
define i1 @cmp_and_swap16(i128 %a, i128 %b, i128* %addr, i32 %n) {
  %dummy = alloca i32, i32 %n
tail call void asm sideeffect "nop", "~{rax},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15}"()
  %cmp = cmpxchg i128* %addr, i128 %a, i128 %b seq_cst seq_cst
  %res = extractvalue { i128, i1 } %cmp, 1
  %idx = getelementptr i32, i32* %dummy, i32 5
  store i32 %n, i32* %idx
  ret i1 %res
}
