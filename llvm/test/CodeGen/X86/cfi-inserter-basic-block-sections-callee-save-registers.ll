;; This test checks if CFI instructions for all callee saved registers are emitted
;; correctly with basic block sections.
; RUN: llc  %s -mtriple=x86_64 -filetype=asm --basicblock-sections=all --frame-pointer=all -o - | FileCheck --check-prefix=SECTIONS_CFI %s

; SECTIONS_CFI:       _Z3foob:
; SECTIONS_CFI:      .cfi_offset %rbp, -16
; SECTIONS_CFI:      .cfi_offset [[RA:%r.+]], -56
; SECTIONS_CFI-NEXT: .cfi_offset [[RB:%r.+]], -48
; SECTIONS_CFI-NEXT: .cfi_offset [[RC:%r.+]], -40
; SECTIONS_CFI-NEXT: .cfi_offset [[RD:%r.+]], -32
; SECTIONS_CFI-NEXT: .cfi_offset [[RE:%r.+]], -24

; SECTIONS_CFI:      _Z3foob.1:
; SECTIONS_CFI:      .cfi_offset %rbp, -16
; SECTIONS_CFI:      .cfi_offset [[RA]], -56
; SECTIONS_CFI-NEXT: .cfi_offset [[RB]], -48
; SECTIONS_CFI-NEXT: .cfi_offset [[RC]], -40
; SECTIONS_CFI-NEXT: .cfi_offset [[RD]], -32
; SECTIONS_CFI-NEXT: .cfi_offset [[RE]], -24

; SECTIONS_CFI:      _Z3foob.2:
; SECTIONS_CFI:      .cfi_offset %rbp, -16
; SECTIONS_CFI:      .cfi_offset [[RA]], -56
; SECTIONS_CFI-NEXT: .cfi_offset [[RB]], -48
; SECTIONS_CFI-NEXT: .cfi_offset [[RC]], -40
; SECTIONS_CFI-NEXT: .cfi_offset [[RD]], -32
; SECTIONS_CFI-NEXT: .cfi_offset [[RE]], -24


;; void foo(bool b) {
;;   if (b) // adds a basic block
;;     // clobber all callee-save registers to force them to be callee-saved and to
;;     // be described by cfi_offset directives.
;;     asm("nop" ::: "r12", "r13", "r14", "r15", "rbx");
;; }

define dso_local void @_Z3foob(i1 zeroext %b) {
entry:
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  %0 = load i8, i8* %b.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void asm sideeffect "nop", "~{r12},~{r13},~{r14},~{r15},~{rbx},~{dirflag},~{fpsr},~{flags}"() #1, !srcloc !2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
!2 = !{i32 38}
