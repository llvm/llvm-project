; RUN: llc < %s -O2 --code-model=kernel | FileCheck %s
; The intent of the test is that we do not generate conditional
; tail calls to the thunk.

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo(ptr %something) #0 {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    movq (%rdi), %r11
; CHECK-NEXT:    testq %r11, %r11
; Make sure that a JNE was not generated instead of a JE + JMP sequence
; CHECK-NOT:     jne
; CHECK-NEXT:    je .LBB0_1
; CHECK-NEXT:    bb.2: # %if.then
; CHECK-NEXT:    jmp __x86_indirect_thunk_r11
; CHECK-NEXT:    LBB0_1:
; CHECK-NEXT:    retq
entry:
  %0 = load ptr, ptr %something, align 8
  %tobool.not = icmp eq ptr %0, null
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  tail call void %0()
  br label %if.end

if.end:
  ret void
}

attributes #0 = { optsize "target-features"="+retpoline-external-thunk" }
