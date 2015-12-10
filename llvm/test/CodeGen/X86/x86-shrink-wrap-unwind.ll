; RUN: llc %s -o - | FileCheck %s --check-prefix=CHECK
;
; This test checks that we do not use shrink-wrapping when
; the function does not have any frame pointer and may unwind.
; This is a workaround for a limitation in the emission of
; the CFI directives, that are not correct in such case.
; PR25614
;
; Note: This test cannot be merged with the shrink-wrapping tests
; because the booleans set on the command line take precedence on
; the target logic that disable shrink-wrapping.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-apple-macosx"


; No shrink-wrapping should occur here, until the CFI information are fixed.
; CHECK-LABEL: framelessUnwind:
;
; Prologue code.
; (What we push does not matter. It should be some random sratch register.)
; CHECK: pushq
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; CHECK: movl %edi, [[ARG0CPY:%e[a-z]+]]
; CHECK-NEXT: cmpl %esi, [[ARG0CPY]]
; CHECK-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], 4(%rsp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq 4(%rsp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; CHECK: [[EXIT_LABEL]]:
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; CHECK-NEXT: popq
;
; CHECK-NEXT: retq
define i32 @framelessUnwind(i32 %a, i32 %b) #0 {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

declare i32 @doSomething(i32, i32*)

attributes #0 = { "no-frame-pointer-elim"="false" }

; Shrink-wrapping should occur here. We have a frame pointer.
; CHECK-LABEL: frameUnwind:
;
; Compare the arguments and jump to exit.
; No prologue needed.
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; CHECK: movl %edi, [[ARG0CPY:%e[a-z]+]]
; CHECK-NEXT: cmpl %esi, [[ARG0CPY]]
; CHECK-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], -4(%rbp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq -4(%rbp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; Epilogue code. (What we pop does not matter.)
; CHECK: popq %rbp
;
; CHECK: [[EXIT_LABEL]]:
; CHECK-NEXT: retq
define i32 @frameUnwind(i32 %a, i32 %b) #1 {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

attributes #1 = { "no-frame-pointer-elim"="true" }

; Shrink-wrapping should occur here. We do not have to unwind.
; CHECK-LABEL: framelessnoUnwind:
;
; Compare the arguments and jump to exit.
; No prologue needed.
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; CHECK: movl %edi, [[ARG0CPY:%e[a-z]+]]
; CHECK-NEXT: cmpl %esi, [[ARG0CPY]]
; CHECK-NEXT: jge [[EXIT_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; (What we push does not matter. It should be some random sratch register.)
; CHECK: pushq
;
; Store %a in the alloca.
; CHECK: movl [[ARG0CPY]], 4(%rsp)
; Set the alloca address in the second argument.
; CHECK-NEXT: leaq 4(%rsp), %rsi
; Set the first argument to zero.
; CHECK-NEXT: xorl %edi, %edi
; CHECK-NEXT: callq _doSomething
;
; Epilogue code.
; CHECK-NEXT: addq
;
; CHECK: [[EXIT_LABEL]]:
; CHECK-NEXT: retq
define i32 @framelessnoUnwind(i32 %a, i32 %b) #2 {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

attributes #2 = { "no-frame-pointer-elim"="false" nounwind }
