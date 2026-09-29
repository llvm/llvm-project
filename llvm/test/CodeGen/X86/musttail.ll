; RUN: llc -mtriple=i686-- < %s | FileCheck %s
; RUN: llc -mtriple=i686-- -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-- -disable-tail-calls < %s | FileCheck %s

declare void @t1_callee(ptr)
define void @t1(ptr %a) {
; CHECK-LABEL: t1:
; CHECK: jmp {{_?}}t1_callee
  musttail call void @t1_callee(ptr %a)
  ret void
}

declare ptr @t2_callee()
define ptr @t2() {
; CHECK-LABEL: t2:
; CHECK: jmp {{_?}}t2_callee
  %v = musttail call ptr @t2_callee()
  ret ptr %v
}

; Complex frame layout: stack realignment with dynamic alloca.
define void @t3(i32 %n) alignstack(32) nounwind {
entry:
; CHECK: t3:
; CHECK: pushl %ebp
; CHECK: pushl %esi
; CHECK: andl $-32, %esp
; CHECK: movl %esp, %esi
; CHECK: popl %esi
; CHECK: popl %ebp
; CHECK-NEXT: jmp {{_?}}t3_callee
  %a = alloca i8, i32 %n
  call void @capture(ptr %a)
  musttail call void @t3_callee(i32 %n) nounwind
  ret void
}

declare void @capture(ptr)
declare void @t3_callee(i32)

; Test that we actually copy in and out stack arguments that aren't forwarded
; without modification.
define i32 @t4(ptr %fn, i32 %n, i32 %r) {
; CHECK-LABEL: t4:
; CHECK: incl %[[r:.*]]
; CHECK: decl %[[n:.*]]
; CHECK-DAG: movl %[[r]], {{[0-9]+}}(%esp)
; CHECK-DAG: movl %[[n]], {{[0-9]+}}(%esp)
; CHECK: jmpl *%{{.*}}

entry:
  %r1 = add i32 %r, 1
  %n1 = sub i32 %n, 1
  %r2 = musttail call i32 %fn(ptr %fn, i32 %n1, i32 %r1)
  ret i32 %r2
}

; Combine the complex stack frame with the parameter modification.
define i32 @t5(ptr %fn, i32 %n, i32 %r) alignstack(32) {
; CHECK-LABEL: t5:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: pushl %esi
; 	Align the stack.
; CHECK: andl $-32, %esp
; CHECK: movl %esp, %esi
; 	Modify the args.
; CHECK: incl %[[r:.*]]
; CHECK: decl %[[n:.*]]
; 	Store them through ebp, since that's the only stable arg pointer.
; CHECK-DAG: movl %[[r]], {{[0-9]+}}(%ebp)
; CHECK-DAG: movl %[[n]], {{[0-9]+}}(%ebp)
; 	Epilogue.
; CHECK: leal {{[-0-9]+}}(%ebp), %esp
; CHECK: popl %esi
; CHECK: popl %ebp
; CHECK: jmpl *%{{.*}}

entry:
  %a = alloca i8, i32 %n
  call void @capture(ptr %a)
  %r1 = add i32 %r, 1
  %n1 = sub i32 %n, 1
  %r2 = musttail call i32 %fn(ptr %fn, i32 %n1, i32 %r1)
  ret i32 %r2
}
