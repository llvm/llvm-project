; RUN: llc < %s -mtriple=i686-linux   -mattr=+sse2 -asm-verbose=false | FileCheck %s -check-prefix=32
; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse2 -asm-verbose=false | FileCheck %s -check-prefix=64

define void @t1(i32 %x) nounwind ssp {
entry:
; 32: t1:
; 32: jmp {{_?}}foo

; 64: t1:
; 64: jmp {{_?}}foo
  tail call void @foo() nounwind
  ret void
}

declare void @foo()

define void @t2() nounwind ssp {
entry:
; 32: t2:
; 32: jmp {{_?}}foo2

; 64: t2:
; 64: jmp {{_?}}foo2
  %0 = tail call i32 @foo2() nounwind
  ret void
}

declare i32 @foo2()

define void @t3() nounwind ssp {
entry:
; 32: t3:
; 32: jmp {{_?}}foo3

; 64: t3:
; 64: jmp {{_?}}foo3
  %0 = tail call i32 @foo3() nounwind
  ret void
}

declare i32 @foo3()

define void @t4(void (i32)* nocapture %x) nounwind ssp {
entry:
; 32: t4:
; 32: calll *
; FIXME: gcc can generate a tailcall for this. But it's tricky.

; 64: t4:
; 64-NOT: call
; 64: jmpq *
  tail call void %x(i32 0) nounwind
  ret void
}

define void @t5(void ()* nocapture %x) nounwind ssp {
entry:
; 32: t5:
; 32-NOT: call
; 32: jmpl *4(%esp)

; 64: t5:
; 64-NOT: call
; 64: jmpq *%rdi
  tail call void %x() nounwind
  ret void
}

define i32 @t6(i32 %x) nounwind ssp {
entry:
; 32: t6:
; 32: calll {{_?}}t6
; 32: jmp {{_?}}bar

; 64: t6:
; 64: jmp {{_?}}t6
; 64: jmp {{_?}}bar
  %0 = icmp slt i32 %x, 10
  br i1 %0, label %bb, label %bb1

bb:
  %1 = add nsw i32 %x, -1
  %2 = tail call i32 @t6(i32 %1) nounwind ssp
  ret i32 %2

bb1:
  %3 = tail call i32 @bar(i32 %x) nounwind
  ret i32 %3
}

declare i32 @bar(i32)

define i32 @t7(i32 %a, i32 %b, i32 %c) nounwind ssp {
entry:
; 32: t7:
; 32: jmp {{_?}}bar2

; 64: t7:
; 64: jmp {{_?}}bar2
  %0 = tail call i32 @bar2(i32 %a, i32 %b, i32 %c) nounwind
  ret i32 %0
}

declare i32 @bar2(i32, i32, i32)

define signext i16 @t8() nounwind ssp {
entry:
; 32: t8:
; 32: calll {{_?}}bar3

; 64: t8:
; 64: callq {{_?}}bar3
  %0 = tail call signext i16 @bar3() nounwind      ; <i16> [#uses=1]
  ret i16 %0
}

declare signext i16 @bar3()

define signext i16 @t9(i32 (i32)* nocapture %x) nounwind ssp {
entry:
; 32: t9:
; 32: calll *

; 64: t9:
; 64: callq *
  %0 = bitcast i32 (i32)* %x to i16 (i32)*
  %1 = tail call signext i16 %0(i32 0) nounwind
  ret i16 %1
}

define void @t10() nounwind ssp {
entry:
; 32: t10:
; 32: calll

; 64: t10:
; 64: callq
  %0 = tail call i32 @foo4() noreturn nounwind
  unreachable
}

declare i32 @foo4()

define i32 @t11(i32 %x, i32 %y, i32 %z.0, i32 %z.1, i32 %z.2) nounwind ssp {
; In 32-bit mode, it's emitting a bunch of dead loads that are not being
; eliminated currently.

; 32: t11:
; 32-NOT: subl ${{[0-9]+}}, %esp
; 32: je
; 32-NOT: movl
; 32-NOT: addl ${{[0-9]+}}, %esp
; 32: jmp {{_?}}foo5

; 64: t11:
; 64-NOT: subq ${{[0-9]+}}, %esp
; 64-NOT: addq ${{[0-9]+}}, %esp
; 64: jmp {{_?}}foo5
entry:
  %0 = icmp eq i32 %x, 0
  br i1 %0, label %bb6, label %bb

bb:
  %1 = tail call i32 @foo5(i32 %x, i32 %y, i32 %z.0, i32 %z.1, i32 %z.2) nounwind
  ret i32 %1

bb6:
  ret i32 0
}

declare i32 @foo5(i32, i32, i32, i32, i32)

%struct.t = type { i32, i32, i32, i32, i32 }

define i32 @t12(i32 %x, i32 %y, %struct.t* byval align 4 %z) nounwind ssp {
; 32: t12:
; 32-NOT: subl ${{[0-9]+}}, %esp
; 32-NOT: addl ${{[0-9]+}}, %esp
; 32: jmp {{_?}}foo6

; 64: t12:
; 64-NOT: subq ${{[0-9]+}}, %esp
; 64-NOT: addq ${{[0-9]+}}, %esp
; 64: jmp {{_?}}foo6
entry:
  %0 = icmp eq i32 %x, 0
  br i1 %0, label %bb2, label %bb

bb:
  %1 = tail call i32 @foo6(i32 %x, i32 %y, %struct.t* byval align 4 %z) nounwind
  ret i32 %1

bb2:
  ret i32 0
}

declare i32 @foo6(i32, i32, %struct.t* byval align 4)

; rdar://r7717598
%struct.ns = type { i32, i32 }
%struct.cp = type { float, float, float, float, float }

define %struct.ns* @t13(%struct.cp* %yy) nounwind ssp {
; 32: t13:
; 32-NOT: jmp
; 32: calll
; 32: ret

; 64: t13:
; 64-NOT: jmp
; 64: callq
; 64: ret
entry:
  %0 = tail call fastcc %struct.ns* @foo7(%struct.cp* byval align 4 %yy, i8 signext 0) nounwind
  ret %struct.ns* %0
}

; rdar://6195379
; llvm can't do sibcall for this in 32-bit mode (yet).
declare fastcc %struct.ns* @foo7(%struct.cp* byval align 4, i8 signext) nounwind ssp

%struct.__block_descriptor = type { i64, i64 }
%struct.__block_descriptor_withcopydispose = type { i64, i64, i8*, i8* }
%struct.__block_literal_1 = type { i8*, i32, i32, i8*, %struct.__block_descriptor* }
%struct.__block_literal_2 = type { i8*, i32, i32, i8*, %struct.__block_descriptor_withcopydispose*, void ()* }

define void @t14(%struct.__block_literal_2* nocapture %.block_descriptor) nounwind ssp {
entry:
; 64: t14:
; 64: movq 32(%rdi)
; 64-NOT: movq 16(%rdi)
; 64: jmpq *16({{%rdi|%rax}})
  %0 = getelementptr inbounds %struct.__block_literal_2* %.block_descriptor, i64 0, i32 5 ; <void ()**> [#uses=1]
  %1 = load void ()** %0, align 8                 ; <void ()*> [#uses=2]
  %2 = bitcast void ()* %1 to %struct.__block_literal_1* ; <%struct.__block_literal_1*> [#uses=1]
  %3 = getelementptr inbounds %struct.__block_literal_1* %2, i64 0, i32 3 ; <i8**> [#uses=1]
  %4 = load i8** %3, align 8                      ; <i8*> [#uses=1]
  %5 = bitcast i8* %4 to void (i8*)*              ; <void (i8*)*> [#uses=1]
  %6 = bitcast void ()* %1 to i8*                 ; <i8*> [#uses=1]
  tail call void %5(i8* %6) nounwind
  ret void
}

; rdar://7726868
%struct.foo = type { [4 x i32] }

define void @t15(%struct.foo* noalias sret %agg.result) nounwind  {
; 32: t15:
; 32: calll {{_?}}f
; 32: ret $4

; 64: t15:
; 64: callq {{_?}}f
; 64: ret
  tail call fastcc void @f(%struct.foo* noalias sret %agg.result) nounwind
  ret void
}

declare void @f(%struct.foo* noalias sret) nounwind

define void @t16() nounwind ssp {
entry:
; 32: t16:
; 32: calll {{_?}}bar4
; 32: fstp

; 64: t16:
; 64: jmp {{_?}}bar4
  %0 = tail call double @bar4() nounwind
  ret void
}

declare double @bar4()

; rdar://6283267
define void @t17() nounwind ssp {
entry:
; 32: t17:
; 32: jmp {{_?}}bar5

; 64: t17:
; 64: xorb %al, %al
; 64: jmp {{_?}}bar5
  tail call void (...)* @bar5() nounwind
  ret void
}

declare void @bar5(...)

; rdar://7774847
define void @t18() nounwind ssp {
entry:
; 32: t18:
; 32: calll {{_?}}bar6
; 32: fstp %st(0)

; 64: t18:
; 64: xorb %al, %al
; 64: jmp {{_?}}bar6
  %0 = tail call double (...)* @bar6() nounwind
  ret void
}

declare double @bar6(...)

define void @t19() alignstack(32) nounwind {
entry:
; CHECK: t19:
; CHECK: andl $-32
; CHECK: calll {{_?}}foo
  tail call void @foo() nounwind
  ret void
}

; If caller / callee calling convention mismatch then check if the return
; values are returned in the same registers.
; rdar://7874780

define double @t20(double %x) nounwind {
entry:
; 32: t20:
; 32: calll {{_?}}foo20
; 32: fldl (%esp)

; 64: t20:
; 64: jmp {{_?}}foo20
  %0 = tail call fastcc double @foo20(double %x) nounwind
  ret double %0
}

declare fastcc double @foo20(double) nounwind
