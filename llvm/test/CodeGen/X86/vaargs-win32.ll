; RUN: llc -mcpu=generic -mtriple=i686-pc-windows-msvc -mattr=+sse < %s | FileCheck %s --check-prefix=MSVC
; RUN: llc -mcpu=generic -mtriple=i686-pc-mingw32 -mattr=+sse < %s | FileCheck %s --check-prefix=MINGW

@a = external dso_local global <4 x float>, align 16

define dso_local void @testPastArguments() nounwind {
; MSVC-LABEL: testPastArguments:
; MSVC:       # %bb.0: # %entry
; MSVC-NEXT:    subl $20, %esp
; MSVC-NEXT:    movaps _a, %xmm0
; MSVC-NEXT:    movups %xmm0, 4(%esp)
; MSVC-NEXT:    movl $1, (%esp)
; MSVC-NEXT:    calll _testm128
; MSVC-NEXT:    addl $20, %esp
; MSVC-NEXT:    retl
;
; MINGW-LABEL: testPastArguments:
; MINGW:       # %bb.0: # %entry
; MINGW-NEXT:    pushl %ebp
; MINGW-NEXT:    movl %esp, %ebp
; MINGW-NEXT:    andl $-16, %esp
; MINGW-NEXT:    subl $48, %esp
; MINGW-NEXT:    movaps _a, %xmm0
; MINGW-NEXT:    movaps %xmm0, 16(%esp)
; MINGW-NEXT:    movl $1, (%esp)
; MINGW-NEXT:    calll _testm128
; MINGW-NEXT:    movl %ebp, %esp
; MINGW-NEXT:    popl %ebp
; MINGW-NEXT:    retl
entry:
  %0 = load <4 x float>, <4 x float>* @a, align 16
  %call = tail call i32 (i32, ...) @testm128(i32 1, <4 x float> inreg %0)
  ret void
}

declare i32 @testm128(i32, ...) nounwind
