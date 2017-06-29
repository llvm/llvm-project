; RUN: llc < %s -mcpu=atom -mtriple=i686-linux  | FileCheck -check-prefix=ATOM32 %s
; RUN: llc < %s -mcpu=core2 -mtriple=i686-linux | FileCheck -check-prefix=ATOM-NOT32 %s
; RUN: llc < %s -mcpu=atom -mtriple=x86_64-linux  | FileCheck -check-prefix=ATOM64 %s
; RUN: llc < %s -mcpu=core2 -mtriple=x86_64-linux | FileCheck -check-prefix=ATOM-NOT64 %s
; RUN: llc < %s -mcpu=slm -mtriple=i686-linux  | FileCheck -check-prefix=SLM32 %s
; RUN: llc < %s -mcpu=slm -mtriple=x86_64-linux  | FileCheck -check-prefix=SLM64 %s
; RUN: llc < %s -mcpu=goldmont -mtriple=i686-linux  | FileCheck -check-prefix=SLM32 %s
; RUN: llc < %s -mcpu=goldmont -mtriple=x86_64-linux  | FileCheck -check-prefix=SLM64 %s


; fn_ptr.ll
%class.A = type { i32 (...)** }

define i32 @test1() #0 {
  ;ATOM-LABEL: test1:
entry:
  %call = tail call %class.A* @_Z3facv()
  %0 = bitcast %class.A* %call to void (%class.A*)***
  %vtable = load void (%class.A*)**, void (%class.A*)*** %0, align 8
  %1 = load void (%class.A*)*, void (%class.A*)** %vtable, align 8
  ;ATOM32: movl (%ecx), %ecx
  ;ATOM32: calll *%ecx
  ;ATOM-NOT32: calll *(%ecx)
  ;ATOM64: movq (%rcx), %rcx
  ;ATOM64: callq *%rcx
  ;ATOM-NOT64: callq *(%rcx)
  ;SLM32: movl (%ecx), %ecx
  ;SLM32: calll *%ecx
  ;SLM64: movq (%rcx), %rcx
  ;SLM64: callq *%rcx
  tail call void %1(%class.A* %call)
  ret i32 0
}

declare %class.A* @_Z3facv() #1

; virt_fn.ll
@p = external global void (i32)**

define i32 @test2() #0 {
  ;ATOM-LABEL: test2:
entry:
  %0 = load void (i32)**, void (i32)*** @p, align 8
  %1 = load void (i32)*, void (i32)** %0, align 8
  ;ATOM32: movl (%eax), %eax
  ;ATOM32: calll *%eax
  ;ATOM-NOT: calll *(%eax)
  ;ATOM64: movq (%rax), %rax
  ;ATOM64: callq *%rax
  ;ATOM-NOT64: callq *(%rax)
  ;SLM32: movl (%eax), %eax
  ;SLM32: calll *%eax
  ;SLM64: movq (%rax), %rax
  ;SLM64: callq *%rax
  tail call void %1(i32 2)
  ret i32 0
}
