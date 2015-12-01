; RUN: llc < %s -verify-coalescing | FileCheck %s
target triple = "x86_64-pc-win32"

@fnptr = external global void ()*

define void @test1() {
entry:
  %p = load void ()*, void ()** @fnptr
  tail call void %p()
  ret void
}

; CHECK-LABEL: test1{{$}}
; CHECK: .seh_proc test1{{$}}
; CHECK: rex64 jmpq *fnptr(%rip)
; CHECK: .seh_endproc
