; RUN: llc < %s | FileCheck %s
; PR2936

target triple = "i386-pc-mingw32"

define dllexport x86_fastcallcc i32 @foo() nounwind  {
entry:
	ret i32 0
}

; CHECK: .section .drectve
; CHECK: -export:@foo@0
