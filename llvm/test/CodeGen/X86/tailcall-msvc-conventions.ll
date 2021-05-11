; RUN: llc -mtriple=i686-unknown-linux-gnu -O1 < %s | FileCheck %s
; RUN: llc -mtriple=i686-unknown-linux-gnu -O0 < %s | FileCheck %s

; The MSVC family of x86 calling conventions makes tail calls really tricky.
; Tests of all the various combinations should live here.

declare i32 @cdecl_i32()
declare void @cdecl_void()

; Don't allow tail calling these cdecl functions, because we need to clear the
; incoming stack arguments for these argument-clearing conventions.

define x86_thiscallcc void @thiscall_cdecl_notail(i32 %a, i32 %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: thiscall_cdecl_notail
; CHECK: calll cdecl_void
; CHECK: retl $8

define x86_stdcallcc void @stdcall_cdecl_notail(i32 %a, i32 %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: stdcall_cdecl_notail
; CHECK: calll cdecl_void
; CHECK: retl $12

define x86_vectorcallcc void @vectorcall_cdecl_notail(i32 inreg %a, i32 inreg %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: vectorcall_cdecl_notail
; CHECK: calll cdecl_void
; CHECK: retl $4

define x86_fastcallcc void @fastcall_cdecl_notail(i32 inreg %a, i32 inreg %b, i32 %c) {
  tail call void @cdecl_void()
  ret void
}
; CHECK-LABEL: fastcall_cdecl_notail
; CHECK: calll cdecl_void
; CHECK: retl $4


; Tail call to/from callee pop functions can work under the right circumstances:

declare x86_thiscallcc void @no_args_method(i8*)
declare x86_thiscallcc void @one_arg_method(i8*, i32)
declare x86_thiscallcc void @two_args_method(i8*, i32, i32)
declare void @ccall_func()
declare void @ccall_func1(i32)

define x86_thiscallcc void @thiscall_thiscall_tail(i8* %this) {
entry:
  tail call x86_thiscallcc void @no_args_method(i8* %this)
  ret void
}
; CHECK-LABEL: thiscall_thiscall_tail:
; CHECK: jmp no_args_method

define x86_thiscallcc void @thiscall_thiscall_tail2(i8* %this, i32 %a, i32 %b) {
entry:
  tail call x86_thiscallcc void @two_args_method(i8* %this, i32 %a, i32 %b)
  ret void
}
; @two_args_method will take care of popping %a and %b from the stack for us.
; CHECK-LABEL: thiscall_thiscall_tail2:
; CHECK: jmp two_args_method

define x86_thiscallcc void @thiscall_thiscall_notail(i8* %this, i32 %a, i32 %b, i32 %x) {
entry:
  tail call x86_thiscallcc void @two_args_method(i8* %this, i32 %a, i32 %b)
  ret void
}
; @two_args_method would not pop %x.
; CHECK-LABEL: thiscall_thiscall_notail:
; CHECK: calll two_args_method
; CHECK: retl $12

define x86_thiscallcc void @thiscall_thiscall_notail2(i8* %this, i32 %a) {
entry:
  tail call x86_thiscallcc void @no_args_method(i8* %this)
  ret void
}
; @no_args_method would not pop %x for us. Make sure this is checked even
; when there are no arguments to the call.
; CHECK-LABEL: thiscall_thiscall_notail2:
; CHECK: calll no_args_method
; CHECK: retl $4

define void @ccall_thiscall_tail(i8* %x) {
entry:
  tail call x86_thiscallcc void @no_args_method(i8* %x)
  ret void
}
; Tail calling from ccall to thiscall works.
; CHECK-LABEL: ccall_thiscall_tail:
; CHECK: jmp no_args_method

define void @ccall_thiscall_notail(i8* %x, i32 %y) {
entry:
  tail call x86_thiscallcc void @one_arg_method(i8* %x, i32 %y);
  ret void
}
; @one_arg_method would pop %y off the stack.
; CHECK-LABEL: ccall_thiscall_notail:
; CHECK: calll one_arg_method

define x86_thiscallcc void @thiscall_ccall_tail(i8* %this) {
entry:
  tail call void @ccall_func()
  ret void
}
; Tail call from thiscall to ccall works if no arguments need popping.
; CHECK-LABEL: thiscall_ccall_tail:
; CHECK: jmp ccall_func

define x86_thiscallcc void @thiscall_ccall_notail(i8* %this, i32 %x) {
entry:
  tail call void @ccall_func1(i32 %x)
  ret void
}
; No tail call: %x needs to be popped.
; CHECK-LABEL: thiscall_ccall_notail:
; CHECK: calll ccall_func1
; CHECK: retl $4

%S = type { i32 (...)** }
define x86_thiscallcc void @tailcall_through_pointer(%S* %this, i32 %a) {
entry:
  %0 = bitcast %S* %this to void (%S*, i32)***
  %vtable = load void (%S*, i32)**, void (%S*, i32)*** %0
  %1 = load void (%S*, i32)*, void (%S*, i32)** %vtable
  tail call x86_thiscallcc void %1(%S* %this, i32 %a)
  ret void
}
; Tail calling works through function pointers too.
; CHECK-LABEL: tailcall_through_pointer:
; CHECK: jmpl

define x86_stdcallcc void @stdcall_cdecl_tail() {
  tail call void @ccall_func()
  ret void
}
; stdcall to cdecl works if no arguments need popping.
; CHECK-LABEL: stdcall_cdecl_tail
; CHECK: jmp ccall_func

define x86_vectorcallcc void @vectorcall_cdecl_tail(i32 inreg %a, i32 inreg %b) {
  tail call void @ccall_func()
  ret void
}
; vectorcall to cdecl works if no arguments need popping.
; CHECK-LABEL: vectorcall_cdecl_tail
; CHECK: jmp ccall_func

define x86_fastcallcc void @fastcall_cdecl_tail(i32 inreg %a, i32 inreg %b) {
  tail call void @ccall_func()
  ret void
}
; fastcall to cdecl works if no arguments need popping.
; CHECK-LABEL: fastcall_cdecl_tail
; CHECK: jmp ccall_func

define x86_stdcallcc void @stdcall_thiscall_notail(i8* %this, i32 %a, i32 %b) {
  tail call x86_thiscallcc void @two_args_method(i8* %this, i32 %a, i32 %b)
  ret void
}
; two_args_method will not pop %this.
; CHECK-LABEL: stdcall_thiscall_notail
; CHECK: calll two_args_method

define x86_stdcallcc void @stdcall_thiscall_tail(i32 %a, i32 %b) {
  tail call x86_thiscallcc void @two_args_method(i8* null, i32 %a, i32 %b)
  ret void
}
; The callee pop amounts match up.
; CHECK-LABEL: stdcall_thiscall_tail
; CHECK: jmp two_args_method

declare x86_fastcallcc void @fastcall2(i32 inreg %a, i32 inreg %b)
define void @cdecl_fastcall_tail(i32 %a, i32 %b) {
  tail call x86_fastcallcc void @fastcall2(i32 inreg %a, i32 inreg %b)
  ret void
}
; fastcall2 won't pop anything.
; CHECK-LABEL: cdecl_fastcall_tail
; CHECK: jmp fastcall2
