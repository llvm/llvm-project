; RUN: llc -mtriple=i686-pc-windows -no-x86-call-frame-opt < %s | FileCheck %s

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

declare x86_thiscallcc void @thiscall_thunk(ptr %this, ...)
define i32 @call_varargs_thiscall_thunk(ptr %a, i32 %b, i32 %c, i32 %d) {
  call x86_thiscallcc void (ptr, ...) @thiscall_thunk(ptr %a, i32 1, i32 2)
  call x86_thiscallcc void (ptr, ...) @thiscall_thunk(ptr %a, i32 1, i32 2)
  %t1 = add i32 %b, %c
  %r = add i32 %t1, %d
  ret i32 %r
}

; CHECK: _call_varargs_thiscall_thunk:
; CHECK: calll _thiscall_thunk
; CHECK-NEXT: subl $8, %esp

; We don't mangle the argument size into variadic callee cleanup functions.

declare x86_stdcallcc void @stdcall_thunk(ptr %this, ...)
define i32 @call_varargs_stdcall_thunk(ptr %a, i32 %b, i32 %c, i32 %d) {
  call x86_stdcallcc void (ptr, ...) @stdcall_thunk(ptr %a, i32 1, i32 2)
  call x86_stdcallcc void (ptr, ...) @stdcall_thunk(ptr %a, i32 1, i32 2)
  %t1 = add i32 %b, %c
  %r = add i32 %t1, %d
  ret i32 %r
}

; CHECK: _call_varargs_stdcall_thunk:
; CHECK: calll _stdcall_thunk{{$}}
; CHECK-NEXT: subl $12, %esp

declare x86_fastcallcc void @fastcall_thunk(ptr %this, ...)
define i32 @call_varargs_fastcall_thunk(ptr %a, i32 %b, i32 %c, i32 %d) {
  call x86_fastcallcc void (ptr, ...) @fastcall_thunk(ptr inreg %a, i32 inreg 1, i32 2)
  call x86_fastcallcc void (ptr, ...) @fastcall_thunk(ptr inreg %a, i32 inreg 1, i32 2)
  %t1 = add i32 %b, %c
  %r = add i32 %t1, %d
  ret i32 %r
}

; CHECK: _call_varargs_fastcall_thunk:
; CHECK: calll @fastcall_thunk{{$}}
; CHECK-NEXT: subl $4, %esp

; If you actually return from such a thunk, it will only pop the non-variadic
; portion of the arguments, which is different from what the callee passes.

define x86_stdcallcc void @varargs_stdcall_return(i32, i32, ...) {
  ret void
}

; CHECK: _varargs_stdcall_return:
; CHECK: retl $8
