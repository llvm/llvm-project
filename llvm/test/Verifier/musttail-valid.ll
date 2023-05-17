; RUN: llvm-as %s -o /dev/null

; Should assemble without error.

declare void @similar_param_ptrty_callee(ptr)
define void @similar_param_ptrty(ptr) {
  musttail call void @similar_param_ptrty_callee(ptr null)
  ret void
}

declare ptr @similar_ret_ptrty_callee()
define ptr @similar_ret_ptrty() {
  %v = musttail call ptr @similar_ret_ptrty_callee()
  ret ptr %v
}

declare x86_thiscallcc void @varargs_thiscall(ptr, ...)
define x86_thiscallcc void @varargs_thiscall_thunk(ptr %this, ...) {
  musttail call x86_thiscallcc void (ptr, ...) @varargs_thiscall(ptr %this, ...)
  ret void
}

declare x86_fastcallcc void @varargs_fastcall(ptr, ...)
define x86_fastcallcc void @varargs_fastcall_thunk(ptr %this, ...) {
  musttail call x86_fastcallcc void (ptr, ...) @varargs_fastcall(ptr %this, ...)
  ret void
}

define x86_thiscallcc void @varargs_thiscall_unreachable(ptr %this, ...) {
  unreachable
}

define x86_thiscallcc void @varargs_thiscall_ret_unreachable(ptr %this, ...) {
  musttail call x86_thiscallcc void (ptr, ...) @varargs_thiscall(ptr %this, ...)
  ret void
bb1:
  ret void
}
