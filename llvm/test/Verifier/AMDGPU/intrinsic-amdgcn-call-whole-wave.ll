; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define amdgpu_cs void @indirect(ptr %fn, i32 %x) {
  ; CHECK: Indirect whole wave calls are not allowed
  %whatever = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr %fn, i32 %x)
  ret void
}

declare amdgpu_gfx_whole_wave void @variadic_callee(i1 %active, i32 %x, ...)

define amdgpu_cs void @variadic(ptr %fn, i32 %x) {
  ; CHECK: Variadic whole wave calls are not allowed
  %whatever = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @variadic_callee, i32 %x)
  ret void
}

declare amdgpu_gfx void @bad_cc_callee(i1 %active, i32 %x)

define amdgpu_cs void @bad_cc(i32 %x) {
  ; CHECK: Callee must have the amdgpu_gfx_whole_wave calling convention
  %whatever = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @bad_cc_callee, i32 %x)
  ret void
}

declare amdgpu_gfx_whole_wave i32 @no_i1_callee(i32 %active, i32 %y, i32 %z)

define amdgpu_cs void @no_i1(i32 %x) {
  ; CHECK: Callee must have i1 as its first argument
  %whatever = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @no_i1_callee, i32 %x, i32 0)
  ret void
}

declare amdgpu_gfx_whole_wave i32 @good_callee(i1 %active, i32 %x, i32 inreg %y)

define amdgpu_cs void @bad_args(i32 %x) {
  ; CHECK: Call argument count must match callee argument count
  %whatever.0 = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @good_callee, i32 %x)

  ; CHECK: Argument types must match
  %whatever.1 = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @good_callee, i32 %x, i64 inreg 0)

  ; CHECK: Argument inreg attributes must match
  %whatever.2 = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @good_callee, i32 %x, i32 0)

  ret void
}
