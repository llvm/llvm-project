; RUN: opt  -passes='inline,sroa' -max-devirt-iterations=1 -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.3"

declare ptr @f1(ptr) ssp align 2

define linkonce_odr void @f2(ptr %t) inlinehint ssp {
entry:
  unreachable
}

define linkonce_odr void @f3(ptr %__f) ssp {
entry:
  %__f_addr = alloca ptr, align 8
  store ptr %__f, ptr %__f_addr

  %0 = load ptr, ptr %__f_addr, align 8
  call void %0(ptr undef)
  call ptr @f1(ptr undef) ssp
  unreachable
}

define linkonce_odr void @f4(ptr %this) ssp align 2 {
entry:
  %0 = alloca i32
  call void @f3(ptr @f2) ssp
  ret void
}

