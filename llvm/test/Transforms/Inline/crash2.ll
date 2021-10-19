; RUN: opt  -inline -sroa -max-devirt-iterations=1 -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.3"

declare i8* @f1(i8*) ssp align 2

define linkonce_odr void @f2(i8* %t) inlinehint ssp {
entry:
  unreachable
}

define linkonce_odr void @f3(void (i8*)* %__f) ssp {
entry:
  %__f_addr = alloca void (i8*)*, align 8
  store void (i8*)* %__f, void (i8*)** %__f_addr

  %0 = load void (i8*)*, void (i8*)** %__f_addr, align 8
  call void %0(i8* undef)
  call i8* @f1(i8* undef) ssp
  unreachable
}

define linkonce_odr void @f4(i8* %this) ssp align 2 {
entry:
  %0 = alloca i32
  call void @f3(void (i8*)* @f2) ssp
  ret void
}

