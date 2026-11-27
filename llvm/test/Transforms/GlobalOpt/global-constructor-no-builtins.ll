; RUN: opt -S -p globalopt < %s | FileCheck %s

; Test that the "no-builtins" attribute prevents constant evaluation of the call
; to `pow` in `bar`, leaving the global `@s` zero-initialised.

%struct.S = type { double }

; CHECK: @s = global %struct.S zeroinitializer, align 8
@s = global %struct.S zeroinitializer, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_x.cc, ptr null }]

declare double @pow(double, double)

define double @foo(double %x) {
entry:
  %call = call double @pow(double %x, double f0x3FF3333340000000)
  ret double %call
}

define double @bar(double %x) "no-builtins" {
entry:
  %call = call double @pow(double %x, double f0x3FF3333340000000)
  ret double %call
}

define internal void @__cxx_global_var_init() {
entry:
  call void @ctor(ptr @s, double 2.000000e+00)
  ret void
}

define void @ctor(ptr %this, double %f) {
entry:
  %call1 = call double @foo(double %f)
  %call2 = call double @bar(double %f)
  %add = fadd double %call1, %call2
  store double %add, ptr %this, align 8
  ret void
}

define internal void @_GLOBAL__sub_I_x.cc()  {
entry:
  call void @__cxx_global_var_init()
  ret void
}

