; PR48055. Check that this does not crash.
; RUN: opt -passes=globalopt %s -disable-output
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.g = type opaque
%struct.a = type { i32 (...)** }

@l = dso_local global i32 0, align 4
@h = external dso_local global %struct.g, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_bug48055.cc, i8* null }]

; Function Attrs: uwtable
define internal void @__cxx_global_var_init() {
entry:
  %vtable = load i32 (%struct.a*)**, i32 (%struct.a*)*** bitcast (%struct.g* @h to i32 (%struct.a*)***), align 1
  %0 = load i32 (%struct.a*)*, i32 (%struct.a*)** %vtable, align 8
  %call = call i32 %0(%struct.a* nonnull dereferenceable(8) bitcast (%struct.g* @h to %struct.a*))
  store i32 %call, i32* @l, align 4
  ret void
}

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_bug48055.cc() {
entry:
  call void @__cxx_global_var_init()
  ret void
}

