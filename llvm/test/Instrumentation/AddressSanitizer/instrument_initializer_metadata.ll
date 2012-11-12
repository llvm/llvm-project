; RUN: opt < %s -asan -asan-initialization-order -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
@xxx = global i32 0, align 4
; Clang will emit the following metadata identifying @xxx as dynamically
; initialized.
!0 = metadata !{i32* @xxx}
!llvm.asan.dynamically_initialized_globals = !{!0}

define i32 @initializer() uwtable {
entry:
  ret i32 42
}

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  %call = call i32 @initializer()
  store i32 %call, i32* @xxx, align 4
  ret void
}

define internal void @_GLOBAL__I_a() address_safety section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

; Clang indicated that @xxx was dynamically initailized.
; __asan_{before,after}_dynamic_init should be called from _GLOBAL__I_a

; CHECK: define internal void @_GLOBAL__I_a
; CHECK-NOT: ret
; CHECK: call void @__asan_before_dynamic_init
; CHECK: call void @__cxx_global_var_init
; CHECK: call void @__asan_after_dynamic_init
; CHECK: ret
