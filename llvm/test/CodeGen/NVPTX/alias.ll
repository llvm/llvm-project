; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx63 | FileCheck %s

define i32 @a() { ret i32 0 }
@b = internal alias i32 (), ptr @a
@c = internal alias i32 (), ptr @a

define void @foo(i32 %0, ptr %1) { ret void }
@bar = alias i32 (), ptr @foo

; CHECK: .visible .func  (.param .b32 func_retval0) a()

;      CHECK: .visible .func foo(
; CHECK-NEXT:         .param .b32 foo_param_0,
; CHECK-NEXT:         .param .b64 foo_param_1
; CHECK-NEXT: )

;      CHECK: .visible .func  (.param .b32 func_retval0) b();
; CHECK-NEXT: .alias b, a;

;      CHECK: .visible .func  (.param .b32 func_retval0) c();
; CHECK-NEXT: .alias c, a;

;      CHECK: .visible .func bar(
; CHECK-NEXT:         .param .b32 foo_param_0,
; CHECK-NEXT:         .param .b64 foo_param_1
; CHECK-NEXT: );
; CHECK-NEXT: .alias bar, foo;
