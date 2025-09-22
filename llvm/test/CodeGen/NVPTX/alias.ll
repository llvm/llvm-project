; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_30 -mattr=+ptx64 | FileCheck %s
; RUN: %if ptxas-isa-6.4 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_30 -mattr=+ptx64 | %ptxas-verify %}

define i32 @a() { ret i32 0 }
@b = internal alias i32 (), ptr @a
@c = internal alias i32 (), ptr @a
@d = internal alias i32 (), ptr @c

define void @foo(i32 %0, ptr %1) { ret void }
@bar = alias i32 (), ptr @foo

define void @noreturn() #0 {
  ret void
}
@noreturn_alias = alias i32 (), ptr @noreturn

define i32 @z() {
  %val = call i32 @b()
  ret i32 %val
}


attributes #0 = { noreturn }

;      CHECK: .visible .func  (.param .b32 func_retval0) b
; CHECK-NEXT: ()
; CHECK-NEXT: ;

;      CHECK: .visible .func  (.param .b32 func_retval0) c
; CHECK-NEXT: ()
; CHECK-NEXT: ;

;      CHECK: .visible .func  (.param .b32 func_retval0) d
; CHECK-NEXT: ()
; CHECK-NEXT: ;

;      CHECK: .visible .func bar
; CHECK-NEXT: (
; CHECK-NEXT:         .param .b32 foo_param_0,
; CHECK-NEXT:         .param .b64 foo_param_1
; CHECK-NEXT: )
; CHECK-NEXT: ;

;      CHECK: .visible .func noreturn_alias
; CHECK-NEXT: ()
; CHECK-NEXT: .noreturn;

; CHECK: .visible .func  (.param .b32 func_retval0) a()

;      CHECK: .visible .func foo(
; CHECK-NEXT:         .param .b32 foo_param_0,
; CHECK-NEXT:         .param .b64 foo_param_1
; CHECK-NEXT: )

;      CHECK: .visible .func noreturn()
; CHECK-NEXT: .noreturn

;      CHECK: .visible .func  (.param .b32 func_retval0) z()
;      CHECK:      call.uni (retval0), b,


; CHECK: .alias b, a;
; CHECK: .alias c, a;
; CHECK: .alias d, a;
; CHECK: .alias bar, foo;
; CHECK: .alias noreturn_alias, noreturn;
