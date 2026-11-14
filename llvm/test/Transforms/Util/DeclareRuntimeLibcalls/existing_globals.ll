; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --implicit-check-not='{{memcpy|malloc|free|sqrt|sqrtf|cosf|__stack_chk_guard|__safestack_unsafe_stack_ptr}}.1' %s

; Existing globals with the name of a runtime libcall are left unmodified,
; instead of inserting a new declaration.

; A variable with the name of a function.
; CHECK: @memcpy = global i32 0
@memcpy = global i32 0

; A variable with the wrong type.
; CHECK: @__stack_chk_guard = global i32 0
@__stack_chk_guard = global i32 0

; A function with the name of a variable.
; CHECK: declare void @__safestack_unsafe_stack_ptr()
declare void @__safestack_unsafe_stack_ptr()

; Functions with the wrong type, for which the libcall attributes would be
; invalid.
; CHECK: declare i32 @malloc(i32){{$}}
declare i32 @malloc(i32)

; CHECK: declare void @free(i64){{$}}
declare void @free(i64)

; CHECK: declare i32 @sqrt(i32){{$}}
declare i32 @sqrt(i32)

; A function with the wrong calling convention.
; CHECK: declare fastcc float @sqrtf(float){{$}}
declare fastcc float @sqrtf(float)

; A definition with the correct type and different attributes.
; CHECK: define noundef nofpclass(nan) float @cosf(float %x) [[COSF_ATTRS:#[0-9]+]] {
define noundef nofpclass(nan) float @cosf(float %x) "foo" {
  %ret = call float asm "; $0 = cos($1)", "=r,r"(float %x)
  ret float %ret
}

; CHECK: attributes [[COSF_ATTRS]] = { "foo" }
