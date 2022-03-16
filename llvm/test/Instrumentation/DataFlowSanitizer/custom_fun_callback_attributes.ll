; RUN: opt < %s -dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Declare custom functions.  Inputs/abilist.txt causes any function with a
; name matching /custom.*/ to be a custom function.
declare i32 @custom_fun_one_callback(i8 (i32, double)* %callback_arg)
declare i32 @custom_fun_two_callbacks(
  i8 (i32, double)* %callback_arg1,
  i64 %an_int,
  i8 (i32, double)* %callback_arg2
)

declare i8 @a_callback_fun(i32, double)

; CHECK-LABEL: @call_custom_funs_with_callbacks.dfsan
define void @call_custom_funs_with_callbacks(i8 (i32, double)* %callback_arg) {
  ;; The callback should have attribute 'nonnull':
  ; CHECK: call signext i32 @__dfsw_custom_fun_one_callback(
  %call1 = call signext i32 @custom_fun_one_callback(
    i8 (i32, double)* nonnull @a_callback_fun
  )

  ;; Call a custom function with two callbacks.  Check their annotations.
  ; CHECK: call i32 @__dfsw_custom_fun_two_callbacks(
  ; CHECK: i64 12345
  %call2 = call i32 @custom_fun_two_callbacks(
    i8 (i32, double)* nonnull @a_callback_fun,
    i64 12345,
    i8 (i32, double)* noalias @a_callback_fun
  )
  ret void
}
