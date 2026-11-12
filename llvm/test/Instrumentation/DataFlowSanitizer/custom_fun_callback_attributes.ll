; RUN: opt < %s -passes=dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Declare custom functions.  Inputs/abilist.txt causes any function with a
; name matching /custom.*/ to be a custom function.
declare i32 @custom_fun_one_callback(ptr %callback_arg)
declare i32 @custom_fun_two_callbacks(
  ptr %callback_arg1,
  i64 %an_int,
  ptr %callback_arg2
)

declare i8 @a_callback_fun(i32, double)

; CHECK-LABEL: @call_custom_funs_with_callbacks.dfsan
define void @call_custom_funs_with_callbacks(ptr %callback_arg) {
  ;; The callback should have attribute 'nonnull':
  ; CHECK: call signext i32 @__dfsw_custom_fun_one_callback(
  %call1 = call signext i32 @custom_fun_one_callback(
    ptr nonnull @a_callback_fun
  )

  ;; Call a custom function with two callbacks.  Check their annotations.
  ; CHECK: call i32 @__dfsw_custom_fun_two_callbacks(
  ; CHECK: i64 12345
  %call2 = call i32 @custom_fun_two_callbacks(
    ptr nonnull @a_callback_fun,
    i64 12345,
    ptr noalias @a_callback_fun
  )
  ret void
}
