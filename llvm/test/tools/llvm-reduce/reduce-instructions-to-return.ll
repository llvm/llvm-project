; Test that llvm-reduce can move intermediate values by inserting
; early returns
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions-to-return --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,RESULT %s < %t

@gv = global i32 0, align 4
@gv_struct = global { i32, float } zeroinitializer, align 4
@gv_array = global [3 x i32] zeroinitializer, align 4
@gv_empty_struct = global { } zeroinitializer, align 4

; CHECK: @global.func.user = global ptr @store_instruction_to_return_with_uses
@global.func.user = global ptr @store_instruction_to_return_with_uses

; INTERESTING-LABEL: @store_instruction_to_return_with_uses(
; INTERESTING-NEXT: = load

; RESULT-LABEL: define i32 @store_instruction_to_return_with_uses(ptr %arg) {
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @store_instruction_to_return_with_uses(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: define void @user(
; INTERESTING: call

; RESULT-LABEL: define void @user(
; RESULT-NEXT: call i32 @store_instruction_to_return_with_uses(ptr %a, ptr %b)
; RESULT-NEXT: ret void
; RESULT-NEXT: }
define void @user(ptr %a, ptr %b) {
  call void @store_instruction_to_return_with_uses(ptr %a, ptr %b)
  ret void
}

; INTERESTING-LABEL: @store_instruction_to_return_no_uses(
; INTERESTING: = load i32

; RESULT-LABEL: define i32 @store_instruction_to_return_no_uses(
; RESULT-NEXT: %load = load i32
; RESULT-NEXT: ret i32 %load
define void @store_instruction_to_return_no_uses(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: @store_instruction_to_return_preserve_attrs(
; INTERESTING: = load

; RESULT: ; Function Attrs: nounwind
; RESULT-NEXT: define weak i32 @store_instruction_to_return_preserve_attrs(ptr byref(i32) %arg) #0 {
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define weak void @store_instruction_to_return_preserve_attrs(ptr byref(i32) %arg) nounwind "some-attr" {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: @store_instruction_to_return_preserve_addrspace(
; INTERESTING: = load

; RESULT-LABEL: define i32 @store_instruction_to_return_preserve_addrspace(ptr %arg) addrspace(1) {
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @store_instruction_to_return_preserve_addrspace(ptr %arg) addrspace(1) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: @store_instruction_to_return_no_uses_unreachable(
; INTERESTING: = load

; RESULT-LABEL: define i32 @store_instruction_to_return_no_uses_unreachable(ptr %arg) {
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @store_instruction_to_return_no_uses_unreachable(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  unreachable
}

; INTERESTING-LABEL: @store_instruction_to_return_with_non_callee_use(
; INTERESTING: = load

; RESULT-LABEL: define i32 @store_instruction_to_return_with_non_callee_use(ptr %arg) {
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @store_instruction_to_return_with_non_callee_use(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

declare void @takes_fptr(ptr)

; CHECK: @non_callee_user(
; CHECK: ret void
define void @non_callee_user(ptr %a, ptr %b) {
  call void @takes_fptr(ptr @store_instruction_to_return_with_non_callee_use)
  ret void
}

declare i32 @convergent_call() convergent

; CHECK-LABEL: @no_return_token_def(
; CHECK: call token
; RESULT: ret void
define void @no_return_token_def(ptr %arg) convergent {
  %t = call token @llvm.experimental.convergence.entry()
  ret void
}

; INTERESTING-LABEL: @no_return_token_def_other(
; INTERESTING: call token

; RESULT-LABEL: define i32 @no_return_token_def_other(
; RESULT: call token
; RESULT: call i32
; RESULT: ret i32
define void @no_return_token_def_other(ptr %arg) convergent {
  %t = call token @llvm.experimental.convergence.entry()
  %call = call i32 @convergent_call() [ "convergencectrl"(token %t) ]
  store i32 %call, ptr @gv
  ret void
}

; INTERESTING-LABEL: @store_instruction_to_return_variadic_func(
; INTERESTING: = load

; RESULT-LABEL: define i32 @store_instruction_to_return_variadic_func(ptr %arg, ...)
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @store_instruction_to_return_variadic_func(ptr %arg, ...) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; Has a callsite use that is invoking the function with a non-void
; return type, that does not match the new return type.

; INTERESTING-LABEL: @inst_to_return_has_nonvoid_wrong_type_caller(

; RESULT-LABEL: define void @inst_to_return_has_nonvoid_wrong_type_caller(
; RESULT-NEXT: %load = load i32, ptr %arg
; RESULT-NEXT: store i32 %load, ptr @gv
; RESULT-NEXT: ret void
define void @inst_to_return_has_nonvoid_wrong_type_caller(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: @wrong_callsite_return_type(

; RESULT-LABEL: define i64 @wrong_callsite_return_type(
; RESULT-NEXT: %ret = call i64 @inst_to_return_has_nonvoid_wrong_type_caller(ptr %arg)
; RESULT-NEXT: ret i64 %ret
define i64 @wrong_callsite_return_type(ptr %arg) {
  %ret = call i64 @inst_to_return_has_nonvoid_wrong_type_caller(ptr %arg)
  ret i64 %ret
}

; INTERESTING-LABEL: @inst_to_return_already_has_new_type_caller(

; RESULT-LABEL: define i32 @inst_to_return_already_has_new_type_caller(
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @inst_to_return_already_has_new_type_caller(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; Callsite has UB signature mismatch, but the return type happens to
; match the new return type.
;
; INTERESTING-LABEL: @callsite_already_new_return_type(

; RESULT-LABEL: define i32 @callsite_already_new_return_type(
; RESULT-NEXT: %ret = call i32 @inst_to_return_already_has_new_type_caller(ptr %arg)
; RESULT-NEXT: ret i32 %ret
define i32 @callsite_already_new_return_type(ptr %arg) {
  %ret = call i32 @inst_to_return_already_has_new_type_caller(ptr %arg)
  ret i32 %ret
}

; INTERESTING-LABEL: @non_void_no_op(
; INTERESTING: = load
; INTERESTING: ret

; RESULT-LABEL: define ptr @non_void_no_op(
; RESULT-NEXT: %load = load i32, ptr %arg
; RESULT-NEXT: store i32 %load, ptr @gv
; RESULT-NEXT: ret ptr null
define ptr @non_void_no_op(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret ptr null
}

; INTERESTING-LABEL: @non_void_no_op_caller(

; RESULT-LABEL: define ptr @non_void_no_op_caller(ptr %arg) {
; RESULT-NEXT: %call = call ptr @non_void_no_op(ptr %arg)
; RESULT-NEXT: ret ptr %call
define ptr @non_void_no_op_caller(ptr %arg) {
  %call = call ptr @non_void_no_op(ptr %arg)
  ret ptr %call
}

; INTERESTING-LABEL: @non_void_same_type_use(
; INTERESTING: = load
; INTERESTING: ret

; RESULT-LABEL: define i32 @non_void_same_type_use(
; RESULT-NEXT: %load = load i32, ptr %arg
; RESULT-NEXT: ret i32 %load
define i32 @non_void_same_type_use(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret i32 0
}

; INTERESTING-LABEL: @non_void_bitcastable_type_use(
; INTERESTING: = load
; INTERESTING: ret

; RESULT-LABEL: define i32 @non_void_bitcastable_type_use(
; RESULT-NEXT: %load = load float, ptr %arg
; RESULT-NEXT: store float %load,
; RESULT-NEXT: ret i32 0
define i32 @non_void_bitcastable_type_use(ptr %arg) {
  %load = load float, ptr %arg
  store float %load, ptr @gv
  ret i32 0
}

; INTERESTING-LABEL: @non_void_bitcastable_type_use_caller(
define i32 @non_void_bitcastable_type_use_caller(ptr %arg) {
  %ret = call i32 @non_void_bitcastable_type_use(ptr %arg)
  ret i32 %ret
}

; INTERESTING-LABEL: @form_return_struct(
; INTERESTING: = load { i32, float }

; RESULT-LABEL: define { i32, float } @form_return_struct(ptr %arg) {
; RESULT-NEXT: %load = load { i32, float }, ptr %arg, align 4
; RESULT-NEXT: ret { i32, float } %load
define void @form_return_struct(ptr %arg) {
  %load = load { i32, float }, ptr %arg
  store { i32, float } %load, ptr @gv_struct
  ret void
}

; INTERESTING-LABEL: define void @return_struct_user(
; INTERESTING-NEXT: call
; RESULT: call { i32, float } @form_return_struct(ptr %arg)
define void @return_struct_user(ptr %arg) {
  call void @form_return_struct(ptr %arg)
  ret void
}

; INTERESTING-LABEL: @form_return_array(
; INTERESTING: = load

; RESULT-LABEL: define [3 x i32] @form_return_array(
; RESULT-NEXT: %load = load [3 x i32]
; RESULT-NEXT: ret [3 x i32] %load
define void @form_return_array(ptr %arg) {
  %load = load [3 x i32], ptr %arg
  store [3 x i32] %load, ptr @gv_array
  ret void
}

; CHECK-LABEL: @return_array_user(
; RESULT: call [3 x i32] @form_return_array(ptr %arg)
define void @return_array_user(ptr %arg) {
  call void @form_return_array(ptr %arg)
  ret void
}

; INTERESTING-LABEL: @form_return_empty_struct(
; INTERESTING: = load

; RESULT: define {} @form_return_empty_struct(
; RESULT-NEXT: %load = load {}
; RESULT-NEXT: ret {} %load
define void @form_return_empty_struct(ptr %arg) {
  %load = load { }, ptr %arg
  store { } %load, ptr @gv_empty_struct
  ret void
}

; CHECK-LABEL: define void @return_empty_struct_user(
; RESULT: call {} @form_return_empty_struct(ptr %arg)
define void @return_empty_struct_user(ptr %arg) {
  call void @form_return_empty_struct(ptr %arg)
  ret void
}

define target("sometarget.sometype") @target_type_func() {
  ret target("sometarget.sometype") poison
}

define void @target_type_user(target("sometarget.sometype") %a) {
  ret void
}

; INTERESTING-LABEL: @form_return_target_ty(
; INTERESTING: call target("sometarget.sometype") @target_type_func()

; RESULT: define target("sometarget.sometype") @form_return_target_ty(
; RESULT-NEXT: %call = call target("sometarget.sometype") @target_type_func()
; RESULT-NEXT:  ret target("sometarget.sometype") %call
define void @form_return_target_ty(ptr %arg) {
  %call = call target("sometarget.sometype") @target_type_func()
  call void @target_type_user(target("sometarget.sometype") %call)
  ret void
}

; CHECK-LABEL: define void @return_target_ty_user(
; RESULT-NEXT: %1 = call target("sometarget.sometype") @form_return_target_ty(ptr %arg)
; RESULT-NEXT: ret void
define void @return_target_ty_user(ptr %arg) {
  call void @form_return_target_ty(ptr %arg)
  ret void
}

; Make sure an invalid reduction isn't attempted for a function with
; an sret argument

; CHECK-LABEL: @no_sret_nonvoid_return
define void @no_sret_nonvoid_return(ptr sret(i32) %out.sret, ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr %out.sret
  ret void
}

; Test a calling convention where it's illegal to use a non-void
; return. No invalid reduction should be introduced.

; INTERESTING-LABEL: @no_void_return_callingconv(
; INTERESTING: = load i32

; RESULT-LABEL: define amdgpu_kernel void @no_void_return_callingconv(
; RESULT-NEXT: %load = load i32
; RESULT-NEXT: store i32 %load
; RESULT-NEXT: ret void
define amdgpu_kernel void @no_void_return_callingconv(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: @keep_first_of_3(
; INTERESTING: %load0 = load i32, ptr %arg0
; INTERESTING: ret

; RESULT-LABEL: define i32 @keep_first_of_3(
; RESULT-NEXT: %load0 = load i32, ptr %arg0, align 4
; RESULT-NEXT: ret i32 %load0
define void @keep_first_of_3(ptr %arg0, ptr %arg1, ptr %arg2) {
  %load0 = load i32, ptr %arg0
  %load1 = load i32, ptr %arg1
  %load2 = load i32, ptr %arg2
  store i32 %load0, ptr @gv
  store i32 %load1, ptr @gv
  store i32 %load2, ptr @gv
  ret void
}

; INTERESTING-LABEL: @keep_second_of_3(
; INTERESTING: %load1 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @keep_second_of_3(
; RESULT-NEXT: %load0 = load i32, ptr %arg0
; RESULT-NEXT: %load1 = load i32, ptr %arg1
; RESULT-NEXT: ret i32 %load1
define void @keep_second_of_3(ptr %arg0, ptr %arg1, ptr %arg2) {
  %load0 = load i32, ptr %arg0
  %load1 = load i32, ptr %arg1
  %load2 = load i32, ptr %arg2
  store i32 %load0, ptr @gv
  store i32 %load1, ptr @gv
  store i32 %load2, ptr @gv
  ret void
}

; INTERESTING-LABEL: @keep_third_of_3(
; INTERESTING: %load2 = load i32, ptr %arg2

; RESULT-LABEL: define i32 @keep_third_of_3(
; RESULT-NEXT: %load0 = load i32, ptr %arg0, align 4
; RESULT-NEXT: %load1 = load i32, ptr %arg1, align 4
; RESULT-NEXT: %load2 = load i32, ptr %arg2, align 4
; RESULT-NEXT: ret i32 %load2
define void @keep_third_of_3(ptr %arg0, ptr %arg1, ptr %arg2) {
  %load0 = load i32, ptr %arg0
  %load1 = load i32, ptr %arg1
  %load2 = load i32, ptr %arg2
  store i32 %load0, ptr @gv
  store i32 %load1, ptr @gv
  store i32 %load2, ptr @gv
  ret void
}

; INTERESTING-LABEL: @keep_first_2_of_3(
; INTERESTING: %load0 = load i32, ptr %arg0
; INTERESTING: %load1 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @keep_first_2_of_3(
; RESULT-NEXT: %load0 = load i32, ptr %arg0
; RESULT-NEXT: %load1 = load i32, ptr %arg1
; RESULT-NEXT: ret i32 %load1
define void @keep_first_2_of_3(ptr %arg0, ptr %arg1, ptr %arg2) {
  %load0 = load i32, ptr %arg0
  %load1 = load i32, ptr %arg1
  %load2 = load i32, ptr %arg2
  store i32 %load0, ptr @gv
  store i32 %load1, ptr @gv
  store i32 %load2, ptr @gv
  ret void
}

; INTERESTING-LABEL: @keep_second_of_3_already_ret_constexpr(
; INTERESTING: %load1 = load i32, ptr %arg1
; INTERESTING: ret

; RESULT-LABEL: define i32 @keep_second_of_3_already_ret_constexpr(
; RESULT-NEXT: %load0 = load i32, ptr %arg0, align 4
; RESULT-NEXT: %load1 = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load1
define i32 @keep_second_of_3_already_ret_constexpr(ptr %arg0, ptr %arg1, ptr %arg2) {
  %load0 = load i32, ptr %arg0
  %load1 = load i32, ptr %arg1
  %load2 = load i32, ptr %arg2
  store i32 %load0, ptr @gv
  store i32 %load1, ptr @gv
  store i32 %load2, ptr @gv
  ret i32 ptrtoint (ptr @gv to i32)
}

; INTERESTING-LABEL: @self_recursive(
; INTERESTING:  %load = load i32, ptr %arg

; RESULT-LABEL: define i32 @self_recursive(
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @self_recursive(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  call void @self_recursive(ptr %arg)
  ret void
}

; INTERESTING-LABEL: @has_invoke_user(

; RESULT-LABEL: define i32 @has_invoke_user(
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define void @has_invoke_user(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

declare i32 @__gxx_personality_v0(...)

; INTERESTING-LABEL: @invoker(
; RESULT:   %0 = invoke i32 @has_invoke_user(ptr %arg)
define void @invoker(ptr %arg) personality ptr @__gxx_personality_v0 {
bb:
  invoke void @has_invoke_user(ptr %arg)
    to label %bb3 unwind label %bb1

bb1:
  landingpad { ptr, i32 }
  catch ptr null
  br label %bb3

bb3:
  ret void
}

; INTERESTING-LABEL: @return_from_nonentry_block(

; RESULT-LABEL: define i32 @return_from_nonentry_block(
; RESULT: br i1 %arg0, label %bb0, label %bb1

; RESULT: bb0:
; RESULT-NEXT: %load = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load

; RESULT: bb1:
; RESULT-NEXT: unreachable
define void @return_from_nonentry_block(i1 %arg0, ptr %arg1) {
entry:
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %load = load i32, ptr %arg1
  store i32 %load, ptr @gv
  ret void

bb1:
  unreachable
}

; INTERESTING-LABEL: @multi_void_return(
; INTERESTING: %load = load i32, ptr %arg1

; RESULT-LABEL: define i32 @multi_void_return(i1 %arg0, ptr %arg1) {
; RESULT-NEXT: entry:
; RESULT-NEXT: br i1 %arg0, label %bb0, label %bb1

; RESULT: bb0:
; RESULT-NEXT: %load = load i32, ptr %arg1
; RESULT-NEXT: ret i32 %load

; RESULT: bb1:
; RESULT-NEXT: ret i32 0
define void @multi_void_return(i1 %arg0, ptr %arg1) {
entry:
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %load = load i32, ptr %arg1
  store i32 %load, ptr @gv
  ret void

bb1:
  ret void
}

; INTERESTING-LABEL: @multi_void_return_dominates_all(
; INTERESTING: %load = load i32, ptr %arg1

; RESULT-LABEL: define i32 @multi_void_return_dominates_all(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load
; RESULT-NEXT: }
define void @multi_void_return_dominates_all(i1 %arg0, ptr %arg1) {
entry:
  %load = load i32, ptr %arg1
  br i1 %arg0, label %bb0, label %bb1

bb0:
  store i32 %load, ptr @gv
  ret void

bb1:
  ret void
}

; INTERESTING-LABEL: @multi_unreachable_dominates_all(
; INTERESTING: %load = load i32, ptr %arg1

; RESULT-LABEL: define i32 @multi_unreachable_dominates_all(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load
; RESULT-NEXT: }
define void @multi_unreachable_dominates_all(i1 %arg0, ptr %arg1) {
entry:
  %load = load i32, ptr %arg1
  br i1 %arg0, label %bb0, label %bb1

bb0:
  store i32 %load, ptr @gv
  unreachable

bb1:
  unreachable
}

; We want to mutate %bb0 to return %load0, and not break the ret in
; %bb1

; INTERESTING-LABEL: @multi_nonvoid_return(
; INTERESTING: %other = load i32, ptr %arg2
; INTERESTING: br i1 %arg0
; INTERESTING: %load = load i32, ptr %arg1


; RESULT-LABEL: define i32 @multi_nonvoid_return(

; RESULT: entry:
; RESULT-NEXT: %other = load i32, ptr %arg2
; RESULT-NEXT: br i1 %arg0, label %bb0, label %bb1

; RESULT: bb0:
; RESULT-NEXT: %load = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load

; RESULT: bb1:
; RESULT-NEXT: ret i32 99
define i32 @multi_nonvoid_return(i1 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %other = load i32, ptr %arg2
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %load = load i32, ptr %arg1
  store i32 %load, ptr @gv
  ret i32 %other

bb1:
  ret i32 99
}

; TODO: Could handle this better if we avoided eliminating code that
; was already dead

; INTERESTING-LABEL: @interesting_in_unreachable_code(
; INTERESTING: %load = load i32, ptr %arg

; RESULT-LABEL: define void @interesting_in_unreachable_code(
; RESULT-NEXT: entry:
; RESULT-NEXT:  ret void

; RESULT: bb: ; No predecessors!
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: store i32 %load,
; RESULT-NEXT: ret void
define void @interesting_in_unreachable_code(ptr %arg) {
entry:
  ret void

bb:
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret void
}

; INTERESTING-LABEL: @use_in_successor_phi(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @use_in_successor_phi(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load0
; RESULT-NEXT: }
define void @use_in_successor_phi(i1 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %load0 = load i32, ptr %arg1
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %phi = phi i32 [ %load0, %entry ], [ %load1, %bb1 ]
  store i32 %phi, ptr @gv
  br label %bb2

bb1:
  %load1 = load i32, ptr %arg2
  br label %bb0

bb2:
  ret void
}

; INTERESTING-LABEL: @use_in_successor_phi_repeated(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @use_in_successor_phi_repeated(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load0
; RESULT-NEXT: }
define void @use_in_successor_phi_repeated(i1 %arg0, ptr %arg1, ptr %arg2, i32 %switch.val) {
entry:
  %load0 = load i32, ptr %arg1
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %phi = phi i32 [ %load0, %entry ], [ %load1, %bb1 ], [ %load1, %bb1 ]
  store i32 %phi, ptr @gv
  br label %bb2

bb1:
  %load1 = load i32, ptr %arg2
  switch i32 %switch.val, label %bb2 [
    i32 1, label %bb0
    i32 2, label %bb0
  ]

bb2:
  ret void
}

; INTERESTING-LABEL: @replace_cond_br_with_ret(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @replace_cond_br_with_ret(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load0
define void @replace_cond_br_with_ret(i1 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %load0 = load i32, ptr %arg1
  br i1 %arg0, label %bb0, label %bb1

bb0:
  store i32 %load0, ptr %arg2
  ret void

bb1:
  store i32 %load0, ptr %arg2
  ret void
}

; INTERESTING-LABEL: @replace_switch_with_ret(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @replace_switch_with_ret(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load0
; RESULT-NEXT: }
define void @replace_switch_with_ret(i32 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %load0 = load i32, ptr %arg1
  switch i32 %arg0, label %bb2 [
    i32 1, label %bb0
    i32 2, label %bb1
  ]

bb0:
  store i32 9, ptr %arg2
  ret void

bb1:
  store i32 10, ptr %arg2
  unreachable

bb2:
  ret void
}

; INTERESTING-LABEL: @replace_uncond_br_with_ret(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @replace_uncond_br_with_ret(
; RESULT-NEXT: entry:
; RESULT: %load0 = load i32, ptr %arg1, align 4
; RESULT-NEXT: ret i32 %load0
; RESULT-NEXT: }
define void @replace_uncond_br_with_ret(i1 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %load0 = load i32, ptr %arg1
  br label %bb0

bb0:
  store i32 %load0, ptr %arg2
  ret void
}

; INTERESTING-LABEL: @replace_uncond_br_with_ret_with_phi(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @replace_uncond_br_with_ret_with_phi(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1
; RESULT-NEXT: ret i32 %load0
; RESULT-NEXT: }
define void @replace_uncond_br_with_ret_with_phi(i1 %arg0, ptr %arg1, ptr %arg2, ptr %arg3) {
entry:
  %load0 = load i32, ptr %arg1
  br label %bb0

bb0:
  %phi = phi i32 [ %load0, %entry ]
  store i32 %phi, ptr %arg2
  store i32 %load0, ptr %arg3
  ret void
}

; INTERESTING-LABEL: @use_tail_instr_in_successor_phi(
; INTERESTING: %load0 = load i32, ptr %arg1

; RESULT-LABEL: define i32 @use_tail_instr_in_successor_phi(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1
; RESULT-NEXT: ret i32 %load0
; RESULT-NEXT: }
define void @use_tail_instr_in_successor_phi(i1 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %load0 = load i32, ptr %arg1
  %load1 = load i32, ptr %arg2
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %phi = phi i32 [ %load0, %entry ], [ %load1, %bb1 ]
  store i32 %phi, ptr @gv
  br label %bb2

bb1:
  br label %bb0

bb2:
  ret void
}

; INTERESTING-LABEL: @use_before_instr_in_successor_phi(
; INTERESTING: %load1 = load i32, ptr %arg2

; RESULT-LABEL: define i32 @use_before_instr_in_successor_phi(
; RESULT-NEXT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg1
; RESULT-NEXT: %load1 = load i32, ptr %arg2
; RESULT-NEXT: ret i32 %load1
; RESULT-NEXT: }
define void @use_before_instr_in_successor_phi(i1 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %load0 = load i32, ptr %arg1
  %load1 = load i32, ptr %arg2
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %phi = phi i32 [ %load0, %entry ], [ %load1, %bb1 ]
  store i32 %phi, ptr @gv
  br label %bb2

bb1:
  br label %bb0

bb2:
  ret void
}

declare i32 @maybe_throwing_callee(i32)
declare void @thrown()
declare void @did_not_throw(i32)

; TODO: Handle invokes properly
; INTERESTING-LABEL @reduce_invoke_use(
; INTERESTING: call void @did_not_throw(i32 %invoke)

; RESULT-LABEL: define { ptr, i32 } @reduce_invoke_use(

; RESULT: %invoke = invoke i32 @maybe_throwing_callee

; RESULT: bb1: ; preds = %bb
; RESULT-NEXT: %landing = landingpad { ptr, i32 }
; RESULT-NEXT: catch ptr null
; RESULT-NEXT: ret { ptr, i32 } %landing

; RESULT: bb4: ; preds = %bb3
; RESULT-NEXT: ret { ptr, i32 } zeroinitializer
define void @reduce_invoke_use(i32 %arg) personality ptr @__gxx_personality_v0 {
bb:
  %invoke = invoke i32 @maybe_throwing_callee(i32 %arg)
          to label %bb3 unwind label %bb1

bb1:                                              ; preds = %bb
  %landing = landingpad { ptr, i32 }
          catch ptr null
  call void @thrown()
  br label %bb4

bb3:                                              ; preds = %bb
  call void @did_not_throw(i32 %invoke)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb1
  ret void
}

; We can replace the branch in %bb0 with a return, but bb2 will still
; be reachable after

; INTERESTING-LABEL: @successor_block_not_dead_after_ret(
; INTERESTING: %load0 = load i32, ptr %arg2

; RESULT-LABEL: define i32 @successor_block_not_dead_after_ret(
; RESULT: entry:
; RESULT-NEXT: br i1 %arg0, label %bb0, label %bb2

; RESULT: bb0:                                              ; preds = %entry
; RESULT-NEXT: %load0 = load i32, ptr %arg2, align 4
; RESULT-NEXT: ret i32 %load0

; RESULT: bb2:                                              ; preds = %entry
; RESULT-NEXT: %phi = phi i32 [ %arg4, %entry ]
; RESULT-NEXT: ret i32 %phi
; RESULT-NEXT: }
define void @successor_block_not_dead_after_ret(i1 %arg0, i1 %arg1, ptr %arg2, ptr %arg3, i32 %arg4, i32 %arg5) {
entry:
  br i1 %arg0, label %bb0, label %bb2

bb0:
  %load0 = load i32, ptr %arg2
  store i32 %load0, ptr @gv
  br i1 %arg1, label %bb1, label %bb2

bb1:
  %load1 = load i32, ptr %arg3
  store i32 %load1, ptr @gv
  br label %bb0

bb2:
  %phi = phi i32 [ %arg4, %entry ], [ %arg5, %bb0 ]
  store i32 %phi, ptr @gv
  ret void
}

; INTERESTING-LABEL: @successor_block_self_loop_phi(
; INTERESTING: %load0 = load i32, ptr %arg2

; RESULT-LABEL: define i32 @successor_block_self_loop_phi(
; RESULT: entry:
; RESULT-NEXT: br i1 %arg0, label %bb0, label %bb1

; RESULT: bb0:                                              ; preds = %entry
; RESULT-NEXT: %phi = phi i32 [ 12, %entry ]
; RESULT-NEXT: %load0 = load i32, ptr %arg2, align 4
; RESULT-NEXT: ret i32 %load0

; RESULT: bb1:                                              ; preds = %entry
; RESULT-NEXT: ret i32 0
; RESULT-NEXT: }
define void @successor_block_self_loop_phi(i1 %arg0, i1 %arg1, ptr %arg2) {
entry:
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %phi = phi i32 [ 12, %entry ], [ %load0, %bb0 ]
  %load0 = load i32, ptr %arg2
  store i32 %phi, ptr @gv
  br i1 %arg1, label %bb0, label %bb1

bb1:
  ret void
}

; INTERESTING-LABEL: @successor_block_self_loop_phi_2(
; INTERESTING: %phi1 = phi i32

; RESULT-LABEL: define i32 @successor_block_self_loop_phi_2(
; RESULT: entry:
; RESULT-NEXT: %load0 = load i32, ptr %arg2
; RESULT-NEXT: br i1 %arg0, label %bb0, label %bb1

; RESULT: bb0:                                              ; preds = %entry
; RESULT-NEXT: %phi0 = phi i32 [ 12, %entry ]
; RESULT-NEXT: %phi1 = phi i32 [ %arg4, %entry ]
; RESULT-NEXT: ret i32 %phi1

; RESULT-NOT: bb

; RESULT: bb1:                                              ; preds = %entry
; RESULT-NEXT: ret i32 0
; RESULT-NEXT: }
define void @successor_block_self_loop_phi_2(i1 %arg0, i1 %arg1, ptr %arg2, ptr %arg3, i32 %arg4) {
entry:
  %load0 = load i32, ptr %arg2
  br i1 %arg0, label %bb0, label %bb1

bb0:
  %phi0 = phi i32 [ 12, %entry ], [ %load0, %bb0 ]
  %phi1 = phi i32 [ %arg4, %entry ], [ %load1, %bb0 ]
  %load1 = load i32, ptr %arg3
  store i32 %phi0, ptr @gv
  store i32 %phi1, ptr @gv
  br i1 %arg1, label %bb0, label %bb1

bb1:
  ret void
}
