; Test that llvm-reduce can move intermediate values by inserting
; early returns when the function already has a different return type
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions-to-return --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t


@gv = global i32 0, align 4
@ptr_array = global [2 x ptr] [ptr @inst_to_return_has_different_type_but_no_func_call_use,
                               ptr @multiple_callsites_wrong_return_type]

; Should rewrite this return from i64 to i32 since the function has no
; uses.
; INTERESTING-LABEL: @inst_to_return_has_different_type_but_no_func_call_use(
; RESULT-LABEL: define i32 @inst_to_return_has_different_type_but_no_func_call_use(ptr %arg) {
; RESULT-NEXT: %load = load i32, ptr %arg, align 4
; RESULT-NEXT: ret i32 %load
define i64 @inst_to_return_has_different_type_but_no_func_call_use(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret i64 0
}

; INTERESTING-LABEL: @multiple_returns_wrong_return_type_no_callers(
; RESULT-LABEL: define i32 @multiple_returns_wrong_return_type_no_callers(

; RESULT: bb0:
; RESULT-NEXT: %load0 = load i32,
; RESULT-NEXT: ret i32 %load0

; RESULT: bb1:
; RESULT-NEXT: store i32 8, ptr null
; RESULT-NEXT: ret i32 0
define i64 @multiple_returns_wrong_return_type_no_callers(ptr %arg, i1 %cond, i64 %arg2) {
entry:
  br i1 %cond, label %bb0, label %bb1

bb0:
  %load0 = load i32, ptr %arg
  store i32 %load0, ptr @gv
  ret i64 234

bb1:
  store i32 8, ptr null
  ret i64 %arg2

bb2:
  ret i64 34
}

; INTERESTING-LABEL: define {{.+}} @callsite_different_type_unused_0(

; RESULT-LABEL: define i64 @callsite_different_type_unused_0(ptr %arg) {
; RESULT-NEXT: %unused0 = call i64 @inst_to_return_has_different_type_but_call_result_unused(ptr %arg)
; RESULT-NEXT: ret i64 %unused0
define void @callsite_different_type_unused_0(ptr %arg) {
  %unused0 = call i64 @inst_to_return_has_different_type_but_call_result_unused(ptr %arg)
  %unused1 = call i64 @inst_to_return_has_different_type_but_call_result_unused(ptr null)
  ret void
}

; TODO: Could rewrite this return from i64 to i32 since the callsite is unused.
; INTERESTING-LABEL: define {{.+}} @inst_to_return_has_different_type_but_call_result_unused(
; RESULT-LABEL: define i64 @inst_to_return_has_different_type_but_call_result_unused(
; RESULT-NEXT: %load = load i32, ptr %arg
; RESULT-NEXT: store i32 %load, ptr @gv
; RESULT: ret i64 0
define i64 @inst_to_return_has_different_type_but_call_result_unused(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret i64 0
}

; INTERESTING-LABEL: @multiple_callsites_wrong_return_type(
; RESULT-LABEL: define i64 @multiple_callsites_wrong_return_type(
; RESULT: ret i64 0
define i64 @multiple_callsites_wrong_return_type(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret i64 0
}

; INTERESTING-LABEL: @unused_with_wrong_return_types(
; RESULT-LABEL: define i64 @unused_with_wrong_return_types(
; RESULT-NEXT: %unused0 = call i64 @multiple_callsites_wrong_return_type(ptr %arg)
; RESULT-NEXT: ret i64 %unused0
define void @unused_with_wrong_return_types(ptr %arg) {
  %unused0 = call i64 @multiple_callsites_wrong_return_type(ptr %arg)
  %unused1 = call i32 @multiple_callsites_wrong_return_type(ptr %arg)
  %unused2 = call ptr @multiple_callsites_wrong_return_type(ptr %arg)
  ret void
}

; INTERESTING-LABEL: @multiple_returns_wrong_return_type(
; INTERESTING: %load0 = load i32,

; RESULT-LABEL: define i32 @multiple_returns_wrong_return_type(
; RESULT: ret i32
; RESULT: ret i32
; RESULT: ret i32
define i32 @multiple_returns_wrong_return_type(ptr %arg, i1 %cond, i32 %arg2) {
entry:
  br i1 %cond, label %bb0, label %bb1

bb0:
  %load0 = load i32, ptr %arg
  store i32 %load0, ptr @gv
  ret i32 234

bb1:
  ret i32 %arg2

bb2:
  ret i32 34
}

; INTERESTING-LABEL: @call_multiple_returns_wrong_return_type(
; RESULT-LABEL: define <2 x i32> @call_multiple_returns_wrong_return_type(
; RESULT-NEXT: %unused = call <2 x i32> @multiple_returns_wrong_return_type(
; RESULT-NEXT: ret <2 x i32> %unused
define void @call_multiple_returns_wrong_return_type(ptr %arg, i1 %cond, i32 %arg2) {
  %unused = call <2 x i32> @multiple_returns_wrong_return_type(ptr %arg, i1 %cond, i32 %arg2)
  ret void
}
