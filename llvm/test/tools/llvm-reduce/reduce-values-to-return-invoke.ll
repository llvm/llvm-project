; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions-to-return --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t

@gv = global i32 0, align 4


define i32 @has_invoke_user(ptr %arg) {
  %load = load i32, ptr %arg
  store i32 %load, ptr @gv
  ret i32 9
}

declare i32 @__gxx_personality_v0(...)

; INTERESTING-LABEL: @invoker_keep_invoke(
; INTERESTING: %invoke
; RESULT:   %invoke = invoke i32 @has_invoke_user(ptr %arg)
define void @invoker_keep_invoke(ptr %arg) personality ptr @__gxx_personality_v0 {
bb:
  %invoke = invoke i32 @has_invoke_user(ptr %arg)
    to label %bb3 unwind label %bb1

bb1:
  landingpad { ptr, i32 }
  catch ptr null
  ret void

bb3:
  store i32 %invoke, ptr null
  ret void
}

; INTERESTING-LABEL: @invoker_drop_invoke(
; INTERESTING: %add = add i32

; RESULT-LABEL: define i32 @invoker_drop_invoke(i32 %arg0, ptr %arg1) personality ptr @__gxx_personality_v0 {
; RESULT-NEXT: bb:
; RESULT-NEXT: %add = add i32 %arg0, 9
; RESULT-NEXT: ret i32 %add
; RESULT-NEXT: }
define void @invoker_drop_invoke(i32 %arg0, ptr %arg1) personality ptr @__gxx_personality_v0 {
bb:
  %add = add i32 %arg0, 9
  %invoke = invoke i32 @has_invoke_user(ptr %arg1)
    to label %bb3 unwind label %bb1

bb1:
  landingpad { ptr, i32 }
  catch ptr null
  br label %bb3

bb3:
  %phi = phi i32 [ %invoke, %bb ], [ %add, %bb1 ]
  store i32 %phi, ptr null
  ret void
}
