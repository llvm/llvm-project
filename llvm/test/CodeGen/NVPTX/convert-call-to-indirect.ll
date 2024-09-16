; RUN: llc < %s -march=nvptx64 -mcpu=sm_90 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_90 | %ptxas-verify %}

%struct.64 = type <{ i64 }>
declare i64 @callee(ptr %p);
declare i64 @callee_variadic(ptr %p, ...);

define %struct.64 @test_return_type_mismatch(ptr %p) {
; CHECK-LABEL: test_return_type_mismatch(
; CHECK:         .param .align 1 .b8 retval0[8];
; CHECK-NEXT:    prototype_0 : .callprototype (.param .align 1 .b8 _[8]) _ (.param .b64 _);
; CHECK-NEXT:    call (retval0),
; CHECK-NEXT:    %rd
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    )
; CHECK-NEXT:    , prototype_0;
  %ret = call %struct.64 @callee(ptr %p)
  ret %struct.64 %ret
}

define i64 @test_param_type_mismatch(ptr %p) {
; CHECK-LABEL: test_param_type_mismatch(
; CHECK:         .param .b64 retval0;
; CHECK-NEXT:    prototype_1 : .callprototype (.param .b64 _) _ (.param .b64 _);
; CHECK-NEXT:    call (retval0),
; CHECK-NEXT:    %rd
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    )
; CHECK-NEXT:    , prototype_1;
  %ret = call i64 @callee(i64 7)
  ret i64 %ret
}

define i64 @test_param_count_mismatch(ptr %p) {
; CHECK-LABEL: test_param_count_mismatch(
; CHECK:         .param .b64 retval0;
; CHECK-NEXT:    prototype_2 : .callprototype (.param .b64 _) _ (.param .b64 _, .param .b64 _);
; CHECK-NEXT:    call (retval0),
; CHECK-NEXT:    %rd
; CHECK-NEXT:    (
; CHECK-NEXT:    param0,
; CHECK-NEXT:    param1
; CHECK-NEXT:    )
; CHECK-NEXT:    , prototype_2;
  %ret = call i64 @callee(ptr %p, i64 7)
  ret i64 %ret
}

define %struct.64 @test_return_type_mismatch_variadic(ptr %p) {
; CHECK-LABEL: test_return_type_mismatch_variadic(
; CHECK:         .param .align 1 .b8 retval0[8];
; CHECK-NEXT:    prototype_3 : .callprototype (.param .align 1 .b8 _[8]) _ (.param .b64 _);
; CHECK-NEXT:    call (retval0),
; CHECK-NEXT:    %rd
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    )
; CHECK-NEXT:    , prototype_3;
  %ret = call %struct.64 (ptr, ...) @callee_variadic(ptr %p)
  ret %struct.64 %ret
}

define i64 @test_param_type_mismatch_variadic(ptr %p) {
; CHECK-LABEL: test_param_type_mismatch_variadic(
; CHECK:         .param .b64 retval0;
; CHECK-NEXT:    call.uni (retval0),
; CHECK-NEXT:    callee_variadic
; CHECK-NEXT:    (
; CHECK-NEXT:    param0,
; CHECK-NEXT:    param1
; CHECK-NEXT:    )
  %ret = call i64 (ptr, ...) @callee_variadic(ptr %p, i64 7)
  ret i64 %ret
}

define i64 @test_param_count_mismatch_variadic(ptr %p) {
; CHECK-LABEL: test_param_count_mismatch_variadic(
; CHECK:         .param .b64 retval0;
; CHECK-NEXT:    call.uni (retval0),
; CHECK-NEXT:    callee_variadic
; CHECK-NEXT:    (
; CHECK-NEXT:    param0,
; CHECK-NEXT:    param1
; CHECK-NEXT:    )
  %ret = call i64 (ptr, ...) @callee_variadic(ptr %p, i64 7)
  ret i64 %ret
}
