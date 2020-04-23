; Test -sanitizer-coverage-inline-bool-flag=1
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-inline-bool-flag=1  -S | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-bool-flag=1  -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo() {
entry:
; CHECK: @__sancov_gen_ = private global [1 x i1] zeroinitializer, section "__sancov_bools", comdat($foo), align 1, !associated !0
; CHECK: store i1 true, i1* getelementptr inbounds ([1 x i1], [1 x i1]* @__sancov_gen_, i64 0, i64 0), align 1, !nosanitize !1
  ret void
}
; CHECK: call void @__sanitizer_cov_bool_flag_init(i1* bitcast (i1** @__start___sancov_bools to i1*), i1* bitcast (i1** @__stop___sancov_bools to i1*))
