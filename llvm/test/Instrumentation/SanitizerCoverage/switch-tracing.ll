; Test -sanitizer-coverage-trace-compares=1 (instrumenting a switch)
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-compares=1  -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
declare void @_Z3bari(i32)
define void @foo(i32 %x) {
entry:
; CHECK: __sancov_gen_cov_switch_values = internal global [5 x i64] [i64 3, i64 32, i64 1, i64 101, i64 1001]
; CHECK: [[TMP:%[0-9]*]] = zext i32 %x to i64
; CHECK-NEXT: call void @__sanitizer_cov_trace_switch(i64 [[TMP]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* @__sancov_gen_cov_switch_values, i32 0, i32 0))
  switch i32 %x, label %sw.epilog [
    i32 1, label %sw.bb
    i32 1001, label %sw.bb.1
    i32 101, label %sw.bb.2
  ]

sw.bb:                                            ; preds = %entry
  tail call void @_Z3bari(i32 4)
  br label %sw.epilog

sw.bb.1:                                          ; preds = %entry
  tail call void @_Z3bari(i32 5)
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry
  tail call void @_Z3bari(i32 6)
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb.2, %sw.bb.1, %sw.bb
  ret void
}

define void @fooi72(i72 %x) {
entry:
  switch i72 %x, label %sw.epilog [
    i72 1, label %sw.bb
    i72 101, label %sw.bb.1
    i72 1001, label %sw.bb.2
  ]

sw.bb:                                            ; preds = %entry
  tail call void @_Z3bari(i32 4)
  br label %sw.epilog

sw.bb.1:                                          ; preds = %entry
  tail call void @_Z3bari(i32 5)
  br label %sw.epilog

sw.bb.2:                                          ; preds = %entry
  tail call void @_Z3bari(i32 6)
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb.2, %sw.bb.1, %sw.bb
  ret void
}
