; RUN: opt -S -passes=pgo-instr-gen,instrprof < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Test that we use private aliases to reference function addresses inside profile data

; CHECK: @__profd_foo = private global {{.*}}, ptr @foo.1,
; CHECK-NOT: @__profd_foo = private global {{.*}}, ptr @foo,
; CHECK: @foo.1 = private alias i32 (i32), ptr @foo

define i32 @foo(i32 %0) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_foo, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_foo, align 8
; CHECK-NEXT:    ret i32 0
;
entry:
  ret i32 0
}

