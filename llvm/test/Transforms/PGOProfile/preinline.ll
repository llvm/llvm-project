; RUN: opt < %s -O2 -profile-generate=default.profraw -S | FileCheck %s --check-prefix=GEN
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %i) {
entry:
; GEN: %pgocount = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc_foo
; GEN-NOT: %pgocount.i = load i64, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__profc__stdin__bar
  %call = call i32 @bar()
  %add = add nsw i32 %i, %call
  ret i32 %add
}

define internal i32 @bar() {
; check that bar is inlined into foo and eliminiated from IR.
; GEN-NOT: define internal i32 @bar
entry:
  %call = call i32 (...) @bar1()
  ret i32 %call
}

declare i32 @bar1(...)
