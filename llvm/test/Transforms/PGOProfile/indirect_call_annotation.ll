; RUN: llvm-profdata merge %S/Inputs/indirect_call.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=VP-ANNOTATION
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = common global ptr null, align 8

define i32 @func1(i32 %x) {
entry:
  ret i32 %x
}

define i32 @func2(i32 %x) {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define i32 @func3(i32 %x) {
entry:
  %add = add nsw i32 %x, 3
  ret i32 %add
}

define i32 @bar(i32 %i) {
entry:
  %tmp = load ptr, ptr @foo, align 8
  %call = call i32 %tmp(i32 %i)
; VP-ANNOTATION: %call = call i32 %tmp(i32 %i)
; VP-ANNOTATION-SAME: !prof ![[VP:[0-9]+]]
; VP-ANNOTATION: ![[VP]] = !{!"VP", i32 0, i64 140, i64 -4377547752858689819, i64 80, i64 -2545542355363006406, i64 40, i64 -6929281286627296573, i64 20}
  ret i32 %call
}


