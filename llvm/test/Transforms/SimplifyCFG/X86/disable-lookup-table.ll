; RUN: opt < %s -simplifycfg -switch-to-lookup -S -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; In the presence of "-no-jump-tables"="true", simplifycfg should not convert switches to lookup tables.

; CHECK: @switch.table.bar = private unnamed_addr constant [4 x i32] [i32 55, i32 123, i32 0, i32 -1]
; CHECK-LABEL: foo
; CHECK-NOT: @switch.table.foo = private unnamed_addr constant [4 x i32] [i32 55, i32 123, i32 0, i32 -1]

define i32 @foo(i32 %c) "no-jump-tables"="true" {
entry:
  switch i32 %c, label %sw.default [
    i32 42, label %return
    i32 43, label %sw.bb1
    i32 44, label %sw.bb2
    i32 45, label %sw.bb3
  ]

sw.bb1: br label %return
sw.bb2: br label %return
sw.bb3: br label %return
sw.default: br label %return
return:
  %retval.0 = phi i32 [ 15, %sw.default ],  [ -1, %sw.bb3 ], [ 0, %sw.bb2 ], [ 123, %sw.bb1 ], [ 55, %entry ]
  ret i32 %retval.0
}


define i32 @bar(i32 %c) {
entry:
  switch i32 %c, label %sw.default [
    i32 42, label %return
    i32 43, label %sw.bb1
    i32 44, label %sw.bb2
    i32 45, label %sw.bb3
  ]

sw.bb1: br label %return
sw.bb2: br label %return
sw.bb3: br label %return
sw.default: br label %return
return:
  %retval.0 = phi i32 [ 15, %sw.default ],  [ -1, %sw.bb3 ], [ 0, %sw.bb2 ], [ 123, %sw.bb1 ], [ 55, %entry ]
  ret i32 %retval.0
}

