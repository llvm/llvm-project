; RUN: opt -S -O2 < %s | FileCheck %s -check-prefix=DISABLE
; RUN: opt -S -hexagon-emit-lookup-tables=true -O2 < %s | FileCheck %s -check-prefix=DISABLE
; RUN: opt -S -hexagon-emit-lookup-tables=false -O2 < %s | FileCheck %s -check-prefix=DISABLE
; The attribute "no-jump-tables"="true" disables the generation of switch generated lookup tables

; DISABLE-NOT: @{{.*}} = private unnamed_addr constant [6 x i32] [i32 9, i32 20, i32 14, i32 22, i32 12, i32 5]

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; Function Attrs: noinline nounwind
define i32 @foo(i32 %c) #0 {
entry:
  switch i32 %c, label %sw.default [
    i32 42, label %return
    i32 43, label %sw.bb1
    i32 44, label %sw.bb2
    i32 45, label %sw.bb3
    i32 46, label %sw.bb4
    i32 47, label %sw.bb5
    i32 48, label %sw.bb6
  ]

sw.bb1: br label %return
sw.bb2: br label %return
sw.bb3: br label %return
sw.bb4: br label %return
sw.bb5: br label %return
sw.bb6: br label %return
sw.default: br label %return
return:
  %retval.0 = phi i32 [ 15, %sw.default ], [ 1, %sw.bb6 ], [ 62, %sw.bb5 ], [ 27, %sw.bb4 ], [ -1, %sw.bb3 ], [ 0, %sw.bb2 ], [ 123, %sw.bb1 ], [ 55, %entry ]
  ret i32 %retval.0
}

attributes #0 = { noinline nounwind "no-jump-tables"="true"}
