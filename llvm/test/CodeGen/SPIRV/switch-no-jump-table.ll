; The test is to check that jump tables are not generated from switch

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpSwitch %[[#]] %[[Label:]]
; CHECK-4: OpBranch %[[Label]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @foo(i32 noundef %val) {
entry:
  switch i32 %val, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb2
    i32 2, label %sw.bb3
    i32 3, label %sw.bb4
  ]
sw.bb:
  br label %sw.epilog
sw.bb2:
  br label %sw.epilog
sw.bb3:
  br label %sw.epilog
sw.bb4:
  br label %sw.epilog
sw.epilog:
  ret void
}
