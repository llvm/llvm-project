; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s

; CHECK: test_frameindex_cmp:
; CHECK: cmn sp, #{{[0-9]+}}
define void @test_frameindex_cmp() {
  %stack = alloca i8
  %stack.int = ptrtoint i8* %stack to i64
  %cmp = icmp ne i64 %stack.int, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  call void @bar()
  ret void

bb2:
  ret void
}

declare void @bar()
