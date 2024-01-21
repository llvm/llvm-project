; RUN: not llc -march=nvptx < %s 2>&1 | FileCheck %s
; RUN: not llc -march=nvptx64 < %s 2>&1 | FileCheck %s

; CHECK: in function test_dynamic_stackalloc{{.*}}: dynamic alloca unsupported by NVPTX backend

define void @test_dynamic_stackalloc(i64 %n) {
  %alloca = alloca i32, i64 %n
  store volatile i32 0, ptr %alloca
  ret void
}
