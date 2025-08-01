; RUN: not opt -disable-output -mtriple=nvptx64-- -mcpu=sm_90 -passes=safe-stack %s 2>&1 | FileCheck %s

; CHECK: error: no libcall available for stackprotector check fail
define void @foo(i32 %t) #0 {
  %vla = alloca i32, i32 %t, align 4
  call void @baz(ptr %vla)
  ret void
}

declare void @baz(ptr)

attributes #0 = { nounwind safestack sspstrong }
