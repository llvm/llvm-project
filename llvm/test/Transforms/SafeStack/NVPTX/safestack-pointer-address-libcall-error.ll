; RUN: not opt -disable-output -mtriple=nvptx64-- -safestack-use-pointer-address -mcpu=sm_90 -passes=safe-stack %s 2>&1 | FileCheck -check-prefix=ERR1 %s
; RUN: not opt -disable-output -mtriple=nvptx64-unknown-android -mcpu=sm_90 -passes=safe-stack %s 2>&1 | FileCheck -check-prefix=ERR2 %s

; ERR1: error: no libcall available for safestack pointer address
; ERR2: error: no libcall available for stackprotector check fail
define void @foo(i32 %t) #0 {
  %vla = alloca i32, i32 %t, align 4
  call void @baz(ptr %vla)
  ret void
}

declare void @baz(ptr)

attributes #0 = { nounwind safestack sspstrong }
