; RUN: not opt -disable-output -mtriple=nvptx64-- -safestack-use-pointer-address -mcpu=sm_90 -passes='require<libcall-lowering-info>,safe-stack' %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: error: no libcall available for safestack pointer address
define void @foo(i32 %t) #0 {
  %vla = alloca i32, i32 %t, align 4
  call void @baz(ptr %vla)
  ret void
}

declare void @baz(ptr)

attributes #0 = { nounwind safestack sspstrong }
