; RUN: not opt -passes='require<libcall-lowering-info>,safe-stack' -mtriple=x86_64-pc-linux-gnu -safestack-use-pointer-address -disable-output %s 2>&1 | FileCheck %s

; __safestack_pointer_address is only available on Android.

; CHECK: error: no location available for safestack pointer address

define void @foo() nounwind uwtable safestack {
  %a = alloca i8, align 8
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)
