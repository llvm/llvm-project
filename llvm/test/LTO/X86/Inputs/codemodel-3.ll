target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"Code Model", i32 1}

%struct.rtx_def = type { i16, i16 }

define void @bar(ptr %a, i8 %b, i32 %c) {
  call void  @llvm.memset.p0.rtx_def.i32(ptr align 4 %a, i8 %b, i32 %c, i1 true)
  ret void
}

declare void @llvm.memset.p0.rtx_def.i32(ptr, i8, i32, i1)
