target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64"

module asm ".text"
module asm ".balign 16"
module asm ".globl foo"
module asm "foo:"
module asm "  nop"
module asm ".previous"

!llvm.module.flags = !{!0, !1, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 6, !"global-asm-symbols", !2}
!2 = !{!3}
!3 = !{!"foo", i32 2050}
!4 = !{i32 6, !"global-asm-symvers", !5}
!5 = !{}

