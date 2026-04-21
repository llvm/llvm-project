target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64"

module asm ".text"
module asm ".balign 16"
module asm ".globl bar"
module asm "bar:"
module asm "  nop"
module asm ".symver bar, bar@VER"
module asm ".previous"

!llvm.module.flags = !{!0, !1, !5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 6, !"global-asm-symbols", !2}
!2 = !{!3, !4}
!3 = !{!"bar", i32 2050}
!4 = !{!"bar@VER", i32 2050}
!5 = !{i32 6, !"global-asm-symvers", !6}
!6 = !{!7}
!7 = !{!"bar", !"bar@VER"}

