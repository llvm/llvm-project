target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

module asm ".text"
module asm ".balign 16"
module asm ".globl foo"
module asm "foo:"
module asm "pacib     x30, x27"
module asm "retab"
module asm ".symver foo, foo@VER"
module asm ".symver foo, foo@ANOTHERVER"
module asm ".globl bar"
module asm "bar:"
module asm "pacib     x30, x27"
module asm "retab"
module asm ".symver bar, bar@VER"
module asm ".previous"

!llvm.module.flags = !{!1, !8}

!1 = !{i32 6, !"global-asm-symbols", !2}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{!"bar", i32 2050}
!4 = !{!"bar@VER", i32 2050}
!5 = !{!"foo@ANOTHERVER", i32 2050}
!6 = !{!"foo", i32 2050}
!7 = !{!"foo@VER", i32 2050}
!8 = !{i32 6, !"global-asm-symvers", !9}
!9 = !{!10, !11}
!10 = !{!"foo", !"foo@VER", !"foo@ANOTHERVER"}
!11 = !{!"bar", !"bar@VER"}

