;; Symbol bar (defined), symver bar@VER (defined).

module asm ".do.not.parse"

!llvm.module.flags = !{!1, !5}

!1 = !{i32 6, !"global-asm-symbols", !2}
!2 = !{!3, !4}
!3 = !{!"bar", i32 2050}
!4 = !{!"bar@VER", i32 2050}
!5 = !{i32 6, !"global-asm-symvers", !6}
!6 = !{!7}
!7 = !{!"bar", !"bar@VER"}
