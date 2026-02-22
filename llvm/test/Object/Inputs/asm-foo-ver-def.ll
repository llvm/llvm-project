;; Symbol foo is defined, foo@VER is a symver (also defined).

module asm ".do.not.parse"

!llvm.module.flags = !{!1, !5}

!1 = !{i32 6, !"global-asm-symbols", !2}
!2 = !{!3, !4}
!3 = !{!"foo", i32 2050}
!4 = !{!"foo@VER", i32 2050}
!5 = !{i32 6, !"global-asm-symvers", !6}
!6 = !{!7}
!7 = !{!"foo", !"foo@VER"}
