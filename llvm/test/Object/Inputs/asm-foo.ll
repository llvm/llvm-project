;; Symbol foo is defined, no symvers.

module asm ".do.not.parse"

!llvm.module.flags = !{!1, !4}

!1 = !{i32 6, !"global-asm-symbols", !2}
!2 = !{!3}
!3 = !{!"foo", i32 2050}
!4 = !{i32 6, !"global-asm-symvers", !5}
!5 = !{}
