; RUN: llvm-as < %s | llvm-dis | grep "llvm.stuff = "

;; Simple NamedMDNode
!0 = !{i32 42}
!1 = !{!"foo"}
!llvm.stuff = !{!0, !1}

!samename = !{!0, !1}
declare void @samename()
