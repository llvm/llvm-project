; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/linknamedmdnode2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc -S | grep "!llvm.stuff = !{!0, !1}"

!0 = metadata !{i32 42}
!llvm.stuff = !{!0}
