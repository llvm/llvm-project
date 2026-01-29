; RUN: opt < %s -S | FileCheck %s
!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"openmp", i32 51}
!2 = !{!"clang version 19.0.0 (myvendor clang version 99.99.0.59070 deadbabe asserts)"}
