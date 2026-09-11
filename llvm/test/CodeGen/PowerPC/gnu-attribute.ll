; RUN: split-file %s %t
; RUN: llc -mtriple=powerpc64le < %t/fp128.ll | FileCheck %s --check-prefix=IEEE
; RUN: llc -mtriple=powerpc64le < %t/ppc_fp128.ll | FileCheck %s --check-prefix=IBM
; RUN: llc -mtriple=powerpc64le < %t/double.ll | FileCheck %s --check-prefix=DBL
; RUN: llc -mtriple=powerpc64le < %t/none.ll | FileCheck %s --check-prefix=NONE

;--- fp128.ll
!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 1, !"long-double-type", !"fp128"}
; IEEE: .gnu_attribute 4, 13

;--- ppc_fp128.ll
!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 1, !"long-double-type", !"ppc_fp128"}
; IBM: .gnu_attribute 4, 5

;--- double.ll
!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 1, !"long-double-type", !"double"}
; DBL: .gnu_attribute 4, 9

;--- none.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
; NONE-NOT: .gnu_attribute 4,
