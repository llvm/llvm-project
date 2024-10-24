; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto --thinlto-action=run %t.o
; RUN: llvm-objdump --disassemble-symbols=._start %t.o.thinlto.o | FileCheck %s --check-prefix=CHECK-LARGE

target datalayout = "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64-ibm-aix7.2.0.0"

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"Code Model", i32 4}
!3 = !{i32 1, !"EnableSplitLTOUnit", i32 0}

@data = internal constant [0 x i32] []

define ptr @_start() nounwind readonly {
entry:
; CHECK-LARGE-LABEL:  <._start>:
; CHECK-LARGE-NEXT: addis 3, 2, 0
; CHECK-LARGE-NEXT: ld 3, 0(3)

    ret ptr @data
}
