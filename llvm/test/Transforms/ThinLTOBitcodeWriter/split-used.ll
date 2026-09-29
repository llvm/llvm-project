; Test to ensure that @llvm[.compiler].used is cloned to the split module for
; any globals whose defs were cloned to that module.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o %t0.bc %t
; RUN: llvm-modextract -b -n 1 -o %t1.bc %t
; RUN: llvm-dis -o - %t0.bc | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis -o - %t1.bc | FileCheck --check-prefix=M1 %s

; M0: @g1 = external global i8
; M0: @g2 = external global i8
; M0: @g3 = global i8 42
; M0: @g4 = global i8 42
; M1: @g1 = global i8 42, !type !0
; M1: @g2 = global i8 42, !type !0
; M1-NOT: @g
@g1 = global i8 42, !type !0
@g2 = global i8 42, !type !0
@g3 = global i8 42
@g4 = global i8 42

; M0: @llvm.used = appending global [2 x ptr] [ptr @g1, ptr @g3]
; M0: @llvm.compiler.used = appending global [2 x ptr] [ptr @g2, ptr @g4]
; M1: @llvm.used = appending global [1 x ptr] [ptr @g1]
; M1: @llvm.compiler.used = appending global [1 x ptr] [ptr @g2]
@llvm.used = appending global [2 x ptr] [ ptr @g1, ptr @g3]
@llvm.compiler.used = appending global [2 x ptr] [ ptr @g2, ptr @g4]

; M1: !0 = !{i32 0, !"typeid"}
!0 = !{i32 0, !"typeid"}
