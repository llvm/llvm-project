; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @init1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @init2, ptr null }]
@llvm.global_dtors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @destruct1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @destruct2, ptr null }]

define i32 @extFunc() {
entry:
  ret i32 3
}

define internal void @init1() {
  ret void
}

define internal void @destruct1() {
  ret void
}

define internal void @init2() {
  ret void
}

define internal void @destruct2() {
  ret void
}

; CHECK:       .lglobl	init1[DS]
; CHECK:       .lglobl	.init1
; CHECK:       .csect init1[DS]
; CHECK: __sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0: # @init1
; CHECK: .init1:
; CHECK: .__sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0:
; CHECK:       .lglobl	destruct1[DS]
; CHECK:       .lglobl	.destruct1
; CHECK:       .csect destruct1[DS]
; CHECK: __sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0: # @destruct1
; CHECK: .destruct1:
; CHECK: .__sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0:
; CHECK:       .lglobl	init2[DS]
; CHECK:       .lglobl	.init2
; CHECK:       .csect init2[DS]
; CHECK: __sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1: # @init2
; CHECK: .init2:
; CHECK: .__sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1:
; CHECK:       .lglobl	destruct2[DS]
; CHECK:       .lglobl	.destruct2
; CHECK:       .csect destruct2[DS]
; CHECK: __sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1: # @destruct2
; CHECK: .destruct2:
; CHECK: .__sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1:

; CHECK: 	.globl	__sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0
; CHECK: 	.globl	.__sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0
; CHECK: 	.globl	__sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1
; CHECK: 	.globl	.__sinit80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1
; CHECK: 	.globl	__sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0
; CHECK: 	.globl	.__sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_0
; CHECK: 	.globl	__sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1
; CHECK: 	.globl	.__sterm80000000_clang_ac404299654d2af7eae71e75c17f7c9b_1
