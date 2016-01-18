; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/testlink.ll > %t2.bc
; RUN: llvm-link %t.bc %t.bc %t2.bc -o %t1.bc
; RUN: llvm-dis < %t1.bc | FileCheck %s
; CHECK: kallsyms_names = extern_weak
; CHECK: Inte = global i32
; CHECK: MyVar = external global i32

@kallsyms_names = extern_weak global [0 x i8]		; <[0 x i8]*> [#uses=0]
@MyVar = extern_weak global i32		; <i32*> [#uses=0]
@Inte = extern_weak global i32		; <i32*> [#uses=0]

