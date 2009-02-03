; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/testlink1.ll > %t2.bc
; RUN: llvm-link %t.bc %t.bc %t2.bc -o %t1.bc -f
; RUN: llvm-dis < %t1.bc | grep {kallsyms_names = extern_weak}
; RUN: llvm-dis < %t1.bc | grep {MyVar = external global i32}
; RUN: llvm-dis < %t1.bc | grep {Inte = global i32}

@kallsyms_names = extern_weak global [0 x i8]		; <[0 x i8]*> [#uses=0]
@MyVar = extern_weak global i32		; <i32*> [#uses=0]
@Inte = extern_weak global i32		; <i32*> [#uses=0]

