; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define spir_func void @foo() {
        ret void
}

define spir_kernel void @bar() {
        call spir_func void @foo( )
        call spir_kernel void @bar( )
        ret void
}
