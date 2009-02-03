; RUN: llvm-as < %s | opt -dse | llvm-dis | grep store

define double @foo(i8* %X) {
        %X_addr = alloca i8*            ; <i8**> [#uses=2]
        store i8* %X, i8** %X_addr
        %tmp.0 = va_arg i8** %X_addr, double            ; <double> [#uses=1]
        ret double %tmp.0
}

