; RUN: llvm-as < %s | llc -mtriple=arm-linux | \
; RUN:   grep {__DTOR_END__:}
; RUN: llvm-as < %s | llc -mtriple=arm-linux | \
; RUN:   grep {\\.section.\\.dtors,"aw",.progbits}

@__DTOR_END__ = internal global [1 x i32] zeroinitializer, section ".dtors"       ; <[1 x i32]*> [#uses=0]

