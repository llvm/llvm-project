; RUN: llvm-as < %s | llvm-dis | llvm-as > /dev/null

%t = type opaque
@x = global %t undef
