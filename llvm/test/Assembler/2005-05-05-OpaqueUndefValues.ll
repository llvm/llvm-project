; RUN: llvm-as < %s | llvm-dis | llvm-as > /dev/null
; RUN: verify-uselistorder %s

%t = type opaque
@x = global %t undef
