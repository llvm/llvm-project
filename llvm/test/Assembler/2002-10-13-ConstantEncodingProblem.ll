; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s

%Domain = type { ptr, ptr }
@D = global %Domain zeroinitializer             ; <ptr> [#uses=0]

