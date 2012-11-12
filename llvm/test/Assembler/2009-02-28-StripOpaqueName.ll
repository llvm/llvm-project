; RUN: opt < %s -strip -S | llvm-as | llvm-dis

; Stripping the name from A should not break references to it.
%A = type opaque
@g1 = external global %A
@g2 = global %A* @g1
