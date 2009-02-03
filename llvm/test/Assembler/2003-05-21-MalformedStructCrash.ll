; Found by inspection of the code
; RUN: not llvm-as < %s  > /dev/null |& grep {constant expression type mismatch}

global {} { i32 7, float 1.0, i32 7, i32 8 }
