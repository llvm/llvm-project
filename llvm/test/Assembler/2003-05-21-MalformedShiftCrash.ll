; Found by inspection of the code
; RUN: not llvm-as < %s > /dev/null 2> %t
; RUN: grep "constexpr requires integer operands" %t

@0 = global i32 ashr (float 1.0, float 2.0)
