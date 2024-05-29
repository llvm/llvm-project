; Found by inspection of the code
; RUN: not llvm-as < %s > /dev/null 2> %t
; RUN: grep "constexpr requires integer or integer vector operands" %t

@0 = global i32 shl (float 1.0, float 2.0)
