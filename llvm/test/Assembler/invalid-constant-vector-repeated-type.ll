; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: error: constant vector elements must begin with a type

; test for issue: (https://github.com/llvm/llvm-project/issues/222931) and PR: (https://github.com/llvm/llvm-project/pull/224680)

@g = constant <2 x i16> <123, 456>
@g2 = constant <2 x float> <1.0, 2.0>
