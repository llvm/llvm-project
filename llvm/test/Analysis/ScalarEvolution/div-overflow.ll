; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

; CHECK: -->  ((-128 * %a) /u -128)

; Don't let ScalarEvolution fold this div away.

define i8 @foo(i8 %a) {
        %t0 = shl i8 %a, 7
        %t1 = lshr i8 %t0, 7
        ret i8 %t1
}
