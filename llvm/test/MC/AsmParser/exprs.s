// RUN: llvm-mc -triple i386-apple-darwin %s

.macro check_expr
  .if ($0) != ($1)
        .abort Unexpected $0 != $1.
  .endif
.endmacro

        .text
g:
h:
j:
k:
        .data
        check_expr !1 + 2, 2
        check_expr !0, 1
        check_expr ~0, -1
        check_expr -1, ~0
        check_expr +1, 1
        check_expr 1 + 2, 3
        check_expr 1 & 3, 1
        check_expr 4 / 2, 2
        check_expr 4 / -2, -2
        check_expr 1 == 1, -1
        check_expr 1 == 0, 0
        check_expr 1 > 0, -1
        check_expr 1 >= 1, -1
        check_expr 1 < 2, -1
        check_expr 1 <= 1, -1
        check_expr 4 % 3, 1
        check_expr 2 * 2, 4
        check_expr 2 != 2, 0
        check_expr 2 <> 2, 0
        check_expr 1 | 2, 3
        check_expr 1 << 1, 2
        check_expr 2 >> 1, 1
        check_expr (~0 >> 62), 3
        check_expr 3 - 2, 1
        check_expr 1 ^ 3, 2
        check_expr 1 && 2, 1
        check_expr 3 && 0, 0
        check_expr 0 && 1, 0
        check_expr 1 || 2, 1
        check_expr 0 || 1, 1
        check_expr 0 || 0, 0
        check_expr 1 + 2 < 3 + 4, -1
        check_expr 1 << 8 - 1, 128
        check_expr 3 * 9 - 2 * 9 + 1, 10

        .set c, 10
        check_expr c + 1, 11

        d = e + 10
        .long d

        f = g - h + 5
        .long f

        i = (j + 10) - (k + 2)
        .long i

        l = m - n + 4

        .text
m:
n:
        nop


        movw	$8, (42)+66(%eax)

// "." support:
_f0:
L0:
        jmp L1
        .long . - L0
L1:
        jmp A
        .long . - L1
