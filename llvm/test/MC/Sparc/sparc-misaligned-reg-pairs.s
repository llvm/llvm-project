! RUN: llvm-mc %s -triple=sparc --verify
! RUN: llvm-mc %s -triple=sparcv9 --verify

!! Register pair operands must be aligned.
ldd [%i0], %o7        ! expected-error:12 {{invalid operand for instruction}}
faddd %f1, %f2, %f4   ! expected-error:7 {{invalid operand for instruction}}
faddq %f2, %f4, %f8   ! expected-error:7 {{invalid operand for instruction}}
faddq %f34, %f4, %f8  ! expected-error:7 {{invalid operand for instruction}}
ldd [%i0], %c1        ! expected-error:12 {{invalid operand for instruction}}
