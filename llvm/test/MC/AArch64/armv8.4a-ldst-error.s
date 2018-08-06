// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// Armv8.4-A LDAPR and STLR instructions with immediate offsets
//------------------------------------------------------------------------------

STLURB   SP, [X1, #1]
ldapurb  SP, [x12, #1]
STLURB   W1, [XZR, #1]

STLURB   W1, [X10, #-257]
stlurb   w2, [x11, #256]
STLURB   W3, [SP, #-257]

ldapurb  w4, [x12, #-257]
LDAPURB  W5, [X13, #256]
LDAPURB  W6, [SP, #256]

LDAPURSB W7, [X14, #-257]
ldapursb w8, [x15, #256]
ldapursb w9, [sp, #-257]

LDAPURSB X0, [X16, #-257]
LDAPURSB X1, [X17, #256]
ldapursb x2, [sp, #256]

stlurh   w10, [x18, #-257]
STLURH   W11, [X19, #256]
STLURH   W12, [SP, #-257]

LDAPURH  W13, [X20, #-257]
ldapurh  w14, [x21, #256]
LDAPURH  W15, [SP, #256]

LDAPURSH W16, [X22, #-257]
LDAPURSH W17, [X23, #256]
ldapursh w18, [sp, #-257]

ldapursh x3, [x24, #-257]
LDAPURSH X4, [X25, #256]
LDAPURSH X5, [SP, #256]

STLUR    W19, [X26, #-257]
stlur    w20, [x27, #256]
STLUR    W21, [SP, #-257]

LDAPUR   W22, [X28, #-257]
LDAPUR   W23, [X29, #256]
ldapur   w24, [sp, #256]

ldapursw x6, [x30, #-257]
LDAPURSW X7, [X0, #256]
LDAPURSW X8, [SP, #-257]

STLUR    X9, [X1, #-257]
stlur    x10, [x2, #256]
STLUR    X11, [SP, #256]

LDAPUR   X12, [X3, #-257]
LDAPUR   X13, [X4, #256]
ldapur   x14, [sp, #-257]

//CHECK-ERROR:      error: invalid operand for instruction
//CHECK-ERROR-NEXT: STLURB   SP, [X1, #1]
//CHECK-ERROR-NEXT:          ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: ldapurb  SP, [x12, #1]
//CHECK-ERROR-NEXT:          ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: STLURB   W1, [XZR, #1]
//CHECK-ERROR-NEXT:               ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLURB   W1, [X10, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: stlurb   w2, [x11, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLURB   W3, [SP, #-257]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapurb  w4, [x12, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURB  W5, [X13, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURB  W6, [SP, #256]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSB W7, [X14, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapursb w8, [x15, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapursb w9, [sp, #-257]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSB X0, [X16, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSB X1, [X17, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapursb x2, [sp, #256]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: stlurh   w10, [x18, #-257]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLURH   W11, [X19, #256]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLURH   W12, [SP, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURH  W13, [X20, #-257]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapurh  w14, [x21, #256]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURH  W15, [SP, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSH W16, [X22, #-257]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSH W17, [X23, #256]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapursh w18, [sp, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapursh x3, [x24, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSH X4, [X25, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSH X5, [SP, #256]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLUR    W19, [X26, #-257]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: stlur    w20, [x27, #256]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLUR    W21, [SP, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPUR   W22, [X28, #-257]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPUR   W23, [X29, #256]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapur   w24, [sp, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapursw x6, [x30, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSW X7, [X0, #256]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPURSW X8, [SP, #-257]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLUR    X9, [X1, #-257]
//CHECK-ERROR-NEXT:                   ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: stlur    x10, [x2, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: STLUR    X11, [SP, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPUR   X12, [X3, #-257]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: LDAPUR   X13, [X4, #256]
//CHECK-ERROR-NEXT:                    ^
//CHECK-ERROR-NEXT: error: index must be an integer in range [-256, 255].
//CHECK-ERROR-NEXT: ldapur   x14, [sp, #-257]
//CHECK-ERROR-NEXT:                    ^
