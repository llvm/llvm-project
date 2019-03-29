# RUN: not llvm-mc -triple i386-unknown-unknown %s 2> /dev/null | FileCheck %s
# RUN: not llvm-mc -triple i386-unknown-unknown %s 2>&1 > /dev/null| FileCheck %s --check-prefix=CHECK-ERROR

# CHECK: .long	1067412619
# CHECK: .long	1075000115
# CHECK: .long	1077936128
# CHECK: .long	1082549862
.single 1.2455, +2.3, 3, + 4.2

# CHECK: .long	2139095040
.single InFinIty

# CHECK: .quad	9218868437227405312
.double infinity

# CHECK: .long	4286578688
.single -iNf

# CHECK: .quad	-4503599627370496
.double -inf

# CHECK: .long	2147483647
.single nAN

# CHECK: .quad	9223372036854775807
.double NaN

# CHECK: .long  1067928519
.float 1.307

# CHECK: .quad	4617315517961601024
# CHECK: .quad	4597526701198935065
# CHECK: .quad	-4600933674317040845
.double 5, .232, -11.1

# CHECK: .quad  0
.double 0.0

# CHECK: .quad  -4570379565595099136
.double -1.2e3
# CHECK: .quad  -4690170861623122860
.double -1.2e-5
# CHECK: .quad  -4465782973978902528
.double -1.2e+10
# CHECK: .quad  4681608360884174848
.double 1e5
# CHECK: .quad  4681608360884174848
.double 1.e5
# CHECK: .quad  4611686018427387904
.double 2.
# CHECK: .quad  4611686018427387904
.double 2.e
# CHECK: .quad  4611686018427387904
.double 2.e+
# CHECK: .quad  4611686018427387904
.double 2.e-
# CHECK: .quad  -4615288898129284301
.double -1.2e
# CHECK: .quad  4621819117588971520
.double 1e1
# CHECK: .quad	4591870180066957722
.double 1e-1


# CHECK: .quad  -4570379565595099136
.double -1.2E3
# CHECK: .quad  -4690170861623122860
.double -1.2E-5
# CHECK: .quad  -4465782973978902528
.double -1.2E+10
# CHECK: .quad  4681608360884174848
.double 1E5
# CHECK: .quad  4681608360884174848
.double 1.E5
# CHECK: .quad  4611686018427387904
.double 2.E
# CHECK: .quad  4611686018427387904
.double 2.E+
# CHECK: .quad  4611686018427387904
.double 2.E-
# CHECK: .quad  -4615288898129284301
.double -1.2E
# CHECK: .quad  4621819117588971520
.double 1E1
# CHECK: .quad  4591870180066957722
.double 1E-1


#CHECK-ERROR: unexpected token in '.double' directive
.double 1e1e
# CHECK-ERROR: Invalid sign in float literal
.double 2.+1

# CHECK: .long 1310177520
.float 0x12f7.1ep+17
# CHECK: .long 1084227584
.float 0x.ap+3
# CHECK: .quad 4602678819172646912
.double 0x2.p-2
# CHECK: .long 1094713344
.float 0x3p2
# CHECK: .long 872284160
.float 0x7fp-30
# CHECK: .long 3212836864
.float -0x1.0p0

# CHECK-ERROR: invalid hexadecimal floating-point constant: expected at least one exponent digit
.float 0xa.apa

# CHECK-ERROR: invalid hexadecimal floating-point constant: expected at least one exponent digit
.double -0x1.2p+

# CHECK-ERROR: invalid hexadecimal floating-point constant: expected at least one exponent digit
.double -0x1.2p

# CHECK-ERROR: invalid hexadecimal floating-point constant: expected at least one significand digit
.float 0xp2

# CHECK-ERROR: invalid hexadecimal floating-point constant: expected at least one significand digit
.float 0x.p5

# CHECK-ERROR: error: invalid hexadecimal floating-point constant: expected exponent part 'p'
.float 0x1.2

