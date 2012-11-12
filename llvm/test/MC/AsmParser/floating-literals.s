# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: .long	1067412619
# CHECK: .long	1075000115
# CHECK: .long	1077936128
# CHECK: .long	1082549862
.single 1.2455, +2.3, 3, + 4.2

# CHECK: .long	2139095040
.single InFinIty

# CHECK: .long	4286578688
.single -iNf

# CHECK: .long	2147483647
.single nAN

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

// APFloat should reject these with an error, not crash:
//.double -1.2e+
//.double -1.2e
