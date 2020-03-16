# RUN: llvm-mc -triple riscv32 -mattr=+f < %s -riscv-arch-reg-names \
# RUN:     | FileCheck -check-prefix=CHECK-NUMERIC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump --mattr=+f -d -M numeric - \
# RUN:     | FileCheck -check-prefix=CHECK-NUMERIC %s

# CHECK-NUMERIC: fsqrt.s f10, f0
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f0
fsqrt.s fa0, f0
fsqrt.s fa0, ft0

# CHECK-NUMERIC: fsqrt.s f10, f1
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f1
fsqrt.s fa0, f1
fsqrt.s fa0, ft1

# CHECK-NUMERIC: fsqrt.s f10, f2
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f2
fsqrt.s fa0, f2
fsqrt.s fa0, ft2

# CHECK-NUMERIC: fsqrt.s f10, f3
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f3
fsqrt.s fa0, f3
fsqrt.s fa0, ft3

# CHECK-NUMERIC: fsqrt.s f10, f4
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f4
fsqrt.s fa0, f4
fsqrt.s fa0, ft4

# CHECK-NUMERIC: fsqrt.s f10, f5
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f5
fsqrt.s fa0, f5
fsqrt.s fa0, ft5

# CHECK-NUMERIC: fsqrt.s f10, f6
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f6
fsqrt.s fa0, f6
fsqrt.s fa0, ft6

# CHECK-NUMERIC: fsqrt.s f10, f7
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f7
fsqrt.s fa0, f7
fsqrt.s fa0, ft7

# CHECK-NUMERIC: fsqrt.s f10, f8
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f8
fsqrt.s fa0, f8
fsqrt.s fa0, fs0

# CHECK-NUMERIC: fsqrt.s f10, f9
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f9
fsqrt.s fa0, f9
fsqrt.s fa0, fs1

# CHECK-NUMERIC: fsqrt.s f10, f10
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f10
fsqrt.s fa0, f10
fsqrt.s fa0, fa0

# CHECK-NUMERIC: fsqrt.s f10, f11
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f11
fsqrt.s fa0, f11
fsqrt.s fa0, fa1

# CHECK-NUMERIC: fsqrt.s f10, f12
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f12
fsqrt.s fa0, f12
fsqrt.s fa0, fa2

# CHECK-NUMERIC: fsqrt.s f10, f13
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f13
fsqrt.s fa0, f13
fsqrt.s fa0, fa3

# CHECK-NUMERIC: fsqrt.s f10, f14
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f14
fsqrt.s fa0, f14
fsqrt.s fa0, fa4

# CHECK-NUMERIC: fsqrt.s f10, f15
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f15
fsqrt.s fa0, f15
fsqrt.s fa0, fa5

# CHECK-NUMERIC: fsqrt.s f10, f16
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f16
fsqrt.s fa0, f16
fsqrt.s fa0, fa6

# CHECK-NUMERIC: fsqrt.s f10, f17
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f17
fsqrt.s fa0, f17
fsqrt.s fa0, fa7

# CHECK-NUMERIC: fsqrt.s f10, f18
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f18
fsqrt.s fa0, f18
fsqrt.s fa0, fs2

# CHECK-NUMERIC: fsqrt.s f10, f19
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f19
fsqrt.s fa0, f19
fsqrt.s fa0, fs3

# CHECK-NUMERIC: fsqrt.s f10, f20
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f20
fsqrt.s fa0, f20
fsqrt.s fa0, fs4

# CHECK-NUMERIC: fsqrt.s f10, f21
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f21
fsqrt.s fa0, f21
fsqrt.s fa0, fs5

# CHECK-NUMERIC: fsqrt.s f10, f22
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f22
fsqrt.s fa0, f22
fsqrt.s fa0, fs6

# CHECK-NUMERIC: fsqrt.s f10, f23
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f23
fsqrt.s fa0, f23
fsqrt.s fa0, fs7

# CHECK-NUMERIC: fsqrt.s f10, f24
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f24
fsqrt.s fa0, f24
fsqrt.s fa0, fs8

# CHECK-NUMERIC: fsqrt.s f10, f25
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f25
fsqrt.s fa0, f25
fsqrt.s fa0, fs9

# CHECK-NUMERIC: fsqrt.s f10, f26
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f26
fsqrt.s fa0, f26
fsqrt.s fa0, fs10

# CHECK-NUMERIC: fsqrt.s f10, f27
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f27
fsqrt.s fa0, f27
fsqrt.s fa0, fs11

# CHECK-NUMERIC: fsqrt.s f10, f28
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f28
fsqrt.s fa0, f28
fsqrt.s fa0, ft8

# CHECK-NUMERIC: fsqrt.s f10, f29
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f29
fsqrt.s fa0, f29
fsqrt.s fa0, ft9

# CHECK-NUMERIC: fsqrt.s f10, f30
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f30
fsqrt.s fa0, f30
fsqrt.s fa0, ft10

# CHECK-NUMERIC: fsqrt.s f10, f31
# CHECK-NUMERIC-NEXT: fsqrt.s f10, f31
fsqrt.s fa0, f31
fsqrt.s fa0, ft11
