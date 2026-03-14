// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+mte < %s 2>&1| FileCheck %s

irg
irg x0
irg q0, x0
irg w0, x0
irg x0, q0
irg x0, w0
irg x0, x1, q0
irg x0, x1, w0
irg x0, x1, sp
irg x0, x1, #1
irg x0, #1, x1
irg #1, x0, x1
irg x0, x1, x2, x3

// CHECK: too few operands for instruction
// CHECK-NEXT: irg
// CHECK: too few operands for instruction
// CHECK-NEXT: irg x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg q0, x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg w0, x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, q0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, w0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, x1, q0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, x1, w0
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, x1, sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, x1, #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, #1, x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg #1, x0, x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: irg x0, x1, x2, x3

addg
addg x0
addg x0, x1
addg x0, x1, #0
addg x0, x1, #1024, #0
addg x0, x1, #8, #0
addg x0, x1, #-16, #0
addg x0, x1, #0, #16
addg q0, x1, #0, #0
addg w0, x1, #0, #0
addg x0, q1, #0, #0
addg x0, w1, #0, #0
addg #0, x1, #0, #0
addg x0, #0, #0, #0
addg x0, x1, x0, #0
addg x0, x1, #0, x0
addg x0, x1, #16, #2, #99

subg
subg x0
subg x0, x1
subg x0, x1, #0
subg x0, x1, #1024, #0
subg x0, x1, #8, #0
subg x0, x1, #-16, #0
subg x0, x1, #0, #16
subg q0, x1, #0, #0
subg w0, x1, #0, #0
subg x0, q1, #0, #0
subg x0, w1, #0, #0
subg #0, x1, #0, #0
subg x0, #0, #0, #0
subg x0, x1, x0, #0
subg x0, x1, #0, x0
subg x0, x1, #16, #2, #99

// CHECK: too few operands for instruction
// CHECK-NEXT: addg
// CHECK: too few operands for instruction
// CHECK-NEXT: addg x0
// CHECK: too few operands for instruction
// CHECK-NEXT: addg x0, x1
// CHECK: too few operands for instruction
// CHECK-NEXT: addg x0, x1, #0
// CHECK: index must be a multiple of 16 in range [0, 1008]
// CHECK-NEXT: addg x0, x1, #1024, #0
// CHECK: index must be a multiple of 16 in range [0, 1008]
// CHECK-NEXT: addg x0, x1, #8, #0
// CHECK: index must be a multiple of 16 in range [0, 1008]
// CHECK-NEXT: addg x0, x1, #-16, #0
// CHECK: immediate must be an integer in range [0, 15]
// CHECK-NEXT: addg x0, x1, #0, #16
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg q0, x1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg w0, x1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg x0, q1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg x0, w1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg #0, x1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg x0, #0, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg x0, x1, x0, #0
// CHECK: immediate must be an integer in range [0, 15]
// CHECK-NEXT: addg x0, x1, #0, x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: addg x0, x1, #16, #2, #99

// CHECK: too few operands for instruction
// CHECK-NEXT: subg
// CHECK: too few operands for instruction
// CHECK-NEXT: subg x0
// CHECK: too few operands for instruction
// CHECK-NEXT: subg x0, x1
// CHECK: too few operands for instruction
// CHECK-NEXT: subg x0, x1, #0
// CHECK: index must be a multiple of 16 in range [0, 1008]
// CHECK-NEXT: subg x0, x1, #1024, #0
// CHECK: index must be a multiple of 16 in range [0, 1008]
// CHECK-NEXT: subg x0, x1, #8, #0
// CHECK: index must be a multiple of 16 in range [0, 1008]
// CHECK-NEXT: subg x0, x1, #-16, #0
// CHECK: immediate must be an integer in range [0, 15]
// CHECK-NEXT: subg x0, x1, #0, #16
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg q0, x1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg w0, x1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg x0, q1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg x0, w1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg #0, x1, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg x0, #0, #0, #0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg x0, x1, x0, #0
// CHECK: immediate must be an integer in range [0, 15]
// CHECK-NEXT: subg x0, x1, #0, x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: subg x0, x1, #16, #2, #99

gmi
gmi x0
gmi x0, x1
gmi sp, x0, x1
gmi x0, x1, sp
gmi x0, x1, x2, #1
gmi w0, x1, x2
gmi x0, w1, x2
gmi x0, x1, w2
gmi #1, x1, x2
gmi x0, #1, x2
gmi x0, x1, #1

// CHECK: too few operands for instruction
// CHECK-NEXT: gmi
// CHECK: too few operands for instruction
// CHECK-NEXT: gmi x0
// CHECK: too few operands for instruction
// CHECK-NEXT: gmi x0, x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi sp, x0, x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi x0, x1, sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi x0, x1, x2, #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi w0, x1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi x0, w1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi x0, x1, w2
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi #1, x1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi x0, #1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: gmi x0, x1, #1
stg
stg x0
stg w0
stg sp
stg #1
stg x0, [x0, #-4112]
stg x0, [x0, #4096]
stg x0, [x0, #8]
stg x0, [x0, x1]
stg x0, [w0]
stg x0, [xzr]
stg x0, [#1]

// CHECK: too few operands for instruction
// CHECK-NEXT: stg
// CHECK: too few operands for instruction
// CHECK-NEXT: stg x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg w0
// CHECK: too few operands for instruction
// CHECK-NEXT: stg sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg #1
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [x0, x1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [w0]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [xzr]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [#1]

stzg
stzg x0
stzg w0
stzg sp
stzg #1
stzg x0, [x0, #-4112]
stzg x0, [x0, #4096]
stzg x0, [x0, #8]
stzg x0, [x0, x1]
stzg x0, [w0]
stzg x0, [xzr]
stzg x0, [#1]

// CHECK: too few operands for instruction
// CHECK-NEXT: stzg
// CHECK: too few operands for instruction
// CHECK-NEXT: stzg x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg w0
// CHECK: too few operands for instruction
// CHECK-NEXT: stzg sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg #1
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [x0, x1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [w0]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [xzr]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [#1]

stg x0, [x0, #-4112]!
stg x0, [x0, #4096]!
stg x0, [x0, #8]!
stg x0, [x0, x1]!
stg x0, [w0]!
stg x0, [xzr]!
stg x0, [#1]!

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [x0, x1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [w0]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [xzr]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [#1]!

stzg x0, [x0, #-4112]!
stzg x0, [x0, #4096]!
stzg x0, [x0, #8]!
stzg x0, [x0, x1]!
stzg x0, [w0]!
stzg x0, [xzr]!
stzg x0, [#1]!

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [x0, x1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [w0]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [xzr]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [#1]!

stg x0, [x0], #-4112
stg x0, [x0], #4096
stg x0, [x0], #8
stg x0, [x0], x1
stg x0, [w0], #255
stg x0, [xzr], #255
stg x0, [#1], #255

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [x0], x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [w0], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [xzr], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: stg x0, [#1], #255

stzg x0, [x0], #-4112
stzg x0, [x0], #4096
stzg x0, [x0], #8
stzg x0, [x0], x1
stzg x0, [w0], #255
stzg x0, [xzr], #255
stzg x0, [#1], #255

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [x0], x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [w0], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [xzr], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: stzg x0, [#1], #255

st2g
st2g x0
st2g w0
st2g sp
st2g #1
st2g x0, [x0, #-4112]
st2g x0, [x0, #4096]
st2g x0, [x0, #8]
st2g x0, [x0, x1]
st2g x0, [w0]
st2g x0, [xzr]
st2g x0, [#1]

// CHECK: too few operands for instruction
// CHECK-NEXT: st2g
// CHECK: too few operands for instruction
// CHECK-NEXT: st2g x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g w0
// CHECK: too few operands for instruction
// CHECK-NEXT: st2g sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g #1
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [x0, x1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [w0]
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [xzr]
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [#1]

stz2g
stz2g x0
stz2g w0
stz2g sp
stz2g #1
stz2g x0, [x0, #-4112]
stz2g x0, [x0, #4096]
stz2g x0, [x0, #8]
stz2g x0, [x0, x1]
stz2g x0, [w0]
stz2g x0, [xzr]
stz2g x0, [#1]

// CHECK: too few operands for instruction
// CHECK-NEXT: stz2g
// CHECK: too few operands for instruction
// CHECK-NEXT: stz2g x0
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g w0
// CHECK: too few operands for instruction
// CHECK-NEXT: stz2g sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g #1
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [x0, x1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [w0]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [xzr]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [#1]

st2g x0, [x0, #-4112]!
st2g x0, [x0, #4096]!
st2g x0, [x0, #8]!
st2g x0, [x0, x1]!
st2g x0, [w0, #256]!
st2g x0, [xzr, #256]!
st2g x0, [#1, #256]!

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [x0, x1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [w0, #256]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [xzr, #256]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [#1, #256]!

stz2g x0, [x0, #-4112]!
stz2g x0, [x0, #4096]!
stz2g x0, [x0, #8]!
stz2g x0, [x0, x1]!
stz2g x0, [w0, #255]!
stz2g x0, [xzr, #255]!
stz2g x0, [#1, #255]!

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [x0, x1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [w0, #255]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [xzr, #255]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [#1, #255]!

st2g x0, [x0], #-4112
st2g x0, [x0], #4096
st2g x0, [x0], #8
st2g x0, [x0], x1
st2g x0, [w0], #255
st2g x0, [xzr], #255
st2g x0, [#1], #255

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [x0], x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [w0], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [xzr], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: st2g x0, [#1], #255

stz2g x0, [x0], #-4112
stz2g x0, [x0], #4096
stz2g x0, [x0], #8
stz2g x0, [x0], x1
stz2g x0, [w0], #255
stz2g x0, [xzr], #255
stz2g x0, [#1], #255

// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #-4112
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #4096
// CHECK: index must be a multiple of 16 in range [-4096, 4080]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [x0], x1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [w0], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [xzr], #255
// CHECK: invalid operand for instruction
// CHECK-NEXT: stz2g x0, [#1], #255

stgp sp, x1, [x2, #16]
stgp x0, sp, [x2, #16]
stgp x0, x1, [xzr, #16]
stgp x0, x1, [x2, #-1040]
stgp x0, x1, [x2, #1024]
stgp x0, x1, [x2, #8]
stgp x0, x1, [x2, x3]
stgp w0, x1, [x2, #1]
stgp x0, w1, [x2, #1]
stgp x0, x1, [w2, #1]
stgp #1, x1, [x3, #1]
stgp x0, #1, [x3, #1]
stgp x0, x1, [#1, #1]

// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [xzr
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #-1040
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #1024
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [x2, x3]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp w0, x1, [x2, #1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, w1, [x2, #1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [w2, #1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp #1, x1, [x3, #1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, #1, [x3, #1]
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [#1, #1]

stgp sp, x1, [x2, #16]!
stgp x0, sp, [x2, #16]!
stgp x0, x1, [xzr, #16]!
stgp x0, x1, [x2, #-1040]!
stgp x0, x1, [x2, #1024]!
stgp x0, x1, [x2, #8]!
stgp x0, x1, [x2, x3]!
stgp w0, x1, [x2, #1]!
stgp x0, w1, [x2, #1]!
stgp x0, x1, [w2, #1]!
stgp #1, x1, [x3, #1]!
stgp x0, #1, [x3, #1]!
stgp x0, x1, [#1, #1]!

// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [xzr
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #-1040
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #1024
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [x2, x3]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp w0, x1, [x2, #1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, w1, [x2, #1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [w2, #1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp #1, x1, [x3, #1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, #1, [x3, #1]!
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [#1, #1]!

stgp sp, x1, [x2], #16
stgp x0, sp, [x2], #16
stgp x0, x1, [xzr], #16
stgp x0, x1, [x2], #-1040
stgp x0, x1, [x2], #1024
stgp x0, x1, [x2], #8
stgp x0, x1, [x2], x3
stgp w0, x1, [x2], #1
stgp x0, w1, [x2], #1
stgp x0, x1, [w2], #1
stgp #1, x1, [x2], #1
stgp x0, #1, [x2], #1
stgp x0, x1, [#1], #1

// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, sp
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [xzr
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #-1040
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #1024
// CHECK: index must be a multiple of 16 in range [-1024, 1008]
// CHECK-NEXT: #8
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [x2], x3
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp w0, x1, [x2], #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, w1, [x2], #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [w2], #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp #1, x1, [x2], #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, #1, [x2], #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: stgp x0, x1, [#1], #1

mrs tco
mrs gcr_el1
mrs rgsr_el1
mrs tfsr_el1
mrs tfsr_el2
mrs tfsr_el3
mrs tfsr_el12
mrs tfsre0_el1
mrs gmid_el1

// CHECK:      invalid operand for instruction
// CHECK-NEXT: tco
// CHECK:      invalid operand for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsre0_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: gmid_el1

mrs tco, #0
mrs tco, x0
mrs gcr_el1, x1
mrs rgsr_el1, x2
mrs tfsr_el1, x3
mrs tfsr_el2, x4
mrs tfsr_el3, x5
mrs tfsr_el12, x6
mrs tfsre0_el1, x7
mrs gmid_el1, x7

// CHECK:      invalid operand for instruction
// CHECK-NEXT: tco, #0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tco, x0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsre0_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: gmid_el1

msr tco
msr gcr_el1
msr rgsr_el1
msr tfsr_el1
msr tfsr_el2
msr tfsr_el3
msr tfsr_el12
msr tfsre0_el1
msr gmid_el1

// CHECK:      too few operands for instruction
// CHECK-NEXT: tco
// CHECK:      too few operands for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      too few operands for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      too few operands for instruction
// CHECK-NEXT: tfsre0_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: gmid_el1

msr x0, tco
msr x1, gcr_el1
msr x2, rgsr_el1
msr x3, tfsr_el1
msr x4, tfsr_el2
msr x5, tfsr_el3
msr x6, tfsr_el12
msr x7, tfsre0_el1
msr x7, gmid_el1

// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tco
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: gcr_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: rgsr_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el2
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el3
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsr_el12
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: tfsre0_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: gmid_el1

msr gmid_el1, x7

// CHECK:      expected writable system register or pstate
// CHECK-NEXT: gmid_el1

// Among the system registers added by MTE, only TCO can be used with MSR (imm).
// The rest can only be used with MSR (reg).
msr gcr_el1, #1
msr rgsr_el1, #2
msr tfsr_el1, #3
msr tfsr_el2, #4
msr tfsr_el3, #5
msr tfsr_el12, #6
msr tfsre0_el1, #7
// Is read only
msr gmid_el1, #7

// CHECK:      invalid operand for instruction
// CHECK-NEXT: gcr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rgsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el1
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el3
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsr_el12
// CHECK:      invalid operand for instruction
// CHECK-NEXT: tfsre0_el1
// CHECK:      expected writable system register or pstate
// CHECK-NEXT: gmid_el1

// Xd cannot be the stack pointer, the rest can
subps sp, x0, x1
subp  sp, x2, x3
subp  w0, x1, x2
subp  x0, w1, x2
subp  x0, x1, w2
subps w0, x1, x2
subps x0, w1, x2
subps x0, x1, w2
subp  #1, x1, x2
subp  x0, #1, x2
subp  x0, x1, #1
subps #1, x1, x2
subps x0, #1, x2
subps x0, x1, #1

// CHECK:      invalid operand for instruction
// CHECK-NEXT: sp
// CHECK:      invalid operand for instruction
// CHECK-NEXT: sp
// CHECK:      invalid operand for instruction
// CHECK-NEXT: subp  w0, x1, x2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: subp  x0, w1, x2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: subp  x0, x1, w2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: subps w0, x1, x2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: subps x0, w1, x2
// CHECK:      invalid operand for instruction
// CHECK-NEXT: subps x0, x1, w2
// CHECK: invalid operand for instruction
// CHECK-NEXT: subp  #1, x1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: subp  x0, #1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: subp  x0, x1, #1
// CHECK: invalid operand for instruction
// CHECK-NEXT: subps #1, x1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: subps x0, #1, x2
// CHECK: invalid operand for instruction
// CHECK-NEXT: subps x0, x1, #1

subps x0, x1, x2, x3
subp  x0, x1, x2, x3

// CHECK: invalid operand for instruction
// CHECK-NEXT: subps x0, x1, x2, x3
// CHECK: invalid operand for instruction
// CHECK-NEXT: subp x0, x1, x2, x3

subps
subp
cmpp
subps x0
subp  x0
cmpp  x0
subps x0, x1
subp  x0, x1

// CHECK:      too few operands for instruction
// CHECK-NEXT: subps
// CHECK:      too few operands for instruction
// CHECK-NEXT: subp
// CHECK:      too few operands for instruction
// CHECK-NEXT: cmpp
// CHECK:      too few operands for instruction
// CHECK-NEXT: subps x0
// CHECK:      too few operands for instruction
// CHECK-NEXT: subp x0
// CHECK:      too few operands for instruction
// CHECK-NEXT: cmpp x0
// CHECK:      too few operands for instruction
// CHECK-NEXT: subps x0, x1
// CHECK:      too few operands for instruction
// CHECK-NEXT: subp x0, x1

ldg sp, [x0, #0]
ldg x0, [x0, x0]
ldg x0, [x0, #4096]
ldg x0, [x0, #-4112]
ldg #1, [x0, #255]
ldg x0, [#1, #255]

// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg sp, [x0, #0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg x0, [x0, x0]
// CHECK:      index must be a multiple of 16 in range [-4096, 4080].
// CHECK-NEXT: ldg x0, [x0, #4096]
// CHECK:      index must be a multiple of 16 in range [-4096, 4080].
// CHECK-NEXT: ldg x0, [x0, #-4112]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg #1, [x0, #255]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg x0, [#1, #255]

ldg
ldg x0
ldg x0, [#0]
ldg w0, [x1]
ldg x0, [w1]

// CHECK:      too few operands for instruction
// CHECK-NEXT: ldg
// CHECK:      too few operands for instruction
// CHECK-NEXT: ldg x0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg x0, [#0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg w0, [x1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldg x0, [w1]

ldgm
ldgm x0
ldgm w0, [x1]
ldgm x0, [w1]
ldgm #1, [x1]
ldgm x0, [#1]
ldgm sp, [x0]

// CHECK:      too few operands for instruction
// CHECK-NEXT: ldgm
// CHECK:      too few operands for instruction
// CHECK-NEXT: ldgm x0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldgm w0, [x1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldgm x0, [w1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldgm #1, [x1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldgm x0, [#1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: ldgm sp, [x0]

stgm
stgm x0
stgm sp, [x0]
stgm w0, [x0]
stgm x0, [w0]
stgm #1, [x1]
stgm x0, [#1]

// CHECK:      too few operands for instruction
// CHECK-NEXT: stgm
// CHECK:      too few operands for instruction
// CHECK-NEXT: stgm x0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stgm sp, [x0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stgm w0, [x0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stgm x0, [w0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stgm #1, [x1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stgm x0, [#1]

stzgm
stzgm x0
stzgm sp, [x0]
stzgm w0, [x0]
stzgm x0, [w0]
stzgm #1, [x1]
stzgm x0, [#1]

// CHECK:      too few operands for instruction
// CHECK-NEXT: stzgm
// CHECK:      too few operands for instruction
// CHECK-NEXT: stzgm x0
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stzgm sp, [x0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stzgm w0, [x0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stzgm x0, [w0]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stzgm #1, [x1]
// CHECK:      invalid operand for instruction
// CHECK-NEXT: stzgm x0, [#1]
