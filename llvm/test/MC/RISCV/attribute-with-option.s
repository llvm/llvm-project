## When a user specifies an architecture extension which conflicts with an
## architecture attribute, we use the architecture attribute instead of the
## command line option.
##
## This test uses option '-mattr=+e' to specify the "e" extension. However,
## there is an architecture attribute in the file to specify rv32i. We will
## use rv32i to assemble the file instead of rv32e.

# RUN: llvm-mc %s -triple=riscv32 -mattr=+e -filetype=obj -o - \
# RUN:   | llvm-readobj -A - | FileCheck %s

.attribute arch, "rv32i2p0"
## Invalid operand for RV32E, because x16 is an invalid register for RV32E.
## Use RV32I to assemble, since it will not trigger an assembly error.
lui x16, 1

## Check that the architecture attribute is not overridden by the command line
## option.
# CHECK:      Tag: 5
# CHECK-NEXT: TagName: arch
# CHECK-NEXT: Value: rv32i2p0
