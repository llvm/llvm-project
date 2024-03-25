
# RUN: not llvm-mc -triple powerpc64-unknown-unknown < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple powerpc64le-unknown-unknown < %s 2> %t
# RUN: FileCheck < %t %s

# Register operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: add 32, 32, 32
              add 32, 32, 32

# CHECK: error: invalid register name
# CHECK-NEXT: add %r32, %r32, %r32
              add %r32, %r32, %r32

# TLS register operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: add 3, symbol@tls, 4
              add 3, symbol@tls, 4

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: subf 3, 4, symbol@tls
              subf 3, 4, symbol@tls

# Unsigned 1-bit immediate operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: mtmsr 1, 2
              mtmsr 1, 2

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: mtmsrd 1, 2
              mtmsrd 1, 2

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: mtfsf 1, 2, 2, 1
              mtfsf 1, 2, 2, 1

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: mtfsf. 1, 2, 2, 1
              mtfsf. 1, 2, 2, 1

# Unsigned 2-bit immediate operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: darn 1, 4
              darn 1, 4

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: wait 4
              wait 4

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: sync 8
              sync 8

# Unsigned 3-bit immediate operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: dcbf 0, 1, 8
              dcbf 0, 1, 8

# Signed 16-bit immediate operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: addi 1, 0, -32769
              addi 1, 0, -32769

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: addi 1, 0, 32768
              addi 1, 0, 32768

# Unsigned 16-bit immediate operands

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ori 1, 2, -1
              ori 1, 2, -1

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ori 1, 2, 65536
              ori 1, 2, 65536

# Signed 16-bit immediate operands (extended range for addis)

# CHECK: error: invalid operand for instruction
         addis 1, 0, -65537

# CHECK: error: invalid operand for instruction
         addis 1, 0, 65536

# D-Form memory operands

# CHECK: error: invalid register number
# CHECK-NEXT: lwz 1, 0(32)
              lwz 1, 0(32)

# CHECK: error: invalid register name
# CHECK-NEXT: lwz 1, 0(%r32)
              lwz 1, 0(%r32)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: lwz 1, -32769(2)
              lwz 1, -32769(2)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: lwz 1, 32768(2)
              lwz 1, 32768(2)

# CHECK: error: invalid register number
# CHECK-NEXT: ld 1, 0(32)
              ld 1, 0(32)

# CHECK: error: invalid register name
# CHECK-NEXT: ld 1, 0(%r32)
              ld 1, 0(%r32)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ld 1, 1(2)
              ld 1, 1(2)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ld 1, 2(2)
              ld 1, 2(2)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ld 1, 3(2)
              ld 1, 3(2)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ld 1, -32772(2)
              ld 1, -32772(2)

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: ld 1, 32768(2)
              ld 1, 32768(2)

# CHECK: error: invalid modifier 'got' (no symbols present)
         addi 4, 3, 123@got
# CHECK-NEXT: addi 4, 3, 123@got

# CHECK: error: invalid operand for instruction
# CHECK-NEXT: lwarx 1, 2, 3, a
              lwarx 1, 2, 3, a
