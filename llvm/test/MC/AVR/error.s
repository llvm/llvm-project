# RUN: not llvm-mc -triple avr -mcpu=avrtiny %s 2>&1 \
# RUN:     | FileCheck -check-prefix=AVRTINY %s
# RUN: not llvm-mc -triple avr -mcpu=avr2 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=AVR2 %s

# AVRTINY: error: invalid register on avrtiny
mov r0, r16

# AVRTINY: error: invalid register on avrtiny
mov 1, r16

# AVR2: error: invalid operand for instruction
ldi r1, 15

# AVR2: error: instruction requires a CPU feature not currently enabled
lpm r8, Z+

## Extra immediate operands in 0..15 should not be diagnosed as invalid
## registers on avrtiny when matching against InvalidMatchClass.
# AVRTINY: :[[#@LINE+2]]:5: error: invalid operand for instruction
# AVR2: :[[#@LINE+1]]:5: error: invalid operand for instruction
ret 0

# AVRTINY: :[[#@LINE+2]]:13: error: invalid operand for instruction
# AVR2: :[[#@LINE+1]]:13: error: invalid operand for instruction
ldi r16, 0, 0
