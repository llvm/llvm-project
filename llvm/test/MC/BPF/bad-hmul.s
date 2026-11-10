# RUN: not llvm-mc -mcpu=v5 -triple bpfel < %s 2>&1 \
# RUN:   | grep 'error: invalid operand for instruction' \
# RUN:   | count 2
        w1 uh*= w2
        w3 sh*= -2
