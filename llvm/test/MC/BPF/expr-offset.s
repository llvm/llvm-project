# RUN: llvm-mc -triple bpfel -filetype=obj < %s \
# RUN:       | llvm-objdump --no-print-imm-hex --no-show-raw-insn -d - \
# RUN:       | FileCheck %s

.equ foo, -1
        if r1 > r2 goto foo + 2
        exit
        exit

# CHECK: if r1 > r2 goto +1
# CHECK: exit
# CHECK: exit
