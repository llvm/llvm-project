# RUN: llvm-mc --verify -triple riscv32 -mattr=+e,+zcmp < %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcmp < %s \
# RUN:     | llvm-objdump --mattr=+e,+zcmp -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV32E, when both assembling and disassembling.


# CHECK-DIS: b872 <unknown>
cm.push {ra,s0-s2}, -16 # expected-error:16 {{invalid register}}
# CHECK-DIS: be72 <unknown>
cm.popret {ra,s0-s2}, 16 # expected-error:18 {{invalid register}}
# CHECK-DIS: ba72 <unknown>
cm.pop {x1, x8-x9, x18}, 16 # expected-error:20 {{invalid register}}
# CHECK-DIS: ba72 <unknown>
cm.pop {ra, s0-s2}, 16 # expected-error:16 {{invalid register}}
