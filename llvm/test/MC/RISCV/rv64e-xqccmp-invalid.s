# RUN: llvm-mc --verify -triple riscv64 -mattr=+e,+xqccmp < %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xqccmp < %s \
# RUN:     | llvm-objdump --mattr=+e,+xqccmp -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV64E, when both assembling and disassembling.


# CHECK-DIS: b872 <unknown>
qc.cm.push {ra,s0-s2}, -32 # expected-error:19 {{invalid register}}
# CHECK-DIS: be72 <unknown>
qc.cm.popret {ra,s0-s2}, 32 # expected-error:21 {{invalid register}}
# CHECK-DIS: ba72 <unknown>
qc.cm.pop {x1, x8-x9, x18}, 32 # expected-error:23 {{invalid register}}
# CHECK-DIS: b972 <unknown>
qc.cm.pushfp {x1, x8-x9, x18}, -32 # expected-error:26 {{invalid register}}
# CHECK-DIS: b972 <unknown>
qc.cm.pushfp {ra, s0-s2}, -32 # expected-error:22 {{invalid register}}
