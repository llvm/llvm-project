# RUN: llvm-mc -filetype=obj -triple=hexagon %s | llvm-objdump -d --print-imm-hex - | FileCheck %s

# CHECK: r3 = ##0x70000240
r3 = ##1879048768
# CHECK: r3 = ##-0x70000240
r3 = ##-1879048768
