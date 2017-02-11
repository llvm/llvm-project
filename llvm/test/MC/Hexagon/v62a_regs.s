# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv62 -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-V62
# RUN: not llvm-mc -arch=hexagon -mcpu=hexagonv60 -filetype=asm %s 2>%t; FileCheck -check-prefix=CHECK-NOV62 %s < %t
#

# Assure that v62 added registers are understood

r0=framelimit
r0=framekey
r1:0=c17:16

# CHECK-V62:  6a10c000 { r0 = framelimit }
# CHECK-V62:  6a11c000 { r0 = framekey }
# CHECK-V62:  6810c000 { r1:0 = c17:16 }
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction

r0=pktcountlo
r0=pktcounthi
r1:0=c19:18
r1:0=pktcount

# CHECK-V62:  6a12c000 { r0 = pktcountlo }
# CHECK-V62:  6a13c000 { r0 = pktcounthi }
# CHECK-V62:  6812c000 { r1:0 = c19:18 }
# CHECK-V62:  6812c000 { r1:0 = c19:18 }
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction

r0=utimerlo
r0=utimerhi
r1:0=c31:30
r1:0=UTIMER

# CHECK-V62:  6a1ec000 { r0 = utimerlo }
# CHECK-V62:  6a1fc000 { r0 = utimerhi }
# CHECK-V62:  681ec000 { r1:0 = c31:30 }
# CHECK-V62:  681ec000 { r1:0 = c31:30 }
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction
# CHECK-NOV62: rror: invalid operand for instruction
