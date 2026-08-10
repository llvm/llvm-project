# RUN: llvm-mc -triple=hexagon -show-encoding %s | FileCheck %s
# Check that we generate a duplex for this packet.
# CHECK: encoding: [A,0x40'A',A,A,0x01'B',0x28'B',B,0x20'B']

.data
g:
.long 0

.text
  {
    r0 = add(r0,##g)
    r1 = #0
  }


