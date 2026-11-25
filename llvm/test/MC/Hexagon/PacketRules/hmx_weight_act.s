# RUN: not llvm-mc -triple=hexagon -mhmx -filetype=obj %s 2>&1 | FileCheck --implicit-check-not=error %s

activation.hf = mxmem(r0,r0)
# CHECK: hmx_weight_act.s:[[#@LINE-1]]:1: error: packet has HMX activation without a weight

weight.hf = mxmem(r0, r1)
# CHECK: hmx_weight_act.s:[[#@LINE-1]]:1: error: packet has HMX weight without an activation

{
  weight.hf = mxmem(r0, r1)
  r0 = r1
}
# CHECK: hmx_weight_act.s:[[#@LINE-1]]:1: error: packet has HMX weight without an activation

## Valid pairings, none of the below should result in errors.

{
    activation.hf = mxmem(r0,r0)
    weight.hf = mxmem(r0,r1) }

{   activation.hf = mxmem(r0,r0)
    r0 = r3
    weight.hf = mxmem(r0,r0) }

{   activation.hf = mxmem(r0,r0):deep
    weight.hf = mxmem(r0,r0) }

{   activation.hf = mxmem(r0,r0):single
    weight.hf = mxmem(r0,r0) }

{   activation.hf = mxmem(r0,r0):single
    r0 = r3
    weight.hf = mxmem(r0,r0) }
