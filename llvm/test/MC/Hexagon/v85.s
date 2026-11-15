# RUN: llvm-mc -triple=hexagon -mv85 -filetype=obj %s | \
# RUN:   llvm-objdump --mcpu=hexagonv85 -d - | FileCheck %s

# Test the new V85 instructions. CHECK lines verify the encoding bytes.

# AM_addw_aq
r0 = atomw_add(r1,r2):aq
# CHECK: 9001e220

# AM_addw_rl
r0 = atomw_add(r1,r2):rl
# CHECK: 9001e240

# AM_andw_aq
r0 = atomw_and(r1,r2):aq
# CHECK: 9001e2a0

# AM_andw_rl
r0 = atomw_and(r1,r2):rl
# CHECK: 9001e2c0

# AM_orw_aq
r0 = atomw_or(r1,r2):aq
# CHECK: 90c1c220

# AM_orw_rl
r0 = atomw_or(r1,r2):rl
# CHECK: 90c1c240

# AM_swapd_aq
r1:0 = atomd_swap(r3,r5:4):aq
# CHECK: 90c3e4a0

# AM_swapd_rl
r1:0 = atomd_swap(r3,r5:4):rl
# CHECK: 90c3e4c0

# AM_swapw_aq
r0 = atomw_swap(r1,r2):aq
# CHECK: 90c1e220

# AM_swapw_rl
r0 = atomw_swap(r1,r2):rl
# CHECK: 90c1e240

# AM_xorw_aq
r0 = atomw_xor(r1,r2):aq
# CHECK: 90c1c2a0

# AM_xorw_rl
r0 = atomw_xor(r1,r2):rl
# CHECK: 90c1c2c0

# Y4_dcfetch_multi
dcfetch_multi(r1,r2)
# CHECK: 9401c880
