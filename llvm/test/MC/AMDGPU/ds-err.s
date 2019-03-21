// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s

// offset too big
// CHECK: error: invalid operand for instruction
ds_add_u32 v2, v4 offset:1000000000

// offset too big
// CHECK: error: invalid operand for instruction
ds_add_u32 v2, v4 offset:0x10000

// offset0 twice
// CHECK:  error: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset0:4 offset0:8

// offset1 twice
// CHECK:  error: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset1:4 offset1:8

// offset0 too big
// CHECK: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset0:1000000000

// offset0 too big
// CHECK: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset0:0x100

// offset1 too big
// CHECK: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset1:1000000000

// offset1 too big
// CHECK: invalid operand for instruction
ds_write2_b32 v2, v4, v6 offset1:0x100

//===----------------------------------------------------------------------===//
// swizzle
//===----------------------------------------------------------------------===//

// CHECK: error: expected a colon
ds_swizzle_b32 v8, v2 offset

// CHECK: error: failed parsing operand
ds_swizzle_b32 v8, v2 offset:

// CHECK: error: expected a colon
ds_swizzle_b32 v8, v2 offset-

// CHECK: error: expected absolute expression
ds_swizzle_b32 v8, v2 offset:SWIZZLE(QUAD_PERM, 0, 1, 2, 3)

// CHECK: error: expected a swizzle mode
ds_swizzle_b32 v8, v2 offset:swizzle(quad_perm, 0, 1, 2, 3)

// CHECK: error: expected a swizzle mode
ds_swizzle_b32 v8, v2 offset:swizzle(XXX,1)

// CHECK: error: expected a comma
ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM

// CHECK: error: expected a comma
ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2)

// CHECK: error: expected a closing parentheses
ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2, 3

// CHECK: error: expected a closing parentheses
ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 0, 1, 2, 3, 4)

// CHECK: error: expected a 2-bit lane id
ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, -1, 1, 2, 3)

// CHECK: error: expected a 2-bit lane id
ds_swizzle_b32 v8, v2 offset:swizzle(QUAD_PERM, 4, 1, 2, 3)

// CHECK: error: group size must be in the interval [1,16]
ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,0)

// CHECK: error: group size must be a power of two
ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,3)

// CHECK: error: group size must be in the interval [1,16]
ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,17)

// CHECK: error: group size must be in the interval [1,16]
ds_swizzle_b32 v8, v2 offset:swizzle(SWAP,32)

// CHECK: error: group size must be in the interval [2,32]
ds_swizzle_b32 v8, v2 offset:swizzle(REVERSE,1)

// CHECK: error: group size must be a power of two
ds_swizzle_b32 v8, v2 offset:swizzle(REVERSE,3)

// CHECK: error: group size must be in the interval [2,32]
ds_swizzle_b32 v8, v2 offset:swizzle(REVERSE,33)

// CHECK: error: group size must be in the interval [2,32]
ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,1,0)

// CHECK: error: group size must be a power of two
ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,3,1)

// CHECK: error: group size must be in the interval [2,32]
ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,33,1)

// CHECK: error: lane id must be in the interval [0,group size - 1]
ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,2,-1)

// CHECK: error: lane id must be in the interval [0,group size - 1]
ds_swizzle_b32 v8, v2 offset:swizzle(BROADCAST,2,2)

// CHECK: error: expected a string
ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, pppii)

// CHECK: error: expected a 5-character mask
ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "")

// CHECK: error: expected a 5-character mask
ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "ppii")

// CHECK: error: expected a 5-character mask
ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "pppiii")

// CHECK: invalid mask
ds_swizzle_b32 v8, v2 offset:swizzle(BITMASK_PERM, "pppi2")
