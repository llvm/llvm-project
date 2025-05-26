// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefix=NOGCN --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=fiji %s 2>&1 | FileCheck %s --check-prefix=NOGCN --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=NOGFX9 --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck %s --check-prefix=NOGFX90A --implicit-check-not=error:

//===----------------------------------------------------------------------===//
// Image Load/Store
//===----------------------------------------------------------------------===//

image_load    v[4:6], v[237:240], s[28:35] dmask:0x7 tfe
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_load    v[4:5], v[237:240], s[28:35] dmask:0x7
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask and d16

image_store   v[4:7], v[237:240], s[28:35] dmask:0x7
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask and d16

image_store   v[4:7], v[237:240], s[28:35] dmask:0xe
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask and d16

image_load    v4, v[237:240], s[28:35] tfe
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

//===----------------------------------------------------------------------===//
// Image Sample
//===----------------------------------------------------------------------===//

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 tfe
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x3
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask and d16

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0xf
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask and d16

//===----------------------------------------------------------------------===//
// Image Atomics
//===----------------------------------------------------------------------===//

image_atomic_add v252, v2, s[8:15] dmask:0x1 tfe
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_add v[6:7], v255, s[8:15] dmask:0x2
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask

image_atomic_add v[6:7], v255, s[8:15] dmask:0xf
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf tfe
// NOGCN:    error: image data size does not match dmask and tfe
// NOGFX9:   error: image data size does not match dmask and tfe
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_add v252, v2, s[8:15]
// NOGCN:    error: invalid atomic image dmask
// NOGFX9:   error: invalid atomic image dmask
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid atomic image dmask

image_atomic_add v[6:7], v255, s[8:15] dmask:0x2 tfe
// NOGCN:    error: invalid atomic image dmask
// NOGFX9:   error: invalid atomic image dmask
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xe tfe
// NOGCN:    error: invalid atomic image dmask
// NOGFX9:   error: invalid atomic image dmask
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

//===----------------------------------------------------------------------===//
// Image Gather
//===----------------------------------------------------------------------===//

image_gather4_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x3
// NOGCN:    error: invalid image_gather dmask: only one bit must be set
// NOGFX9:   error: invalid image_gather dmask: only one bit must be set
// NOGFX90A: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// R128
//===----------------------------------------------------------------------===//

image_atomic_add v5, v1, s[8:11] dmask:0x1
// NOGCN:    error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX9:   error: operands are not valid for this GPU or mode
// NOGFX90A: error: operands are not valid for this GPU or mode

image_atomic_add v5, v1, s[8:15] dmask:0x1 r128
// NOGCN:    error: r128 not allowed with 256-bit RSRC reg
// NOGFX9:   error: r128 modifier is not supported on this GPU
// NOGFX90A: error: r128 modifier is not supported on this GPU

image_sample  v[193:195], v[237:240], s[28:31], s[4:7] dmask:0x3
// NOGCN:    error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX9:   error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX90A: error: operands are not valid for this GPU or mode

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x3 r128
// NOGCN:    error: r128 not allowed with 256-bit RSRC reg
// NOGFX9:   error: r128 modifier is not supported on this GPU
// NOGFX90A: error: r128 modifier is not supported on this GPU

image_gather4 v[5:8], v[1:4], s[8:11], s[12:15] dmask:0x3
// NOGCN:    error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX9:   error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX90A: error: instruction not supported on this GPU

image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x3 r128
// NOGCN:    error: r128 not allowed with 256-bit RSRC reg
// NOGFX9:   error: r128 modifier is not supported on this GPU
// NOGFX90A: error: instruction not supported on this GPU

image_load v[5:6], v1, s[8:11] dmask:0x1
// NOGCN:    error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX9:   error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX90A: error: operands are not valid for this GPU or mode

image_load v[5:6], v1, s[8:15] dmask:0x1 r128
// NOGCN:    error: r128 not allowed with 256-bit RSRC reg
// NOGFX9:   error: r128 modifier is not supported on this GPU
// NOGFX90A: error: r128 modifier is not supported on this GPU

image_store   v[4:7], v[237:240], s[28:31] dmask:0x7
// NOGCN:    error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX9:   error: the RSRC reg should be 256-bit, or the r128 flag is required
// NOGFX90A: error: operands are not valid for this GPU or mode

image_store   v[4:7], v[237:240], s[28:35] dmask:0x7 r128
// NOGCN:    error: r128 not allowed with 256-bit RSRC reg
// NOGFX9:   error: r128 modifier is not supported on this GPU
// NOGFX90A: error: r128 modifier is not supported on this GPU
