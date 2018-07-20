// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN

//===----------------------------------------------------------------------===//
// Image Load/Store
//===----------------------------------------------------------------------===//

image_load    v[4:6], v[237:240], s[28:35] dmask:0x7 tfe
// NOGCN: error: image data size does not match dmask and tfe

image_load    v[4:5], v[237:240], s[28:35] dmask:0x7
// NOGCN: error: image data size does not match dmask and tfe

image_store   v[4:7], v[237:240], s[28:35] dmask:0x7
// NOGCN: error: image data size does not match dmask and tfe

image_store   v[4:7], v[237:240], s[28:35] dmask:0xe
// NOGCN: error: image data size does not match dmask and tfe

image_load    v4, v[237:240], s[28:35] tfe
// NOGCN: error: image data size does not match dmask and tfe

//===----------------------------------------------------------------------===//
// Image Sample
//===----------------------------------------------------------------------===//

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 tfe
// NOGCN: error: image data size does not match dmask and tfe

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x3
// NOGCN: error: image data size does not match dmask and tfe

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0xf
// NOGCN: error: image data size does not match dmask and tfe

//===----------------------------------------------------------------------===//
// Image Atomics
//===----------------------------------------------------------------------===//

image_atomic_add v252, v2, s[8:15] dmask:0x1 tfe
// NOGCN: error: image data size does not match dmask and tfe

image_atomic_add v[6:7], v255, s[8:15] dmask:0x2
// NOGCN: error: image data size does not match dmask and tfe

image_atomic_add v[6:7], v255, s[8:15] dmask:0xf
// NOGCN: error: image data size does not match dmask and tfe

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf tfe
// NOGCN: error: image data size does not match dmask and tfe

image_atomic_add v252, v2, s[8:15]
// NOGCN: error: invalid atomic image dmask

image_atomic_add v[6:7], v255, s[8:15] dmask:0x2 tfe
// NOGCN: error: invalid atomic image dmask

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xe tfe
// NOGCN: error: invalid atomic image dmask

//===----------------------------------------------------------------------===//
// Image Gather
//===----------------------------------------------------------------------===//

image_gather4_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x3
// NOGCN: error: invalid image_gather dmask: only one bit must be set
