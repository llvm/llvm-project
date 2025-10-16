// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=bonaire < %s | FileCheck --check-prefix=GFX7 %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck --check-prefix=GFX90A %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck --check-prefix=GFX942 %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx1010 < %s | FileCheck --check-prefix=GFX10 %s

// gfx942 has architected flat scratch enabled.

// GFX7: .set extrasgpr_none, 0
// GFX7: .set extrasgpr_vcc, 2
// GFX7: .set extrasgpr_flatscr, 4
// GFX7: .set extrasgpr_xnack, 0

// GFX90A: .set extrasgpr_none, 0
// GFX90A: .set extrasgpr_vcc, 2
// GFX90A: .set extrasgpr_flatscr, 6
// GFX90A: .set extrasgpr_xnack, 4

// GFX942: .set extrasgpr_none, 6
// GFX942: .set extrasgpr_vcc, 6
// GFX942: .set extrasgpr_flatscr, 6
// GFX942: .set extrasgpr_xnack, 6

// GFX10: .set extrasgpr_none, 0
// GFX10: .set extrasgpr_vcc, 2
// GFX10: .set extrasgpr_flatscr, 0
// GFX10: .set extrasgpr_xnack, 0

.set extrasgpr_none, extrasgprs(0, 0, 0)
.set extrasgpr_vcc, extrasgprs(1, 0, 0)
.set extrasgpr_flatscr, extrasgprs(0, 1, 0)
.set extrasgpr_xnack, extrasgprs(0, 0, 1)
