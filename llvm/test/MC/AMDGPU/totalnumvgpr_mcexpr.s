// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck --check-prefix=GFX90A %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx1010 < %s | FileCheck --check-prefix=GFX10 %s

// GFX10: .set totalvgpr_none, 0
// GFX10: .set totalvgpr_one, 1
// GFX10: .set totalvgpr_two, 2

.set totalvgpr_none, totalnumvgprs(0, 0)
.set totalvgpr_one, totalnumvgprs(1, 0)
.set totalvgpr_two, totalnumvgprs(1, 2)

// GFX90A: .set totalvgpr90a_none, 0
// GFX90A: .set totalvgpr90a_one, 1
// GFX90A: .set totalvgpr90a_two, 2

.set totalvgpr90a_none, totalnumvgprs(0, 0)
.set totalvgpr90a_one, totalnumvgprs(0, 1)
.set totalvgpr90a_two, totalnumvgprs(0, 2)

// GFX90A: .set totalvgpr90a_agpr_minimal, 1
// GFX90A: .set totalvgpr90a_agpr_rounded_eight, 8
// GFX90A: .set totalvgpr90a_agpr_exact_eight, 8

.set totalvgpr90a_agpr_minimal, totalnumvgprs(1, 0)
.set totalvgpr90a_agpr_rounded_eight, totalnumvgprs(4, 2)
.set totalvgpr90a_agpr_exact_eight, totalnumvgprs(4, 4)
