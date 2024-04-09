// RUN: llvm-mc -triple amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s

// ASM: .set totalvgpr_none, 0
// ASM: .set totalvgpr_one, 1
// ASM: .set totalvgpr_two, 2

.set totalvgpr_none, totalnumvgprs(0, 0)
.set totalvgpr_one, totalnumvgprs(1, 0)
.set totalvgpr_two, totalnumvgprs(1, 2)

// ASM: .set totalvgpr90a_none, 0
// ASM: .set totalvgpr90a_one, 1
// ASM: .set totalvgpr90a_two, 2

.set totalvgpr90a_none, totalnumvgprs90a(0, 0)
.set totalvgpr90a_one, totalnumvgprs90a(0, 1)
.set totalvgpr90a_two, totalnumvgprs90a(0, 2)

// ASM: .set totalvgpr90a_agpr_minimal, 1
// ASM: .set totalvgpr90a_agpr_rounded_eight, 8
// ASM: .set totalvgpr90a_agpr_exact_eight, 8

.set totalvgpr90a_agpr_minimal, totalnumvgprs90a(1, 0)
.set totalvgpr90a_agpr_rounded_eight, totalnumvgprs90a(4, 2)
.set totalvgpr90a_agpr_exact_eight, totalnumvgprs90a(4, 4)
