// RUN: llvm-mc -triple amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s

// ASM: .set occupancy_init_one, 1
// ASM: .set occupancy_init_seven, 7
// ASM: .set occupancy_init_eight, 8

.set occupancy_init_one, occupancy(0, 0, 0, 0, 1, 0, 0)
.set occupancy_init_seven, occupancy(0, 0, 0, 0, 7, 0, 0)
.set occupancy_init_eight, occupancy(0, 0, 0, 0, 8, 0, 0)

// ASM: .set occupancy_numsgpr_seaisle_ten, 10
// ASM: .set occupancy_numsgpr_seaisle_nine, 9
// ASM: .set occupancy_numsgpr_seaisle_eight, 8
// ASM: .set occupancy_numsgpr_seaisle_seven, 7
// ASM: .set occupancy_numsgpr_seaisle_six, 6
// ASM: .set occupancy_numsgpr_seaisle_five, 5

.set occupancy_numsgpr_seaisle_ten, occupancy(0, 0, 0, 6, 11, 1, 0)
.set occupancy_numsgpr_seaisle_nine, occupancy(0, 0, 0, 6, 11, 49, 0)
.set occupancy_numsgpr_seaisle_eight, occupancy(0, 0, 0, 6, 11, 57, 0)
.set occupancy_numsgpr_seaisle_seven, occupancy(0, 0, 0, 6, 11, 65, 0)
.set occupancy_numsgpr_seaisle_six, occupancy(0, 0, 0, 6, 11, 73, 0)
.set occupancy_numsgpr_seaisle_five, occupancy(0, 0, 0, 6, 11, 81, 0)

// ASM: .set occupancy_numsgpr_gfx9_ten, 10
// ASM: .set occupancy_numsgpr_gfx9_nine, 9
// ASM: .set occupancy_numsgpr_gfx9_eight, 8
// ASM: .set occupancy_numsgpr_gfx9_seven, 7

.set occupancy_numsgpr_gfx9_ten, occupancy(0, 0, 0, 8, 11, 1, 0)
.set occupancy_numsgpr_gfx9_nine, occupancy(0, 0, 0, 8, 11, 81, 0)
.set occupancy_numsgpr_gfx9_eight, occupancy(0, 0, 0, 8, 11, 89, 0)
.set occupancy_numsgpr_gfx9_seven, occupancy(0, 0, 0, 8, 11, 101, 0)

// ASM: .set occupancy_numsgpr_gfx10_one, 1
// ASM: .set occupancy_numsgpr_gfx10_seven, 7
// ASM: .set occupancy_numsgpr_gfx10_eight, 8

.set occupancy_numsgpr_gfx10_one, occupancy(1, 0, 0, 9, 11, 1, 0)
.set occupancy_numsgpr_gfx10_seven, occupancy(7, 0, 0, 9, 11, 1, 0)
.set occupancy_numsgpr_gfx10_eight, occupancy(8, 0, 0, 9, 11, 1, 0)

// ASM: .set occupancy_numvgpr_high_granule_one, 1
// ASM: .set occupancy_numvgpr_high_granule_seven, 7
// ASM: .set occupancy_numvgpr_high_granule_eight, 8

.set occupancy_numvgpr_high_granule_one, occupancy(1, 2, 0, 0, 11, 0, 1)
.set occupancy_numvgpr_high_granule_seven, occupancy(7, 2, 0, 0, 11, 0, 1)
.set occupancy_numvgpr_high_granule_eight, occupancy(8, 2, 0, 0, 11, 0, 1)

// ASM: .set occupancy_numvgpr_low_total_one, 1
// ASM: .set occupancy_numvgpr_one, 1
// ASM: .set occupancy_numvgpr_seven, 7
// ASM: .set occupancy_numvgpr_eight, 8
// ASM: .set occupancy_numvgpr_ten, 10

.set occupancy_numvgpr_low_total_one, occupancy(11, 4, 2, 0, 11, 0, 4)
.set occupancy_numvgpr_one, occupancy(11, 4, 4, 0, 11, 0, 4)
.set occupancy_numvgpr_seven, occupancy(11, 4, 28, 0, 11, 0, 4)
.set occupancy_numvgpr_eight, occupancy(11, 4, 32, 0, 11, 0, 4)
.set occupancy_numvgpr_ten, occupancy(11, 4, 40, 0, 11, 0, 4)
