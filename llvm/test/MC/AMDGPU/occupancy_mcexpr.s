// RUN: llvm-mc -triple amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s

// The occupancy() MCExpr arguments are:
//   occupancy(MaxWaves, VGPRGranule, TotalVGPRs, Generation, InitOcc,
//             NumSGPRs, NumVGPRs, SGPRTotal, SGPRGranule, SGPRTrapReserve)
// The SGPR budget (SGPRTotal/SGPRGranule/SGPRTrapReserve) is passed explicitly
// so the SGPR-limited occupancy matches the code generator, including the
// trap-handler reservation, independent of the assembler's subtarget.

// ASM: .set occupancy_init_one, 1
// ASM: .set occupancy_init_seven, 7
// ASM: .set occupancy_init_eight, 8

.set occupancy_init_one, occupancy(0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
.set occupancy_init_seven, occupancy(0, 0, 0, 0, 7, 0, 0, 0, 0, 0)
.set occupancy_init_eight, occupancy(0, 0, 0, 0, 8, 0, 0, 0, 0, 0)

// ASM: .set occupancy_numsgpr_seaisle_ten, 10
// ASM: .set occupancy_numsgpr_seaisle_nine, 9
// ASM: .set occupancy_numsgpr_seaisle_eight, 8
// ASM: .set occupancy_numsgpr_seaisle_seven, 7
// ASM: .set occupancy_numsgpr_seaisle_six, 6
// ASM: .set occupancy_numsgpr_seaisle_five, 5

.set occupancy_numsgpr_seaisle_ten, occupancy(10, 0, 0, 6, 10, 1, 0, 512, 8, 0)
.set occupancy_numsgpr_seaisle_nine, occupancy(10, 0, 0, 6, 10, 49, 0, 512, 8, 0)
.set occupancy_numsgpr_seaisle_eight, occupancy(10, 0, 0, 6, 10, 57, 0, 512, 8, 0)
.set occupancy_numsgpr_seaisle_seven, occupancy(10, 0, 0, 6, 10, 65, 0, 512, 8, 0)
.set occupancy_numsgpr_seaisle_six, occupancy(10, 0, 0, 6, 10, 73, 0, 512, 8, 0)
.set occupancy_numsgpr_seaisle_five, occupancy(10, 0, 0, 6, 10, 81, 0, 512, 8, 0)

// ASM: .set occupancy_numsgpr_gfx9_ten, 10
// ASM: .set occupancy_numsgpr_gfx9_eight, 8
// ASM: .set occupancy_numsgpr_gfx9_eight_b, 8
// ASM: .set occupancy_numsgpr_gfx9_seven, 7

.set occupancy_numsgpr_gfx9_ten, occupancy(10, 0, 0, 8, 10, 1, 0, 800, 16, 0)
.set occupancy_numsgpr_gfx9_eight, occupancy(10, 0, 0, 8, 10, 81, 0, 800, 16, 0)
.set occupancy_numsgpr_gfx9_eight_b, occupancy(10, 0, 0, 8, 10, 89, 0, 800, 16, 0)
.set occupancy_numsgpr_gfx9_seven, occupancy(10, 0, 0, 8, 10, 101, 0, 800, 16, 0)

// Same SGPR budget as gfx9 above, but with the trap handler enabled the 16
// reserved SGPRs per wave lower the achievable occupancy.

// ASM: .set occupancy_numsgpr_gfx9_trap_ten, 10
// ASM: .set occupancy_numsgpr_gfx9_trap_eight, 8
// ASM: .set occupancy_numsgpr_gfx9_trap_seven, 7
// ASM: .set occupancy_numsgpr_gfx9_trap_six, 6

.set occupancy_numsgpr_gfx9_trap_ten, occupancy(10, 0, 0, 8, 10, 1, 0, 800, 16, 16)
.set occupancy_numsgpr_gfx9_trap_eight, occupancy(10, 0, 0, 8, 10, 65, 0, 800, 16, 16)
.set occupancy_numsgpr_gfx9_trap_seven, occupancy(10, 0, 0, 8, 10, 81, 0, 800, 16, 16)
.set occupancy_numsgpr_gfx9_trap_six, occupancy(10, 0, 0, 8, 10, 97, 0, 800, 16, 16)

// On GFX10+ (Generation 9) the SGPR file is large enough that SGPRs never
// limit occupancy, so the result is just the init occupancy.

// ASM: .set occupancy_numsgpr_gfx10_one, 1
// ASM: .set occupancy_numsgpr_gfx10_seven, 7
// ASM: .set occupancy_numsgpr_gfx10_eight, 8

.set occupancy_numsgpr_gfx10_one, occupancy(10, 0, 0, 9, 1, 200, 0, 800, 16, 0)
.set occupancy_numsgpr_gfx10_seven, occupancy(10, 0, 0, 9, 7, 200, 0, 800, 16, 0)
.set occupancy_numsgpr_gfx10_eight, occupancy(10, 0, 0, 9, 8, 200, 0, 800, 16, 0)

// ASM: .set occupancy_numvgpr_high_granule_one, 1
// ASM: .set occupancy_numvgpr_high_granule_seven, 7
// ASM: .set occupancy_numvgpr_high_granule_eight, 8

.set occupancy_numvgpr_high_granule_one, occupancy(1, 2, 0, 0, 11, 0, 1, 0, 0, 0)
.set occupancy_numvgpr_high_granule_seven, occupancy(7, 2, 0, 0, 11, 0, 1, 0, 0, 0)
.set occupancy_numvgpr_high_granule_eight, occupancy(8, 2, 0, 0, 11, 0, 1, 0, 0, 0)

// ASM: .set occupancy_numvgpr_low_total_one, 1
// ASM: .set occupancy_numvgpr_one, 1
// ASM: .set occupancy_numvgpr_seven, 7
// ASM: .set occupancy_numvgpr_eight, 8
// ASM: .set occupancy_numvgpr_ten, 10

.set occupancy_numvgpr_low_total_one, occupancy(11, 4, 2, 0, 11, 0, 4, 0, 0, 0)
.set occupancy_numvgpr_one, occupancy(11, 4, 4, 0, 11, 0, 4, 0, 0, 0)
.set occupancy_numvgpr_seven, occupancy(11, 4, 28, 0, 11, 0, 4, 0, 0, 0)
.set occupancy_numvgpr_eight, occupancy(11, 4, 32, 0, 11, 0, 4, 0, 0, 0)
.set occupancy_numvgpr_ten, occupancy(11, 4, 40, 0, 11, 0, 4, 0, 0, 0)
