// RUN: llvm-mc -triple amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s

// The occupancy() MCExpr arguments are:
//   occupancy(MaxWaves, VGPRGranule, TotalVGPRs, InitOcc,
//             SGPRTotal, SGPRGranule, SGPRTrapReserve, NumSGPRs, NumVGPRs)
// The known constants (the wave/VGPR caps and the SGPR budget) come first; only
// NumSGPRs/NumVGPRs may be symbolic. The SGPR budget is passed explicitly so
// the SGPR-limited occupancy matches the code generator, including the
// trap-handler reservation, independent of the assembler's subtarget. On
// targets where SGPRs do not limit occupancy the asm printer passes a zero
// NumSGPRs (see createOccupancy), so the expression never tests the generation.

// ASM: .set occupancy_init_one, 1
// ASM: .set occupancy_init_seven, 7
// ASM: .set occupancy_init_eight, 8

.set occupancy_init_one, occupancy(0, 0, 0, 1, 0, 0, 0, 0, 0)
.set occupancy_init_seven, occupancy(0, 0, 0, 7, 0, 0, 0, 0, 0)
.set occupancy_init_eight, occupancy(0, 0, 0, 8, 0, 0, 0, 0, 0)

// ASM: .set occupancy_numsgpr_seaisle_ten, 10
// ASM: .set occupancy_numsgpr_seaisle_nine, 9
// ASM: .set occupancy_numsgpr_seaisle_eight, 8
// ASM: .set occupancy_numsgpr_seaisle_seven, 7
// ASM: .set occupancy_numsgpr_seaisle_six, 6
// ASM: .set occupancy_numsgpr_seaisle_five, 5

.set occupancy_numsgpr_seaisle_ten, occupancy(10, 0, 0, 10, 512, 8, 0, 1, 0)
.set occupancy_numsgpr_seaisle_nine, occupancy(10, 0, 0, 10, 512, 8, 0, 49, 0)
.set occupancy_numsgpr_seaisle_eight, occupancy(10, 0, 0, 10, 512, 8, 0, 57, 0)
.set occupancy_numsgpr_seaisle_seven, occupancy(10, 0, 0, 10, 512, 8, 0, 65, 0)
.set occupancy_numsgpr_seaisle_six, occupancy(10, 0, 0, 10, 512, 8, 0, 73, 0)
.set occupancy_numsgpr_seaisle_five, occupancy(10, 0, 0, 10, 512, 8, 0, 81, 0)

// ASM: .set occupancy_numsgpr_gfx9_ten, 10
// ASM: .set occupancy_numsgpr_gfx9_eight, 8
// ASM: .set occupancy_numsgpr_gfx9_eight_b, 8
// ASM: .set occupancy_numsgpr_gfx9_seven, 7

.set occupancy_numsgpr_gfx9_ten, occupancy(10, 0, 0, 10, 800, 16, 0, 1, 0)
.set occupancy_numsgpr_gfx9_eight, occupancy(10, 0, 0, 10, 800, 16, 0, 81, 0)
.set occupancy_numsgpr_gfx9_eight_b, occupancy(10, 0, 0, 10, 800, 16, 0, 89, 0)
.set occupancy_numsgpr_gfx9_seven, occupancy(10, 0, 0, 10, 800, 16, 0, 101, 0)

// Same SGPR budget as gfx9 above, but with the trap handler enabled the 16
// reserved SGPRs per wave lower the achievable occupancy.

// ASM: .set occupancy_numsgpr_gfx9_trap_ten, 10
// ASM: .set occupancy_numsgpr_gfx9_trap_eight, 8
// ASM: .set occupancy_numsgpr_gfx9_trap_seven, 7
// ASM: .set occupancy_numsgpr_gfx9_trap_six, 6

.set occupancy_numsgpr_gfx9_trap_ten, occupancy(10, 0, 0, 10, 800, 16, 16, 1, 0)
.set occupancy_numsgpr_gfx9_trap_eight, occupancy(10, 0, 0, 10, 800, 16, 16, 65, 0)
.set occupancy_numsgpr_gfx9_trap_seven, occupancy(10, 0, 0, 10, 800, 16, 16, 81, 0)
.set occupancy_numsgpr_gfx9_trap_six, occupancy(10, 0, 0, 10, 800, 16, 16, 97, 0)

// On a target where SGPRs do not limit occupancy the asm printer passes a zero
// NumSGPRs, so the SGPR term is skipped and the result is the init occupancy --
// the expression no longer needs to know the target generation.

// ASM: .set occupancy_numsgpr_unlimited_one, 1
// ASM: .set occupancy_numsgpr_unlimited_seven, 7
// ASM: .set occupancy_numsgpr_unlimited_eight, 8

.set occupancy_numsgpr_unlimited_one, occupancy(10, 0, 0, 1, 800, 16, 0, 0, 0)
.set occupancy_numsgpr_unlimited_seven, occupancy(10, 0, 0, 7, 800, 16, 0, 0, 0)
.set occupancy_numsgpr_unlimited_eight, occupancy(10, 0, 0, 8, 800, 16, 0, 0, 0)

// ASM: .set occupancy_numvgpr_high_granule_one, 1
// ASM: .set occupancy_numvgpr_high_granule_seven, 7
// ASM: .set occupancy_numvgpr_high_granule_eight, 8

.set occupancy_numvgpr_high_granule_one, occupancy(1, 2, 0, 11, 0, 0, 0, 0, 1)
.set occupancy_numvgpr_high_granule_seven, occupancy(7, 2, 0, 11, 0, 0, 0, 0, 1)
.set occupancy_numvgpr_high_granule_eight, occupancy(8, 2, 0, 11, 0, 0, 0, 0, 1)

// ASM: .set occupancy_numvgpr_low_total_one, 1
// ASM: .set occupancy_numvgpr_one, 1
// ASM: .set occupancy_numvgpr_seven, 7
// ASM: .set occupancy_numvgpr_eight, 8
// ASM: .set occupancy_numvgpr_ten, 10

.set occupancy_numvgpr_low_total_one, occupancy(11, 4, 2, 11, 0, 0, 0, 0, 4)
.set occupancy_numvgpr_one, occupancy(11, 4, 4, 11, 0, 0, 0, 0, 4)
.set occupancy_numvgpr_seven, occupancy(11, 4, 28, 11, 0, 0, 0, 0, 4)
.set occupancy_numvgpr_eight, occupancy(11, 4, 32, 11, 0, 0, 0, 0, 4)
.set occupancy_numvgpr_ten, occupancy(11, 4, 40, 11, 0, 0, 0, 0, 4)
