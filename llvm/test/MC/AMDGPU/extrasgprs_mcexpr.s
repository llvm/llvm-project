// RUN: llvm-mc -triple amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=ASM %s

// ASM: .set extrasgpr_none_gfx7, 0
// ASM: .set extrasgpr_none_gfx9, 0
// ASM: .set extrasgpr_none_gfx10, 0

.set extrasgpr_none_gfx7, extrasgprs(7, 0, 0, 0, 0)
.set extrasgpr_none_gfx9, extrasgprs(9, 0, 0, 0, 0)
.set extrasgpr_none_gfx10, extrasgprs(10, 0, 0, 0, 0)

// ASM: .set extrasgpr_vcc_gfx7, 2
// ASM: .set extrasgpr_vcc_gfx9, 2
// ASM: .set extrasgpr_vcc_gfx10, 2

.set extrasgpr_vcc_gfx7, extrasgprs(7, 1, 0, 0, 0)
.set extrasgpr_vcc_gfx9, extrasgprs(9, 1, 0, 0, 0)
.set extrasgpr_vcc_gfx10, extrasgprs(10, 1, 0, 0, 0)

// ASM: .set extrasgpr_flatscr_gfx7, 4
// ASM: .set extrasgpr_flatscr_gfx9, 6
// ASM: .set extrasgpr_flatscr_gfx10, 0

.set extrasgpr_flatscr_gfx7, extrasgprs(7, 0, 1, 0, 0)
.set extrasgpr_flatscr_gfx9, extrasgprs(9, 0, 1, 0, 0)
.set extrasgpr_flatscr_gfx10, extrasgprs(10, 0, 1, 0, 0)

// ASM: .set extrasgpr_xnack_gfx7, 0
// ASM: .set extrasgpr_xnack_gfx9, 4
// ASM: .set extrasgpr_xnack_gfx10, 0

.set extrasgpr_xnack_gfx7, extrasgprs(7, 0, 0, 1, 0)
.set extrasgpr_xnack_gfx9, extrasgprs(9, 0, 0, 1, 0)
.set extrasgpr_xnack_gfx10, extrasgprs(10, 0, 0, 1, 0)

// ASM: .set extrasgpr_archflatscr_gfx7, 0
// ASM: .set extrasgpr_archflatscr_gfx9, 6
// ASM: .set extrasgpr_archflatscr_gfx10, 0

.set extrasgpr_archflatscr_gfx7, extrasgprs(7, 0, 0, 0, 1)
.set extrasgpr_archflatscr_gfx9, extrasgprs(9, 0, 0, 0, 1)
.set extrasgpr_archflatscr_gfx10, extrasgprs(10, 0, 0, 0, 1)
