.ascii "APS2"
.sleb128 8    // Number of relocations
.sleb128 4096 // Initial offset

.sleb128 2 // Number of relocations in group
.sleb128 1 // RELOCATION_GROUPED_BY_INFO_FLAG
.sleb128 8 // R_X86_RELATIVE

.sleb128 256 // Reloc 1: r_offset delta
.sleb128 128 // Reloc 2: r_offset delta

.sleb128 2 // Number of relocations in group
.sleb128 2 // RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG
.sleb128 8 // offset delta

.sleb128 (1 << 32) | 1 // R_X86_64_64 (sym index 1)
.sleb128 (2 << 32) | 1 // R_X86_64_64 (sym index 2)

.sleb128 2 // Number of relocations in group
.sleb128 8 // RELOCATION_GROUP_HAS_ADDEND_FLAG

.sleb128 1             // offset delta
.sleb128 (1 << 32) | 1 // R_X86_64_64 (sym index 1)
.sleb128 8             // addend delta

.sleb128 2             // offset delta
.sleb128 (2 << 32) | 1 // R_X86_64_64 (sym index 2)
.sleb128 4             // addend delta

.sleb128 2  // Number of relocations in group
.sleb128 12 // RELOCATION_GROUP_HAS_ADDEND_FLAG | RELOCATION_GROUPED_BY_ADDEND_FLAG
.sleb128 -2 // addend delta

.sleb128 4             // offset delta
.sleb128 (1 << 32) | 1 // R_X86_64_64 (sym index 1)
.sleb128 8             // offset delta
.sleb128 (2 << 32) | 1 // R_X86_64_64 (sym index 2)
