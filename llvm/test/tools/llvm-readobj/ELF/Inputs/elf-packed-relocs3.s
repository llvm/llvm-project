.ascii "APS2"
.sleb128 6    // Number of relocations
.sleb128 4096 // Initial offset

.sleb128 2 // Number of relocations in group
.sleb128 8 // RELOCATION_GROUP_HAS_ADDEND_FLAG

.sleb128 256 // Reloc 1: r_offset delta
.sleb128 8   // Reloc 1: r_info R_X86_RELATIVE
.sleb128 0   // Reloc 1: r_addend delta
.sleb128 128 // Reloc 2: r_offset delta
.sleb128 8   // Reloc 2: r_info R_X86_RELATIVE
.sleb128 8   // Reloc 2: r_addend delta

.sleb128 2 // Number of relocations in group
.sleb128 0 // No RELOCATION_GROUP_HAS_ADDEND_FLAG

.sleb128 128           // reloc 1: r_offset delta
.sleb128 (1 << 32) | 1 // r_x86_64_64 (sym index 1)
.sleb128 8             // reloc 2: r_offset delta
.sleb128 (2 << 32) | 1 // r_x86_64_64 (sym index 2)

.sleb128 2 // Number of relocations in group
.sleb128 8 // RELOCATION_GROUP_HAS_ADDEND_FLAG

.sleb128 8             // reloc 1: r_offset delta
.sleb128 (1 << 32) | 1 // r_x86_64_64 (sym index 1)
.sleb128 0             // reloc 1: r_addend delta
.sleb128 8             // reloc 2: r_offset delta
.sleb128 (2 << 32) | 1 // r_x86_64_64 (sym index 2)
.sleb128 8             // reloc 2: r_addend delta
