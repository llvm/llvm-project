.ascii "APS2"
.sleb128 10   // Number of relocations
.sleb128 4096 // Initial offset

.sleb128 2 // Number of relocations in group
.sleb128 2 // RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG
.sleb128 8 // offset delta

.sleb128 (1 << 8) | 1 // R_386_32    (sym index 1)
.sleb128 (2 << 8) | 3 // R_386_GOT32 (sym index 2)

.sleb128 8  // Number of relocations in group
.sleb128 3  // RELOCATION_GROUPED_BY_OFFSET_DELTA_FLAG | RELOCATION_GROUPED_BY_INFO_FLAG
.sleb128 -4 // offset delta
.sleb128 8  // R_386_RELATIVE
