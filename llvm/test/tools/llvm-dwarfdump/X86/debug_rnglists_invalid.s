# RUN: llvm-mc %S/Inputs/debug_rnglists_short_section.s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN: not llvm-dwarfdump --debug-rnglists - 2>&1 | FileCheck %s --check-prefix=SHORT
# SHORT-NOT: error:
# SHORT-NOT: range list header
# SHORT: error: parsing .debug_rnglists table at offset 0x0: unexpected end of data at offset 0x3
# SHORT-NOT: range list header
# SHORT-NOT: error:

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN: not llvm-dwarfdump --debug-rnglists - 2> %t.err | FileCheck %s --check-prefix=GOOD
# RUN: FileCheck %s --input-file %t.err

# GOOD: .debug_rnglists contents:
# GOOD-NEXT: range list header: length = 0x0000001e, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000001
# GOOD-NEXT: offsets: [
# GOOD-NEXT:    0x00000004
# GOOD-NEXT: ]
# GOOD-NEXT: ranges:
# GOOD-NEXT: [0x0000000000000010, 0x0000000000000020)
# GOOD-NEXT: <End of list>
# GOOD-NEXT: range list header: length = 0x0000001a, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
# GOOD-NEXT: ranges:
# GOOD-NEXT: [0x0000000000000030, 0x0000000000000040)
# GOOD-NEXT: <End of list>
# GOOD-NOT:  range list header

# CHECK-NOT: error:
# CHECK: error: .debug_rnglists table at offset 0x22 has too small length (0xb) to contain a complete header
# CHECK-NEXT: error: unrecognised .debug_rnglists table version 4 in table at offset 0x2d
# CHECK-NEXT: error: .debug_rnglists table at offset 0x39 has unsupported address size: 3
# CHECK-NEXT: error: .debug_rnglists table at offset 0x45 has unsupported segment selector size 4
# CHECK-NEXT: error: .debug_rnglists table at offset 0x51 has more offset entries (12345678) than there is space for
# CHECK-NEXT: error: read past end of table when reading DW_RLE_start_end encoding at offset 0x69
# CHECK-NEXT: error: read past end of table when reading DW_RLE_start_length encoding at offset 0x82
# CHECK-NEXT: error: unknown rnglists encoding 0x2a at offset 0x98
# CHECK-NEXT: error: no end of list marker detected at end of .debug_rnglists table starting at offset 0xaa
# CHECK-NEXT: error: section is not large enough to contain a .debug_rnglists table of length 0x1f at offset 0xe5
# CHECK-NOT: error:

.section .debug_rnglists,"",@progbits

# Table 1 (good)
.long 30 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 1  # Offset entry count

# Offsets
.long 4

# First range list
.byte 6         # DW_RLE_start_length
.quad 0x10, 0x20   # Encoding payload
.byte 0         # DW_RLE_end_of_list

# Table 2 (length too small for header)
.long 7  # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.byte 0, 0, 0  # Truncated offset entry count

# Table 3 (unknown version)
.long 8  # Table length
.short 4 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# Table 4 (unsupported address size)
.long 8  # Table length
.short 5 # Version
.byte 3  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# Table 5 (unsupported segment selector size)
.long 8  # Table length
.short 5 # Version
.byte 8  # Address size
.byte 4  # Segment selector size
.long 0  # Offset entry count

# Table 6 (bad offset entry count)
.long 8  # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 12345678  # Offset entry count

# Table 7 (malformed DW_RLE_start_end)
.long 21 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 6          # DW_RLE_start_end
.quad 1            # Start address
.long 4            # Truncated end address

# Table 8 (malformed DW_RLE_start_length)
.long 18 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 7          # DW_RLE_start_length
.quad 1            # Start address
.byte 0xFF         # Length - invalid ULEB, so will continue reading past the end

# Table 9 (unknown encoding)
.long 26 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 42         # Unknown encoding
.quad 0x10, 0x20   # Encoding payload
.byte 0          # DW_RLE_end_of_list

# Table 10 (missing end of list marker)
.long 25 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 6         # DW_RLE_start_length
.quad 0x10, 0x20   # Encoding payload

# Table 11 (good)
.long 26 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 6         # DW_RLE_start_length
.quad 0x30, 0x40   # Encoding payload
.byte 0         # DW_RLE_end_of_list

# Table 12 (length too long)
.long 27 # Table length - 1 greater than actual contents
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 6          # DW_RLE_start_end
.quad 1, 2         # Start, end address
.byte 0          # DW_RLE_end_of_list
