# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump --debug-rnglists %t.o 2> %t.err | FileCheck %s --check-prefixes=TERSE,BOTH
# RUN: FileCheck %s --allow-empty --input-file %t.err --check-prefix=ERR
# RUN: llvm-dwarfdump -v --debug-rnglists %t.o 2> %t.err | FileCheck %s --check-prefixes=VERBOSE,BOTH
# RUN: FileCheck %s --allow-empty --input-file %t.err --check-prefix=ERR

# BOTH:         .debug_rnglists contents:
# TERSE-NEXT:     range list header: length = 0x00000037, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x00000037, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NEXT:    ranges:
# TERSE-NEXT:   [0x0000000000000010, 0x0000000000000020)
# TERSE-NEXT:   [0x0000000000000025, 0x00000000000000a5)
# TERSE-NEXT:   <End of list>
# TERSE-NEXT:   [0x0000000000000100, 0x0000000000000200)
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x0000000c: [DW_RLE_start_end   ]: [0x0000000000000010, 0x0000000000000020)
# VERBOSE-NEXT: 0x0000001d: [DW_RLE_start_length]:  0x0000000000000025, 0x0000000000000080
# VERBOSE-SAME: => [0x0000000000000025, 0x00000000000000a5)
# VERBOSE-NEXT: 0x00000028: [DW_RLE_end_of_list ]
# VERBOSE-NEXT: 0x00000029: [DW_RLE_start_end   ]: [0x0000000000000100, 0x0000000000000200)
# VERBOSE-NEXT: 0x0000003a: [DW_RLE_end_of_list ]

# TERSE-NEXT:   range list header: length = 0x0000002b, format = DWARF32, version = 0x0005, addr_size = 0x04, seg_size = 0x00, offset_entry_count = 0x00000002

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x0000002b, format = DWARF32, version = 0x0005, addr_size = 0x04, seg_size = 0x00, offset_entry_count = 0x00000002

# BOTH-NEXT:    offsets: [
# BOTH-NEXT:      0x00000008
# VERBOSE-SAME:   => 0x0000004f
# BOTH-NEXT:      0x0000001b
# VERBOSE-SAME:   => 0x00000062
# BOTH-NEXT:    ]
# BOTH-NEXT:    ranges:

# TERSE-NEXT:   [0x00000000, 0x00000000)
# TERSE-NEXT:   [0x00000002, 0x00000006)
# TERSE-NEXT:   <End of list>
# TERSE-NEXT:   [0x00000036, 0x00000136)
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x0000004f: [DW_RLE_start_end   ]: [0x00000000, 0x00000000)
# VERBOSE-NEXT: 0x00000058: [DW_RLE_start_end   ]: [0x00000002, 0x00000006)
# VERBOSE-NEXT: 0x00000061: [DW_RLE_end_of_list ]
# VERBOSE-NEXT: 0x00000062: [DW_RLE_start_length]:  0x00000036, 0x00000100 => [0x00000036, 0x00000136)
# VERBOSE-NEXT: 0x00000069: [DW_RLE_end_of_list ]

# TERSE-NEXT:   range list header: length = 0x00000008, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x00000008, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NOT:     offsets:
# BOTH:         ranges:
# BOTH-NOT:     [

# TERSE-NEXT:   range list header: length = 0x0000000b, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x0000000b, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NEXT:    ranges:
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x00000082: [DW_RLE_base_addressx]:  0x0000000000000000
# VERBOSE-NEXT: 0x00000084: [DW_RLE_end_of_list ]

# TERSE-NEXT:   range list header: length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NEXT:    ranges:
# TERSE-NEXT:   [0x0000000000000000, 0x0000000000000000)
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x00000091: [DW_RLE_startx_endx]:  0x0000000000000001, 0x000000000000000a => [0x0000000000000000, 0x0000000000000000)
# VERBOSE-NEXT: 0x00000094: [DW_RLE_end_of_list]

# TERSE-NEXT:   range list header: length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NEXT:    ranges:
# TERSE-NEXT:   [0x0000000000000000, 0x000000000000002a)
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x000000a1: [DW_RLE_startx_length]:  0x0000000000000002, 0x000000000000002a => [0x0000000000000000, 0x000000000000002a)
# VERBOSE-NEXT: 0x000000a4: [DW_RLE_end_of_list ]

# TERSE-NEXT:   range list header: length = 0x0000000e, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x0000000e, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NEXT:    ranges:
# TERSE-NEXT:   [0x0000000000000800, 0x0000000000001000)
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x000000b1: [DW_RLE_offset_pair]:  0x0000000000000800, 0x0000000000001000 =>
# VERBOSE-SAME:                                   [0x0000000000000800, 0x0000000000001000)
# VERBOSE-NEXT: 0x000000b6: [DW_RLE_end_of_list]

# TERSE-NEXT:   range list header: length = 0x00000017, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# VERBOSE-NEXT: 0x{{[0-9a-f]*}}:
# VERBOSE-SAME: range list header: length = 0x00000017, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000

# BOTH-NEXT:    ranges:
# TERSE-NEXT:   [0x0000000000001800, 0x0000000000002000)
# TERSE-NEXT:   <End of list>

# VERBOSE-NEXT: 0x000000c3: [DW_RLE_base_address]:  0x0000000000001000
# VERBOSE-NEXT: 0x000000cc: [DW_RLE_offset_pair ]:  0x0000000000000800, 0x0000000000001000 =>
# VERBOSE-SAME:                                    [0x0000000000001800, 0x0000000000002000)
# VERBOSE-NEXT: 0x000000d1: [DW_RLE_end_of_list ]

# BOTH-NOT:     range list header:

# ERR-NOT:  error:

.section .debug_rnglists,"",@progbits

# First table (tests DW_RLE_end_of_list, start_end, and start_length encodings)
.long 55 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 6          # DW_RLE_start_end
.quad 0x10, 0x20   # Start, end address
.byte 7          # DW_RLE_start_length
.quad 0x25         # Start address
.byte 0x80, 0x01   # Length
.byte 0          # DW_RLE_end_of_list

# Second range list
.byte 6          # DW_RLE_start_end
.quad 0x100, 0x200 # Start, end address
.byte 0          # DW_RLE_end_of_list

# Second table (shows support for size 4 addresses)
.long 43 # Table length
.short 5 # Version
.byte 4  # Address size
.byte 0  # Segment selector size
.long 2  # Offset entry count

# Offset array
.long 8  # Offset Entry 0
.long 27 # Offset Entry 1

# First range list
.byte 6          # DW_RLE_start_end
.long 0, 0         # Start, end address
.byte 6          # DW_RLE_start_end
.long 0x2, 0x6     # Start, end address
.byte 0          # DW_RLE_end_of_list

# Second range list
.byte 7          # DW_RLE_start_length
.long 0x36         # Start address
.byte 0x80, 0x02   # Length
.byte 0          # DW_RLE_end_of_list

# Third (empty) table
.long 8  # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# The following entries are for encodings unsupported at the time of writing.
# The test should be updated as these encodings are supported.

# Fourth table (testing DW_RLE_base_addressx)
.long 11 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 1          # DW_RLE_base_addressx
.byte 0            # Base address (index 0 in .debug_addr)
.byte 0          # DW_RLE_end_of_list

# Fifth table (testing DW_RLE_startx_endx)
.long 12 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 2          # DW_RLE_startx_endx
.byte 1            # Start address (index in .debug_addr)
.byte 10           # End address (index in .debug_addr)
.byte 0          # DW_RLE_end_of_list

# Sixth table (testing DW_RLE_startx_length)
.long 12 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 3          # DW_RLE_startx_length
.byte 2            # Start address (index in .debug_addr)
.byte 42           # Length
.byte 0          # DW_RLE_end_of_list

# Seventh table (testing DW_RLE_offset_pair)
.long 14 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 4          # DW_RLE_offset_pair
.byte 0x80, 0x10   # Start offset
.byte 0x80, 0x20   # End offset (index in .debug_addr)
.byte 0          # DW_RLE_end_of_list

# Eigth table (testing DW_RLE_base_address and its impact
# on DW_RLE_offset_pair)
.long 23 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 5          # DW_RLE_base_address
.quad 0x1000       # Base address
.byte 4          # DW_RLE_offset_pair
.byte 0x80, 0x10 # Start offset
.byte 0x80, 0x20 # End offset
.byte 0          # DW_RLE_end_of_list
