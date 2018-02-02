# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN: llvm-dwarfdump --debug-rnglists - 2> %t.err | FileCheck %s
# RUN: FileCheck %s --input-file %t.err --check-prefix=ERR

# CHECK: .debug_rnglists contents:
# CHECK-NEXT: Range List Header: length = 0x0000003f, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000002
# CHECK-NEXT: Offsets: [
# CHECK-NEXT:    0x00000008
# CHECK-NEXT:    0x0000002b
# CHECK-NEXT: ]
# CHECK-NEXT: Ranges:
# CHECK-NEXT: [0x0000000000000010, 0x0000000000000020)
# CHECK-NEXT: [0x0000000000000025, 0x00000000000000a5)
# CHECK-NEXT: <End of list>
# CHECK-NEXT: [0x0000000000000100, 0x0000000000000200)
# CHECK-NEXT: <End of list>
# CHECK-NEXT: Range List Header: length = 0x0000001b, version = 0x0005, addr_size = 0x04, seg_size = 0x00, offset_entry_count = 0x00000000
# CHECK-NEXT: Ranges:
# CHECK-NEXT: [0x00000000, 0x00000000)
# CHECK-NEXT: [0x00000002, 0x00000006)
# CHECK-NEXT: <End of list>
# CHECK-NEXT: Range List Header: length = 0x00000008, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
# CHECK-NOT: Offsets:
# CHECK: Ranges:
# CHECK-NOT: [
# CHECK-NOT: Range List Header:

# ERR-NOT:  error:
# ERR:      error: unsupported rnglists encoding DW_RLE_base_addressx at offset 0x7a
# ERR-NEXT: error: unsupported rnglists encoding DW_RLE_startx_endx at offset 0x89
# ERR-NEXT: error: unsupported rnglists encoding DW_RLE_startx_length at offset 0x99
# ERR-NEXT: error: unsupported rnglists encoding DW_RLE_offset_pair at offset 0xa9
# ERR-NEXT: error: unsupported rnglists encoding DW_RLE_base_address at offset 0xb9
# ERR-NOT:  error:

.section .debug_rnglists,"",@progbits

# First table (tests DW_RLE_end_of_list, start_end, and start_length encodings)
.long 63 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 2  # Offset entry count

# Offset array
.long 8  # Offset Entry 0
.long 43 # Offset Entry 1

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
.long 27 # Table length
.short 5 # Version
.byte 4  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 6          # DW_RLE_start_end
.long 0, 0         # Start, end address
.byte 6          # DW_RLE_start_end
.long 0x2, 0x6     # Start, end address
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
.long 12 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 4          # DW_RLE_offset_pair
.byte 3            # Start offset (index in .debug_addr)
.byte 19           # End offset (index in .debug_addr)
.byte 0          # DW_RLE_end_of_list

# Eigth table (testing DW_RLE_base_address)
.long 18 # Table length
.short 5 # Version
.byte 8  # Address size
.byte 0  # Segment selector size
.long 0  # Offset entry count

# First range list
.byte 5          # DW_RLE_base_address
.quad 0x1000       # Base address
.byte 0          # DW_RLE_end_of_list
