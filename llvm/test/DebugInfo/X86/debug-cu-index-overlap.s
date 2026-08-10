# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   not llvm-dwarfdump -debug-cu-index -debug-tu-index --verify - | FileCheck %s

# FIXME: The verifier should probably be handled to verify the hash table
# itself - in which case this test would need to be updated to have a correct
# hash table (currently hand crafted with no attempt at correct allocation of
# hashes to buckets) - and probably to verify that the section ranges apply to
# sections that exist, which currently they don't

# This tests that an index that describes units as being in overlapping
# sections is invalid (this was observed in the wild due to overflow due to the
# 32 bit limit of the indexes (a DWARF spec bug - there should be a 64 bit
# version of the index format with 64 bit offsets/sizes)) - but Type Units will
# generally share all the sections other than the info section with each other
# (and with their originating CU) since the dwo format has no way to describe
# which part of non-info-section contributions are used by which units, so
# they're all shared. So demonstrate that the TU index ignores non-info overlap,
# but the CU index diagnoses such overlap (in the abbrev section, in this case)

# This doesn't currently check for info section overlap between the CU and TU
# index, but that could be an extension of this work in the future.

# CHECK: Verifying .debug_cu_index... 
# CHECK: error: overlapping index entries for entries 0x0000000000000001 and 0x0000000000000002 for column DW_SECT_ABBREV
# CHECK: Verifying .debug_tu_index... 
# CHECK: error: overlapping index entries for entries 0x0000000000000001 and 0x0000000000000003 for column DW_SECT_INFO

    .section .debug_cu_index, "", @progbits
## Header:
    .long 5             # Version
    .long 2             # Section count
    .long 3             # Unit count
    .long 4             # Slot count
## Hash Table of Signatures:
    .quad 0x0000000000000001
    .quad 0x0000000000000002
    .quad 0x0000000000000003
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 2
    .long 3
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1             # DW_SECT_INFO
    .long 3             # DW_SECT_ABBREV
## Row 1:
    .long 0x1           # Offset in .debug_info.dwo
    .long 0x1           # Offset in .debug_abbrev.dwo
## Row 2:
    .long 0x2           # Offset in .debug_info.dwo
    .long 0x1           # Offset in .debug_abbrev.dwo
## Row 3:
    .long 0x1           # Offset in .debug_info.dwo
    .long 0x1           # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
    .long 0x1          # Size in .debug_info.dwo
    .long 0x1          # Size in .debug_abbrev.dwo
    .long 0x1          # Size in .debug_info.dwo
    .long 0x1          # Size in .debug_abbrev.dwo
    .long 0x1          # Size in .debug_info.dwo
    .long 0x1          # Size in .debug_abbrev.dwo

    .section .debug_tu_index, "", @progbits
## Header:
    .long 5             # Version
    .long 2             # Section count
    .long 3             # Unit count
    .long 4             # Slot count
## Hash Table of Signatures:
    .quad 0x0000000000000001
    .quad 0x0000000000000002
    .quad 0x0000000000000003
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 2
    .long 3
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1             # DW_SECT_INFO
    .long 3             # DW_SECT_ABBREV
## Row 1:
    .long 0x1           # Offset in .debug_info.dwo
    .long 0x1           # Offset in .debug_abbrev.dwo
## Row 2:
    .long 0x2           # Offset in .debug_info.dwo
    .long 0x1           # Offset in .debug_abbrev.dwo
## Row 3:
    .long 0x1           # Offset in .debug_info.dwo
    .long 0x1           # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
    .long 0x1          # Size in .debug_info.dwo
    .long 0x1          # Size in .debug_abbrev.dwo
    .long 0x1          # Size in .debug_info.dwo
    .long 0x1          # Size in .debug_abbrev.dwo
    .long 0x1          # Size in .debug_info.dwo
    .long 0x1          # Size in .debug_abbrev.dwo
