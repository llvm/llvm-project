# RUN: llvm-mc -filetype=obj -triple=mips-linux-gnu -g %s -o - \
# RUN:   | llvm-readobj -S - | FileCheck %s

# MIPS .debug_* sections should have SHT_MIPS_DWARF section type
# to distinguish among sections contain DWARF and ECOFF debug formats,
# but in assembly files these sections have SHT_PROGBITS type.

.section        .debug_abbrev,"",@progbits
.section        .debug_addr,"",@progbits
.section        .debug_aranges,"",@progbits
.section        .debug_info,"",@progbits
.section        .debug_line,"",@progbits
.section        .debug_loclists,"",@progbits
.section        .debug_pubnames,"",@progbits
.section        .debug_pubtypes,"",@progbits
.section        .debug_ranges,"",@progbits
.section        .debug_rnglists,"",@progbits
.section        .debug_str,"MS",@progbits,1

# CHECK:      Section {
# CHECK:        Name: .debug_abbrev
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_addr
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_aranges
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_info
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_line
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_loclists
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_pubnames
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_pubtypes
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_ranges
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_rnglists
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
# CHECK:        Name: .debug_str
# CHECK-NEXT:   Type: SHT_MIPS_DWARF
