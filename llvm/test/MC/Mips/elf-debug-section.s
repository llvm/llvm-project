# RUN: llvm-mc -filetype=obj -triple=mips-linux-gnu -g %s -o - \
# RUN:   | llvm-readobj -S - | FileCheck %s

# CHECK:      Section {
# CHECK:        Name: .debug_line
# CHECK-NEXT:   Type: SHT_MIPS_DWARF (0x7000001E)
