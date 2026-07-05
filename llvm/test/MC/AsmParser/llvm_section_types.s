## Verify that LLVM-specific section types are correctly inferred from assembly input and printed.
# RUN: llvm-mc -triple i386-pc-linux %s | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc -triple i386-pc-linux -filetype=obj -o %t %s
# RUN: llvm-readobj -S %t | FileCheck %s
# ASM: .section    .section1,"",@llvm_bb_addr_map
.section    .section1,"",@llvm_bb_addr_map
.byte 1
# ASM: .section    .section2,"",@llvm_call_graph_profile
.section    .section2,"",@llvm_call_graph_profile
.byte 1
# ASM: .section    .section3,"",@llvm_odrtab
.section    .section3,"",@llvm_odrtab
.byte 1
# ASM: .section    .section4,"",@llvm_linker_options
.section    .section4,"",@llvm_linker_options
.byte 1
# ASM: .section    .section5,"",@llvm_sympart
.section    .section5,"",@llvm_sympart
.byte 1
# ASM: .section    .section6,"",@llvm_dependent_libraries
.section    .section6,"",@llvm_dependent_libraries
.byte 1
# ASM: .section    .section7,"",@llvm_offloading
.section    .section7,"",@llvm_offloading
.byte 1
# ASM: .section    .section8,"",@llvm_lto
.section    .section8,"",@llvm_lto
.byte 1
# ASM: .section    .section9,"",@llvm_cfi_jump_table,1
.section    .section9,"",@llvm_cfi_jump_table,1
.byte 1

# CHECK:        Name: .section1
# CHECK-NEXT:   Type: SHT_LLVM_BB_ADDR_MAP
# CHECK:        Name: .section2
# CHECK-NEXT:   Type: SHT_LLVM_CALL_GRAPH_PROFILE
# CHECK:        Name: .section3
# CHECK-NEXT:   Type: SHT_LLVM_ODRTAB
# CHECK:        Name: .section4
# CHECK-NEXT:   Type: SHT_LLVM_LINKER_OPTIONS
# CHECK:        Name: .section5
# CHECK-NEXT:   Type: SHT_LLVM_SYMPART
# CHECK:        Name: .section6
# CHECK-NEXT:   Type: SHT_LLVM_DEPENDENT_LIBRARIES
# CHECK:        Name: .section7
# CHECK-NEXT:   Type: SHT_LLVM_OFFLOADING
# CHECK:        Name: .section8
# CHECK-NEXT:   Type: SHT_LLVM_LTO
# CHECK:        Name: .section9
# CHECK-NEXT:   Type: SHT_LLVM_CFI_JUMP_TABLE
# CHECK:        EntrySize: 1
