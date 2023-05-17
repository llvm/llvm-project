## Verify that LLVM-specific section types are correctly inferred from assembly input.
# RUN: llvm-mc -triple i386-pc-linux -filetype=obj -o %t %s
# RUN: llvm-readobj -S %t | FileCheck %s
.section    .section1,"",@llvm_bb_addr_map
.byte 1
.section    .section2,"",@llvm_call_graph_profile
.byte 1
.section    .section3,"",@llvm_odrtab
.byte 1
.section    .section4,"",@llvm_linker_options
.byte 1
.section    .section5,"",@llvm_sympart
.byte 1
.section    .section6,"",@llvm_dependent_libraries
.byte 1
.section    .section7,"",@llvm_offloading
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
