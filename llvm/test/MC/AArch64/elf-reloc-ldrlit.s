// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s
// RUN: llvm-mc -target-abi=ilp32 -triple=aarch64-none-linux-gnu \
// RUN:   -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ-ILP32 %s

        ldr x0, some_label
        ldr w3, some_label
        ldrsw x9, some_label
        prfm pldl3keep, some_label

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0 R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:     0x4 R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:     0x8 R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:     0xC R_AARCH64_LD_PREL_LO19 some_label 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]

// OBJ-ILP32:      Relocations [
// OBJ-ILP32-NEXT:   Section {{.*}} .rela.text {
// OBJ-ILP32-NEXT:     0x0 R_AARCH64_P32_LD_PREL_LO19 some_label 0x0
// OBJ-ILP32-NEXT:     0x4 R_AARCH64_P32_LD_PREL_LO19 some_label 0x0
// OBJ-ILP32-NEXT:     0x8 R_AARCH64_P32_LD_PREL_LO19 some_label 0x0
// OBJ-ILP32-NEXT:     0xC R_AARCH64_P32_LD_PREL_LO19 some_label 0x0
// OBJ-ILP32-NEXT:   }
// OBJ-ILP32-NEXT: ]
