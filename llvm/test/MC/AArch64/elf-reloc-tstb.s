// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ %s
// RUN: llvm-mc -target-abi=ilp32 -triple=aarch64-none-linux-gnu \
// RUN:   -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ-ILP32 %s

        tbz x6, #45, somewhere
        tbnz w3, #15, somewhere

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0  R_AARCH64_TSTBR14 somewhere 0x0
// OBJ-NEXT:     0x4  R_AARCH64_TSTBR14 somewhere 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]

// OBJ-ILP32:      Relocations [
// OBJ-ILP32-NEXT:   Section {{.*}} .rela.text {
// OBJ-ILP32-NEXT:     0x0  R_AARCH64_P32_TSTBR14 somewhere 0x0
// OBJ-ILP32-NEXT:     0x4  R_AARCH64_P32_TSTBR14 somewhere 0x0
// OBJ-ILP32-NEXT:   }
// OBJ-ILP32-NEXT: ]
