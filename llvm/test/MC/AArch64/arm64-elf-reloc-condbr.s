// RUN: llvm-mc -triple=arm64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ %s
// RUN: llvm-mc -triple=arm64-none-linux-gnu_ilp32 -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ-ILP32 %s

        b.eq somewhere

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0 R_AARCH64_CONDBR19 somewhere 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]

// OBJ-ILP32:      Relocations [
// OBJ-ILP32-NEXT:   Section {{.*}} .rela.text {
// OBJ-ILP32-NEXT:     0x0 R_AARCH64_P32_CONDBR19 somewhere 0x0
// OBJ-ILP32-NEXT:   }
// OBJ-ILP32-NEXT: ]
