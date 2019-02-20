// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r | FileCheck -check-prefix=OBJ %s

        adr x2, some_label
        adrp x5, some_label

        adrp x5, :got:some_label
        ldr x0, [x5, #:got_lo12:some_label]

        ldr x0, :got:some_label

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0 R_AARCH64_ADR_PREL_LO21    some_label 0x0
// OBJ-NEXT:     0x4 R_AARCH64_ADR_PREL_PG_HI21 some_label 0x0
// OBJ-NEXT:     0x8 R_AARCH64_ADR_GOT_PAGE     some_label 0x0
// OBJ-NEXT:     0xC R_AARCH64_LD64_GOT_LO12_NC some_label 0x0
// OBJ-NEXT:     0x10 R_AARCH64_GOT_LD_PREL19   some_label 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
