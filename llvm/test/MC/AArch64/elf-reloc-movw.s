// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ %s

        movz x0, #:abs_g0:some_label
        movk x0, #:abs_g0_nc:some_label

        movz x3, #:abs_g1:some_label
        movk x5, #:abs_g1_nc:some_label

        movz x3, #:abs_g2:some_label
        movk x5, #:abs_g2_nc:some_label

        movz x7, #:abs_g3:some_label
        movk x11, #:abs_g3:some_label

        movz x13, #:abs_g0_s:some_label
        movn x17, #:abs_g0_s:some_label

        movz x19, #:abs_g1_s:some_label
        movn x19, #:abs_g1_s:some_label

        movz x19, #:abs_g2_s:some_label
        movn x19, #:abs_g2_s:some_label

        movz x19, #:abs_g1:some_label + 20 - 10
        movn x19, #:abs_g2_s:some_label - 20 + 10
        movk x11, #:abs_g3:some_label + 2 * 10

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0  R_AARCH64_MOVW_UABS_G0    some_label 0x0
// OBJ-NEXT:     0x4  R_AARCH64_MOVW_UABS_G0_NC some_label 0x0
// OBJ-NEXT:     0x8  R_AARCH64_MOVW_UABS_G1    some_label 0x0
// OBJ-NEXT:     0xC  R_AARCH64_MOVW_UABS_G1_NC some_label 0x0
// OBJ-NEXT:     0x10 R_AARCH64_MOVW_UABS_G2    some_label 0x0
// OBJ-NEXT:     0x14 R_AARCH64_MOVW_UABS_G2_NC some_label 0x0
// OBJ-NEXT:     0x18 R_AARCH64_MOVW_UABS_G3    some_label 0x0
// OBJ-NEXT:     0x1C R_AARCH64_MOVW_UABS_G3    some_label 0x0
// OBJ-NEXT:     0x20 R_AARCH64_MOVW_SABS_G0    some_label 0x0
// OBJ-NEXT:     0x24 R_AARCH64_MOVW_SABS_G0    some_label 0x0
// OBJ-NEXT:     0x28 R_AARCH64_MOVW_SABS_G1    some_label 0x0
// OBJ-NEXT:     0x2C R_AARCH64_MOVW_SABS_G1    some_label 0x0
// OBJ-NEXT:     0x30 R_AARCH64_MOVW_SABS_G2    some_label 0x0
// OBJ-NEXT:     0x34 R_AARCH64_MOVW_SABS_G2    some_label 0x0
// OBJ-NEXT:     0x38 R_AARCH64_MOVW_UABS_G1    some_label 0xA
// OBJ-NEXT:     0x3C R_AARCH64_MOVW_SABS_G2    some_label 0xFFFFFFFFFFFFFFF6
// OBJ-NEXT:     0x40 R_AARCH64_MOVW_UABS_G3    some_label 0x14
// OBJ-NEXT:   }
// OBJ-NEXT: ]
