// RUN: llvm-mc -triple=armv7-linux-gnueabi -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ %s

        bleq some_label
        bl some_label
        blx some_label
        beq some_label
        b some_label

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rel.text {
// OBJ-NEXT:     0x0  R_ARM_JUMP24 some_label
// OBJ-NEXT:     0x4  R_ARM_CALL   some_label
// OBJ-NEXT:     0x8  R_ARM_CALL   some_label
// OBJ-NEXT:     0xC  R_ARM_JUMP24 some_label
// OBJ-NEXT:     0x10 R_ARM_JUMP24 some_label
// OBJ-NEXT:   }
// OBJ-NEXT: ]
