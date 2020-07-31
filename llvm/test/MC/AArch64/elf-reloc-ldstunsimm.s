// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+fp-armv8 -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ %s

        ldrb w0, [sp, #:lo12:some_label]
        ldrh w0, [sp, #:lo12:some_label]
        ldr w0, [sp, #:lo12:some_label]
        ldr x0, [sp, #:lo12:some_label]
        str q0, [sp, #:lo12:some_label]

// OBJ:      Relocations [
// OBJ-NEXT:   Section {{.*}} .rela.text {
// OBJ-NEXT:     0x0  R_AARCH64_LDST8_ABS_LO12_NC   some_label 0x0
// OBJ-NEXT:     0x4  R_AARCH64_LDST16_ABS_LO12_NC  some_label 0x0
// OBJ-NEXT:     0x8  R_AARCH64_LDST32_ABS_LO12_NC  some_label 0x0
// OBJ-NEXT:     0xC  R_AARCH64_LDST64_ABS_LO12_NC  some_label 0x0
// OBJ-NEXT:     0x10 R_AARCH64_LDST128_ABS_LO12_NC some_label 0x0
// OBJ-NEXT:   }
// OBJ-NEXT: ]
