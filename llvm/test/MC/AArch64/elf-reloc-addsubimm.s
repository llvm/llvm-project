// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj %s -o - | \
// RUN:   llvm-objdump -r -d - | FileCheck --check-prefix=OBJ %s

// OBJ-LABEL: Disassembly of section .text:

        add x2, x3, #:lo12:some_label
// OBJ: [[addr:[0-9a-f]+]]: 91000062 add x2, x3, #0
// OBJ-NEXT: [[addr]]: R_AARCH64_ADD_ABS_LO12_NC	some_label

        add x2, x3, #:lo12:some_label, lsl #12
// OBJ: [[addr:[0-9a-f]+]]: 91400062 add x2, x3, #0, lsl #12
// OBJ-NEXT: [[addr]]: R_AARCH64_ADD_ABS_LO12_NC	some_label

