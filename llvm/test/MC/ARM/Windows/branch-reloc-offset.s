// RUN: llvm-mc -triple thumbv7-windows-gnu -filetype obj %s -o - | llvm-objdump -D -r - | FileCheck %s
// RUN: not llvm-mc -triple thumbv7-windows-gnu -filetype obj --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

    .text
main:
    nop
    b .Ltarget
    b .Lother_target

// A private label target in the same section
    .def .Ltarget
    .scl 3
    .type 32
    .endef
    .p2align 2
.Ltarget:
    bx lr

// A private label target in another section
    .section "other", "xr"
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    .def .Lother_target
    .scl 3
    .type 32
    .endef
    .p2align 2
.Lother_target:
    bx lr

// Check that both branches have a relocation with a zero offset.
//
// CHECK: 00000000 <main>:
// CHECK:        0: bf00          nop
// CHECK:        2: f000 b800     b.w     0x6 <main+0x6>          @ imm = #0x0
// CHECK:                         00000002:  IMAGE_REL_ARM_BRANCH24T      .Ltarget
// CHECK:        6: f000 b800     b.w     0xa <main+0xa>          @ imm = #0x0
// CHECK:                         00000006:  IMAGE_REL_ARM_BRANCH24T      .Lother_target
// CHECK:        a: bf00          nop
// CHECK: 0000000c <.Ltarget>:
// CHECK:        c: 4770          bx      lr
// CHECK: 00000000 <other>:
// CHECK:        0: bf00          nop
// CHECK:        2: bf00          nop
// CHECK:        4: bf00          nop
// CHECK:        6: bf00          nop
// CHECK:        8: bf00          nop
// CHECK:        a: bf00          nop
// CHECK:        c: bf00          nop
// CHECK:        e: bf00          nop
// CHECK: 00000010 <.Lother_target>:
// CHECK:       10: 4770          bx      lr

.ifdef ERR
    .section "other2", "xr"
err:
    nop

// Test errors, if referencing a symbol with an offset

    b .Lerr_target+4
// ERR: [[#@LINE-1]]:5: error: cannot perform a PC-relative fixup with a non-zero symbol offset
    bl .Lerr_target+4
// ERR: [[#@LINE-1]]:5: error: cannot perform a PC-relative fixup with a non-zero symbol offset
    blx .Lerr_target+4
// ERR: [[#@LINE-1]]:5: error: cannot perform a PC-relative fixup with a non-zero symbol offset

// Test errors, if referencing a private label which lacks .def/.scl/.type/.endef, in another
// section, without an offset. Such symbols are omitted from the output symbol table, so the
// relocation can't reference them. Such relocations usually are made towards the base of the
// section plus an offset, but such an offset is not supported with this relocation.

    b .Lerr_target2
// ERR: [[#@LINE-1]]:5: error: cannot perform a PC-relative fixup with a non-zero symbol offset

    .def .Lerr_target
    .scl 3
    .type 32
    .endef
.Lerr_target:
    nop
    nop
    bx lr

    .section "other3", "xr"
    nop
.Lerr_target2:
    bx lr
.endif
