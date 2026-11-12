// RUN: llvm-mc -triple aarch64-unknown-windows-msvc -filetype obj %s -o - | llvm-objdump -D -r - | FileCheck %s

    .text
main:
    nop
    b .Ltarget
    b .Lother_target

// A privte label target in the same section
    .def .Ltarget
    .scl 3
    .type 32
    .endef
    .p2align 2
.Ltarget:
    ret

// A privte label target in another section
    .section "other"
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
    ret

// Check that both branches have a relocation with a zero offset.
//
// CHECK: 0000000000000000 <main>:
// CHECK:        0: d503201f      nop
// CHECK:        4: 14000000      b       0x4 <main+0x4>
// CHECK:                 0000000000000004:  IMAGE_REL_ARM64_BRANCH26     .Ltarget
// CHECK:        8: 14000000      b       0x8 <main+0x8>
// CHECK:                 0000000000000008:  IMAGE_REL_ARM64_BRANCH26     .Lother_target
// CHECK: 000000000000000c <.Ltarget>:
// CHECK:        c: d65f03c0      ret
// CHECK: 0000000000000000 <other>:
// CHECK:        0: d503201f      nop
// CHECK:        4: d503201f      nop
// CHECK:        8: d503201f      nop
// CHECK:        c: d503201f      nop
// CHECK:       10: d503201f      nop
// CHECK:       14: d503201f      nop
// CHECK:       18: d503201f      nop
// CHECK:       1c: d503201f      nop
// CHECK: 0000000000000020 <.Lother_target>:
// CHECK:       20: d65f03c0      ret

// CHECK: 0000000000000000 <foo>:
// CHECK:        0: d503201f      nop
// CHECK:        4: 14000001      b       0x8 <.Lpcrel_target>
// CHECK:                 0000000000000004:  IMAGE_REL_ARM64_BRANCH26     .Lpcrel_target

    .section "foo"
    nop
    b .Lpcrel_target+4

    .def .Lpcrel_target
    .scl 3
    .type 32
    .p2align 2
    .endef
.Lpcrel_target:
    nop
    nop
    ret
