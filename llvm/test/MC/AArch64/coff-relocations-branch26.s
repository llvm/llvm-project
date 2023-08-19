// RUN: llvm-mc -triple aarch64-unknown-windows-msvc -filetype obj %s -o - | llvm-objdump -D -r - | FileCheck %s
// RUN: not llvm-mc -triple aarch64-unknown-windows-msvc -filetype obj --defsym ERR=1 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

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

.ifdef ERR
    .section "err"
err:
    nop
    b .Lerr_target+4
// ERR: [[#@LINE-1]]:5: error: cannot perform a PC-relative fixup with a non-zero symbol offset

    .def .Lerr_target
    .scl 3
    .type 32
    .p2align 2
    .endef
.Lerr_target:
    nop
    nop
    ret
.endif
