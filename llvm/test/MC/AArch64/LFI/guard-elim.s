// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

// Consecutive loads from the same register share a guard.
ldr x0, [x1, #8]
ldr x2, [x1, #16]
// CHECK:      add x28, x27, w1, uxtw
// CHECK-NEXT: ldr x0, [x28, #8]
// CHECK-NEXT: ldr x2, [x28, #16]

// Modifying the base register invalidates the guard.
ldr x4, [x3, #8]
add x3, x3, #24
ldr x5, [x3, #8]
// CHECK:      add x28, x27, w3, uxtw
// CHECK-NEXT: ldr x4, [x28, #8]
// CHECK-NEXT: add x3, x3, #24
// CHECK-NEXT: add x28, x27, w3, uxtw
// CHECK-NEXT: ldr x5, [x28, #8]

// A different base register requires a new guard.
ldr x6, [x4, #8]
ldr x7, [x5, #8]
// CHECK:      add x28, x27, w4, uxtw
// CHECK-NEXT: ldr x6, [x28, #8]
// CHECK-NEXT: add x28, x27, w5, uxtw
// CHECK-NEXT: ldr x7, [x28, #8]

// Labels invalidate the guard.
label_boundary_test:
ldr x8, [x6, #8]
label1:
ldr x9, [x6, #16]
// CHECK-LABEL: label_boundary_test:
// CHECK-NEXT: add x28, x27, w6, uxtw
// CHECK-NEXT: ldr x8, [x28, #8]
// CHECK-NEXT: label1:
// CHECK-NEXT: add x28, x27, w6, uxtw
// CHECK-NEXT: ldr x9, [x28, #16]

// Branches invalidate the guard.
control_flow_test:
ldr x10, [x7, #8]
b label2
ldr x11, [x7, #16]
label2:
// CHECK-LABEL: control_flow_test:
// CHECK-NEXT: add x28, x27, w7, uxtw
// CHECK-NEXT: ldr x10, [x28, #8]
// CHECK-NEXT: b label2
// CHECK-NEXT: add x28, x27, w7, uxtw
// CHECK-NEXT: ldr x11, [x28, #16]
// CHECK-NEXT: label2:

// Modifying the W subregister invalidates the X guard.
w_reg_modification:
ldr x12, [x8, #8]
mov w8, #0
ldr x13, [x8, #16]
// CHECK-LABEL: w_reg_modification:
// CHECK-NEXT: add x28, x27, w8, uxtw
// CHECK-NEXT: ldr x12, [x28, #8]
// CHECK-NEXT: mov w8, #0
// CHECK-NEXT: add x28, x27, w8, uxtw
// CHECK-NEXT: ldr x13, [x28, #16]

// Multiple consecutive accesses share a single guard.
multiple_accesses:
ldr x14, [x9, #8]
ldr x15, [x9, #16]
ldr x16, [x9, #24]
str x17, [x9, #32]
// CHECK-LABEL: multiple_accesses:
// CHECK-NEXT: add x28, x27, w9, uxtw
// CHECK-NEXT: ldr x14, [x28, #8]
// CHECK-NEXT: ldr x15, [x28, #16]
// CHECK-NEXT: ldr x16, [x28, #24]
// CHECK-NEXT: str x17, [x28, #32]

// Mixed loads and stores share a guard.
mixed_load_store:
str x18, [x10, #8]
ldr x19, [x10, #16]
str x20, [x10, #24]
// CHECK-LABEL: mixed_load_store:
// CHECK-NEXT: add x28, x27, w10, uxtw
// CHECK-NEXT: str x18, [x28, #8]
// CHECK-NEXT: ldr x19, [x28, #16]
// CHECK-NEXT: str x20, [x28, #24]

// Instructions that don't touch x28 or the guarded register keep the guard.
non_modifying_between:
ldr x21, [x11, #8]
mov x0, x1
add x2, x3, x4
ldr x22, [x11, #16]
// CHECK-LABEL: non_modifying_between:
// CHECK-NEXT: add x28, x27, w11, uxtw
// CHECK-NEXT: ldr x21, [x28, #8]
// CHECK-NEXT: mov x0, x1
// CHECK-NEXT: add x2, x3, x4
// CHECK-NEXT: ldr x22, [x28, #16]

// Post-index pair writeback invalidates the guard.
prepost_ldp:
ldp x0, x1, [x2]
ldp x3, x4, [x2], #16
ldp x5, x6, [x2]
// CHECK-LABEL: prepost_ldp:
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]
// CHECK-NEXT: ldp x3, x4, [x28]
// CHECK-NEXT: add x2, x2, #16
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x5, x6, [x28]

prepost_stp:
stp x0, x1, [x2]
stp x3, x4, [x2], #16
stp x5, x6, [x2]
// CHECK-LABEL: prepost_stp:
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: stp x0, x1, [x28]
// CHECK-NEXT: stp x3, x4, [x28]
// CHECK-NEXT: add x2, x2, #16
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: stp x5, x6, [x28]

// Pre-index pair writeback invalidates the guard.
prepost_ldp_pre:
ldp x0, x1, [x2]
ldp x3, x4, [x2, #16]!
ldp x5, x6, [x2]
// CHECK-LABEL: prepost_ldp_pre:
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x0, x1, [x28]
// CHECK-NEXT: ldp x3, x4, [x28, #16]
// CHECK-NEXT: add x2, x2, #16
// CHECK-NEXT: add x28, x27, w2, uxtw
// CHECK-NEXT: ldp x5, x6, [x28]

// A load into the base register invalidates the guard.
load_into_base:
ldr x1, [x1, #8]
ldr x2, [x1, #16]
// CHECK-LABEL: load_into_base:
// CHECK-NEXT: add x28, x27, w1, uxtw
// CHECK-NEXT: ldr x1, [x28, #8]
// CHECK-NEXT: add x28, x27, w1, uxtw
// CHECK-NEXT: ldr x2, [x28, #16]

// The guard for x28 carries across an LR mask, since the mask touches only x30.
lr_mask_between:
ldr x0, [x12, #8]
ldr x30, [x12, #16]
ldr x1, [x12, #24]
ret
// CHECK-LABEL: lr_mask_between:
// CHECK-NEXT: add x28, x27, w12, uxtw
// CHECK-NEXT: ldr x0, [x28, #8]
// CHECK-NEXT: ldr x30, [x28, #16]
// CHECK-NEXT: ldr x1, [x28, #24]
// CHECK-NEXT: add x30, x27, w30, uxtw
// CHECK-NEXT: ret

// A .lfi_rewrite_disable region invalidates the guard, since the instructions
// inside it bypass the rewriter and may modify the base register or x28.
rewrite_disable_boundary:
ldr x0, [x14, #8]
.lfi_rewrite_disable
mov x14, x15
.lfi_rewrite_enable
ldr x1, [x14, #16]
// CHECK-LABEL: rewrite_disable_boundary:
// CHECK-NEXT: add x28, x27, w14, uxtw
// CHECK-NEXT: ldr x0, [x28, #8]
// CHECK-NEXT: mov x14, x15
// CHECK-NEXT: add x28, x27, w14, uxtw
// CHECK-NEXT: ldr x1, [x28, #16]
