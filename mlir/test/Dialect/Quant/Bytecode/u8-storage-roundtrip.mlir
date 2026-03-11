// Verify that the u8 keyword in quant types creates unsigned (ui8) storage,
// not signless (i8). The quant printer always outputs "u8" regardless of the
// underlying IntegerType signedness, so the only way to detect a mismatch is
// to compare bytecodes: one from the explicit "ui8" syntax (parsed via
// parseOptionalType, always unsigned) and one from a text roundtrip through
// the "u8" keyword path.

// RUN: mlir-opt %s -allow-unregistered-dialect --strip-debuginfo -emit-bytecode -o %t.bc
// RUN: mlir-opt %t.bc -allow-unregistered-dialect | mlir-opt -allow-unregistered-dialect --strip-debuginfo -emit-bytecode -o %t2.bc
// RUN: cmp %t.bc %t2.bc

module @u8StorageRoundtrip attributes {
  bytecode.test = !quant.uniform<ui8:f32, 9.987200e-01:127>
} {}
