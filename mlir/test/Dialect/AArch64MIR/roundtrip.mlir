// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// The aarch64_mir dialect (selected AArch64 ops) is closely coupled to the mir
// dialect: its ops reuse the mir LLT type system and intermix with generic
// (mir.g_*) ops inside a single mir.func container. This is a parse/print
// round-trip check of a partially-selected function.

// CHECK-LABEL: mir.func @selected
mir.func @selected {
  // CHECK: %[[P:.*]] = mir.copy_from_phys #mir.physreg<"x0"> -> !mir.pointer<0, 64>
  %p = mir.copy_from_phys #mir.physreg<"x0"> -> !mir.pointer<0, 64>
  // Generic (not-yet-selected) load.
  // CHECK: %[[L:.*]] = mir.g_load %[[P]] : (!mir.pointer<0, 64>) -> !mir.scalar<64>
  %l = mir.g_load %p : (!mir.pointer<0, 64>) -> !mir.scalar<64>

  // Selected AArch64 ops, collapsed to one op per mnemonic.
  // CHECK: %[[A:.*]] = aarch64_mir.add %[[L]], %[[L]] : (!mir.scalar<64>, !mir.scalar<64>) -> !mir.scalar<64>
  %a = aarch64_mir.add %l, %l : (!mir.scalar<64>, !mir.scalar<64>) -> !mir.scalar<64>
  // CHECK: %[[N:.*]] = aarch64_mir.and %[[A]], %[[L]] {variant = "ANDXrr"} : (!mir.scalar<64>, !mir.scalar<64>) -> !mir.scalar<64>
  %n = aarch64_mir.and %a, %l {variant = "ANDXrr"} : (!mir.scalar<64>, !mir.scalar<64>) -> !mir.scalar<64>

  // CHECK: mir.copy_to_phys %[[N]] : !mir.scalar<64> to #mir.physreg<"x0">
  mir.copy_to_phys %n : !mir.scalar<64> to #mir.physreg<"x0">
}
