// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Exercises the MIR dialect LLT type system, attributes, generic-opcode stubs,
// and the ABI-boundary physical-register copies. This is a parse/print
// round-trip (idempotence) check.

// CHECK-LABEL: mir.func @scalar_flow
mir.func @scalar_flow {
  // CHECK: %[[C:.*]] = mir.g_constant 42 : !mir.scalar<32>
  %c = mir.g_constant 42 : !mir.scalar<32>
  // CHECK: %{{.*}} = mir.g_add %[[C]], %[[C]] : !mir.scalar<32>
  %s = mir.g_add %c, %c : !mir.scalar<32>

  // CHECK: %[[P:.*]] = mir.copy_from_phys #mir.physreg<"x0"> -> !mir.pointer<0, 64>
  %p = mir.copy_from_phys #mir.physreg<"x0"> -> !mir.pointer<0, 64>
  // CHECK: %[[L:.*]] = mir.g_load %[[P]] {mmo = #mir.mmo<size = 4, align = 4>} : !mir.pointer<0, 64> -> !mir.scalar<32>
  %l = mir.g_load %p {mmo = #mir.mmo<size = 4, align = 4>} : !mir.pointer<0, 64> -> !mir.scalar<32>
  // CHECK: mir.g_store %[[L]], %[[P]] {mmo = #mir.mmo<size = 8, align = 8, volatile true>} : !mir.scalar<32>, !mir.pointer<0, 64>
  mir.g_store %l, %p {mmo = #mir.mmo<size = 8, align = 8, volatile true>} : !mir.scalar<32>, !mir.pointer<0, 64>

  // CHECK: %[[CP:.*]] = mir.copy %[[L]] : !mir.scalar<32>
  %cp = mir.copy %l : !mir.scalar<32>
  // CHECK: mir.copy_to_phys %[[CP]] : !mir.scalar<32> to #mir.physreg<"x0">
  mir.copy_to_phys %cp : !mir.scalar<32> to #mir.physreg<"x0">
}

// CHECK-LABEL: mir.func @vector_type
mir.func @vector_type {
  // CHECK: mir.copy %{{.*}} : !mir.vector<4 x !mir.scalar<32>>
  %v = mir.copy_from_phys #mir.physreg<"q0"> -> !mir.vector<4 x !mir.scalar<32>>
  %w = mir.copy %v : !mir.vector<4 x !mir.scalar<32>>
}
