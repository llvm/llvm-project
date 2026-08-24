// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Exercises the MIR dialect LLT type system, attributes, generated generic
// (G_*) opcodes, and the ABI-boundary physical-register copies. This is a
// parse/print round-trip (idempotence) check.

// CHECK-LABEL: mir.func @scalar_flow
mir.func @scalar_flow {
  // CHECK: %[[P:.*]] = mir.copy_from_phys #mir.physreg<"x0"> -> !mir.pointer<0, 64>
  %p = mir.copy_from_phys #mir.physreg<"x0"> -> !mir.pointer<0, 64>

  // Generated generic opcodes use the uniform functional-type assembly form.
  // CHECK: %[[L:.*]] = mir.g_load %[[P]] {mmo = #mir.mmo<size = 4, align = 4>} : (!mir.pointer<0, 64>) -> !mir.scalar<32>
  %l = mir.g_load %p {mmo = #mir.mmo<size = 4, align = 4>} : (!mir.pointer<0, 64>) -> !mir.scalar<32>
  // CHECK: %[[A:.*]] = mir.g_add %[[L]], %[[L]] : (!mir.scalar<32>, !mir.scalar<32>) -> !mir.scalar<32>
  %a = mir.g_add %l, %l : (!mir.scalar<32>, !mir.scalar<32>) -> !mir.scalar<32>
  // CHECK: mir.g_store %[[A]], %[[P]] : (!mir.scalar<32>, !mir.pointer<0, 64>) -> ()
  mir.g_store %a, %p : (!mir.scalar<32>, !mir.pointer<0, 64>) -> ()

  // CHECK: %[[CP:.*]] = mir.copy %[[A]] : !mir.scalar<32>
  %cp = mir.copy %a : !mir.scalar<32>
  // CHECK: mir.copy_to_phys %[[CP]] : !mir.scalar<32> to #mir.physreg<"x0">
  mir.copy_to_phys %cp : !mir.scalar<32> to #mir.physreg<"x0">
}

// CHECK-LABEL: mir.func @vector_and_phi
mir.func @vector_and_phi {
  // CHECK: %[[V:.*]] = mir.copy_from_phys #mir.physreg<"q0"> -> !mir.vector<4 x !mir.scalar<32>>
  %v = mir.copy_from_phys #mir.physreg<"q0"> -> !mir.vector<4 x !mir.scalar<32>>
  // Variadic-operand generic op (PHI).
  // CHECK: %{{.*}} = mir.g_phi %[[V]], %[[V]] : (!mir.vector<4 x !mir.scalar<32>>, !mir.vector<4 x !mir.scalar<32>>) -> !mir.vector<4 x !mir.scalar<32>>
  %phi = mir.g_phi %v, %v : (!mir.vector<4 x !mir.scalar<32>>, !mir.vector<4 x !mir.scalar<32>>) -> !mir.vector<4 x !mir.scalar<32>>
}
