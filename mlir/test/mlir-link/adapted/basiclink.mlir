// RUN: mlir-link %S/Inputs/basiclink.a.mlir %S/Inputs/basiclink.b.mlir -o %t.mlir
// RUN: mlir-link %S/Inputs/basiclink.b.mlir %S/Inputs/basiclink.a.mlir -o %t.mlir
