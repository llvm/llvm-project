// This file contains test cases related to roundtripping.

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: target=s390x-{{.*}}

//===--------------------------------------------------------------------===//
// Test roundtrip
//===--------------------------------------------------------------------===//

// RUN: mlir-opt %S/versioned-op-1.12.mlirbc -emit-bytecode \
// RUN:   -emit-bytecode-version=0 | mlir-opt -o %t.1 && \
// RUN: mlir-opt %S/versioned-op-1.12.mlirbc -o %t.2 && \
// RUN: diff %t.1 %t.2

