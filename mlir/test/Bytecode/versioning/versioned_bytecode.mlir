// This file contains test cases related to roundtripping.

//===--------------------------------------------------------------------===//
// Test roundtrip
//===--------------------------------------------------------------------===//

// RUN: mlir-opt %S/versioned-op-1.12.mlirbc -emit-bytecode \
// RUN:   -emit-bytecode-version=0 | mlir-opt -o %t.1 && \
// RUN: mlir-opt %S/versioned-op-1.12.mlirbc -o %t.2 && \
// RUN: diff %t.1 %t.2

//===--------------------------------------------------------------------===//
// Test invalid versions
//===--------------------------------------------------------------------===//

// RUN: not mlir-opt %S/versioned-op-1.12.mlirbc -emit-bytecode \
// RUN:   -emit-bytecode-version=-1 2>&1 | FileCheck %s --check-prefix=ERR_VERSION_NEGATIVE
// ERR_VERSION_NEGATIVE: unsupported version requested -1, must be in range [{{[0-9]+}}, {{[0-9]+}}]

// RUN: not mlir-opt %S/versioned-op-1.12.mlirbc -emit-bytecode \
// RUN:   -emit-bytecode-version=999 2>&1 | FileCheck %s --check-prefix=ERR_VERSION_FUTURE
// ERR_VERSION_FUTURE: unsupported version requested 999, must be in range [{{[0-9]+}}, {{[0-9]+}}]
