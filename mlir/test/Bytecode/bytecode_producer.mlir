// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-producer="MyCustomProducer" | llvm-strings | FileCheck %s

// CHECK: MyCustomProducer

module {}
