// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-producer="MyCustomProducer" | strings | FileCheck %s

// CHECK: MyCustomProducer

module {}
