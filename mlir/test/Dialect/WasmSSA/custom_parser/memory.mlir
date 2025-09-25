// RUN: mlir-opt %s | FileCheck %s

// CHECK:   wasmssa.memory @mem0 public !wasmssa<limit[0: 65536]>
wasmssa.memory @mem0 public !wasmssa<limit[0:65536]>

// CHECK:   wasmssa.memory @mem1 nested !wasmssa<limit[512:]>
wasmssa.memory @mem1 !wasmssa<limit[512:]>
