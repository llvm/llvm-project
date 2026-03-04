// RUN: mlir-opt %s | FileCheck %s

// CHECK:   wasmssa.memory @mem1 !wasmssa<limit[512:]>
wasmssa.memory @mem1 !wasmssa<limit[512:]>

// CHECK:   wasmssa.memory exported @mem2 !wasmssa<limit[0: 65536]>
wasmssa.memory exported @mem2 !wasmssa<limit[0:65536]>
