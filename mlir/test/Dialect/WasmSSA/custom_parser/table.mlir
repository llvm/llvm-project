// RUN: mlir-opt %s | FileCheck %s

// CHECK:   wasmssa.table @tab0 public !wasmssa<tabletype !wasmssa.externref [0: 65536]>
wasmssa.table @tab0 public !wasmssa<tabletype !wasmssa.externref [0:65536]>

// CHECK:   wasmssa.table @tab1 nested !wasmssa<tabletype !wasmssa.funcref [348:]>
wasmssa.table @tab1 !wasmssa<tabletype !wasmssa.funcref [348:]>
