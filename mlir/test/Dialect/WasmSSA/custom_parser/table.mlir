// RUN: mlir-opt %s | FileCheck %s

// CHECK:   wasmssa.table exported @tab0 !wasmssa<tabletype !wasmssa.externref [0: 65536]>
wasmssa.table exported @tab0 !wasmssa<tabletype !wasmssa.externref [0:65536]>

// CHECK:   wasmssa.table @tab1 !wasmssa<tabletype !wasmssa.funcref [348:]>
wasmssa.table @tab1 !wasmssa<tabletype !wasmssa.funcref [348:]>
