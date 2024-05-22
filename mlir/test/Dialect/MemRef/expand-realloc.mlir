// RUN: mlir-opt --expand-realloc %s --split-input-file --verify-diagnostics | FileCheck %s
// RUN: mlir-opt --expand-realloc=emit-deallocs=false %s --split-input-file --verify-diagnostics | FileCheck %s --check-prefix=NODEALLOC

func.func @reallow_lowering_example(%init_size: index, %new_size: index) -> memref<?xf32> {
  %alloc = memref.alloc(%init_size) : memref<?xf32>
  %realloc = memref.realloc %alloc (%new_size) {alignment = 8}: memref<?xf32> to memref<?xf32>
  return %realloc : memref<?xf32>
}

// CHECK-LABEL: func @reallow_lowering_example
//  CHECK-SAME: ([[INIT_SIZE:%.+]]: index, [[NEW_SIZE:%.+]]: index)
//  CHECK-NEXT:   [[OLD_ALLOC:%.+]] = memref.alloc([[INIT_SIZE]]) : memref<?xf32>
//  CHECK-NEXT:   [[C0:%.+]] = arith.constant 0
//  CHECK-NEXT:   [[CURR_SIZE:%.+]] = memref.dim [[OLD_ALLOC]], [[C0]]
//  CHECK-NEXT:   [[COND:%.+]] = arith.cmpi ult, [[CURR_SIZE]], [[NEW_SIZE]]
//  CHECK-NEXT:   [[REALLOC:%.+]] = scf.if [[COND]]
//  CHECK-NEXT:     [[NEW_ALLOC:%.+]] = memref.alloc([[NEW_SIZE]]) {alignment = 8 : i64} : memref<?xf32>
//  CHECK-NEXT:     [[SUBVIEW:%.+]] = memref.subview [[NEW_ALLOC]][0] [[[CURR_SIZE]]] [1]
//  CHECK-NEXT:     memref.copy [[OLD_ALLOC]], [[SUBVIEW]]
//  CHECK-NEXT:     memref.dealloc [[OLD_ALLOC]]
//  CHECK-NEXT:     scf.yield [[NEW_ALLOC]]
//       CHECK:     [[CAST:%.+]] = memref.reinterpret_cast [[OLD_ALLOC]] to offset: [0], sizes: [[[NEW_SIZE]]], strides: [1] : memref<?xf32> to memref<?xf32>
//  CHECK-NEXT:     scf.yield [[CAST]]
//       CHECK:   return [[REALLOC]]

// NODEALLOC-LABEL: func @reallow_lowering_example
// NODEALLOC-NOT: dealloc

// -----

func.func @reallow_lowering_example() -> memref<4xf32> {
  %alloc = memref.alloc() : memref<2xf32>
  %realloc = memref.realloc %alloc {alignment = 8}: memref<2xf32> to memref<4xf32>
  return %realloc : memref<4xf32>
}

// CHECK-LABEL: func @reallow_lowering_example
//  CHECK-NEXT:   [[OLD_ALLOC:%.+]] = memref.alloc() : memref<2xf32>
//  CHECK-NEXT:   [[CURR_SIZE:%.+]] = arith.constant 2
//  CHECK-NEXT:   [[NEW_SIZE:%.+]] = arith.constant 4
//  CHECK-NEXT:   [[COND:%.+]] = arith.cmpi ult, [[CURR_SIZE]], [[NEW_SIZE]]
//  CHECK-NEXT:   [[REALLOC:%.+]] = scf.if [[COND]]
//  CHECK-NEXT:     [[NEW_ALLOC:%.+]] = memref.alloc() {alignment = 8 : i64} : memref<4xf32>
//  CHECK-NEXT:     [[SUBVIEW:%.+]] = memref.subview [[NEW_ALLOC]][0] [2] [1]
//  CHECK-NEXT:     memref.copy [[OLD_ALLOC]], [[SUBVIEW]]
//  CHECK-NEXT:     memref.dealloc [[OLD_ALLOC]]
//  CHECK-NEXT:     scf.yield [[NEW_ALLOC]]
//       CHECK:     [[CAST:%.+]] = memref.reinterpret_cast [[OLD_ALLOC]] to offset: [0], sizes: [4], strides: [1] : memref<2xf32> to memref<4xf32>
//  CHECK-NEXT:     scf.yield [[CAST]]
//       CHECK:   return [[REALLOC]]

// NODEALLOC-LABEL: func @reallow_lowering_example
// NODEALLOC-NOT: dealloc

// -----

func.func @reallow_lowering_example(%init_size: index) -> memref<4xf32> {
  %alloc = memref.alloc(%init_size) : memref<?xf32>
  %realloc = memref.realloc %alloc : memref<?xf32> to memref<4xf32>
  return %realloc : memref<4xf32>
}

// CHECK-LABEL: func @reallow_lowering_example
//  CHECK-SAME: ([[INIT_SIZE:%.+]]: index)
//  CHECK-NEXT:   [[OLD_ALLOC:%.+]] = memref.alloc([[INIT_SIZE]]) : memref<?xf32>
//  CHECK-NEXT:   [[C0:%.+]] = arith.constant 0
//  CHECK-NEXT:   [[CURR_SIZE:%.+]] = memref.dim [[OLD_ALLOC]], [[C0]]
//  CHECK-NEXT:   [[NEW_SIZE:%.+]] = arith.constant 4
//  CHECK-NEXT:   [[COND:%.+]] = arith.cmpi ult, [[CURR_SIZE]], [[NEW_SIZE]]
//  CHECK-NEXT:   [[REALLOC:%.+]] = scf.if [[COND]]
//  CHECK-NEXT:     [[NEW_ALLOC:%.+]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT:     [[SUBVIEW:%.+]] = memref.subview [[NEW_ALLOC]][0] [[[CURR_SIZE]]] [1]
//  CHECK-NEXT:     memref.copy [[OLD_ALLOC]], [[SUBVIEW]]
//  CHECK-NEXT:     memref.dealloc [[OLD_ALLOC]]
//  CHECK-NEXT:     scf.yield [[NEW_ALLOC]]
//       CHECK:     [[CAST:%.+]] = memref.reinterpret_cast [[OLD_ALLOC]] to offset: [0], sizes: [4], strides: [1] : memref<?xf32> to memref<4xf32>
//  CHECK-NEXT:     scf.yield [[CAST]]
//       CHECK:   return [[REALLOC]]

// NODEALLOC-LABEL: func @reallow_lowering_example
// NODEALLOC-NOT: dealloc

// -----

func.func @reallow_lowering_example(%new_size: index) -> memref<?xf32> {
  %alloc = memref.alloc() : memref<2xf32>
  %realloc = memref.realloc %alloc(%new_size) : memref<2xf32> to memref<?xf32>
  return %realloc : memref<?xf32>
}

// CHECK-LABEL: func @reallow_lowering_example
//  CHECK-SAME: ([[NEW_SIZE:%.+]]: index)
//  CHECK-NEXT:   [[OLD_ALLOC:%.+]] = memref.alloc() : memref<2xf32>
//  CHECK-NEXT:   [[CURR_SIZE:%.+]] = arith.constant 2
//  CHECK-NEXT:   [[COND:%.+]] = arith.cmpi ult, [[CURR_SIZE]], [[NEW_SIZE]]
//  CHECK-NEXT:   [[REALLOC:%.+]] = scf.if [[COND]]
//  CHECK-NEXT:     [[NEW_ALLOC:%.+]] = memref.alloc([[NEW_SIZE]]) : memref<?xf32>
//  CHECK-NEXT:     [[SUBVIEW:%.+]] = memref.subview [[NEW_ALLOC]][0] [2] [1]
//  CHECK-NEXT:     memref.copy [[OLD_ALLOC]], [[SUBVIEW]]
//  CHECK-NEXT:     memref.dealloc [[OLD_ALLOC]]
//  CHECK-NEXT:     scf.yield [[NEW_ALLOC]]
//       CHECK:     [[CAST:%.+]] = memref.reinterpret_cast [[OLD_ALLOC]] to offset: [0], sizes: [[[NEW_SIZE]]], strides: [1] : memref<2xf32> to memref<?xf32>
//  CHECK-NEXT:     scf.yield [[CAST]]
//       CHECK:   return [[REALLOC]]

// NODEALLOC-LABEL: func @reallow_lowering_example
// NODEALLOC-NOT: dealloc
