// RUN: mlir-opt %s -convert-openacc-to-scf -split-input-file | FileCheck %s

func @testenterdataop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.enter_data if(%ifCond) create(%a: memref<10xf32>)
  return
}

// CHECK:      func @testenterdataop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.enter_data create(%{{.*}} : memref<10xf32>)
// CHECK-NEXT:   }

// -----

func @testexitdataop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.exit_data if(%ifCond) delete(%a: memref<10xf32>)
  return
}

// CHECK:      func @testexitdataop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.exit_data delete(%{{.*}} : memref<10xf32>)
// CHECK-NEXT:   }

// -----

func @testupdateop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.update if(%ifCond) host(%a: memref<10xf32>)
  return
}

// CHECK:      func @testupdateop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.update host(%{{.*}} : memref<10xf32>)
// CHECK-NEXT:   }
