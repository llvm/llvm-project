// RUN: mlir-opt %s -convert-openacc-to-scf -split-input-file | FileCheck %s

func.func @testenterdataop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.enter_data if(%ifCond) create(%a: memref<10xf32>)
  return
}

// CHECK:      func @testenterdataop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.enter_data create(%{{.*}} : memref<10xf32>)
// CHECK-NEXT:   }

// -----

func.func @testexitdataop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.exit_data if(%ifCond) delete(%a: memref<10xf32>)
  return
}

// CHECK:      func @testexitdataop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.exit_data delete(%{{.*}} : memref<10xf32>)
// CHECK-NEXT:   }

// -----

func.func @testupdateop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.update if(%ifCond) host(%a: memref<10xf32>)
  return
}

// CHECK:      func @testupdateop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.update host(%{{.*}} : memref<10xf32>)
// CHECK-NEXT:   }

// -----

func.func @update_true(%arg0: memref<10xf32, #spirv.storage_class<StorageBuffer>>) {
  %true = arith.constant true
  acc.update if(%true) host(%arg0 : memref<10xf32, #spirv.storage_class<StorageBuffer>>)
  return
}

// CHECK-LABEL: func.func @update_true
// CHECK-NOT:if
// CHECK:acc.update host

// -----

func.func @update_false(%arg0: memref<10xf32, #spirv.storage_class<StorageBuffer>>) {
  %false = arith.constant false
  acc.update if(%false) host(%arg0 : memref<10xf32, #spirv.storage_class<StorageBuffer>>)
  return
}

// CHECK-LABEL: func.func @update_false
// CHECK-NOT:acc.update

// -----

func.func @enter_data_true(%d1 : memref<10xf32>) {
  %true = arith.constant true
  acc.enter_data if(%true) create(%d1 : memref<10xf32>) attributes {async}
  return
}

// CHECK-LABEL: func.func @enter_data_true
// CHECK-NOT:if
// CHECK:acc.enter_data create

// -----

func.func @enter_data_false(%d1 : memref<10xf32>) {
  %false = arith.constant false
  acc.enter_data if(%false) create(%d1 : memref<10xf32>) attributes {async}
  return
}

// CHECK-LABEL: func.func @enter_data_false
// CHECK-NOT:acc.enter_data

// -----

func.func @exit_data_true(%d1 : memref<10xf32>) {
  %true = arith.constant true
  acc.exit_data if(%true)  delete(%d1 : memref<10xf32>) attributes {async}
  return
}

// CHECK-LABEL: func.func @exit_data_true
// CHECK-NOT:if
// CHECK:acc.exit_data delete

// -----

func.func @exit_data_false(%d1 : memref<10xf32>) {
  %false = arith.constant false
  acc.exit_data if(%false)  delete(%d1 : memref<10xf32>) attributes {async}
  return
}

// CHECK-LABEL: func.func @exit_data_false
// CHECK-NOT:acc.exit_data

// -----
