// RUN: mlir-opt -convert-openacc-to-llvm='use-opaque-pointers=1' -split-input-file %s | FileCheck %s

func.func @testenterdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.enter_data copyin(%b : memref<10xf32>) create(%a : memref<10xf32>)
  return
}

// CHECK: acc.enter_data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testenterdataop(%a: !llvm.ptr, %b: memref<10xf32>) -> () {
  acc.enter_data copyin(%b : memref<10xf32>) create(%a : !llvm.ptr)
  return
}

// CHECK: acc.enter_data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create(%{{.*}} : !llvm.ptr)

// -----

func.func @testenterdataop(%a: memref<10xi64>, %b: memref<10xf32>) -> () {
  acc.enter_data copyin(%b : memref<10xf32>) create_zero(%a : memref<10xi64>) attributes {async}
  return
}

// CHECK: acc.enter_data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create_zero(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) attributes {async}

// -----

func.func @testenterdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.enter_data if(%ifCond) copyin(%b : memref<10xf32>) create(%a : memref<10xf32>)
  return
}

// CHECK: acc.enter_data if(%{{.*}}) copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testexitdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.exit_data copyout(%b : memref<10xf32>) delete(%a : memref<10xf32>)
  return
}

// CHECK: acc.exit_data copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) delete(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testexitdataop(%a: !llvm.ptr, %b: memref<10xf32>) -> () {
  acc.exit_data copyout(%b : memref<10xf32>) delete(%a : !llvm.ptr)
  return
}

// CHECK: acc.exit_data copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) delete(%{{.*}} : !llvm.ptr)

// -----

func.func @testexitdataop(%a: memref<10xi64>, %b: memref<10xf32>) -> () {
  acc.exit_data copyout(%b : memref<10xf32>) delete(%a : memref<10xi64>) attributes {async}
  return
}

// CHECK: acc.exit_data copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) delete(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) attributes {async}

// -----

func.func @testexitdataop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.exit_data if(%ifCond) copyout(%b : memref<10xf32>) delete(%a : memref<10xf32>)
  return
}

// CHECK: acc.exit_data if(%{{.*}}) copyout(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) delete(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testupdateop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.update host(%b : memref<10xf32>) device(%a : memref<10xf32>)
  return
}

// CHECK: acc.update host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) device(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testupdateop(%a: !llvm.ptr, %b: memref<10xf32>) -> () {
  acc.update host(%b : memref<10xf32>) device(%a : !llvm.ptr)
  return
}

// CHECK: acc.update host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) device(%{{.*}} : !llvm.ptr)

// -----

func.func @testupdateop(%a: memref<10xi64>, %b: memref<10xf32>) -> () {
  acc.update host(%b : memref<10xf32>) device(%a : memref<10xi64>) attributes {async}
  return
}

// CHECK: acc.update host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) device(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) attributes {async}

// -----

func.func @testupdateop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.update if(%ifCond) host(%b : memref<10xf32>) device(%a : memref<10xf32>)
  return
}

// CHECK: acc.update if(%{{.*}}) host(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) device(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testdataregion(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.data copy(%b : memref<10xf32>) copyout(%a : memref<10xf32>) {
  }
  return
}

// CHECK: acc.data copy(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) copyout(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testdataregion(%a: !llvm.ptr, %b: memref<10xf32>, %c: !llvm.ptr) -> () {
  acc.data copyin(%b : memref<10xf32>) deviceptr(%c: !llvm.ptr) attach(%a : !llvm.ptr) {
  }
  return
}

// CHECK: acc.data copyin(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) deviceptr(%{{.*}} : !llvm.ptr) attach(%{{.*}} : !llvm.ptr)

// -----

func.func @testdataregion(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.data if(%ifCond) copyin_readonly(%b : memref<10xf32>) copyout_zero(%a : memref<10xf32>) {
  }
  return
}

// CHECK: acc.data if(%{{.*}}) copyin_readonly(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) copyout_zero(%{{.*}} : !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testdataregion(%a: !llvm.ptr, %b: memref<10xf32>, %c: !llvm.ptr) -> () {
  acc.data create(%b : memref<10xf32>) create_zero(%c: !llvm.ptr) no_create(%a : !llvm.ptr) {
  }
  return
}

// CHECK: acc.data create(%{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create_zero(%{{.*}} : !llvm.ptr) no_create(%{{.*}} : !llvm.ptr)

// -----

func.func @testdataregion(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.data present(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  return
}

// CHECK: acc.data present(%{{.*}}, %{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>, !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.parallel copy(%b : memref<10xf32>) copyout(%a : memref<10xf32>) {
  }
  return
}

// CHECK: acc.parallel copy(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) copyout(%{{.*}}: !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testparallelop(%a: !llvm.ptr, %b: memref<10xf32>, %c: !llvm.ptr) -> () {
  acc.parallel copyin(%b : memref<10xf32>) deviceptr(%c: !llvm.ptr) attach(%a : !llvm.ptr) {
  }
  return
}

// CHECK: acc.parallel attach(%{{.*}}: !llvm.ptr) copyin(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) deviceptr(%{{.*}} : !llvm.ptr) 

// -----

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.parallel if(%ifCond) copyin_readonly(%b : memref<10xf32>) copyout_zero(%a : memref<10xf32>) {
  }
  return
}

// CHECK: acc.parallel copyin_readonly(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) copyout_zero(%{{.*}}: !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) if(%{{.*}})

// -----

func.func @testparallelop(%a: !llvm.ptr, %b: memref<10xf32>, %c: !llvm.ptr) -> () {
  acc.parallel create(%b : memref<10xf32>) create_zero(%c: !llvm.ptr) no_create(%a : !llvm.ptr) {
  }
  return
}

// CHECK: acc.parallel create(%{{.*}}: !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>) create_zero(%{{.*}}: !llvm.ptr) no_create(%{{.*}}: !llvm.ptr)

// -----

func.func @testparallelop(%a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.parallel present(%a, %b : memref<10xf32>, memref<10xf32>) {
  }
  return
}

// CHECK: acc.parallel present(%{{.*}}, %{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>, !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)

// -----

func.func @testparallelop(%i: i64, %a: memref<10xf32>, %b: memref<10xf32>) -> () {
  acc.parallel num_gangs(%i: i64) present(%a, %b : memref<10xf32>, memref<10xf32>) {
  } attributes {async}
  return
}

// CHECK: acc.parallel num_gangs(%{{.*}}: i64) present(%{{.*}}, %{{.*}} : !llvm.struct<"openacc_data", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>, !llvm.struct<"openacc_data.1", (struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, ptr, i64)>)
// CHECK-NEXT: } attributes {async}
