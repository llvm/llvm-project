// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ops
// CHECK-SAME:  %[[I32:.*]]: i32
func.func @ops(%arg0: i32) {
// Memory-related operations.
//
// CHECK-NEXT:  %[[ALLOCA:.*]] = llvm.alloca %[[I32]] x f64 : (i32) -> !llvm.ptr<f64>
// CHECK-NEXT:  %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%[[I32]], %[[I32]]] : (!llvm.ptr<f64>, i32, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:  %[[VALUE:.*]] = llvm.load %[[GEP]] : !llvm.ptr<f64>
// CHECK-NEXT:  llvm.store %[[VALUE]], %[[ALLOCA]] : !llvm.ptr<f64>
// CHECK-NEXT:  %{{.*}} = llvm.bitcast %[[ALLOCA]] : !llvm.ptr<f64> to !llvm.ptr<i64>
  %13 = llvm.alloca %arg0 x f64 : (i32) -> !llvm.ptr<f64>
  %14 = llvm.getelementptr %13[%arg0, %arg0] : (!llvm.ptr<f64>, i32, i32) -> !llvm.ptr<f64>
  %15 = llvm.load %14 : !llvm.ptr<f64>
  llvm.store %15, %13 : !llvm.ptr<f64>
  %16 = llvm.bitcast %13 : !llvm.ptr<f64> to !llvm.ptr<i64>
  llvm.return
}

// CHECK-LABEL: @gep
llvm.func @gep(%ptr: !llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, %idx: i64,
               %ptr2: !llvm.ptr<struct<(array<10 x f32>)>>) {
  // CHECK: llvm.getelementptr %{{.*}}[%{{.*}}, 1, 0] : (!llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, i64) -> !llvm.ptr<i32>
  llvm.getelementptr %ptr[%idx, 1, 0] : (!llvm.ptr<struct<(i32, struct<(i32, f32)>)>>, i64) -> !llvm.ptr<i32>
  // CHECK: llvm.getelementptr inbounds %{{.*}}[%{{.*}}, 0, %{{.*}}] : (!llvm.ptr<struct<(array<10 x f32>)>>, i64, i64) -> !llvm.ptr<f32>
  llvm.getelementptr inbounds %ptr2[%idx, 0, %idx] : (!llvm.ptr<struct<(array<10 x f32>)>>, i64, i64) -> !llvm.ptr<f32>
  llvm.return
}

// CHECK-LABEL: @alloca
func.func @alloca(%size : i64) {
  // CHECK: llvm.alloca %{{.*}} x i32 : (i64) -> !llvm.ptr<i32>
  llvm.alloca %size x i32 {alignment = 0} : (i64) -> (!llvm.ptr<i32>)
  // CHECK: llvm.alloca inalloca %{{.*}} x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i32>
  llvm.alloca inalloca %size x i32 {alignment = 8} : (i64) -> (!llvm.ptr<i32>)
  llvm.return
}

// CHECK-LABEL: @null
func.func @null() {
  // CHECK: llvm.mlir.null : !llvm.ptr<i8>
  %0 = llvm.mlir.null : !llvm.ptr<i8>
  // CHECK: llvm.mlir.null : !llvm.ptr<struct<(ptr<func<void (i32, ptr<func<void ()>>)>>, i64)>>
  %1 = llvm.mlir.null : !llvm.ptr<struct<(ptr<func<void (i32, ptr<func<void ()>>)>>, i64)>>
  llvm.return
}

// CHECK-LABEL: llvm.func @vararg_func
llvm.func @vararg_func(%arg0: i32, ...) {
  // CHECK: %{{.*}} = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %{{.*}} = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA0:.+]] = llvm.alloca %{{.*}} x !llvm.struct<"struct.va_list", (ptr<i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"struct.va_list", (ptr<i8>)>>
  // CHECK: %[[CAST0:.+]] = llvm.bitcast %[[ALLOCA0]] : !llvm.ptr<struct<"struct.va_list", (ptr<i8>)>> to !llvm.ptr<i8>
  %2 = llvm.alloca %1 x !llvm.struct<"struct.va_list", (ptr<i8>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<"struct.va_list", (ptr<i8>)>>
  %3 = llvm.bitcast %2 : !llvm.ptr<struct<"struct.va_list", (ptr<i8>)>> to !llvm.ptr<i8>
  // CHECK: llvm.intr.vastart %[[CAST0]]
  llvm.intr.vastart %3 : !llvm.ptr<i8>
  // CHECK: %[[ALLOCA1:.+]] = llvm.alloca %{{.*}} x !llvm.ptr<i8> {alignment = 8 : i64} : (i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[CAST1:.+]] = llvm.bitcast %[[ALLOCA1]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %4 = llvm.alloca %0 x !llvm.ptr<i8> {alignment = 8 : i64} : (i32) -> !llvm.ptr<ptr<i8>>
  %5 = llvm.bitcast %4 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  // CHECK: llvm.intr.vacopy %[[CAST0]] to %[[CAST1]]
  llvm.intr.vacopy %3 to %5 : !llvm.ptr<i8>, !llvm.ptr<i8>
  // CHECK: llvm.intr.vaend %[[CAST1]]
  // CHECK: llvm.intr.vaend %[[CAST0]]
  llvm.intr.vaend %5 : !llvm.ptr<i8>
  llvm.intr.vaend %3 : !llvm.ptr<i8>
  // CHECK: llvm.return
  llvm.return
}
