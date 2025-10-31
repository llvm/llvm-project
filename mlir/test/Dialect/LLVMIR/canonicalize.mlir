// RUN: mlir-opt --pass-pipeline='builtin.module(llvm.func(canonicalize{test-convergence}))' %s -split-input-file | FileCheck %s

// CHECK-LABEL: @fold_icmp_eq
llvm.func @fold_icmp_eq(%arg0 : i32) -> i1 {
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(true) : i1
  %0 = llvm.icmp "eq" %arg0, %arg0 : i32
  // CHECK: llvm.return %[[C0]]
  llvm.return %0 : i1
}

// -----

// CHECK-LABEL: @fold_icmp_ne
llvm.func @fold_icmp_ne(%arg0 : vector<2xi32>) -> vector<2xi1> {
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(dense<false> : vector<2xi1>) : vector<2xi1>
  %0 = llvm.icmp "ne" %arg0, %arg0 : vector<2xi32>
  // CHECK: llvm.return %[[C0]]
  llvm.return %0 : vector<2xi1>
}

// -----

// CHECK-LABEL: @fold_icmp_alloca
llvm.func @fold_icmp_alloca() -> i1 {
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(true) : i1
  %c0 = llvm.mlir.zero : !llvm.ptr
  %c1 = arith.constant 1 : i64
  %0 = llvm.alloca %c1 x i32 : (i64) -> !llvm.ptr
  %1 = llvm.icmp "ne" %c0, %0 : !llvm.ptr
  // CHECK: llvm.return %[[C0]]
  llvm.return %1 : i1
}

// -----

// CHECK-LABEL: fold_extractvalue
llvm.func @fold_extractvalue() -> i32 {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  //  CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32

  %0 = llvm.mlir.undef : !llvm.struct<(i32, i32)>

  // CHECK-NOT: insertvalue
  %1 = llvm.insertvalue %c0, %0[0] : !llvm.struct<(i32, i32)>
  %2 = llvm.insertvalue %c1, %1[1] : !llvm.struct<(i32, i32)>

  // CHECK-NOT: extractvalue
  %3 = llvm.extractvalue %2[0] : !llvm.struct<(i32, i32)>
  %4 = llvm.extractvalue %2[1] : !llvm.struct<(i32, i32)>

  //     CHECK: llvm.add %[[C0]], %[[C1]]
  %5 = llvm.add %3, %4 : i32
  llvm.return %5 : i32
}

// -----

// CHECK-LABEL: fold_extractvalue(
//  CHECK-SAME:     %[[arg1:.*]]: i32, %[[arg2:.*]]: i32, %[[arg3:.*]]: i32)
//  CHECK-NEXT:   llvm.return %[[arg1]] : i32
llvm.func @fold_extractvalue(%arg1: i32, %arg2: i32, %arg3: i32) -> i32{
  %3 = llvm.mlir.undef : !llvm.struct<(struct<(i32, i32, i32)>, struct<(i32, i32)>)>
  %5 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32)>
  %6 = llvm.insertvalue %arg1, %5[0] : !llvm.struct<(i32, i32, i32)>
  %7 = llvm.insertvalue %arg1, %6[1] : !llvm.struct<(i32, i32, i32)>
  %8 = llvm.insertvalue %arg1, %7[2] : !llvm.struct<(i32, i32, i32)>
  %11 = llvm.insertvalue %8, %3[0] : !llvm.struct<(struct<(i32, i32, i32)>, struct<(i32, i32)>)>
  %13 = llvm.extractvalue %11[0, 0] : !llvm.struct<(struct<(i32, i32, i32)>, struct<(i32, i32)>)>
  llvm.return %13 : i32
}

// -----

// CHECK-LABEL: no_fold_extractvalue
llvm.func @no_fold_extractvalue(%arr: !llvm.array<4 x f32>) -> f32 {
  %f0 = arith.constant 0.0 : f32
  %0 = llvm.mlir.undef : !llvm.array<4 x !llvm.array<4 x f32>>

  // CHECK: insertvalue
  // CHECK: insertvalue
  // CHECK: extractvalue
  %1 = llvm.insertvalue %f0, %0[0, 0] : !llvm.array<4 x !llvm.array<4 x f32>>
  %2 = llvm.insertvalue %arr, %1[0] : !llvm.array<4 x !llvm.array<4 x f32>>
  %3 = llvm.extractvalue %2[0, 0] : !llvm.array<4 x !llvm.array<4 x f32>>

  llvm.return %3 : f32
}

// -----

// CHECK-LABEL: fold_unrelated_extractvalue
llvm.func @fold_unrelated_extractvalue(%arr: !llvm.array<4 x f32>) -> f32 {
  %f0 = arith.constant 0.0 : f32
  // CHECK-NOT: insertvalue
  // CHECK: extractvalue
  %2 = llvm.insertvalue %f0, %arr[0] : !llvm.array<4 x f32>
  %3 = llvm.extractvalue %2[1] : !llvm.array<4 x f32>
  llvm.return %3 : f32
}

// -----
// CHECK-LABEL: fold_extract_extractvalue
llvm.func @fold_extract_extractvalue(%arr: !llvm.struct<(i64, array<1 x ptr<1>>)>) -> !llvm.ptr<1> {
  // CHECK: llvm.extractvalue %{{.*}}[1, 0] 
  // CHECK-NOT: extractvalue
  %a = llvm.extractvalue %arr[1] : !llvm.struct<(i64, array<1 x ptr<1>>)> 
  %b = llvm.extractvalue %a[0] : !llvm.array<1 x ptr<1>> 
  llvm.return %b : !llvm.ptr<1>
}

// -----

// CHECK-LABEL: fold_extract_const
// CHECK-NOT: extractvalue
// CHECK:  llvm.mlir.constant(5.000000e-01 : f64)
llvm.func @fold_extract_const() -> f64 {
  %a = llvm.mlir.constant(dense<[-8.900000e+01, 5.000000e-01]> : tensor<2xf64>) : !llvm.array<2 x f64>
  %b = llvm.extractvalue %a[1] : !llvm.array<2 x f64>
  llvm.return %b : f64
}

// -----

// CHECK-LABEL: fold_extract_splat
// CHECK-NOT: extractvalue
// CHECK: llvm.mlir.constant(-8.900000e+01 : f64)
llvm.func @fold_extract_splat() -> f64 {
  %a = llvm.mlir.constant(dense<-8.900000e+01> : tensor<2xf64>) : !llvm.array<2 x f64>
  %b = llvm.extractvalue %a[1] : !llvm.array<2 x f64>
  llvm.return %b : f64
}

// -----

// CHECK-LABEL: fold_bitcast
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[ARG]]
llvm.func @fold_bitcast(%x : !llvm.ptr) -> !llvm.ptr {
  %c = llvm.bitcast %x : !llvm.ptr to !llvm.ptr
  llvm.return %c : !llvm.ptr
}

// -----

// CHECK-LABEL: fold_bitcast2
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[ARG]]
llvm.func @fold_bitcast2(%x : i32) -> i32 {
  %c = llvm.bitcast %x : i32 to f32
  %d = llvm.bitcast %c : f32 to i32
  llvm.return %d : i32
}

// -----

// CHECK-LABEL: fold_bitcast_chain
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
llvm.func @fold_bitcast_chain(%x : i32) -> vector<2xi16> {
  %c = llvm.bitcast %x : i32 to f32
  %d = llvm.bitcast %c : f32 to vector<2xi16>
  // CHECK: %[[BITCAST:.*]] = llvm.bitcast %[[ARG]] : i32 to vector<2xi16>
  // CHECK: llvm.return %[[BITCAST]]
  llvm.return %d : vector<2xi16>
}

// -----

// CHECK-LABEL: fold_addrcast
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[ARG]]
llvm.func @fold_addrcast(%x : !llvm.ptr) -> !llvm.ptr {
  %c = llvm.addrspacecast %x : !llvm.ptr to !llvm.ptr
  llvm.return %c : !llvm.ptr
}

// -----

// CHECK-LABEL: fold_addrcast2
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[ARG]]
llvm.func @fold_addrcast2(%x : !llvm.ptr) -> !llvm.ptr {
  %c = llvm.addrspacecast %x : !llvm.ptr to !llvm.ptr<5>
  %d = llvm.addrspacecast %c : !llvm.ptr<5> to !llvm.ptr
  llvm.return %d : !llvm.ptr
}

// -----

// CHECK-LABEL: fold_addrcast_chain
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
llvm.func @fold_addrcast_chain(%x : !llvm.ptr) -> !llvm.ptr<2> {
  %c = llvm.addrspacecast %x : !llvm.ptr to !llvm.ptr<1>
  %d = llvm.addrspacecast %c : !llvm.ptr<1> to !llvm.ptr<2>
  // CHECK: %[[ADDRCAST:.*]] = llvm.addrspacecast %[[ARG]] : !llvm.ptr to !llvm.ptr<2>
  // CHECK: llvm.return %[[ADDRCAST]]
  llvm.return %d : !llvm.ptr<2>
}

// -----

// CHECK-LABEL: fold_gep
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: llvm.return %[[ARG]]
llvm.func @fold_gep(%x : !llvm.ptr) -> !llvm.ptr {
  %c0 = arith.constant 0 : i32
  %c = llvm.getelementptr %x[%c0] : (!llvm.ptr, i32) -> !llvm.ptr, i8
  llvm.return %c : !llvm.ptr
}

// -----

// CHECK-LABEL: fold_gep_neg
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = llvm.getelementptr inbounds %[[ARG]][0, 1]
// CHECK-NEXT: llvm.return %[[RES]]
llvm.func @fold_gep_neg(%x : !llvm.ptr) -> !llvm.ptr {
  %c0 = arith.constant 0 : i32
  %0 = llvm.getelementptr inbounds %x[%c0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, i32)>
  llvm.return %0 : !llvm.ptr
}

// -----

// CHECK-LABEL: fold_gep_canon
// CHECK-SAME: %[[ARG:[[:alnum:]]+]]
// CHECK-NEXT: %[[RES:.*]] = llvm.getelementptr %[[ARG]][2]
// CHECK-NEXT: llvm.return %[[RES]]
llvm.func @fold_gep_canon(%x : !llvm.ptr) -> !llvm.ptr {
  %c2 = arith.constant 2 : i32
  %c = llvm.getelementptr %x[%c2] : (!llvm.ptr, i32) -> !llvm.ptr, i8
  llvm.return %c : !llvm.ptr
}

// -----

// CHECK-LABEL: fold_shufflevector
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]: vector<1xf32>, %[[ARG2:[[:alnum:]]+]]: vector<1xf32>
llvm.func @fold_shufflevector(%v1 : vector<1xf32>, %v2 : vector<1xf32>) -> vector<1xf32> {
  // CHECK-NOT: llvm.shufflevector
  %c = llvm.shufflevector %v1, %v2 [0] : vector<1xf32>
  // CHECK: llvm.return %[[ARG1]]
  llvm.return %c : vector<1xf32>
}

// -----

// Check that LLVM constants participate in cross-dialect constant folding. The
// resulting constant is created in the arith dialect because the last folded
// operation belongs to it.
// CHECK-LABEL: llvm_constant
llvm.func @llvm_constant() -> i32 {
  // CHECK-NOT: llvm.mlir.constant
  %0 = llvm.mlir.constant(40 : i32) : i32
  %1 = llvm.mlir.constant(42 : i32) : i32
  // CHECK: %[[RES:.*]] = arith.constant 82 : i32
  // CHECK-NOT: arith.addi
  %2 = arith.addi %0, %1 : i32
  // CHECK: return %[[RES]]
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: load_dce
// CHECK-NEXT: llvm.return
llvm.func @load_dce(%x : !llvm.ptr) {
  %0 = llvm.load %x : !llvm.ptr -> i8
  llvm.return
}

// -----

llvm.mlir.global external @fp() : !llvm.ptr

// CHECK-LABEL: addr_dce
// CHECK-NEXT: llvm.return
llvm.func @addr_dce(%x : !llvm.ptr) {
  %0 = llvm.mlir.addressof @fp : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: alloca_dce
// CHECK-NEXT: llvm.return
llvm.func @alloca_dce() {
  %c1_i64 = arith.constant 1 : i64
  %0 = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: func @volatile_load
llvm.func @volatile_load(%x : !llvm.ptr) {
  // A volatile load may have side-effects such as a write operation to arbitrary memory.
  // Make sure it is not removed.
  // CHECK: llvm.load volatile
  %0 = llvm.load volatile %x : !llvm.ptr -> i8
  // Same with monotonic atomics and any stricter modes.
  // CHECK: llvm.load %{{.*}} atomic monotonic
  %2 = llvm.load %x atomic monotonic { alignment = 1 } : !llvm.ptr -> i8
  // But not unordered!
  // CHECK-NOT: llvm.load %{{.*}} atomic unordered
  %3 = llvm.load %x  atomic unordered { alignment = 1 } : !llvm.ptr -> i8
  llvm.return
}

// -----

// CHECK-LABEL: func @inline_asm_side_effects
llvm.func @inline_asm_side_effects(%x : i32) {
  // CHECK-NOT: llvm.inline_asm "pure inline asm"
  llvm.inline_asm "pure inline asm", "r" %x : (i32) -> ()
  // CHECK: llvm.inline_asm has_side_effects "inline asm with side effects"
  llvm.inline_asm has_side_effects "inline asm with side effects", "r" %x : (i32) -> ()
  llvm.return
}
