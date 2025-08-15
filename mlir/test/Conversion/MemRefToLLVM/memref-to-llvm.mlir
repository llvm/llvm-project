// RUN: mlir-opt -finalize-memref-to-llvm %s -split-input-file | FileCheck --check-prefixes=ALL,CHECK %s
// RUN: mlir-opt -finalize-memref-to-llvm='index-bitwidth=32' %s -split-input-file | FileCheck --check-prefix=CHECK32 %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass. This produces slightly different IR
// because the conversion target is set up differently.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=memref" --split-input-file %s | FileCheck --check-prefixes=ALL,CHECK-INTERFACE %s

// TODO: In some (all?) cases, CHECK and CHECK-INTERFACE outputs are identical.
// Use a common prefix instead (e.g. ALL).

// CHECK-LABEL: func @view(
// CHECK: %[[ARG0F:.*]]: index, %[[ARG1F:.*]]: index, %[[ARG2F:.*]]: index
// CHECK-INTERFACE-LABEL: func @view(
// CHECK-INTERFACE-NOT: memref.alloc
// CHECK-INTERFACE-NOT: memref.view
func.func @view(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2F]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0F]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1F]]
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() : memref<2048xi8>

  // Test two dynamic sizes.
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]][%[[ARG2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG0]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %1 = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size.
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_2:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_2:.*]] = llvm.getelementptr %[[BASE_PTR_2]][%[[ARG2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR_2]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_2]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %3 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes.
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_3:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_3:.*]] = llvm.getelementptr %[[BASE_PTR_3]][%[[ARG2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR_3]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_3:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_3]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %5 = memref.view %0[%arg2][] : memref<2048xi8> to memref<64x4xf32>

  // Test view memory space.
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr<4>, ptr<4>, i64, array<1 x i64>, array<1 x i64>)>
  %6 = memref.alloc() : memref<2048xi8, 4>

  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_4:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_4:.*]] = llvm.getelementptr %[[BASE_PTR_4]][%[[ARG2]]] : (!llvm.ptr<4>, i64) -> !llvm.ptr<4>, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR_4]], %{{.*}}[1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_4:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_4]], %{{.*}}[2] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  %7 = memref.view %6[%arg2][] : memref<2048xi8, 4> to memref<64x4xf32, 4>

  return
}

// -----

// CHECK-LABEL: func @view_empty_memref(
// CHECK:        %[[ARG0:.*]]: index,
// CHECK:        %[[ARG1:.*]]: memref<0xi8>)

// CHECK-INTERFACE-LABEL: func @view_empty_memref(
// CHECK-INTERFACE:        %[[ARG0:.*]]: index,
// CHECK-INTERFACE:        %[[ARG1:.*]]: memref<0xi8>)
func.func @view_empty_memref(%offset: index, %mem: memref<0xi8>) {

  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK-INTERFACE: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.constant(0 : index) : i64
  // CHECK-INTERFACE: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.constant(4 : index) : i64
  // CHECK-INTERFACE: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.constant(1 : index) : i64
  // CHECK-INTERFACE: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.constant(0 : index) : i64
  // CHECK-INTERFACE: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.constant(4 : index) : i64
  // CHECK-INTERFACE: = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %0 = memref.view %mem[%offset][] : memref<0xi8> to memref<0x4xf32>

  return
}

// -----

// ALL-LABEL:   func.func @view_memref_as_rank0(
// ALL-SAME:      %[[OFFSET:.*]]: index,
// ALL-SAME:      %[[MEM:.*]]: memref<2xi8>) {
func.func @view_memref_as_rank0(%offset: index, %mem: memref<2xi8>) {

  // ALL:  builtin.unrealized_conversion_cast %[[OFFSET]] : index to i64
  // ALL:  builtin.unrealized_conversion_cast %[[MEM]] : memref<2xi8> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // ALL:  llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  // ALL:  llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // ALL:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64)>
  // ALL:  llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // ALL:  llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // ALL:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
  // ALL:  llvm.mlir.constant(0 : index) : i64
  // ALL:  llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64)>
  %memref_view_bf16 = memref.view %mem[%offset][] : memref<2xi8> to memref<bf16>

  return
}

// -----

// Subviews needs to be expanded outside of the memref-to-llvm pass.
// CHECK-LABEL: func @subview(
// CHECK:         %[[MEMREF:.*]]: memref<{{.*}}>,
// CHECK:         %[[ARG0:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2:.*]]: index)
// CHECK32-LABEL: func @subview(
// CHECK32:         %[[MEMREF:.*]]: memref<{{.*}}>,
// CHECK32:         %[[ARG0:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG1:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG2:.*]]: index)
// CHECK-INTERFACE-LABEL: func @subview(
func.func @subview(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: memref.subview %[[MEMREF]][%[[ARG0]], %[[ARG1]]] [%[[ARG0]], %[[ARG1]]]
  // CHECK32: memref.subview %[[MEMREF]][%[[ARG0]], %[[ARG1]]] [%[[ARG0]], %[[ARG1]]] [%[[ARG0]], %[[ARG1]]]
  // CHECK-INTERFACE: memref.subview
  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
  to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}

// -----

// CHECK-LABEL: func @assume_alignment(
// CHECK-INTERFACE-LABEL: func @assume_alignment(
func.func @assume_alignment(%0 : memref<4x4xf16>) {
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[MEMREF:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-NEXT: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
  // CHECK-NEXT: %[[ALIGN:.*]] = llvm.mlir.constant(16 : index) : i64
  // CHECK-NEXT: llvm.intr.assume %[[TRUE]] ["align"(%[[PTR]], %[[ALIGN]] : !llvm.ptr, i64)] : i1
  // CHECK-INTERFACE: llvm.intr.assume
  %1 = memref.assume_alignment %0, 16 : memref<4x4xf16>
  return
}

// -----

// CHECK-LABEL: func @assume_alignment_w_offset
// CHECK-INTERFACE-LABEL: func @assume_alignment_w_offset
func.func @assume_alignment_w_offset(%0 : memref<4x4xf16, strided<[?, ?], offset: ?>>) {
  // CHECK-DAG: %[[PTR:.*]] = llvm.extractvalue %[[MEMREF:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[OFFSET:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[BUFF_ADDR:.*]] =  llvm.getelementptr %[[PTR]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
  // CHECK-DAG: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
  // CHECK-DAG: %[[ALIGN:.*]] = llvm.mlir.constant(16 : index) : i64
  // CHECK-NEXT: llvm.intr.assume %[[TRUE]] ["align"(%[[BUFF_ADDR]], %[[ALIGN]] : !llvm.ptr, i64)] : i1
  // CHECK-INTERFACE: llvm.intr.assume
  %1 = memref.assume_alignment %0, 16 : memref<4x4xf16, strided<[?, ?], offset: ?>>
  return
}
// -----

// CHECK-LABEL: func @dim_of_unranked
// CHECK32-LABEL: func @dim_of_unranked
// CHECK-INTERFACE-LABEL: func @dim_of_unranked
func.func @dim_of_unranked(%unranked: memref<*xi32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %unranked, %c0 : memref<*xi32>
  return %dim : index
}
// CHECK: %[[UNRANKED_DESC:.*]] = builtin.unrealized_conversion_cast

// CHECK: %[[RANKED_DESC:.*]] = llvm.extractvalue %[[UNRANKED_DESC]][1]
// CHECK-SAME:   : !llvm.struct<(i64, ptr)>

// CHECK: %[[OFFSET_PTR:.*]] = llvm.getelementptr %[[RANKED_DESC]]{{\[}}
// CHECK-SAME:   0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64)>

// CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[INDEX_INC:.*]] = llvm.add %[[C1]], %{{.*}} : i64

// CHECK: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[OFFSET_PTR]]{{\[}}
// CHECK-SAME:   %[[INDEX_INC]]] : (!llvm.ptr, i64) -> !llvm.ptr

// CHECK: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64

// CHECK32: %[[SIZE:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32

// CHECK-INTERFACE-NOT: memref.dim

// -----

// CHECK-LABEL: func @address_space(
// CHECK-INTERFACE-LABEL: func @address_space(
func.func @address_space(%arg0 : memref<32xf32, affine_map<(d0) -> (d0)>, 7>) {
  // CHECK: %[[MEMORY:.*]] = llvm.call @malloc(%{{.*}})
  // CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[MEMORY]] : !llvm.ptr to !llvm.ptr<5>
  // CHECK: llvm.insertvalue %[[CAST]], %{{[[:alnum:]]+}}[0]
  // CHECK: llvm.insertvalue %[[CAST]], %{{[[:alnum:]]+}}[1]
  // CHECK-INTERFACE: llvm.call @malloc
  // CHECK-INTERFACE: llvm.addrspacecast
  %0 = memref.alloc() : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  %1 = arith.constant 7 : index
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK-INTERFACE: llvm.load
  %2 = memref.load %0[%1] : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  func.return
}

// -----

// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-INTERFACE-LABEL: func @transpose
// CHECK-INTERFACE-NOT: memref.transpose
func.func @transpose(%arg0: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  %0 = memref.transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?x?xf32, strided<[1, ?, ?], offset: ?>>
  return
}

// -----

// CHECK:   llvm.mlir.global external @gv0() {addr_space = 0 : i32} : !llvm.array<2 x f32> {
// CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.array<2 x f32>
// CHECK-NEXT:     llvm.return %0 : !llvm.array<2 x f32>
// CHECK-NEXT:   }
// CHECK-INTERFACE: llvm.mlir.global external
memref.global @gv0 : memref<2xf32> = uninitialized

// CHECK: llvm.mlir.global private @gv1() {addr_space = 0 : i32} : !llvm.array<2 x f32>
// CHECK-INTERFACE: llvm.mlir.global private
memref.global "private" @gv1 : memref<2xf32>

// CHECK: llvm.mlir.global external @gv2(dense<{{\[\[}}0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]]> : tensor<2x3xf32>) {addr_space = 0 : i32} : !llvm.array<2 x array<3 x f32>>
// CHECK-INTERFACE: llvm.mlir.global external
memref.global @gv2 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>

// Test 1D memref.
// CHECK-LABEL: func @get_gv0_memref
// CHECK-INTERFACE-LABEL: func @get_gv0_memref
func.func @get_gv0_memref() {
  %0 = memref.get_global @gv0 : memref<2xf32>
  // CHECK: %[[DIM:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv0 : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x f32>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM]], {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE]], {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.addressof
  // CHECK-INTERFACE: llvm.getelementptr
  return
}

// Test 2D memref.
// CHECK-LABEL: func @get_gv2_memref
// CHECK-INTERFACE-LABEL: func @get_gv2_memref
func.func @get_gv2_memref() {
  // CHECK: %[[DIM0:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[DIM1:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv2 : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<3 x f32>>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM0]], {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE1]], {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-INTERFACE: llvm.mlir.addressof
  // CHECK-INTERFACE: llvm.getelementptr

  %0 = memref.get_global @gv2 : memref<2x3xf32>
  return
}

// Test scalar memref.
// CHECK: llvm.mlir.global external @gv3(1.000000e+00 : f32) {addr_space = 0 : i32} : f32
memref.global @gv3 : memref<f32> = dense<1.0>

// CHECK-LABEL: func @get_gv3_memref
// CHECK-INTERFACE-LABEL: func @get_gv3_memref
func.func @get_gv3_memref() {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv3 : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0] : (!llvm.ptr) -> !llvm.ptr, f32
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK-INTERFACE: llvm.mlir.addressof
  // CHECK-INTERFACE: llvm.getelementptr
  %0 = memref.get_global @gv3 : memref<f32>
  return
}

// Test scalar memref with an alignment.
// CHECK: llvm.mlir.global private @gv4(1.000000e+00 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
// CHECK-INTERFACE: llvm.mlir.global private
memref.global "private" @gv4 : memref<f32> = dense<1.0> {alignment = 64}

// -----

// Expand shapes need to be expanded outside of the memref-to-llvm pass.
// CHECK-LABEL: func @expand_shape_static(
// CHECK-SAME:         %[[ARG:.*]]: memref<{{.*}}>)
// CHECK-INTERFACE-LABEL: func @expand_shape_static(
func.func @expand_shape_static(%arg0: memref<3x4x5xf32>) -> memref<1x3x4x1x5xf32> {
  // CHECK: memref.expand_shape %[[ARG]] {{\[}}[0, 1], [2], [3, 4]] output_shape [1, 3, 4, 1, 5]
  // CHECK-INTERFACE: memref.expand_shape
  // Reshapes that expand a contiguous tensor with some 1's.
  %0 = memref.expand_shape %arg0 [[0, 1], [2], [3, 4]] output_shape [1, 3, 4, 1, 5]
      : memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  return %0 : memref<1x3x4x1x5xf32>
}

// -----

// Collapse shapes need to be expanded outside of the memref-to-llvm pass.
// CHECK-LABEL: func @collapse_shape_static
// CHECK-SAME: %[[ARG:.*]]: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
// CHECK-INTERFACE-LABEL: func @collapse_shape_static
func.func @collapse_shape_static(%arg0: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
  // CHECK: memref.collapse_shape %[[ARG]]
  // CHECK-INTERFACE: memref.collapse_shape
  %0 = memref.collapse_shape %arg0 [[0, 1], [2], [3, 4]] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  return %0 : memref<3x4x5xf32>
}

// -----

// CHECK-LABEL: func @rank_of_unranked
// CHECK32-LABEL: func @rank_of_unranked
// CHECK-INTERFACE-LABEL: func @rank_of_unranked
func.func @rank_of_unranked(%unranked: memref<*xi32>) {
  %rank = memref.rank %unranked : memref<*xi32>
  return
}
// CHECK: %[[UNRANKED_DESC:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue %[[UNRANKED_DESC]][0] : !llvm.struct<(i64, ptr)>
// CHECK32: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, ptr)>
// CHECK-INTERFACE-NOT: memref.rank

// -----

// CHECK-LABEL: func @rank_of_ranked
// CHECK32-LABEL: func @rank_of_ranked
// CHECK-INTERFACE-LABEL: func @rank_of_ranked
func.func @rank_of_ranked(%ranked: memref<?xi32>) {
  %rank = memref.rank %ranked : memref<?xi32>
  return
}
// CHECK: llvm.mlir.constant(1 : index) : i64
// CHECK32: llvm.mlir.constant(1 : index) : i32
// CHECK-INTERFACE: llvm.mlir.constant(1 : index) : i64

// -----

// CHECK-LABEL: func @atomic_rmw
// CHECK-INTERFACE-LABEL: func @atomic_rmw
func.func @atomic_rmw(%I : memref<10xi32>, %ival : i32, %F : memref<10xf32>, %fval : f32, %i : index) {
  memref.atomic_rmw assign %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw xchg %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw addi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw add %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxs %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw max %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw mins %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw min %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umax %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw minu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umin %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw addf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maximumf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fmaximum %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxnumf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fmax %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw minimumf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fminimum %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw minnumf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fmin %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw ori %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _or %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw andi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _and %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw xori %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _xor %{{.*}}, %{{.*}} acq_rel
  // CHECK-INTERFACE-COUNT-14: llvm.atomicrmw
  return
}

// -----

func.func @atomic_rmw_with_offset(%I : memref<10xi32, strided<[1], offset: 5>>, %ival : i32, %i : index) {
  memref.atomic_rmw andi %ival, %I[%i] : (i32, memref<10xi32, strided<[1], offset: 5>>) -> i32
  return
}
// CHECK-LABEL:  func @atomic_rmw_with_offset
// CHECK-SAME:   %[[ARG0:.+]]: memref<10xi32, strided<[1], offset: 5>>
// CHECK-SAME:   %[[ARG1:.+]]: i32
// CHECK-SAME:   %[[ARG2:.+]]: index
// CHECK-DAG:    %[[MEMREF_STRUCT:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<10xi32, strided<[1], offset: 5>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:    %[[INDEX:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : index to i64
// CHECK:        %[[BASE_PTR:.+]] = llvm.extractvalue %[[MEMREF_STRUCT]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:        %[[OFFSET:.+]] = llvm.mlir.constant(5 : index) : i64
// CHECK:        %[[OFFSET_PTR:.+]] = llvm.getelementptr %[[BASE_PTR]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:        %[[PTR:.+]] = llvm.getelementptr %[[OFFSET_PTR]][%[[INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:        llvm.atomicrmw _and %[[PTR]], %[[ARG1]] acq_rel

// CHECK-INTERFACE-LABEL:  func @atomic_rmw_with_offset
// CHECK-INTERFACE: llvm.atomicrmw

// -----

// CHECK-LABEL: func @generic_atomic_rmw
// CHECK-INTERFACE-LABEL: func @generic_atomic_rmw
llvm.func @generic_atomic_rmw() {
  %I = "test.foo"() : () -> (memref<10xi32>)
  %i = "test.foo"() : () -> (index)
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xi32> {
    ^bb0(%old_value : i32):
      memref.atomic_yield %old_value : i32
  }
  llvm.return
}
// CHECK:        %[[INIT:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK-NEXT:   llvm.br ^bb1(%[[INIT]] : i32)
// CHECK-NEXT: ^bb1(%[[LOADED:.*]]: i32):
// CHECK-NEXT:   %[[PAIR:.*]] = llvm.cmpxchg %{{.*}}, %[[LOADED]], %[[LOADED]]
// CHECK-SAME:                      acq_rel monotonic : !llvm.ptr, i32
// CHECK-NEXT:   %[[NEW:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK-NEXT:   %[[OK:.*]] = llvm.extractvalue %[[PAIR]][1]
// CHECK-NEXT:   llvm.cond_br %[[OK]], ^bb2, ^bb1(%[[NEW]] : i32)

// CHECK-INTERFACE: llvm.cmpxchg

// -----

// CHECK-LABEL: func @generic_atomic_rmw_in_alloca_scope
// CHECK-INTERFACE-LABEL: func @generic_atomic_rmw_in_alloca_scope
llvm.func @generic_atomic_rmw_in_alloca_scope() {
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<2x3xi32>
  memref.alloca_scope  {
    %0 = memref.generic_atomic_rmw %alloc[%c1, %c1] : memref<2x3xi32> {
    ^bb0(%arg0: i32):
      memref.atomic_yield %arg0 : i32
    }
  }
  llvm.return
}
// CHECK:        %[[STACK_SAVE:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-NEXT:   llvm.br ^bb1
// CHECK:      ^bb1:
// CHECK:        %[[INIT:.*]] = llvm.load %[[BUF:.*]] : !llvm.ptr -> i32
// CHECK-NEXT:   llvm.br ^bb2(%[[INIT]] : i32)
// CHECK-NEXT: ^bb2(%[[LOADED:.*]]: i32):
// CHECK-NEXT:   %[[PAIR:.*]] = llvm.cmpxchg %[[BUF]], %[[LOADED]], %[[LOADED]]
// CHECK-SAME:     acq_rel monotonic : !llvm.ptr, i32
// CHECK-NEXT:   %[[NEW:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK-NEXT:   %[[OK:.*]] = llvm.extractvalue %[[PAIR]][1]
// CHECK-NEXT:   llvm.cond_br %[[OK]], ^bb3, ^bb2(%[[NEW]] : i32)
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   llvm.intr.stackrestore %[[STACK_SAVE]] : !llvm.ptr
// CHECK-NEXT:   llvm.br ^bb4
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   return

// CHECK-INTERFACE: llvm.cmpxchg

// -----

// CHECK-LABEL: func @memref_copy_ranked
// CHECK-INTERFACE-LABEL: func @memref_copy_ranked
func.func @memref_copy_ranked() {
  %0 = memref.alloc() : memref<2xf32>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xf32> to memref<?xf32>
  %2 = memref.alloc() : memref<2xf32>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xf32> to memref<?xf32>
  memref.copy %1, %3 : memref<?xf32> to memref<?xf32>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[EXTRACT0:%.*]] = llvm.extractvalue {{%.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[MUL:%.*]] = llvm.mul [[ONE]], [[EXTRACT0]] : i64
  // CHECK: [[NULL:%.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, f32
  // CHECK: [[PTRTOINT:%.*]] = llvm.ptrtoint [[GEP]] : !llvm.ptr to i64
  // CHECK: [[SIZE:%.*]] = llvm.mul [[MUL]], [[PTRTOINT]] : i64
  // CHECK: [[EXTRACT1P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[EXTRACT1O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GEP1:%.*]] = llvm.getelementptr [[EXTRACT1P]][[[EXTRACT1O]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[EXTRACT2P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[EXTRACT2O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GEP2:%.*]] = llvm.getelementptr [[EXTRACT2P]][[[EXTRACT2O]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: "llvm.intr.memcpy"([[GEP2]], [[GEP1]], [[SIZE]]) <{isVolatile = false}>
  // CHECK-INTERFACE: llvm.intr.memcpy
  return
}


// -----

// CHECK-LABEL: func @memref_copy_contiguous
// CHECK-INTERFACE-LABEL: func @memref_copy_contiguous
func.func @memref_copy_contiguous(%in: memref<16x4xi32>, %offset: index) {
  %buf = memref.alloc() : memref<1x2xi32>
  %sub = memref.subview %in[%offset, 0] [1, 2] [1, 1] : memref<16x4xi32> to memref<1x2xi32, strided<[4, 1], offset: ?>>
  memref.copy %sub, %buf : memref<1x2xi32, strided<[4, 1], offset: ?>> to memref<1x2xi32>
  // Skip the memref descriptor of the alloc.
  // CHECK: llvm.insertvalue {{%.*}}, {{%.*}}[4, 1]
  // Get the memref for the subview.
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [1, 2] [1, 1] : memref<16x4xi32> to memref<1x2xi32, strided<[4, 1], offset: ?>>
  // CHECK: %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[SUBVIEW]] : memref<1x2xi32, strided<[4, 1], offset: ?>> to !llvm.struct<(ptr
  // CHECK: [[EXTRACT0:%.*]] = llvm.extractvalue %[[DESC]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[MUL1:%.*]] = llvm.mul {{.*}}, [[EXTRACT0]] : i64
  // CHECK: [[EXTRACT1:%.*]] = llvm.extractvalue %[[DESC]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[MUL2:%.*]] = llvm.mul [[MUL1]], [[EXTRACT1]] : i64
  // CHECK: [[NULL:%.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, i32
  // CHECK: [[PTRTOINT:%.*]] = llvm.ptrtoint [[GEP]] : !llvm.ptr to i64
  // CHECK: [[SIZE:%.*]] = llvm.mul [[MUL2]], [[PTRTOINT]] : i64
  // CHECK: [[EXTRACT1P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[EXTRACT1O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GEP1:%.*]] = llvm.getelementptr [[EXTRACT1P]][[[EXTRACT1O]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK: [[EXTRACT2P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[EXTRACT2O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GEP2:%.*]] = llvm.getelementptr [[EXTRACT2P]][[[EXTRACT2O]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK: "llvm.intr.memcpy"([[GEP2]], [[GEP1]], [[SIZE]]) <{isVolatile = false}>
  // CHECK-INTERFACE: llvm.intr.memcpy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_0d_offset
// CHECK-INTERFACE-LABEL: func @memref_copy_0d_offset
func.func @memref_copy_0d_offset(%in: memref<2xi32>) {
  %buf = memref.alloc() : memref<i32>
  %sub = memref.subview %in[1] [1] [1] : memref<2xi32> to memref<1xi32, strided<[1], offset: 1>>
  %scalar = memref.collapse_shape %sub [] : memref<1xi32, strided<[1], offset: 1>> into memref<i32, strided<[], offset: 1>>
  memref.copy %scalar, %buf : memref<i32, strided<[], offset: 1>> to memref<i32>
  // CHECK: llvm.intr.memcpy
  // CHECK-INTERFACE: llvm.intr.memcpy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_noncontiguous
// CHECK-INTERFACE-LABEL: func @memref_copy_noncontiguous
func.func @memref_copy_noncontiguous(%in: memref<16x2xi32>, %offset: index) {
  %buf = memref.alloc() : memref<2x1xi32>
  %sub = memref.subview %in[%offset, 0] [2, 1] [1, 1] : memref<16x2xi32> to memref<2x1xi32, strided<[2, 1], offset: ?>>
  memref.copy %sub, %buf : memref<2x1xi32, strided<[2, 1], offset: ?>> to memref<2x1xi32>
  // CHECK: llvm.call @memrefCopy
  // CHECK-INTERFACE: llvm.call @memrefCopy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_unranked
// CHECK-INTERFACE-LABEL: func @memref_copy_unranked
func.func @memref_copy_unranked() {
  %0 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xi1> to memref<*xi1>
  %2 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xi1> to memref<*xi1>
  memref.copy %1, %3 : memref<*xi1> to memref<*xi1>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA:%.*]] = llvm.alloca [[ONE]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
  // CHECK: llvm.store {{%.*}}, [[ALLOCA]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
  // CHECK: [[RANK:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[UNDEF:%.*]] = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
  // CHECK: [[INSERT:%.*]] = llvm.insertvalue [[RANK]], [[UNDEF]][0] : !llvm.struct<(i64, ptr)>
  // CHECK: [[INSERT2:%.*]] = llvm.insertvalue [[ALLOCA]], [[INSERT]][1] : !llvm.struct<(i64, ptr)>
  // CHECK: [[STACKSAVE:%.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: [[RANK2:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA2:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
  // CHECK: llvm.store {{%.*}}, [[ALLOCA2]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
  // CHECK: [[ALLOCA3:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
  // CHECK: llvm.store [[INSERT2]], [[ALLOCA3]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
  // CHECK: [[SIZEPTR:%.*]] = llvm.getelementptr {{%.*}}[1] : (!llvm.ptr) -> !llvm.ptr, i1
  // CHECK: [[SIZE:%.*]] = llvm.ptrtoint [[SIZEPTR]] : !llvm.ptr to i64
  // CHECK: llvm.call @memrefCopy([[SIZE]], [[ALLOCA2]], [[ALLOCA3]]) : (i64, !llvm.ptr, !llvm.ptr) -> ()
  // CHECK: llvm.intr.stackrestore [[STACKSAVE]]
  // CHECK-INTERFACE: llvm.call @memrefCopy
  return
}

// -----

// CHECK-LABEL: func @extract_aligned_pointer_as_index
// CHECK-INTERFACE-LABEL: func @extract_aligned_pointer_as_index
func.func @extract_aligned_pointer_as_index(%m: memref<?xf32>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %m: memref<?xf32> -> index
  // CHECK: %[[E:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[I64:.*]] = llvm.ptrtoint %[[E]] : !llvm.ptr to i64
  // CHECK: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I64]] : i64 to index
  // CHECK-INTERFACE-NOT: memref.extract_aligned_pointer_as_index

  // CHECK: return %[[R:.*]] : index
  return %0: index
}

// -----

// CHECK-LABEL: func @extract_aligned_pointer_as_index_unranked
// CHECK-INTERFACE-LABEL: func @extract_aligned_pointer_as_index_unranked
func.func @extract_aligned_pointer_as_index_unranked(%m: memref<*xf32>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %m: memref<*xf32> -> index
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i64, ptr)>
  // CHECK: %[[ALIGNED_FIELD:.*]] = llvm.getelementptr %[[PTR]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  // CHECK: %[[ALIGNED_PTR:.*]] = llvm.load %[[ALIGNED_FIELD]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[I64:.*]] = llvm.ptrtoint %[[ALIGNED_PTR]] : !llvm.ptr to i64
  // CHECK: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I64]] : i64 to index
  // CHECK-INTERFACE-NOT: memref.extract_aligned_pointer_as_index

  // CHECK: return %[[R]] : index
  return %0: index
}

// -----

// CHECK-LABEL: func @extract_strided_metadata(
// CHECK-SAME: %[[ARG:.*]]: memref
// CHECK: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DESC:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[ALIGNED_BASE]], %[[DESC0]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[OFF0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[BASE_BUFFER_DESC:.*]] = llvm.insertvalue %[[OFF0]], %[[DESC1]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[MEM_DESC]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[SIZE0:.*]] = llvm.extractvalue %[[MEM_DESC]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[SIZE1:.*]] = llvm.extractvalue %[[MEM_DESC]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

// CHECK-INTERFACE-LABEL: func @extract_strided_metadata
// CHECK-INTERFACE-NOT: memref.extract_strided_metadata

func.func @extract_strided_metadata(
    %ref: memref<?x?xf32, strided<[?,?], offset: ?>>) {

  %base, %offset, %sizes:2, %strides:2 =
    memref.extract_strided_metadata %ref : memref<?x?xf32, strided<[?,?], offset: ?>>
    -> memref<f32>, index,
       index, index,
       index, index

  return
}

// -----

// CHECK-LABEL: func @load_non_temporal(
// CHECK-INTERFACE-LABEL: func @load_non_temporal(
func.func @load_non_temporal(%arg0 : memref<32xf32, affine_map<(d0) -> (d0)>>) {
  %1 = arith.constant 7 : index
  // CHECK: llvm.load %{{.*}} {nontemporal} : !llvm.ptr -> f32
  // CHECK-INTERFACE: llvm.load
  %2 = memref.load %arg0[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  func.return
}

// -----

// CHECK-LABEL: func @load_with_alignment(
// CHECK-INTERFACE-LABEL: func @load_with_alignment(
func.func @load_with_alignment(%arg0 : memref<32xf32>, %arg1 : index) {
  // CHECK: llvm.load %{{.*}} {alignment = 32 : i64} : !llvm.ptr -> f32
  // CHECK-INTERFACE: llvm.load
  %1 = memref.load %arg0[%arg1] {alignment = 32} : memref<32xf32>
  func.return
}

// -----

// CHECK-LABEL: func @store_non_temporal(
// CHECK-INTERFACE-LABEL: func @store_non_temporal(
func.func @store_non_temporal(%input : memref<32xf32, affine_map<(d0) -> (d0)>>, %output : memref<32xf32, affine_map<(d0) -> (d0)>>) {
  %1 = arith.constant 7 : index
  %2 = memref.load %input[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  // CHECK: llvm.store %{{.*}}, %{{.*}}  {nontemporal} : f32, !llvm.ptr
  // CHECK-INTERFACE: llvm.store
  memref.store %2, %output[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  func.return
}

// -----

// CHECK-LABEL: func @store_with_alignment(
// CHECK-INTERFACE-LABEL: func @store_with_alignment(
func.func @store_with_alignment(%arg0 : memref<32xf32>, %arg1 : f32, %arg2 : index) {
  // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 32 : i64} : f32, !llvm.ptr
  // CHECK-INTERFACE: llvm.store
  memref.store %arg1, %arg0[%arg2] {alignment = 32} : memref<32xf32>
  func.return
}

// -----

// Ensure unconvertable memory space not cause a crash

// CHECK-LABEL: @alloca_unconvertable_memory_space
// CHECK-INTERFACE-LABEL: @alloca_unconvertable_memory_space
func.func @alloca_unconvertable_memory_space() {
  // CHECK: memref.alloca
  // CHECK-INTERFACE: memref.alloca
  %alloca = memref.alloca() : memref<1x32x33xi32, #spirv.storage_class<StorageBuffer>>
  func.return
}
