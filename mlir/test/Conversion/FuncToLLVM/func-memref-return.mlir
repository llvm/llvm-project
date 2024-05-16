// RUN: mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts %s | FileCheck %s

// RUN: mlir-opt -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1'  %s | FileCheck %s --check-prefix=BAREPTR

// RUN: mlir-opt -transform-interpreter %s | FileCheck %s --check-prefix=BAREPTR

// These tests were separated from func-memref.mlir because applying
// -reconcile-unrealized-casts resulted in `llvm.extractvalue` ops getting
// folded away.

// CHECK-LABEL: func @check_static_return
// CHECK-COUNT-2: !llvm.ptr
// CHECK-COUNT-5: i64
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr) -> !llvm.ptr {
func.func @check_static_return(%static : memref<32x18xf32>) -> memref<32x18xf32> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base0:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(0 : index) : i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base1:.*]] = llvm.extractvalue %[[ins4]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[base1]] : !llvm.ptr
  return %static : memref<32x18xf32>
}

// CHECK-LABEL: func @check_static_return_with_offset
// CHECK-COUNT-2: !llvm.ptr
// CHECK-COUNT-5: i64
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return_with_offset
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr) -> !llvm.ptr {
func.func @check_static_return_with_offset(%static : memref<32x18xf32, strided<[22,1], offset: 7>>) -> memref<32x18xf32, strided<[22,1], offset: 7>> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base0:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(7 : index) : i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(22 : index) : i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base1:.*]] = llvm.extractvalue %[[ins4]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[base1]] : !llvm.ptr
  return %static : memref<32x18xf32, strided<[22,1], offset: 7>>
}


// BAREPTR: llvm.func @foo(!llvm.ptr) -> !llvm.ptr
func.func private @foo(memref<10xi8>) -> memref<20xi8>

// BAREPTR-LABEL: func @check_memref_func_call
// BAREPTR-SAME:    %[[in:.*]]: !llvm.ptr) -> !llvm.ptr
func.func @check_memref_func_call(%in : memref<10xi8>) -> memref<20xi8> {
  // BAREPTR:         %[[inDesc:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0]
  // BAREPTR-NEXT:    %[[barePtr:.*]] = llvm.extractvalue %[[inDesc]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[call:.*]] = llvm.call @foo(%[[barePtr]]) : (!llvm.ptr) -> !llvm.ptr
  // BAREPTR-NEXT:    %[[desc0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[desc1:.*]] = llvm.insertvalue %[[call]], %[[desc0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[desc2:.*]] = llvm.insertvalue %[[call]], %[[desc1]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
  // BAREPTR-NEXT:    %[[desc4:.*]] = llvm.insertvalue %[[c0]], %[[desc2]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c20:.*]] = llvm.mlir.constant(20 : index) : i64
  // BAREPTR-NEXT:    %[[desc6:.*]] = llvm.insertvalue %[[c20]], %[[desc4]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
  // BAREPTR-NEXT:    %[[outDesc:.*]] = llvm.insertvalue %[[c1]], %[[desc6]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %res = call @foo(%in) : (memref<10xi8>) -> (memref<20xi8>)
  // BAREPTR-NEXT:    %[[res:.*]] = llvm.extractvalue %[[outDesc]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    llvm.return %[[res]] : !llvm.ptr
  return %res : memref<20xi8>
}

// BAREPTR-LABEL: func @check_return(
// BAREPTR-SAME: %{{.*}}: memref<?xi8>) -> memref<?xi8>
func.func @check_return(%in : memref<?xi8>) -> memref<?xi8> {
  // BAREPTR: llvm.return {{.*}} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  return %in : memref<?xi8>
}

// BAREPTR-LABEL: func @unconvertible_multiresult
// BAREPTR-SAME: %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>)
func.func @unconvertible_multiresult(%arg0: memref<?xf32> , %arg1: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>) {
  return %arg0, %arg1 : memref<?xf32>, memref<?xf32>
}

// BAREPTR-LABEL: func @unranked_memref(
// BAREPTR-SAME: %{{.*}}: memref<*xi32>)
func.func @unranked_memref(%arg0:memref<*xi32>) {
  // BAREPTR:      call @printMemrefI32(%arg{{.*}}) : (memref<*xi32>) -> ()
  // BAREPTR-NEXT: llvm.return
  call @printMemrefI32(%arg0) : (memref<*xi32>) -> ()
  return
}
func.func private @printMemrefI32(memref<*xi32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %toplevel_module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %toplevel_module
      : (!transform.any_op) -> !transform.any_op
    transform.apply_conversion_patterns to %func {
      transform.apply_conversion_patterns.func.func_to_llvm
    } with type_converter {
      transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
        {use_bare_ptr_call_conv = true, use_opaque_pointers = true}
    } {
      legal_dialects = ["llvm"],
      partial_conversion
    } : !transform.any_op
    transform.yield
  }
}
