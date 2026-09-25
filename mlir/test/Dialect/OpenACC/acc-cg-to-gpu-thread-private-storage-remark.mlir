// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" \
// RUN:   --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

// CHECK: Function=one_array | Remark="Thread-private storage used for buf"
func.func @one_array() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %priv = acc.privatize par_dims(#acc<par_dims[thread_x]>)
      : () -> !acc.private_type<memref<4xi32>>

  acc.compute_region ins(%priv_in = %priv) : (!acc.private_type<memref<4xi32>>) {
    %c0_k = arith.constant 0 : index
    %c1_k = arith.constant 1 : i32
    %local = acc.private_local %priv_in {acc.var_name = #acc.var_name<"buf">}
        : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
    memref.store %c1_k, %local[%c0_k] : memref<4xi32>
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// CHECK: Function=two_arrays | Remark="Thread-private storage used for a,b"
func.func @two_arrays() {
  %priv_a = acc.privatize par_dims(#acc<par_dims[thread_x]>)
      : () -> !acc.private_type<memref<4xi32>>
  %priv_b = acc.privatize par_dims(#acc<par_dims[thread_x]>)
      : () -> !acc.private_type<memref<8xi32>>

  acc.compute_region ins(%a_in = %priv_a, %b_in = %priv_b)
      : (!acc.private_type<memref<4xi32>>, !acc.private_type<memref<8xi32>>) {
    %c0_k = arith.constant 0 : index
    %c1_k = arith.constant 1 : i32
    %a = acc.private_local %a_in {acc.var_name = #acc.var_name<"a">}
        : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
    %b = acc.private_local %b_in {acc.var_name = #acc.var_name<"b">}
        : (!acc.private_type<memref<8xi32>>) -> memref<8xi32>
    memref.store %c1_k, %a[%c0_k] : memref<4xi32>
    memref.store %c1_k, %b[%c0_k] : memref<8xi32>
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// CHECK: Function=scalar | Remark="Thread-private storage used for s"
func.func @scalar() {
  %priv = acc.privatize par_dims(#acc<par_dims[thread_x]>)
      : () -> !acc.private_type<memref<i32>>

  acc.compute_region ins(%priv_in = %priv) : (!acc.private_type<memref<i32>>) {
    %c1_k = arith.constant 1 : i32
    %local = acc.private_local %priv_in {acc.var_name = #acc.var_name<"s">}
        : (!acc.private_type<memref<i32>>) -> memref<i32>
    memref.store %c1_k, %local[] : memref<i32>
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}
