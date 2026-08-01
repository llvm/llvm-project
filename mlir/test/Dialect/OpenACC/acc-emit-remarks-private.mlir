// RUN: mlir-opt %s -acc-emit-remarks-private --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-private | Function=firstpriv_implicit | Remark="Generating implicit firstprivate(t)"
// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-private | Function=firstpriv_explicit | Remark="Generating firstprivate(t)"
// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-private | Function=private_loop_implicit | Remark="Generating implicit private(x)"
// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-private | Function=private_loop_unknown | Remark="Generating implicit private(<unknown>)"
// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-private | Function=private_explicit | Remark="Generating private(x)"
// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-private | Function=multi_private_implicit | Remark="Generating implicit private(a,b)"

acc.firstprivate.recipe @firstprivatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
} copy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
} destroy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  memref.dealloc %arg1 : memref<i32>
  acc.terminator
}

acc.private.recipe @privatization_memref_i64 : memref<i64> init {
^bb0(%arg0: memref<i64>):
  %0 = memref.alloca() : memref<i64>
  acc.yield %0 : memref<i64>
} destroy {
^bb0(%arg0: memref<i64>, %arg1: memref<i64>):
  memref.dealloc %arg1 : memref<i64>
  acc.terminator
}

// CHECK-LABEL: func.func @firstpriv_implicit
// CHECK: acc.firstprivate
// CHECK: acc.parallel firstprivate
func.func @firstpriv_implicit() {
  %c1336 = arith.constant 1336 : i32
  %alloc = memref.alloca() : memref<i32>
  memref.store %c1336, %alloc[] : memref<i32>
  %fp = acc.firstprivate varPtr(%alloc : memref<i32>) recipe(@firstprivatization_memref_i32) -> memref<i32> {implicit = true, name = "t"}
  acc.parallel firstprivate(%fp : memref<i32>) {
    %c1 = arith.constant 1 : i32
    %v = memref.load %fp[] : memref<i32>
    %add = arith.addi %v, %c1 : i32
    memref.store %add, %fp[] : memref<i32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @firstpriv_explicit
func.func @firstpriv_explicit() {
  %c1336 = arith.constant 1336 : i32
  %alloc = memref.alloca() : memref<i32>
  memref.store %c1336, %alloc[] : memref<i32>
  %fp = acc.firstprivate varPtr(%alloc : memref<i32>) recipe(@firstprivatization_memref_i32) -> memref<i32> {name = "t"}
  acc.parallel firstprivate(%fp : memref<i32>) {
    %c1 = arith.constant 1 : i32
    %v = memref.load %fp[] : memref<i32>
    %add = arith.addi %v, %c1 : i32
    memref.store %add, %fp[] : memref<i32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @private_loop_implicit
func.func @private_loop_implicit(%arg0 : memref<i64>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %priv = acc.private varPtr(%arg0 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {implicit = true, name = "x"}
  acc.loop private(%priv : memref<i64>) control(%siv : index) = (%c1 : index) to (%c16 : index) step (%c1 : index) {
    %iv_i64 = arith.index_cast %siv : index to i64
    memref.store %iv_i64, %priv[] : memref<i64>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// CHECK-LABEL: func.func @private_loop_unknown
func.func @private_loop_unknown(%arg0 : memref<i64>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %priv = acc.private varPtr(%arg0 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {implicit = true, name = ""}
  acc.loop private(%priv : memref<i64>) control(%siv : index) = (%c1 : index) to (%c16 : index) step (%c1 : index) {
    %iv_i64 = arith.index_cast %siv : index to i64
    memref.store %iv_i64, %priv[] : memref<i64>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// CHECK-LABEL: func.func @private_explicit
func.func @private_explicit(%arg0 : memref<i64>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %priv = acc.private varPtr(%arg0 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {name = "x"}
  acc.parallel private(%priv : memref<i64>) {
    acc.loop control(%siv : index) = (%c1 : index) to (%c16 : index) step (%c1 : index) {
      %iv_i64 = arith.index_cast %siv : index to i64
      memref.store %iv_i64, %priv[] : memref<i64>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @multi_private_implicit
func.func @multi_private_implicit(%arg0 : memref<i64>, %arg1 : memref<i64>) {
  %privA = acc.private varPtr(%arg0 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {implicit = true, name = "a"}
  %privB = acc.private varPtr(%arg1 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {implicit = true, name = "b"}
  acc.parallel private(%privA, %privB : memref<i64>, memref<i64>) {
    acc.yield
  }
  return
}
