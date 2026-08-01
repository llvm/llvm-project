// RUN: mlir-opt %s -acc-recipe-materialization \
// RUN:   -mlir-print-debuginfo -mlir-pretty-debuginfo | FileCheck %s

acc.private.recipe @privatization_memref_index : memref<index> init {
^bb0(%arg0: memref<index>):
  %alloca = memref.alloca() : memref<index>
  acc.yield %alloca : memref<index>
} destroy {
^bb0(%arg0: memref<index>, %arg1: memref<index>):
  memref.dealloc %arg1 : memref<index>
  acc.terminator
}

acc.firstprivate.recipe @firstprivatization_memref_index : memref<index> init {
^bb0(%arg0: memref<index>):
  %alloca = memref.alloca() : memref<index>
  acc.yield %alloca : memref<index>
} copy {
^bb0(%arg0: memref<index>, %arg1: memref<index>):
  %0 = memref.load %arg0[] : memref<index>
  memref.store %0, %arg1[] : memref<index>
  acc.terminator
} destroy {
^bb0(%arg0: memref<index>, %arg1: memref<index>):
  memref.dealloc %arg1 : memref<index>
  acc.terminator
}

acc.reduction.recipe @reduction_add_memref_index : memref<index> reduction_operator <add> init {
^bb0(%arg0: memref<index>):
  %c0 = arith.constant 0 : index
  %alloca = memref.alloca() : memref<index>
  memref.store %c0, %alloca[] : memref<index>
  acc.yield %alloca : memref<index>
} combiner {
^bb0(%arg0: memref<index>, %arg1: memref<index>):
  %0 = memref.load %arg0[] : memref<index>
  %1 = memref.load %arg1[] : memref<index>
  %2 = arith.addi %0, %1 : index
  memref.store %2, %arg0[] : memref<index>
  acc.yield %arg0 : memref<index>
} destroy {
^bb0(%arg0: memref<index>, %arg1: memref<index>):
  memref.dealloc %arg1 : memref<index>
  acc.terminator
}

func.func @private_loc(%arg0: memref<index>) {
  %c1 = arith.constant 1 : index
  %0 = acc.private varPtr(%arg0 : memref<index>) recipe(@privatization_memref_index) -> memref<index> {implicit = true, name = "priv0"}
  acc.parallel private(%0 : memref<index>) {
    memref.store %c1, %0[] : memref<index>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  %1 = acc.private varPtr(%arg0 : memref<index>) recipe(@privatization_memref_index) -> memref<index> {implicit = true, name = "priv1"}
  acc.parallel private(%1 : memref<index>) {
    memref.store %c1, %1[] : memref<index>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}

  // CHECK-LABEL: func.func @private_loc
  // CHECK:       acc.parallel
  // CHECK:       [[ALLOCA0:%.*]] = memref.alloca() {{.*}}"priv0"{{.*}}:50
  // CHECK:       memref.dealloc [[ALLOCA0]] {{.*}}:50
  // CHECK:       acc.parallel
  // CHECK:       [[ALLOCA1:%.*]] = memref.alloca() {{.*}}"priv1"{{.*}}:55
  // CHECK:       memref.dealloc [[ALLOCA1]] {{.*}}:55
  return
}

func.func @firstprivate_loc() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloca = memref.alloca() : memref<index>
  memref.store %c0, %alloca[] : memref<index>
  %0 = acc.firstprivate varPtr(%alloca : memref<index>) recipe(@firstprivatization_memref_index) -> memref<index> {implicit = true, name = "firstpriv0"}
  acc.parallel firstprivate(%0 : memref<index>) {
    memref.store %c1, %0[] : memref<index>
    acc.yield
  }
  %1 = acc.firstprivate varPtr(%alloca : memref<index>) recipe(@firstprivatization_memref_index) -> memref<index> {implicit = true, name = "firstpriv1"}
  acc.parallel firstprivate(%1 : memref<index>) {
    memref.store %c1, %1[] : memref<index>
    acc.yield
  }
  // CHECK-LABEL: func.func @firstprivate_loc
  // CHECK:       acc.parallel
  // CHECK:       [[ALLOCA2:%.*]] = memref.alloca() {{.*}}"firstpriv0"{{.*}}:76
  // CHECK-NEXT:  [[LOAD2:%.*]] = memref.load{{.*}}:76
  // CHECK-NEXT:  memref.store [[LOAD2]], [[ALLOCA2]][]{{.*}}:76
  // CHECK:       memref.dealloc [[ALLOCA2]] {{.*}}:76
  // CHECK:       acc.parallel
  // CHECK:       [[ALLOCA3:%.*]] = memref.alloca() {{.*}}"firstpriv1"{{.*}}:81
  // CHECK-NEXT:  [[LOAD3:%.*]] = memref.load{{.*}}:81
  // CHECK-NEXT:  memref.store [[LOAD3]], [[ALLOCA3]][]{{.*}}:81
  // CHECK:       memref.dealloc [[ALLOCA3]] {{.*}}:81
  return
}

func.func @reduction_loc(%arg0: memref<index>) {
  %c1 = arith.constant 1 : index
  %0 = acc.reduction varPtr(%arg0 : memref<index>) recipe(@reduction_add_memref_index) -> memref<index> {name = "r0"}
  acc.parallel reduction(%0 : memref<index>) {
    memref.store %c1, %0[] : memref<index>
    acc.yield
  }
  %1 = acc.reduction varPtr(%arg0 : memref<index>) recipe(@reduction_add_memref_index) -> memref<index> {name = "r1"}
  acc.parallel reduction(%1 : memref<index>) {
    memref.store %c1, %1[] : memref<index>
    acc.yield
  }
  // CHECK-LABEL: func.func @reduction_loc
  // CHECK:       acc.parallel
  // CHECK:       [[INIT0:%.*]] = acc.reduction_init {{.*}} {
  // CHECK:         [[ZERO0:%.*]] = arith.constant 0 : index {{.*}}:102
  // CHECK-NEXT:    [[ALLOCA4:%.*]] = memref.alloca() {{.*}}:102
  // CHECK-NEXT:    memref.store [[ZERO0]], [[ALLOCA4]][]{{.*}}:102
  // CHECK-NEXT:    acc.yield [[ALLOCA4]] {{.*}}:102
  // CHECK-NEXT:  } {{.*}}"r0"{{.*}}:102
  // CHECK:       acc.reduction_combine_region {{.*}} {
  // CHECK-NEXT:    [[LOADX:%.*]] = memref.load{{.*}}:102
  // CHECK-NEXT:    [[LOADY:%.*]] = memref.load{{.*}}:102
  // CHECK-NEXT:    [[ADD0:%.*]] = arith.addi [[LOADX]], [[LOADY]]{{.*}}:102
  // CHECK-NEXT:    memref.store [[ADD0]]{{.*}}:102
  // CHECK-NEXT:  } {{.*}}:102
  // CHECK:       memref.dealloc [[INIT0]]{{.*}}:102
  // CHECK:       acc.parallel
  // CHECK:       [[INIT1:%.*]] = acc.reduction_init {{.*}} {
  // CHECK:         [[ZERO1:%.*]] = arith.constant 0 : index {{.*}}:107
  // CHECK-NEXT:    [[ALLOCA5:%.*]] = memref.alloca() {{.*}}:107
  // CHECK-NEXT:    memref.store [[ZERO1]], [[ALLOCA5]][]{{.*}}:107
  // CHECK-NEXT:    acc.yield [[ALLOCA5]] {{.*}}:107
  // CHECK-NEXT:  } {{.*}}"r1"{{.*}}:107
  // CHECK:       acc.reduction_combine_region {{.*}} {
  // CHECK-NEXT:    [[LOADA:%.*]] = memref.load{{.*}}:107
  // CHECK-NEXT:    [[LOADB:%.*]] = memref.load{{.*}}:107
  // CHECK-NEXT:    [[ADD1:%.*]] = arith.addi [[LOADA]], [[LOADB]]{{.*}}:107
  // CHECK-NEXT:    memref.store [[ADD1]]{{.*}}:107
  // CHECK-NEXT:  } {{.*}}:107
  // CHECK:       memref.dealloc [[INIT1]]{{.*}}:107
  return
}
