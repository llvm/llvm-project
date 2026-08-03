// RUN: mlir-opt %s -acc-emit-remarks-data --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

// Remarks are emitted in function order, and within a data construct they are
// ordered by data clause, then by implicitness, then by variable name.
// SSA names describe the clause; the `name` attribute is the source identifier
// that appears in the remark (short Fortran-style names, unique per function).

acc.private.recipe @privatization_memref_f32 : memref<f32> init {
^bb0(%arg0: memref<f32>):
  %0 = memref.alloca() : memref<f32>
  acc.yield %0 : memref<f32>
}

acc.firstprivate.recipe @firstprivatization_memref_f32 : memref<f32> init {
^bb0(%arg0: memref<f32>):
  %0 = memref.alloca() : memref<f32>
  acc.yield %0 : memref<f32>
} copy {
^bb0(%arg0: memref<f32>, %arg1: memref<f32>):
  %0 = memref.load %arg0[] : memref<f32>
  memref.store %0, %arg1[] : memref<f32>
  acc.terminator
}

acc.reduction.recipe @reduction_add_memref_f32 : memref<f32>
    reduction_operator <add> init {
^bb0(%arg0: memref<f32>):
  %cst = arith.constant 0.0 : f32
  %0 = memref.alloca() : memref<f32>
  memref.store %cst, %0[] : memref<f32>
  acc.yield %0 : memref<f32>
} combiner {
^bb0(%arg0: memref<f32>, %arg1: memref<f32>):
  %0 = memref.load %arg0[] : memref<f32>
  %1 = memref.load %arg1[] : memref<f32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg0[] : memref<f32>
  acc.yield %arg0 : memref<f32>
}

func.func @structured_data(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>,
    %arg3: memref<f32>, %arg4: memref<f32>, %arg5: memref<f32>,
    %arg6: memref<f32>, %arg7: memref<f32>, %arg8: memref<f32>,
    %arg9: memref<f32>, %arg10: memref<f32>, %arg11: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating copyin(a) [if not already present]"
  %copyin = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating copyin(readonly:b) [if not already present]"
  %copyin_readonly = acc.copyin varPtr(%arg1 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating copy(c) [if not already present]"
  %copy = acc.copyin varPtr(%arg2 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_copy>, name = "c"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating copyout(d) [if not already present]"
  %copyout = acc.create varPtr(%arg3 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_copyout>, name = "d"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating copyout(zero:e) [if not already present]"
  %copyout_zero = acc.create varPtr(%arg4 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_copyout_zero>, name = "e"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating present(f)"
  %present = acc.present varPtr(%arg5 : memref<f32>) -> memref<f32>
      {name = "f"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating create(g) [if not already present]"
  %create = acc.create varPtr(%arg6 : memref<f32>) -> memref<f32>
      {name = "g"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating create(zero:h) [if not already present]"
  %create_zero = acc.create varPtr(%arg7 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_create_zero>, name = "h"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating attach(i)"
  %attach = acc.attach varPtr(%arg8 : memref<f32>) -> memref<f32>
      {name = "i"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating no_create(j) [if not already present]"
  %no_create = acc.nocreate varPtr(%arg9 : memref<f32>) -> memref<f32>
      {name = "j"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating deviceptr(k)"
  %deviceptr = acc.deviceptr varPtr(%arg10 : memref<f32>) -> memref<f32>
      {name = "k"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_data | Remark="Generating copy(l) [if not already present]"
  %reduction_copy = acc.copyin varPtr(%arg11 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_reduction>, name = "l"}
  acc.data dataOperands(%copyin, %copyin_readonly, %copy, %copyout,
      %copyout_zero, %present, %create, %create_zero, %attach, %no_create,
      %deviceptr, %reduction_copy : memref<f32>, memref<f32>, memref<f32>,
      memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>,
      memref<f32>, memref<f32>, memref<f32>, memref<f32>) {
    acc.terminator
  }
  return
}

// Each data construct in a function is reported independently.
func.func @multiple_data_constructs(%arg0: memref<f32>, %arg1: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=multiple_data_constructs | Remark="Generating copyin(a) [if not already present]"
  %copyin0 = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  acc.data dataOperands(%copyin0 : memref<f32>) {
    acc.terminator
  }
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=multiple_data_constructs | Remark="Generating copyin(b) [if not already present]"
  %copyin1 = acc.copyin varPtr(%arg1 : memref<f32>) -> memref<f32>
      {name = "b"}
  acc.data dataOperands(%copyin1 : memref<f32>) {
    acc.terminator
  }
  return
}

// A kernel environment holds the data clauses of an outlined compute construct
// and thus accepts any kind of data clause operation.
func.func @kernel_environment(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>,
    %arg3: memref<f32>, %arg4: memref<f32>, %arg5: memref<f32>,
    %arg6: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating copyin(a) [if not already present]"
  %copyin = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating private(b)"
  %private = acc.private varPtr(%arg1 : memref<f32>)
      recipe(@privatization_memref_f32) -> memref<f32> {name = "b"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating firstprivate(c)"
  %firstprivate = acc.firstprivate varPtr(%arg2 : memref<f32>)
      recipe(@firstprivatization_memref_f32) -> memref<f32> {name = "c"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating use_device(d)"
  %use_device = acc.use_device varPtr(%arg3 : memref<f32>) -> memref<f32>
      {name = "d"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating reduction(e)"
  %reduction = acc.reduction varPtr(%arg4 : memref<f32>)
      recipe(@reduction_add_memref_f32) -> memref<f32> {name = "e"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating cache(f)"
  %cache = acc.cache varPtr(%arg5 : memref<f32>) -> memref<f32>
      {name = "f"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=kernel_environment | Remark="Generating cache(readonly:g)"
  %cache_readonly = acc.cache varPtr(%arg6 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_cache_readonly>, name = "g"}
  acc.kernel_environment dataOperands(%copyin, %private, %firstprivate,
      %use_device, %reduction, %cache, %cache_readonly : memref<f32>,
      memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>,
      memref<f32>) {
    acc.compute_region {
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// A declare enter whose token is used is structured and is reported.
func.func @structured_declare(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_declare | Remark="Generating create(a) [if not already present]"
  %create = acc.create varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_declare | Remark="Generating device_resident(b)"
  %device_resident = acc.declare_device_resident varPtr(%arg1 : memref<f32>)
      -> memref<f32> {name = "b"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=structured_declare | Remark="Generating link(c)"
  %link = acc.declare_link varPtr(%arg2 : memref<f32>) -> memref<f32>
      {name = "c"}
  %token = acc.declare_enter dataOperands(%create, %device_resident, %link
      : memref<f32>, memref<f32>, memref<f32>)
  acc.declare_exit token(%token) dataOperands(%create : memref<f32>)
  acc.delete accPtr(%create : memref<f32>)
      {dataClause = #acc<data_clause acc_create>}
  return
}

// Unstructured directives get a directive name in the remark.
func.func @enter_data(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=enter_data | Remark="Generating enter data copyin(a) [if not already present]"
  %copyin = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=enter_data | Remark="Generating enter data create(b) [if not already present]"
  %create = acc.create varPtr(%arg1 : memref<f32>) -> memref<f32>
      {name = "b"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=enter_data | Remark="Generating enter data attach(c)"
  %attach = acc.attach varPtr(%arg2 : memref<f32>) -> memref<f32>
      {name = "c"}
  acc.enter_data dataOperands(%copyin, %create, %attach
      : memref<f32>, memref<f32>, memref<f32>)
  return
}

func.func @exit_data(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=exit_data | Remark="Generating exit data copyout(a) [if not already present]"
  %copyout = acc.getdeviceptr varPtr(%arg0 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_copyout>, name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=exit_data | Remark="Generating exit data delete(b)"
  %delete = acc.getdeviceptr varPtr(%arg1 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_delete>, name = "b"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=exit_data | Remark="Generating exit data detach(c)"
  %detach = acc.getdeviceptr varPtr(%arg2 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_detach>, name = "c"}
  acc.exit_data dataOperands(%copyout, %delete, %detach
      : memref<f32>, memref<f32>, memref<f32>)
  acc.copyout accPtr(%copyout : memref<f32>) to varPtr(%arg0 : memref<f32>)
  acc.delete accPtr(%delete : memref<f32>)
  acc.detach accPtr(%detach : memref<f32>)
  return
}

// An update directive does not get a directive name in the remark.
func.func @update(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=update | Remark="Generating update_host(a)"
  %update_host = acc.getdeviceptr varPtr(%arg0 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_update_host>, name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=update | Remark="Generating update_self(b)"
  %update_self = acc.getdeviceptr varPtr(%arg1 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_update_self>, name = "b"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=update | Remark="Generating update_device(c)"
  %update_device = acc.update_device varPtr(%arg2 : memref<f32>) -> memref<f32>
      {name = "c"}
  acc.update dataOperands(%update_host, %update_self, %update_device
      : memref<f32>, memref<f32>, memref<f32>)
  acc.update_host accPtr(%update_host : memref<f32>)
      to varPtr(%arg0 : memref<f32>)
  acc.update_host accPtr(%update_self : memref<f32>)
      to varPtr(%arg1 : memref<f32>)
      {dataClause = #acc<data_clause acc_update_self>}
  return
}

// Clauses added by the compiler are marked as implicit, and clauses that come
// from a default clause are marked as default.
func.func @implicit_and_default(%arg0: memref<f32>, %arg1: memref<f32>) {
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=implicit_and_default | Remark="Generating default copyin(a) [if not already present]"
  %from_default = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {acc.from_default, name = "a"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=implicit_and_default | Remark="Generating implicit present(b)"
  %implicit = acc.present varPtr(%arg1 : memref<f32>) -> memref<f32>
      {implicit = true, name = "b"}
  acc.data dataOperands(%from_default, %implicit : memref<f32>, memref<f32>) {
    acc.terminator
  }
  return
}

// Variables of the same clause and implicitness are reported together, sorted
// by name. Explicit clauses are reported before implicit ones. The grouped
// remark is anchored on the first op after sorting (x).
func.func @grouping_and_sorting(
    %arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>,
    %arg3: memref<f32>, %arg4: memref<f32>) {
  %copyin_z = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "z"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=grouping_and_sorting | Remark="Generating copyin(x, y, z) [if not already present]"
  %copyin_x = acc.copyin varPtr(%arg1 : memref<f32>) -> memref<f32>
      {name = "x"}
  %copyin_y = acc.copyin varPtr(%arg2 : memref<f32>) -> memref<f32>
      {name = "y"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=grouping_and_sorting | Remark="Generating implicit copyin(i) [if not already present]"
  %copyin_i = acc.copyin varPtr(%arg3 : memref<f32>) -> memref<f32>
      {implicit = true, name = "i"}
  // CHECK: [[@LINE+1]]:{{[0-9]+}}: remark: [Passed] openacc | Category:acc-emit-remarks-data | Function=grouping_and_sorting | Remark="Generating present(p)"
  %present = acc.present varPtr(%arg4 : memref<f32>) -> memref<f32>
      {name = "p"}
  acc.data dataOperands(%copyin_z, %copyin_x, %copyin_y, %copyin_i, %present
      : memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) {
    acc.terminator
  }
  return
}

//===----------------------------------------------------------------------===//
// Everything below this point must not produce any remark. Since remarks are
// emitted in order, the check below also catches remarks emitted for any of
// the functions that follow.
//===----------------------------------------------------------------------===//

// CHECK-NOT: Remark=

// The synthetic getdeviceptr clause and clauses without a variable name are
// not reported.
func.func @not_reportable_clauses(%arg0: memref<f32>, %arg1: memref<f32>) {
  %synthetic = acc.getdeviceptr varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "synthetic"}
  %unnamed = acc.copyin varPtr(%arg1 : memref<f32>) -> memref<f32>
  acc.data dataOperands(%synthetic, %unnamed : memref<f32>, memref<f32>) {
    acc.terminator
  }
  return
}

// A data construct without data clauses is not reported.
func.func @default_only_data() {
  acc.data {
    acc.terminator
  } attributes {defaultAttr = #acc<defaultvalue none>}
  return
}

// A declare enter whose token is unused registers the variables for the whole
// program instead of a region, and is not reported.
func.func @unstructured_declare(%arg0: memref<f32>) {
  %create = acc.create varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  acc.declare_enter dataOperands(%create : memref<f32>)
  return
}

// A declare exit is never reported - for a structured declare the variables
// are reported at the declare enter, and an unstructured declare exit only
// unregisters variables.
func.func @declare_exit(%arg0: memref<f32>) {
  %delete = acc.getdeviceptr varPtr(%arg0 : memref<f32>) -> memref<f32>
      {dataClause = #acc<data_clause acc_delete>, name = "a"}
  acc.declare_exit dataOperands(%delete : memref<f32>)
  acc.delete accPtr(%delete : memref<f32>)
  return
}

// Data clauses on a compute construct are not reported by this pass - they are
// reported once they are outlined into a kernel environment.
func.func @compute_construct(%arg0: memref<f32>) {
  %copyin = acc.copyin varPtr(%arg0 : memref<f32>) -> memref<f32>
      {name = "a"}
  acc.parallel dataOperands(%copyin : memref<f32>) {
    acc.yield
  }
  return
}
