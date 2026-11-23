// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @task_then_taskwait(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.task depend(taskdependout -> %x : !llvm.ptr) {
    omp.terminator
  } loc(#loc_task)
  omp.taskwait depend(taskdependin -> %y : !llvm.ptr) loc(#loc_wait)
  llvm.return
} loc(#loc_fn)

// The task is on line 7 and the taskwait on line 11. Each runtime call must
// carry its own line.

// CHECK: define void @task_then_taskwait
// CHECK: call i32 @__kmpc_omp_task_with_deps({{.*}}), !dbg ![[TASK:[0-9]+]]
// CHECK: call void @__kmpc_omp_taskwait_deps_51({{.*}}), !dbg ![[WAIT:[0-9]+]]
// CHECK-DAG: ![[TASK]] = !DILocation(line: 7, column: 9
// CHECK-DAG: ![[WAIT]] = !DILocation(line: 11, column: 9

#di_file = #llvm.di_file<"test.f90" in "">
#di_null_type = #llvm.di_null_type
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>,
  sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang",
  isOptimized = false, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<
  callingConvention = DW_CC_normal, types = #di_null_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>,
  compileUnit = #di_compile_unit, scope = #di_file, name = "sub1",
  file = #di_file, subprogramFlags = "Definition", type = #di_subroutine_type>

#loc1 = loc("test.f90":3:1)
#loc2 = loc("test.f90":7:9)
#loc3 = loc("test.f90":11:9)

#loc_fn = loc(fused<#di_subprogram>[#loc1])
#loc_task = loc(fused<#di_subprogram>[#loc2])
#loc_wait = loc(fused<#di_subprogram>[#loc3])
