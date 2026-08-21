// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Regression test for debug-location preservation across offload-array codegen.
//
// When an omp.target is lowered for the host, OpenMPIRBuilder emits the host
// fallback path that calls the outlined kernel stub directly. Building the
// offload arrays (emitOffloadingArrays) temporarily moves the IRBuilder to the
// alloca block, which clears the builder's current debug location. Before the
// fix that location was never restored, so the inlinable fallback call to the
// kernel stub was emitted with no !dbg attachment. In a function that has debug
// info that call fails the verifier:
//
//   inlinable function call in a function with debug info must have a !dbg
//   location
//
// The nesting below (parallel > taskgroup task_reduction > target in_reduction)
// reproduces the exact state in which the location was dropped. This test pins
// down that the host fallback call carries a debug location (and build does not
// fail).

#di_file = #llvm.di_file<"repro.f90" in "">
#di_null_type = #llvm.di_null_type
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang", isOptimized = false, emissionKind = Full>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_program, types = #di_null_type>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #di_file, name = "foo", file = #di_file, subprogramFlags = "Definition", type = #sp_ty>
#sp_tgt = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #di_file, name = "target_region", file = #di_file, subprogramFlags = "LocalToUnit|Definition", type = #sp_ty>
#loc = loc("repro.f90":1:1)
#loc_foo = loc(fused<#sp>[#loc])
#loc_tgt = loc(fused<#sp_tgt>[#loc])

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 50>} {
  omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }
  llvm.func @foo_() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    omp.parallel {
      omp.taskgroup task_reduction(@add_reduction_i32 %1 -> %arg0 : !llvm.ptr) {
        %2 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit, tofrom) capture(ByRef) -> !llvm.ptr {name = "sum"}
        omp.target kernel_type(generic) in_reduction(@add_reduction_i32 %arg0 : !llvm.ptr) map_entries(%2 -> %arg1 : !llvm.ptr) {
          %c1 = llvm.mlir.constant(1 : i32) : i32
          llvm.store %c1, %arg1 : i32, !llvm.ptr
          omp.terminator
        } loc(#loc_tgt)
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  } loc(#loc_foo)
}

// CHECK-LABEL: define void @foo_(
// CHECK: call void @__omp_offloading_{{.*}}_foo__{{.*}}(ptr %{{.+}}, ptr null), !dbg
