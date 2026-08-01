// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The device reduction helpers emitted by OpenMPIRBuilder have no debug info of
// their own. If the builder's current debug location is left set while their
// bodies are emitted, their instructions get DILocations scoped to the
// wrong subprogram. Check that none of the GPU reduction helpers carry a
// !dbg attachment.

#di_file = #llvm.di_file<"repro.f90" in "">
#di_null_type = #llvm.di_null_type
#loc1 = loc("repro.f90":1:1)
#loc2 = loc("repro.f90":7:9)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang", isOptimized = true, emissionKind = Full>
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_null_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "k", linkageName = "k_", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
#di_subprogram1 = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "__omp_offloading_k_l7", linkageName = "__omp_offloading_k_l7", file = #di_file, line = 7, scopeLine = 7, subprogramFlags = "LocalToUnit|Definition|Optimized", type = #di_subroutine_type>
#loc14 = loc(fused<#di_subprogram>[#loc1])
#loc15 = loc(fused<#di_subprogram1>[#loc2])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<1> = dense<64> : vector<4xi64>, !llvm.ptr<5> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, "dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true, omp.version = #omp.version<version = 31>} {
  omp.declare_reduction @add_reduction_f64 : f64 init {
  ^bb0(%arg0: f64):
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64 loc(#loc2)
    omp.yield(%0 : f64) loc(#loc2)
  } combiner {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = llvm.fadd %arg0, %arg1 : f64 loc(#loc2)
    omp.yield(%0 : f64) loc(#loc2)
  } loc(#loc2)
  llvm.func @k_(%arg0: !llvm.ptr) {
    %1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, f64) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = "s"} loc(#loc2)
    omp.target kernel_type(spmd) map_entries(%1 -> %arg1 : !llvm.ptr) {
      %c1 = llvm.mlir.constant(1 : i32) : i32 loc(#loc2)
      %cn = llvm.mlir.constant(1024 : i32) : i32 loc(#loc2)
      omp.teams reduction(@add_reduction_f64 %arg1 -> %arg2 : !llvm.ptr) {
        omp.parallel {
          omp.distribute {
            omp.wsloop reduction(@add_reduction_f64 %arg2 -> %arg3 : !llvm.ptr) {
              omp.loop_nest (%arg4) : i32 = (%c1) to (%cn) inclusive step (%c1) {
                %2 = llvm.load %arg3 : !llvm.ptr -> f64 loc(#loc2)
                %cst = llvm.mlir.constant(1.000000e+00 : f64) : f64 loc(#loc2)
                %3 = llvm.fadd %2, %cst : f64 loc(#loc2)
                llvm.store %3, %arg3 : f64, !llvm.ptr loc(#loc2)
                omp.yield loc(#loc2)
              } loc(#loc2)
            } {omp.composite} loc(#loc2)
          } {omp.composite} loc(#loc2)
          omp.terminator loc(#loc2)
        } {omp.composite} loc(#loc2)
        omp.terminator loc(#loc2)
      } {omp.combined} loc(#loc2)
      omp.terminator loc(#loc2)
    } {omp.combined} loc(#loc15)
    llvm.return loc(#loc14)
  } loc(#loc14)
} loc(#loc1)

// CHECK-LABEL: define internal void @_omp_reduction_shuffle_and_reduce_func(
// CHECK-NOT:     !dbg
// CHECK:       }
// CHECK-LABEL: define internal void @_omp_reduction_inter_warp_copy_func(
// CHECK-NOT:     !dbg
// CHECK:       }
// CHECK-LABEL: define internal void @_omp_reduction_list_to_global_copy_func(
// CHECK-NOT:     !dbg
// CHECK:       }
// CHECK-LABEL: define internal void @_omp_reduction_global_to_list_copy_func(
// CHECK-NOT:     !dbg
// CHECK:       }
// CHECK-LABEL: define internal void @_omp_reduction_global_to_list_reduce_func(
// CHECK-NOT:     !dbg
// CHECK:       }
