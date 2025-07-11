// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  omp.private {type = private} @_QFEi_private_i32 : i32 loc(#loc1)
  omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32 loc("test.f90":8:7)):
    %0 = llvm.mlir.constant(0 : i32) : i32 loc(#loc2)
    omp.yield(%0 : i32) loc(#loc2)
  } combiner {
  ^bb0(%arg0: i32 loc("test.f90":8:7), %arg1: i32 loc("test.f90":8:7)):
    %0 = llvm.add %arg0, %arg1 : i32 loc(#loc2)
    omp.yield(%0 : i32) loc(#loc2)
  } loc(#loc2)
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i64) : i64 loc(#loc4)
    %1 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr<5> loc(#loc4)
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr loc(#loc4)
    %3 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
    %4 = llvm.alloca %3 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5> loc(#loc1)
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr loc(#loc1)
    %6 = llvm.mlir.constant(8191 : index) : i64 loc(#loc5)
    %7 = llvm.mlir.constant(0 : index) : i64 loc(#loc5)
    %8 = llvm.mlir.constant(1 : index) : i64 loc(#loc5)
    %9 = llvm.mlir.constant(0 : i32) : i32 loc(#loc5)
    %10 = llvm.mlir.constant(8192 : index) : i64 loc(#loc5)
    %11 = llvm.mlir.addressof @_QFEarr : !llvm.ptr<1> loc(#loc6)
    %12 = llvm.addrspacecast %11 : !llvm.ptr<1> to !llvm.ptr loc(#loc6)
    llvm.store %9, %2 : i32, !llvm.ptr loc(#loc7)
    %15 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "x"} loc(#loc4)
    %16 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"} loc(#loc7)
    %17 = omp.map.bounds lower_bound(%7 : i64) upper_bound(%6 : i64) extent(%10 : i64) stride(%8 : i64) start_idx(%8 : i64) loc(#loc7)
    %18 = omp.map.info var_ptr(%12 : !llvm.ptr, !llvm.array<8192 x i32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%17) -> !llvm.ptr {name = "arr"} loc(#loc7)
    omp.target map_entries(%15 -> %arg0, %16 -> %arg1, %18 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %19 = llvm.mlir.constant(8192 : i32) : i32 loc(#loc5)
      %20 = llvm.mlir.constant(1 : i32) : i32 loc(#loc5)
      %21 = llvm.mlir.constant(8192 : index) : i64 loc(#loc6)
      omp.teams reduction(@add_reduction_i32 %arg0 -> %arg3 : !llvm.ptr) {
        omp.parallel private(@_QFEi_private_i32 %arg1 -> %arg4 : !llvm.ptr) {
          omp.distribute {
            omp.wsloop reduction(@add_reduction_i32 %arg3 -> %arg5 : !llvm.ptr) {
              omp.loop_nest (%arg6) : i32 = (%20) to (%19) inclusive step (%20) {
                llvm.store %arg6, %arg4 : i32, !llvm.ptr loc(#loc2)
                %22 = llvm.load %arg5 : !llvm.ptr -> i32 loc(#loc8)
                %23 = llvm.load %arg4 : !llvm.ptr -> i32 loc(#loc8)
                %34 = llvm.add %22, %23 : i32 loc(#loc8)
                llvm.store %34, %arg5 : i32, !llvm.ptr loc(#loc8)
                omp.yield loc(#loc2)
              } loc(#loc2)
            } {omp.composite} loc(#loc2)
          } {omp.composite} loc(#loc2)
          omp.terminator loc(#loc2)
        } {omp.composite} loc(#loc2)
        omp.terminator loc(#loc2)
      } loc(#loc2)
      omp.terminator loc(#loc2)
    } loc(#loc13)
    llvm.return loc(#loc9)
  } loc(#loc12)
  llvm.mlir.global internal @_QFEarr() {addr_space = 1 : i32} : !llvm.array<8192 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<8192 x i32> loc(#loc6)
    llvm.return %0 : !llvm.array<8192 x i32> loc(#loc6)
  } loc(#loc6)
} loc(#loc)

#loc = loc("test.f90":4:18)
#loc1 = loc("test.f90":4:18)
#loc2 = loc("test.f90":8:7)
#loc3 = loc("test.f90":1:7)
#loc4 = loc("test.f90":3:18)
#loc5 = loc(unknown)
#loc6 = loc("test.f90":5:18)
#loc7 = loc("test.f90":6:7)
#loc8 = loc("test.f90":10:7)
#loc9 = loc("test.f90":16:7)

#di_file = #llvm.di_file<"target7.f90" in "">
#di_null_type = #llvm.di_null_type
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang",
 isOptimized = false, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<
  callingConvention = DW_CC_program, types = #di_null_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>,
  compileUnit = #di_compile_unit, scope = #di_file, name = "main",
  file = #di_file, subprogramFlags = "Definition|MainSubprogram",
  type = #di_subroutine_type>
#di_subprogram1 = #llvm.di_subprogram<compileUnit = #di_compile_unit,
  name = "target", file = #di_file, subprogramFlags = "Definition",
  type = #di_subroutine_type>


#loc12 = loc(fused<#di_subprogram>[#loc3])
#loc13 = loc(fused<#di_subprogram1>[#loc2])

// CHECK-DAG: define internal void @_omp_reduction_shuffle_and_reduce_func
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_inter_warp_copy_func
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @"__omp_offloading_{{.*}}__QQmain_l8_omp$reduction$reduction_func.1"
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_shuffle_and_reduce_func.2
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_inter_warp_copy_func.3
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_list_to_global_copy_func
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_list_to_global_reduce_func
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_global_to_list_copy_func
// CHECK-NOT: !dbg
// CHECK: }
// CHECK-DAG: define internal void @_omp_reduction_global_to_list_reduce_func
// CHECK-NOT: !dbg
// CHECK: }
