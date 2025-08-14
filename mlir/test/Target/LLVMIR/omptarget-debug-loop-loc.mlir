// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  omp.private {type = private} @_QFEj_private_i32 : i32 loc(#loc1)
  omp.private {type = private} @_QFEi_private_i32 : i32 loc(#loc1)
  llvm.func @test() {
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr<5> loc(#loc4)
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr loc(#loc4)
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5> loc(#loc4)
    %8 = llvm.addrspacecast %7 : !llvm.ptr<5> to !llvm.ptr
    %9 = llvm.mlir.constant(16383 : index) : i64
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(16384 : i32) : i32
    %14 = llvm.mlir.addressof @_QFEarray : !llvm.ptr
    %18 = omp.map.info var_ptr(%8 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"} loc(#loc3)
    %20 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "j"} loc(#loc3)
    %22 = omp.map.bounds lower_bound(%10 : i64) upper_bound(%9 : i64) extent(%9 : i64) stride(%11 : i64) start_idx(%11 : i64) loc(#loc3)
    %23 = omp.map.info var_ptr(%14 : !llvm.ptr, !llvm.array<16384 x i32>) map_clauses(implicit, tofrom) capture(ByRef) bounds(%22) -> !llvm.ptr {name = "array"} loc(#loc3)
    %24 = omp.map.info var_ptr(%8 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "i"} loc(#loc3)
    omp.target map_entries(%18 -> %arg0, %20 -> %arg2, %23 -> %arg4, %24 -> %arg5 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %25 = llvm.mlir.constant(1 : i32) : i32
      %27 = llvm.mlir.constant(16384 : i32) : i32
      omp.teams {
        omp.distribute private(@_QFEi_private_i32 %arg5 -> %arg6 : !llvm.ptr) {
          omp.loop_nest (%arg7) : i32 = (%25) to (%27) inclusive step (%25) {
            omp.parallel {
              omp.wsloop private(@_QFEj_private_i32 %arg2 -> %arg8 : !llvm.ptr) {
                omp.loop_nest (%arg9) : i32 = (%25) to (%27) inclusive step (%25) {
                  llvm.store %arg9, %arg8 : i32, !llvm.ptr loc(#loc9)
                  omp.yield
                } loc(#loc9)
              } loc(#loc9)
              omp.terminator loc(#loc9)
            } loc(#loc9)
            omp.yield loc(#loc9)
          } loc(#loc9)
        } loc(#loc9)
        omp.terminator loc(#loc9)
      } loc(#loc9)
      omp.terminator loc(#loc9)
    } loc(#loc15)
    llvm.return loc(#loc9)
  } loc(#loc14)
  llvm.mlir.global internal @_QFEarray() {addr_space = 0 : i32} : !llvm.array<16384 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<16384 x i32>
    llvm.return %0 : !llvm.array<16384 x i32>
  } loc(#loc2)
}
#di_file = #llvm.di_file<"test.f90" in "">
#di_null_type = #llvm.di_null_type
#loc1 = loc("test.f90":4:23)
#loc2 = loc("test.f90":4:15)
#loc3 = loc("test.f90":1:7)
#loc4 = loc("test.f90":4:18)
#loc9 = loc("test.f90":13:11)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang", isOptimized = true, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_program, types = #di_null_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, subprogramFlags = "Definition|Optimized|MainSubprogram", type = #di_subroutine_type>
#di_subprogram1 = #llvm.di_subprogram<compileUnit = #di_compile_unit, name = "target", file = #di_file, subprogramFlags = "Definition", type = #di_subroutine_type>
#loc14 = loc(fused<#di_subprogram>[#loc3])
#loc15 = loc(fused<#di_subprogram1>[#loc9])


// CHECK: call void @__kmpc_distribute_static{{.*}}!dbg

