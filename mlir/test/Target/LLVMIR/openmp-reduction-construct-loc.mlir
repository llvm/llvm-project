// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Check that the OpenMP runtime calls of a reducing construct report the
// construct's own line, not that of an enclosing construct or of a reduction
// or privatisation declaration it uses.
//
//   subroutine r_sections(a, s)                 ! line 17
//     !$omp parallel                            ! line 18
//     !$omp sections reduction(+:s)             ! line 19
//   subroutine r_parallel(a, s)                 ! line 27
//     !$omp parallel reduction(+:s)             ! line 29
//   subroutine r_scope(a, s)                    ! line 37
//     !$omp parallel                            ! line 38
//     !$omp scope reduction(+:s)                ! line 39
//   subroutine r_wsloop(x, s)                   ! line 47
//     !$omp parallel                            ! line 48
//     !$omp do firstprivate(x) reduction(+:s)   ! line 49

#file = #llvm.di_file<"repro.f90" in "">
#void = #llvm.di_null_type
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95,
                            file = #file, producer = "flang", isOptimized = false,
                            emissionKind = Full>
#sub_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #void>

#sp_sections = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
                                   name = "r_sections", linkageName = "r_sections_",
                                   file = #file, line = 17, scopeLine = 17,
                                   subprogramFlags = "Definition", type = #sub_type>
#sp_parallel = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #file,
                                   name = "r_parallel", linkageName = "r_parallel_",
                                   file = #file, line = 27, scopeLine = 27,
                                   subprogramFlags = "Definition", type = #sub_type>
#sp_scope = #llvm.di_subprogram<id = distinct[3]<>, compileUnit = #cu, scope = #file,
                                name = "r_scope", linkageName = "r_scope_",
                                file = #file, line = 37, scopeLine = 37,
                                subprogramFlags = "Definition", type = #sub_type>
#sp_wsloop = #llvm.di_subprogram<id = distinct[4]<>, compileUnit = #cu, scope = #file,
                                 name = "r_wsloop", linkageName = "r_wsloop_",
                                 file = #file, line = 47, scopeLine = 47,
                                 subprogramFlags = "Definition", type = #sub_type>

#loc_decl    = loc("repro.f90":100:5)
#loc_fp_decl = loc("repro.f90":110:5)

#loc_sections_par = loc("repro.f90":18:7)
#loc_sections     = loc("repro.f90":19:9)
#loc_sections_bdy = loc("repro.f90":21:3)
#loc_sections_dcl = loc("repro.f90":17:1)
#loc_sections_fn  = loc(fused<#sp_sections>[#loc_sections_dcl])

#loc_parallel     = loc("repro.f90":29:9)
#loc_parallel_bdy = loc("repro.f90":30:3)
#loc_parallel_dcl = loc("repro.f90":27:1)
#loc_parallel_fn  = loc(fused<#sp_parallel>[#loc_parallel_dcl])

#loc_scope_par    = loc("repro.f90":38:7)
#loc_scope        = loc("repro.f90":39:9)
#loc_scope_bdy    = loc("repro.f90":40:3)
#loc_scope_dcl    = loc("repro.f90":37:1)
#loc_scope_fn     = loc(fused<#sp_scope>[#loc_scope_dcl])

#loc_wsloop_par   = loc("repro.f90":48:7)
#loc_wsloop       = loc("repro.f90":49:9)
#loc_wsloop_bdy   = loc("repro.f90":50:3)
#loc_wsloop_dcl   = loc("repro.f90":47:1)
#loc_wsloop_fn    = loc(fused<#sp_wsloop>[#loc_wsloop_dcl])

module attributes {omp.version = #omp.version<version = 51>} {
  omp.declare_reduction @add_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32 loc(#loc_decl)
    omp.yield(%0 : i32) loc(#loc_decl)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32 loc(#loc_decl)
    omp.yield(%0 : i32) loc(#loc_decl)
  } loc(#loc_decl)

  omp.private {type = firstprivate} @fp_i32 : i32 copy {
  ^bb0(%mold: !llvm.ptr, %priv: !llvm.ptr):
    %0 = llvm.load %mold : !llvm.ptr -> i32 loc(#loc_fp_decl)
    llvm.store %0, %priv : i32, !llvm.ptr loc(#loc_fp_decl)
    omp.yield(%priv : !llvm.ptr) loc(#loc_fp_decl)
  } loc(#loc_fp_decl)

  llvm.func @r_sections_(%arg0: !llvm.ptr) {
    omp.parallel {
      omp.sections reduction(@add_i32 %arg0 -> %red : !llvm.ptr) {
        omp.section {
        ^bb0(%prv: !llvm.ptr):
          %c1 = llvm.mlir.constant(1 : i32) : i32 loc(#loc_sections_bdy)
          %0 = llvm.load %prv : !llvm.ptr -> i32 loc(#loc_sections_bdy)
          %1 = llvm.add %0, %c1 : i32 loc(#loc_sections_bdy)
          llvm.store %1, %prv : i32, !llvm.ptr loc(#loc_sections_bdy)
          omp.terminator loc(#loc_sections_bdy)
        } loc(#loc_sections_bdy)
        omp.terminator loc(#loc_sections)
      } loc(#loc_sections)
      omp.terminator loc(#loc_sections_par)
    } loc(#loc_sections_par)
    llvm.return loc(#loc_sections_fn)
  } loc(#loc_sections_fn)

  llvm.func @r_parallel_(%arg0: !llvm.ptr) {
    omp.parallel reduction(@add_i32 %arg0 -> %prv : !llvm.ptr) {
      %c1 = llvm.mlir.constant(1 : i32) : i32 loc(#loc_parallel_bdy)
      %0 = llvm.load %prv : !llvm.ptr -> i32 loc(#loc_parallel_bdy)
      %1 = llvm.add %0, %c1 : i32 loc(#loc_parallel_bdy)
      llvm.store %1, %prv : i32, !llvm.ptr loc(#loc_parallel_bdy)
      omp.terminator loc(#loc_parallel_bdy)
    } loc(#loc_parallel)
    llvm.return loc(#loc_parallel_fn)
  } loc(#loc_parallel_fn)

  llvm.func @r_scope_(%arg0: !llvm.ptr) {
    omp.parallel {
      omp.scope reduction(@add_i32 %arg0 -> %prv : !llvm.ptr) {
        %c1 = llvm.mlir.constant(1 : i32) : i32 loc(#loc_scope_bdy)
        %0 = llvm.load %prv : !llvm.ptr -> i32 loc(#loc_scope_bdy)
        %1 = llvm.add %0, %c1 : i32 loc(#loc_scope_bdy)
        llvm.store %1, %prv : i32, !llvm.ptr loc(#loc_scope_bdy)
        omp.terminator loc(#loc_scope_bdy)
      } loc(#loc_scope)
      omp.terminator loc(#loc_scope_par)
    } loc(#loc_scope_par)
    llvm.return loc(#loc_scope_fn)
  } loc(#loc_scope_fn)

  llvm.func @r_wsloop_(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i32) : i32 loc(#loc_wsloop_par)
    %c10 = llvm.mlir.constant(10 : i32) : i32 loc(#loc_wsloop_par)
    omp.parallel {
      omp.wsloop private(@fp_i32 %arg0 -> %x : !llvm.ptr) private_barrier reduction(@add_i32 %arg1 -> %prv : !llvm.ptr) {
        omp.loop_nest (%i) : i32 = (%c1) to (%c10) inclusive step (%c1) {
          %0 = llvm.load %x : !llvm.ptr -> i32 loc(#loc_wsloop_bdy)
          %1 = llvm.load %prv : !llvm.ptr -> i32 loc(#loc_wsloop_bdy)
          %2 = llvm.add %0, %1 : i32 loc(#loc_wsloop_bdy)
          llvm.store %2, %prv : i32, !llvm.ptr loc(#loc_wsloop_bdy)
          omp.yield loc(#loc_wsloop_bdy)
        } loc(#loc_wsloop)
      } loc(#loc_wsloop)
      omp.terminator loc(#loc_wsloop_par)
    } loc(#loc_wsloop_par)
    llvm.return loc(#loc_wsloop_fn)
  } loc(#loc_wsloop_fn)
}

// CHECK-DAG: @[[SECT_STR:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";repro.f90;r_sections;19;9;;\00"
// CHECK-DAG: @[[PAR_STR:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";repro.f90;r_parallel;29;9;;\00"
// CHECK-DAG: @[[SCOPE_STR:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";repro.f90;r_scope;39;9;;\00"
// CHECK-DAG: @[[WS_STR:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";repro.f90;r_wsloop;49;9;;\00"

// CHECK-DAG: @[[$SECT_ID:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]+}}, ptr @[[SECT_STR]] }
// CHECK-DAG: @[[$PAR_ID:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]+}}, ptr @[[PAR_STR]] }
// CHECK-DAG: @[[$SCOPE_ID:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]+}}, ptr @[[SCOPE_STR]] }
// CHECK-DAG: @[[$WS_ID:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]+}}, ptr @[[WS_STR]] }
// CHECK-DAG: @[[$WS_BARRIER_ID:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 34, i32 0, i32 {{[0-9]+}}, ptr @[[WS_STR]] }
// CHECK-DAG: @[[$WS_LOOP_ID:[0-9]+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 514, i32 0, i32 {{[0-9]+}}, ptr @[[WS_STR]] }

// CHECK-LABEL: define internal void @r_sections_..omp_par(
// CHECK:         %reduce = call i32 @__kmpc_reduce(ptr @[[$SECT_ID]], i32 %[[TID:[^,]*]],
// CHECK:         call void @__kmpc_end_reduce(ptr @[[$SECT_ID]], i32 %[[TID]],

// CHECK-LABEL: define internal void @r_parallel_..omp_par(
// CHECK:         %reduce = call i32 @__kmpc_reduce(ptr @[[$PAR_ID]], i32 %[[TID:[^,]*]],
// CHECK:         call void @__kmpc_end_reduce(ptr @[[$PAR_ID]], i32 %[[TID]],

// CHECK-LABEL: define internal void @r_scope_..omp_par(
// CHECK:         %reduce = call i32 @__kmpc_reduce(ptr @[[$SCOPE_ID]], i32 %[[TID:[^,]*]],
// CHECK:         call void @__kmpc_end_reduce(ptr @[[$SCOPE_ID]], i32 %[[TID]],

// CHECK-LABEL: define internal void @r_wsloop_..omp_par(
// CHECK:         call void @__kmpc_barrier(ptr @[[$WS_BARRIER_ID]],
// CHECK:         call void @__kmpc_for_static_init_4u(ptr @[[$WS_LOOP_ID]],
// CHECK:         %reduce = call i32 @__kmpc_reduce(ptr @[[$WS_ID]], i32 %[[TID:[^,]*]],
// CHECK:         call void @__kmpc_end_reduce(ptr @[[$WS_ID]], i32 %[[TID]],
