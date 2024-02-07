// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x i32 : (i64) -> !llvm.ptr<5>
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 : (i64) -> !llvm.ptr<5>
    %8 = llvm.addrspacecast %7 : !llvm.ptr<5> to !llvm.ptr
    %9 = omp.map_info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %10 = omp.map_info var_ptr(%5 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%9 -> %arg0, %10 -> %arg1 : !llvm.ptr, !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
      %12 = llvm.mlir.constant(2 : i32) : i32
      %13 = llvm.mlir.constant(1 : i32) : i32
      %14 = llvm.load %arg0 : !llvm.ptr -> i32 loc(#loc2)
      %15 = llvm.add %14, %13  : i32 loc(#loc2)
      llvm.store %15, %arg0 : i32, !llvm.ptr loc(#loc2)
      %16 = llvm.load %arg0 : !llvm.ptr -> i32 loc(#loc3)
      %17 = llvm.add %16, %12  : i32 loc(#loc3)
      llvm.store %17, %arg1 : i32, !llvm.ptr loc(#loc3)
      omp.terminator
    }
    %11 = omp.map_info var_ptr(%8 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%11 -> %arg0 : !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr):
      %12 = llvm.mlir.constant(1 : i32) : i32
      omp.parallel {
        %13 = llvm.load %arg0 : !llvm.ptr -> i32 loc(#loc4)
        %14 = llvm.add %13, %12  : i32 loc(#loc4)
        llvm.store %14, %arg0 : i32, !llvm.ptr loc(#loc4)
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  } loc(#loc5)
} loc(#loc)
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "void", encoding = DW_ATE_address>
#di_file = #llvm.di_file<"target.f90" in "">
#loc = loc("target.f90":0:0)
#loc1 = loc("target.f90":1:1)
#loc2 = loc("target.f90":9:3)
#loc3 = loc("target.f90":10:3)
#loc4 = loc("target.f90":14:3)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_basic_type, #di_basic_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "_QQmain", linkageName = "_QQmain", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
#loc5 = loc(fused<#di_subprogram>[#loc1])

// 8:  !$omp target map(tofrom: a, b)
// 9:  a = a + 1
//10:  b = a + 2
//11:  !$omp end target

//13:  !$omp target parallel map(tofrom: a, b)
//14:  c = c + 1
//15:  !$omp end target parallel

// CHECK-DAG: [[FILE:.*]] = !DIFile(filename: "target.f90", directory: "")
// CHECK-DAG: [[CU:.*]] = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: [[FILE]], {{.*}})

// CHECK: [[SP1:.*]] = distinct !DISubprogram(name: "__omp_offloading_{{.*}}", {{.*}}, unit: [[CU]])
// CHECK-DAG: !DILocation(line: 9, column: 3, scope: [[SP1]])
// CHECK-DAG: !DILocation(line: 10, column: 3, scope: [[SP1]])

// CHECK: [[SP2:.*]] = distinct !DISubprogram(name: "__omp_offloading_{{.*}}omp_par", {{.*}}, unit: [[CU]])
// CHECK: !DILocation(line: 14, column: 3, scope: [[SP2]])
