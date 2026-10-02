// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// subroutine reduce(s)
//   integer :: i, s
//   !$omp parallel do reduction(+:s)
//   do i = 1, 10
//      s = s + i
//   end do
// end subroutine

#file = #llvm.di_file<"reduce.f90" in "">
#void = #llvm.di_null_type
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #file, producer = "flang", isOptimized = false, emissionKind = Full>
#sub_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #void>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file, name = "reduce", linkageName = "reduce_", file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition", type = #sub_type>

#loc1 = loc("reduce.f90":1:1)
#loc2 = loc("reduce.f90":3:9)
#loc_fn = loc(fused<#sp>[#loc1])

module attributes {omp.version = #omp.version<version = 31>} {
  omp.declare_reduction @add_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32 loc(#loc2)
    omp.yield(%0 : i32) loc(#loc2)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32 loc(#loc2)
    omp.yield(%0 : i32) loc(#loc2)
  } loc(#loc2)

  llvm.func @reduce_(%arg0: !llvm.ptr) {
    %c1 = llvm.mlir.constant(1 : i32) : i32 loc(#loc2)
    %cn = llvm.mlir.constant(10 : i32) : i32 loc(#loc2)
    omp.parallel {
      omp.wsloop reduction(@add_i32 %arg0 -> %prv : !llvm.ptr) {
        omp.loop_nest (%iv) : i32 = (%c1) to (%cn) inclusive step (%c1) {
          %0 = llvm.load %prv : !llvm.ptr -> i32 loc(#loc2)
          %1 = llvm.add %0, %iv : i32 loc(#loc2)
          llvm.store %1, %prv : i32, !llvm.ptr loc(#loc2)
          omp.yield loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
      omp.terminator loc(#loc2)
    } loc(#loc2)
    llvm.return loc(#loc_fn)
  } loc(#loc_fn)
}

// Test that reduction runtime function calls get a dbg metadata, and that it
// is the directive's own location.
// CHECK-LABEL: define internal void @reduce_..omp_par(
// CHECK-SAME:      !dbg
// CHECK:       call i32 @__kmpc_reduce({{.*}}), !dbg [[LOC:![0-9]+]]
// CHECK:       call void @__kmpc_end_reduce({{.*}}), !dbg [[LOC]]
// CHECK:       [[LOC]] = !DILocation(line: 3, column: 9,
