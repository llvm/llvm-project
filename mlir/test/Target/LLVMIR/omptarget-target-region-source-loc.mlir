// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Verify that the host-side __tgt_target_kernel launch is emitted with the
// target region's real source location (file/line/column) rather than the
// default ";unknown;unknown;0;0;;" identifier. Requires target_triples so that
// an offloading entry (and therefore the kernel launch) is generated.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%2 -> %arg0 : !llvm.ptr) {
      %3 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %3, %arg0 : i32, !llvm.ptr loc(#loc2)
      omp.terminator
    } loc(#loc4)
    llvm.return
  } loc(#loc3)
}
#file = #llvm.di_file<"target.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = LineTablesOnly>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "_QQmain", file = #file, subprogramFlags = "Definition", type = #sp_ty>
#sp1 = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #file,
 name = "__omp_offloading_target", file = #file, subprogramFlags = "Definition",
 type = #sp_ty>
#loc1 = loc("target.f90":10:5)
#loc2 = loc("target.f90":11:7)
#loc3 = loc(fused<#sp>[#loc1])
#loc4 = loc(fused<#sp1>[#loc1])

// The kernel launch identifier must reference a source-location string that
// carries the real file name and position, not ";unknown;unknown;0;0;;".
// CHECK: @[[SRCLOC:[0-9]+]] = private unnamed_addr constant [{{[0-9]+}} x i8] c";target.f90;{{[^;]*}};10;5;;\00"
// CHECK: @[[IDENT:[0-9]+]] = private unnamed_addr constant %struct.ident_t {{.*}}ptr @[[SRCLOC]] }
// CHECK: call i32 @__tgt_target_kernel(ptr @[[IDENT]],
