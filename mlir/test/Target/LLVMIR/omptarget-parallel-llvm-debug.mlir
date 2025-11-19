// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s


#di_file = #llvm.di_file<"target.f90" in "">
#di_null_type = #llvm.di_null_type
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang", isOptimized = false, emissionKind = Full>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_program, types = #di_null_type>
#sp = #llvm.di_subprogram<compileUnit = #cu, scope = #di_file, name = "test", file = #di_file, subprogramFlags = "Definition", type = #sp_ty>
#sp1 = #llvm.di_subprogram<compileUnit = #cu, scope = #di_file, name = "kernel", file = #di_file, subprogramFlags = "Definition", type = #sp_ty>
#int_ty = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
#var_x = #llvm.di_local_variable<scope = #sp, name = "x", file = #di_file, type = #int_ty>
#var_x1 = #llvm.di_local_variable<scope = #sp1, name = "x", file = #di_file, type = #int_ty>
module attributes {dlti.dl_spec = #dlti.dl_spec<i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "x86-64", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 21.0.0 (/home/haqadeer/work/src/aomp-llvm-project/flang 793f9220ab32f92fc3b253efec2e332c18090e53)", llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.requires = #omp<clause_requires none>, omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 52>} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "test", frame_pointer = #llvm.framePointerKind<all>, target_cpu = "x86-64"} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
    llvm.intr.dbg.declare #var_x = %1 : !llvm.ptr loc(#loc2)
    %5 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "x"}
    omp.target map_entries(%5 -> %arg0 : !llvm.ptr) {
      %6 = llvm.mlir.constant(1 : i32) : i32
      llvm.intr.dbg.declare #var_x1 = %arg0 : !llvm.ptr loc(#loc3)
      omp.parallel {
        %7 = llvm.load %arg0 : !llvm.ptr -> i32
        %8 = llvm.add %7, %6 : i32
        llvm.store %8, %arg0 : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    } loc(#loc4)
    llvm.return
  } loc(#loc10)
}
#loc1 = loc("target.f90":1:7)
#loc2 = loc("target.f90":3:18)
#loc3 = loc("target.f90":6:18)
#loc4 = loc(fused<#sp1>[#loc3])
#loc10 = loc(fused<#sp>[#loc1])


// CHECK: define internal void @__omp_offloading{{.*}}omp_par{{.*}} !dbg ![[FN:[0-9]+]] {
// CHECK-NEXT: omp.par.entry:
// CHECK:    #dbg_declare(ptr {{.*}}, ![[VAR:[0-9]+]], {{.*}})
// CHECK-NEXT:  br

// CHECK: ![[FN]] = {{.*}}!DISubprogram(name: "__omp_offloading_{{.*}}omp_par"{{.*}})
// CHECK: ![[VAR]] = !DILocalVariable(name: "x", scope: ![[FN]]{{.*}})

