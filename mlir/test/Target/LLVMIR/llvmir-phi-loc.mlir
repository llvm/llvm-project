// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @test_phi_locations(%arg0: !llvm.ptr)  {
  %0 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
  %1 = llvm.mlir.constant(100 : i32) : i32 loc(#loc1)
  llvm.br ^bb1(%1, %0 : i32, i64) loc(#loc1)
^bb1(%2: i32 loc(#loc2), %3: i64 loc(#loc3)):
  %4 = llvm.icmp "sgt" %3, %0 : i64 loc(#loc1)
  llvm.cond_br %4, ^bb2, ^bb1(%2, %3 : i32, i64) loc(#loc1)
^bb2:
  llvm.return loc(#loc1)
} loc(#loc4)

#file = #llvm.di_file<"test.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = Full>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "test_phi_locations", file = #file, subprogramFlags = Definition,
 type = #sp_ty>

#loc1 = loc("test.f90":15:22)
#loc2 = loc("test.f90":8:2)
#loc3 = loc("test.f90":9:5)
#loc4 = loc(fused<#sp>[#loc1])

// CHECK-LABEL: define void @test_phi_locations
// CHECK: phi i32{{.*}}!dbg ![[LOC1:[0-9]+]]
// CHECK: phi i64{{.*}}!dbg ![[LOC2:[0-9]+]]
// CHECK: ![[LOC1]] = !DILocation(line: 8, column: 2{{.*}})
// CHECK: ![[LOC2]] = !DILocation(line: 9, column: 5{{.*}})
