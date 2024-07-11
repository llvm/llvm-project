// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s


module attributes {} {
  llvm.func @test(%arg0: !llvm.ptr)  {
    %0 = llvm.mlir.constant(1 : i64) : i64 loc(#loc2)
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr loc(#loc2)
    %3 = llvm.mlir.constant(100 : index) : i64 loc(#loc2)
    %7 = llvm.trunc %0 : i64 to i32 loc(#loc2)
    llvm.br ^bb1(%7, %3 : i32, i64) loc(#loc2)
  ^bb1(%8: i32 loc(#loc4), %9: i64 loc(#loc5)):  // 2 preds: ^bb0, ^bb2
    %10 = llvm.icmp "sgt" %9, %0 : i64 loc(#loc3)
    llvm.cond_br %10, ^bb2, ^bb3 loc(#loc3)
  ^bb2:  // pred: ^bb1
    %13 = llvm.load %1 : !llvm.ptr -> i32 loc(#loc3)
    %14 = llvm.add %13, %7 : i32 loc(#loc3)
    %15 = llvm.sub %9, %0 : i64 loc(#loc3)
    llvm.br ^bb1(%14, %15 : i32, i64) loc(#loc3)
  ^bb3:  // pred: ^bb1
    llvm.return loc(#loc3)
  } loc(#loc6)
}

#int_ty = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer",
 sizeInBits = 32, encoding = DW_ATE_signed>
#file = #llvm.di_file<"test.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = Full>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_normal,
 types = #int_ty>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "test", file = #file, line = 1, scopeLine = 1,
 subprogramFlags = Definition, type = #sp_ty>

#loc1 = loc("test.f90":1:1)
#loc2 = loc("test.f90":15:22)
#loc3 = loc("test.f90":26:3)
#loc4 = loc("test.f90":8:2)
#loc5 = loc("test.f90":9:5)
#loc6 = loc(fused<#sp>[#loc1])

// CHECK-LABEl: define void @test
// CHECK: phi i32{{.*}}!dbg ![[LOC1:[0-9]+]]
// CHECK: phi i64{{.*}}!dbg ![[LOC2:[0-9]+]]
// CHECK: ![[LOC1]] = !DILocation(line: 8, column: 2{{.*}})
// CHECK: ![[LOC2]] = !DILocation(line: 9, column: 5{{.*}})
