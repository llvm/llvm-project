//===- DebugInfoCacheTest.cpp - DebugInfoCache unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DebugInfoCache.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

// Forward declare the assembly
extern StringRef MultiCUModule;

const DICompileUnit *findCU(const Module &M, StringRef FileName) {
  for (const auto CU : M.debug_compile_units()) {
    if (CU->getFilename() == FileName)
      return CU;
  }

  return nullptr;
}

class DebugInfoCacheTest : public testing::Test {
protected:
  LLVMContext C;

  std::unique_ptr<Module> makeModule(StringRef Assembly) {
    SMDiagnostic Err;
    auto M = parseAssemblyString(Assembly, Err, C);
    if (!M)
      Err.print("DebugInfoCacheTest", errs());

    verifyModule(*M, &errs());
    return M;
  }
};

TEST_F(DebugInfoCacheTest, TestEmpty) {
  auto M = makeModule("");
  DebugInfoCache DIC{*M};
  EXPECT_EQ(DIC.Result.size(), 0u);
}

TEST_F(DebugInfoCacheTest, TestMultiCU) {
  auto M = makeModule(MultiCUModule);
  DebugInfoCache DIC{*M};
  EXPECT_EQ(DIC.Result.size(), 2u);

  auto *File1CU = findCU(*M, "file1.cpp");
  EXPECT_NE(File1CU, nullptr);

  auto File1DIFinder = DIC.Result.find(File1CU);
  EXPECT_NE(File1DIFinder, DIC.Result.end());

  EXPECT_EQ(File1DIFinder->getSecond().compile_unit_count(), 1u);
  EXPECT_EQ(File1DIFinder->getSecond().type_count(), 6u);
  EXPECT_EQ(File1DIFinder->getSecond().subprogram_count(), 0u);
  EXPECT_EQ(File1DIFinder->getSecond().scope_count(), 1u);

  auto *File2CU = findCU(*M, "file2.cpp");
  EXPECT_NE(File1CU, nullptr);

  auto File2DIFinder = DIC.Result.find(File2CU);
  EXPECT_NE(File2DIFinder, DIC.Result.end());

  EXPECT_EQ(File2DIFinder->getSecond().compile_unit_count(), 1u);
  EXPECT_EQ(File2DIFinder->getSecond().type_count(), 2u);
  EXPECT_EQ(File2DIFinder->getSecond().subprogram_count(), 0u);
  EXPECT_EQ(File2DIFinder->getSecond().scope_count(), 2u);
}

/* Generated roughly by
file1.cpp:
struct file1_extern_type1;
struct file1_extern_type2;

namespace file1 {
typedef struct file1_type1 { int x; float y; } file1_type1;
file1_type1 global{0, 1.};
} // file1

extern struct file1_extern_type1 *file1_extern_func1(struct
file1_extern_type2*);

file1::file1_type1 file1_func1(file1::file1_type1 x) { return x; }
--------
file2.cpp:
struct file2_extern_type1;
struct file2_extern_type2;

namespace file2 {
typedef struct file2_type1 { float x; float y; } file2_type1;
enum class file2_type2 { opt1, opt2 };

namespace inner {
file2_type2 inner_global{file2_type2::opt2};
} // inner
} // file2

extern struct file2_extern_type1 *file2_extern_func1(struct
file2_extern_type2*);

file2::file2_type1 file2_func1(file2::file2_type1 x, file2::file2_type2 y) {
return x; }
--------
$ clang -S -emit-llvm file*.cpp
$ llvm-link -S -o single.ll file*.ll
*/
StringRef MultiCUModule = R"""(
%"struct.file1::file1_type1" = type { i32, float }
%"struct.file2::file2_type1" = type { float, float }

@_ZN5file16globalE = dso_local global %"struct.file1::file1_type1" { i32 0, float 1.000000e+00 }, align 4, !dbg !0
@_ZN5file25inner12inner_globalE = dso_local global i32 1, align 4, !dbg !11

define dso_local i64 @_Z11file1_func1N5file111file1_type1E(i64 %0) !dbg !33 {
  %2 = alloca %"struct.file1::file1_type1", align 4
  %3 = alloca %"struct.file1::file1_type1", align 4
  store i64 %0, ptr %3, align 4
    #dbg_declare(ptr %3, !37, !DIExpression(), !38)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 %3, i64 8, i1 false), !dbg !39
  %4 = load i64, ptr %2, align 4, !dbg !40
  ret i64 %4, !dbg !40
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define dso_local <2 x float> @_Z11file2_func1N5file211file2_type1ENS_11file2_type2E(<2 x float> %0, i32 noundef %1) !dbg !41 {
  %3 = alloca %"struct.file2::file2_type1", align 4
  %4 = alloca %"struct.file2::file2_type1", align 4
  %5 = alloca i32, align 4
  store <2 x float> %0, ptr %4, align 4
    #dbg_declare(ptr %4, !49, !DIExpression(), !50)
  store i32 %1, ptr %5, align 4
    #dbg_declare(ptr %5, !51, !DIExpression(), !52)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %3, ptr align 4 %4, i64 8, i1 false), !dbg !53
  %6 = load <2 x float>, ptr %3, align 4, !dbg !54
  ret <2 x float> %6, !dbg !54
}

!llvm.dbg.cu = !{!20, !22}
!llvm.ident = !{!25, !25}
!llvm.module.flags = !{!26, !27, !28, !29, !30, !31, !32}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", linkageName: "_ZN5file16globalE", scope: !2, file: !3, line: 6, type: !4, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "file1", scope: null)
!3 = !DIFile(filename: "file1.cpp", directory: "")
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "file1_type1", scope: !2, file: !3, line: 5, baseType: !5)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "file1_type1", scope: !2, file: !3, line: 5, size: 64, flags: DIFlagTypePassByValue, elements: !6, identifier: "_ZTSN5file111file1_type1E")
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !3, line: 5, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !3, line: 5, baseType: !10, size: 32, offset: 32)
!10 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "inner_global", linkageName: "_ZN5file25inner12inner_globalE", scope: !13, file: !15, line: 9, type: !16, isLocal: false, isDefinition: true)
!13 = !DINamespace(name: "inner", scope: !14)
!14 = !DINamespace(name: "file2", scope: null)
!15 = !DIFile(filename: "file2.cpp", directory: "")
!16 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "file2_type2", scope: !14, file: !15, line: 6, baseType: !8, size: 32, flags: DIFlagEnumClass, elements: !17, identifier: "_ZTSN5file211file2_type2E")
!17 = !{!18, !19}
!18 = !DIEnumerator(name: "opt1", value: 0)
!19 = !DIEnumerator(name: "opt2", value: 1)
!20 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !21, splitDebugInlining: false, nameTableKind: None)
!21 = !{!0}
!22 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !15, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !23, globals: !24, splitDebugInlining: false, nameTableKind: None)
!23 = !{!16}
!24 = !{!11}
!25 = !{!"clang"}
!26 = !{i32 7, !"Dwarf Version", i32 5}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{i32 1, !"wchar_size", i32 4}
!29 = !{i32 8, !"PIC Level", i32 2}
!30 = !{i32 7, !"PIE Level", i32 2}
!31 = !{i32 7, !"uwtable", i32 2}
!32 = !{i32 7, !"frame-pointer", i32 2}
!33 = distinct !DISubprogram(name: "file1_func1", linkageName: "_Z11file1_func1N5file111file1_type1E", scope: !3, file: !3, line: 11, type: !34, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !20, retainedNodes: !36)
!34 = !DISubroutineType(types: !35)
!35 = !{!4, !4}
!36 = !{}
!37 = !DILocalVariable(name: "x", arg: 1, scope: !33, file: !3, line: 11, type: !4)
!38 = !DILocation(line: 11, column: 51, scope: !33)
!39 = !DILocation(line: 11, column: 63, scope: !33)
!40 = !DILocation(line: 11, column: 56, scope: !33)
!41 = distinct !DISubprogram(name: "file2_func1", linkageName: "_Z11file2_func1N5file211file2_type1ENS_11file2_type2E", scope: !15, file: !15, line: 15, type: !42, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !36)
!42 = !DISubroutineType(types: !43)
!43 = !{!44, !44, !16}
!44 = !DIDerivedType(tag: DW_TAG_typedef, name: "file2_type1", scope: !14, file: !15, line: 5, baseType: !45)
!45 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "file2_type1", scope: !14, file: !15, line: 5, size: 64, flags: DIFlagTypePassByValue, elements: !46, identifier: "_ZTSN5file211file2_type1E")
!46 = !{!47, !48}
!47 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !45, file: !15, line: 5, baseType: !10, size: 32)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !45, file: !15, line: 5, baseType: !10, size: 32, offset: 32)
!49 = !DILocalVariable(name: "x", arg: 1, scope: !41, file: !15, line: 15, type: !44)
!50 = !DILocation(line: 15, column: 51, scope: !41)
!51 = !DILocalVariable(name: "y", arg: 2, scope: !41, file: !15, line: 15, type: !16)
!52 = !DILocation(line: 15, column: 73, scope: !41)
!53 = !DILocation(line: 15, column: 85, scope: !41)
!54 = !DILocation(line: 15, column: 78, scope: !41)
)""";
} // namespace
} // namespace llvm
