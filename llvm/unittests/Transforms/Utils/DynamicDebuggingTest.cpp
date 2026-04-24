//===- DynamicDebuggingTest.cpp - Unit tests for dynamic debugging utils -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DynamicDebugging.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(DynamicDebugging, UnoptimizedModuleAttributes) {
  // Test that `prepareForDynamicDebugging` removes alwaysinline and minsize
  // attributes from cloned functions in the unoptimized module, adding
  // noinline and optnone.
  StringRef IR = R"(
    define dso_local void @f() #0 !dbg !4 {
      ret void
    }

    attributes #0 = { alwaysinline minsize }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.c", directory: "/")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !4 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
    !5 = !DIFile(filename: "test.c", directory: "/")
    !6 = !DISubroutineType(types: !7)
    !7 = !{null}
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  std::unique_ptr<Module> OptModule = parseAssemblyString(IR, Error, Context);
  ASSERT_TRUE(OptModule != nullptr) << "Failed to parse IR\n";
  ASSERT_TRUE(OptModule->getFunction("f"));
  ASSERT_TRUE(
      OptModule->getFunction("f")->hasFnAttribute(Attribute::AlwaysInline));
  ASSERT_TRUE(OptModule->getFunction("f")->hasFnAttribute(Attribute::MinSize));

  std::unique_ptr<Module> UnoptModule =
      prepareForDynamicDebugging(OptModule.get(), "unused");

  // Check that optnone and noinline are added to the unoptimized function
  // clone and that the clashing attributes minsize and alwaysinline are
  // removed.
  Function *F = UnoptModule->getFunction("__dyndbg.f");
  ASSERT_TRUE(F && !F->isDeclaration());
  EXPECT_FALSE(F->hasFnAttribute(Attribute::MinSize));
  EXPECT_FALSE(F->hasFnAttribute(Attribute::AlwaysInline));
  EXPECT_TRUE(F->hasFnAttribute(Attribute::OptimizeNone));
  EXPECT_TRUE(F->hasFnAttribute(Attribute::NoInline));

  // Check the declarations (references to OptModule's functions) are unchanged.
  F = UnoptModule->getFunction("f");
  ASSERT_TRUE(F && F->isDeclaration());
  EXPECT_TRUE(F->hasFnAttribute(Attribute::MinSize));
  EXPECT_TRUE(F->hasFnAttribute(Attribute::AlwaysInline));
  EXPECT_FALSE(F->hasFnAttribute(Attribute::OptimizeNone));
  EXPECT_FALSE(F->hasFnAttribute(Attribute::NoInline));

  // The originl function in OptModule should be unchanged.
  F = OptModule->getFunction("f");
  ASSERT_TRUE(F && !F->isDeclaration());
  EXPECT_TRUE(F->hasFnAttribute(Attribute::MinSize));
  EXPECT_TRUE(F->hasFnAttribute(Attribute::AlwaysInline));
  EXPECT_FALSE(F->hasFnAttribute(Attribute::OptimizeNone));
  EXPECT_FALSE(F->hasFnAttribute(Attribute::NoInline));
}

TEST(DynamicDebugging, OptimizedModuleAttributes) {
  // Test that `prepareForDynamicDebugging` adds necessary attributes to
  // functions in the (to be) optimized module.
  // Note: The test requires a triple as support for targets other than x86_64
  // hasn't been added yet for the tail padding attributes.
  StringRef IR = R"(
    target triple = "x86_64-unknown-linux"

    define dso_local void @f() !dbg !4 {
      ret void
    }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 23.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.c", directory: "/")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !4 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
    !5 = !DIFile(filename: "test.c", directory: "/")
    !6 = !DISubroutineType(types: !7)
    !7 = !{null}
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  std::unique_ptr<Module> M = parseAssemblyString(IR, Error, Context);
  ASSERT_TRUE(M != nullptr) << "Failed to parse IR\n";

  // We're only interested in M (drop returned unoptimzed module).
  prepareForDynamicDebugging(M.get(), "unused");
  Function *F = M->getFunction("f");
  ASSERT_TRUE(F);

  EXPECT_TRUE(F->hasFnAttribute("no-func-spec"));
  EXPECT_EQ(F->getFnAttributeAsParsedInteger("tail-pad-to-size"), 5u);
  EXPECT_EQ(F->getFnAttributeAsParsedInteger("tail-pad-value"), 144u);
}

TEST(DynamicDebugging, FunctionLinkage) {
  // Test functions get expected linkage and names in the unoptimized and
  // optimized modules. Each optimized module function should have a
  // corresponding unoptimized version, and ones with internal linkage should be
  // given external aliases.
  StringRef IR = R"(
    ; Mark as used - unused internal or weak symbols aren't promoted.
    @llvm.compiler.used = appending global [1 x ptr] [ptr @f], section "llvm.metadata"

    define dso_local void @f() !dbg !4 {
      ret void
    }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 23.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.cpp", directory: "/")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
    !5 = !DISubroutineType(types: !6)
    !6 = !{null}
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  std::unique_ptr<Module> M = parseAssemblyString(IR, Error, Context);
  ASSERT_TRUE(M != nullptr) << "Failed to parse IR\n";

  // Input:        define f
  // Opt module:   define f, alias f.promoted
  // Unopt module:         declare f.promoted, define __dyndbg.f.promoted
  auto ExpectPromoted = [&M](GlobalValue::LinkageTypes Linkage,
                             GlobalValue::VisibilityTypes Visibility,
                             int Line) {
#define FAIL_MSG "ExpectPromoted called from " << Line;
    std::unique_ptr<Module> OptModule = CloneModule(*M);
    Function *F = OptModule->getFunction("f");
    ASSERT_TRUE(F && !F->isDeclaration()) << FAIL_MSG;
    F->setLinkage(Linkage);
    F->setVisibility(Visibility);

    std::unique_ptr<Module> UnoptModule =
        prepareForDynamicDebugging(OptModule.get(), ".promoted");

    // Optimized module function linkage and visibility should not change.
    EXPECT_EQ(F->getLinkage(), Linkage) << FAIL_MSG;
    EXPECT_EQ(F->getVisibility(), Visibility) << FAIL_MSG;

    // External alias added to optimized module.
    GlobalAlias *A = OptModule->getNamedAlias("f.promoted");
    ASSERT_TRUE(A) << FAIL_MSG;
    EXPECT_EQ(A->getAliasee(), F) << FAIL_MSG;
    EXPECT_EQ(A->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(A->getVisibility(), GlobalValue::HiddenVisibility) << FAIL_MSG;

    // Unoptimized module's reference to the optimzed module's function.
    F = UnoptModule->getFunction("f.promoted");
    ASSERT_TRUE(F) << FAIL_MSG;
    EXPECT_TRUE(F->isDeclaration()) << FAIL_MSG;
    EXPECT_EQ(F->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(F->getVisibility(), GlobalValue::HiddenVisibility) << FAIL_MSG;

    // Unoptimized module's copy of the function.
    F = UnoptModule->getFunction("__dyndbg.f.promoted");
    ASSERT_TRUE(F) << FAIL_MSG;
    EXPECT_FALSE(F->isDeclaration()) << FAIL_MSG;
    EXPECT_EQ(F->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(F->getVisibility(), GlobalValue::HiddenVisibility) << FAIL_MSG;
#undef FAIL_MSG
  };

  // Input:         define f
  // Opt module:    define f
  // Unopt module: declare f, define __dyndbg.f
  auto ExpectNotPromoted = [&M](GlobalValue::LinkageTypes Linkage,
                                GlobalValue::VisibilityTypes Visibility,
                                int Line) {
#define FAIL_MSG "ExpectNotPromoted called from " << Line;
    std::unique_ptr<Module> OptModule = CloneModule(*M);
    Function *F = OptModule->getFunction("f");
    ASSERT_TRUE(F && !F->isDeclaration()) << FAIL_MSG;
    F->setLinkage(Linkage);
    F->setVisibility(Visibility);

    std::unique_ptr<Module> UnoptModule =
        prepareForDynamicDebugging(OptModule.get(), ".promoted");

    // Optimized module function linkage and visibility should not change.
    EXPECT_EQ(F->getLinkage(), Linkage) << FAIL_MSG;
    EXPECT_EQ(F->getVisibility(), Visibility) << FAIL_MSG;

    // External alias not added to optimized module.
    EXPECT_EQ(OptModule->alias_size(), 0u) << FAIL_MSG;

    // Unoptimized module's reference to the optimzed module's function.
    F = UnoptModule->getFunction("f");
    ASSERT_TRUE(F) << FAIL_MSG;
    EXPECT_TRUE(F->isDeclaration()) << FAIL_MSG;
    EXPECT_EQ(F->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(F->getVisibility(), Visibility) << FAIL_MSG;

    // Unoptimized module's copy of the function.
    F = UnoptModule->getFunction("__dyndbg.f");
    ASSERT_TRUE(F) << FAIL_MSG;
    EXPECT_FALSE(F->isDeclaration()) << FAIL_MSG;
    EXPECT_EQ(F->getLinkage(), Linkage) << FAIL_MSG;
    EXPECT_EQ(F->getVisibility(), Visibility) << FAIL_MSG;
#undef FAIL_MSG
  };

  ExpectPromoted(GlobalValue::InternalLinkage, GlobalValue::DefaultVisibility,
                 __LINE__);
  ExpectPromoted(GlobalValue::PrivateLinkage, GlobalValue::DefaultVisibility,
                 __LINE__);

  ExpectNotPromoted(GlobalValue::ExternalLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::AvailableExternallyLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::AvailableExternallyLinkage,
                    GlobalValue::HiddenVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::AvailableExternallyLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::LinkOnceAnyLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::LinkOnceAnyLinkage,
                    GlobalValue::HiddenVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::LinkOnceAnyLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::WeakAnyLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakAnyLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakAnyLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::WeakODRLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakODRLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakODRLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::ExternalWeakLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalWeakLinkage,
                    GlobalValue::HiddenVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalWeakLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  // Add a comadt group.
  Comdat *Comdat = M->getOrInsertComdat("c");
  Comdat->setSelectionKind(Comdat::Any);
  Function *F = M->getFunction("f");
  ASSERT_TRUE(F);
  F->setComdat(Comdat);
  // These linkage types should no longer be promtoed.
  ExpectNotPromoted(GlobalValue::InternalLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::PrivateLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
}

TEST(DynamicDebugging, GlobalVariableLinkage) {
  // Test global vairables get expected linkage and names in the unoptimized and
  // optimized modules. Each optimized module global should be referenced from
  // the unoptimized version, and ones with internal linkage should be
  // given external aliases to facilitate this.
  StringRef IR = R"(
    ; Mark as used - unused internal or weak symbols aren't promoted.
    @llvm.compiler.used = appending global [1 x ptr] [ptr @g], section "llvm.metadata"

    @g = dso_local global i32 1, align 4

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 23.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.cpp", directory: "/")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  std::unique_ptr<Module> M = parseAssemblyString(IR, Error, Context);
  ASSERT_TRUE(M != nullptr) << "Failed to parse IR\n";

  // If 'g' is a definition, add 'g.promoted' alias which is referenced from
  // the unoptimized module. Declarations don't get promoted (no alias added).
  auto ExpectPromoted = [&M](GlobalValue::LinkageTypes Linkage,
                             GlobalValue::VisibilityTypes Visibility,
                             int Line) {
#define FAIL_MSG "ExpectPromoted called from " << Line;
    std::unique_ptr<Module> OptModule = CloneModule(*M);
    GlobalVariable *G = OptModule->getGlobalVariable("g");
    ASSERT_TRUE(G) << FAIL_MSG;
    G->setLinkage(Linkage);
    G->setVisibility(Visibility);

    std::unique_ptr<Module> UnoptModule =
        prepareForDynamicDebugging(OptModule.get(), ".promoted");

    // Optimized module function linkage and visibility should not change.
    EXPECT_EQ(G->getLinkage(), Linkage) << FAIL_MSG;
    EXPECT_EQ(G->getVisibility(), Visibility) << FAIL_MSG;

    // External alias added to optimized module.
    GlobalAlias *A = OptModule->getNamedAlias("g.promoted");
    ASSERT_TRUE(A) << FAIL_MSG;
    EXPECT_EQ(A->getAliasee(), G) << FAIL_MSG;
    EXPECT_EQ(A->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(A->getVisibility(), GlobalValue::HiddenVisibility) << FAIL_MSG;

    // Unoptimized module's reference to the optimzed module's function.
    G = UnoptModule->getGlobalVariable("g.promoted");
    ASSERT_TRUE(G) << FAIL_MSG;
    EXPECT_TRUE(G->isDeclaration()) << FAIL_MSG;
    EXPECT_EQ(G->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(G->getVisibility(), GlobalValue::HiddenVisibility) << FAIL_MSG;

    // The unoptimized module doesn't contain any global definitions.
    for (GlobalVariable &GV : UnoptModule->globals())
      EXPECT_TRUE(GV.isDeclaration());

#undef FAIL_MSG
  };

  // Add reference to 'g' in the unoptimized module.
  auto ExpectNotPromoted = [&M](GlobalValue::LinkageTypes Linkage,
                                GlobalValue::VisibilityTypes Visibility,
                                int Line) {
#define FAIL_MSG "ExpectPromoted called from " << Line;
    std::unique_ptr<Module> OptModule = CloneModule(*M);
    GlobalVariable *G = OptModule->getGlobalVariable("g");
    ASSERT_TRUE(G) << FAIL_MSG;
    G->setLinkage(Linkage);
    G->setVisibility(Visibility);

    std::unique_ptr<Module> UnoptModule =
        prepareForDynamicDebugging(OptModule.get(), ".promoted");

    // Optimized module function linkage and visibility should not change.
    EXPECT_EQ(G->getLinkage(), Linkage) << FAIL_MSG;
    EXPECT_EQ(G->getVisibility(), Visibility) << FAIL_MSG;

    // External alias not added to optimized module.
    EXPECT_EQ(OptModule->alias_size(), 0u) << FAIL_MSG;

    // Unoptimized module's reference to the optimzed module's function.
    G = UnoptModule->getGlobalVariable("g");
    ASSERT_TRUE(G) << FAIL_MSG;
    EXPECT_TRUE(G->isDeclaration()) << FAIL_MSG;
    EXPECT_EQ(G->getLinkage(), GlobalValue::ExternalLinkage) << FAIL_MSG;
    EXPECT_EQ(G->getVisibility(), Visibility) << FAIL_MSG;

    // The unoptimized module doesn't contain any global definitions.
    for (GlobalVariable &GV : UnoptModule->globals())
      EXPECT_TRUE(GV.isDeclaration());

#undef FAIL_MSG
  };

  ExpectPromoted(GlobalValue::InternalLinkage, GlobalValue::DefaultVisibility,
                 __LINE__);
  ExpectPromoted(GlobalValue::PrivateLinkage, GlobalValue::DefaultVisibility,
                 __LINE__);

  ExpectPromoted(GlobalValue::InternalLinkage, GlobalValue::DefaultVisibility,
                 __LINE__);
  ExpectPromoted(GlobalValue::PrivateLinkage, GlobalValue::DefaultVisibility,
                 __LINE__);

  ExpectNotPromoted(GlobalValue::ExternalLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::AvailableExternallyLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::AvailableExternallyLinkage,
                    GlobalValue::HiddenVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::AvailableExternallyLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::LinkOnceAnyLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::LinkOnceAnyLinkage,
                    GlobalValue::HiddenVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::LinkOnceAnyLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::WeakAnyLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakAnyLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakAnyLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::WeakODRLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakODRLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::WeakODRLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::ExternalWeakLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalWeakLinkage,
                    GlobalValue::HiddenVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::ExternalWeakLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  ExpectNotPromoted(GlobalValue::CommonLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::CommonLinkage, GlobalValue::HiddenVisibility,
                    __LINE__);
  ExpectNotPromoted(GlobalValue::CommonLinkage,
                    GlobalValue::ProtectedVisibility, __LINE__);

  // Add a comadt group.
  Comdat *Comdat = M->getOrInsertComdat("c");
  Comdat->setSelectionKind(Comdat::Any);
  GlobalVariable *G = M->getGlobalVariable("g");
  ASSERT_TRUE(G);
  G->setComdat(Comdat);
  // These linkage types should no longer be promtoed.
  ExpectNotPromoted(GlobalValue::InternalLinkage,
                    GlobalValue::DefaultVisibility, __LINE__);
  ExpectNotPromoted(GlobalValue::PrivateLinkage, GlobalValue::DefaultVisibility,
                    __LINE__);
}

TEST(DynamicDebugging, DiscardableGlobal) {
  // Test discardable symbols are added to the @llvm.compiler.used global,
  // which prevents them being discarded (including by globalopt replacing) them
  // with their external linkage aliases.
  StringRef IR = R"(
    @not_discardable_used = dso_local global i32 1, align 4
    @not_discardable_unused = dso_local global i32 1, align 4

    @discardable_used = linkonce_odr dso_local global i32 2, align 4
    @discardable_unused = linkonce_odr dso_local global i32 2, align 4

    define internal void @use() {
      %1 = load i32, ptr @not_discardable_used, align 4
      %2 = load i32, ptr @discardable_used, align 4
      ret void
    }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 23.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.cpp", directory: "/")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Error, Context);
  ASSERT_TRUE(M != nullptr) << Error.getMessage();

  // We're only interested in M (drop returned unoptimzed module).
  prepareForDynamicDebugging(M.get(), ".whatever");
  auto *CompilerUsedGV = M->getGlobalVariable("llvm.compiler.used");
  ASSERT_TRUE(CompilerUsedGV);
  ASSERT_TRUE(CompilerUsedGV->hasInitializer());
  ASSERT_TRUE(isa<ConstantArray>(CompilerUsedGV->getInitializer()));

  ConstantArray *CompilerUsedArr =
      cast<ConstantArray>(CompilerUsedGV->getInitializer());

  GlobalVariable *DiscardableUsed = M->getGlobalVariable("discardable_used");
  ASSERT_TRUE(DiscardableUsed);

  // Expect to see @llvm.compiler.used = appending global [1 x ptr] [ptr @discardable_used].
  EXPECT_EQ(CompilerUsedArr->getNumOperands(), 1u);
  EXPECT_TRUE(llvm::find(CompilerUsedArr->operands(), DiscardableUsed));
}

} // namespace