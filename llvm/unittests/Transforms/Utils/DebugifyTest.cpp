//===- DebugifyTest.cpp - Debugify unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Debugify.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("DebugifyTest", errs());
  return Mod;
}

namespace llvm {
void initializeDebugInfoDropPass(PassRegistry &);
void initializeDebugInfoDummyAnalysisPass(PassRegistry &);

namespace {
struct DebugInfoDrop : public FunctionPass {
  static char ID;
  bool runOnFunction(Function &F) override {
    // Drop DISubprogram.
    F.setSubprogram(nullptr);
    for (BasicBlock &BB : F) {
      // Remove debug locations.
      for (Instruction &I : BB)
        I.setDebugLoc(DebugLoc());
    }

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  DebugInfoDrop() : FunctionPass(ID) {}
};

struct DebugValueDrop : public FunctionPass {
  static char ID;
  bool runOnFunction(Function &F) override {
    SmallVector<DbgVariableIntrinsic *, 4> Dbgs;
    for (BasicBlock &BB : F) {
      // Remove dbg var intrinsics.
      for (Instruction &I : BB) {
        if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&I))
          Dbgs.push_back(DVI);
        // If there are any non-intrinsic records (DPValues), drop those too.
        I.dropDbgValues();
      }
    }

    for (auto &I : Dbgs)
      I->eraseFromParent();

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  DebugValueDrop() : FunctionPass(ID) {}
};

struct DebugInfoDummyAnalysis : public FunctionPass {
  static char ID;
  bool runOnFunction(Function &F) override {
    // Do nothing, so debug info stays untouched.
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  DebugInfoDummyAnalysis() : FunctionPass(ID) {}
};
}

char DebugInfoDrop::ID = 0;
char DebugValueDrop::ID = 0;
char DebugInfoDummyAnalysis::ID = 0;

TEST(DebugInfoDrop, DropOriginalDebugInfo) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
  )");

  DebugInfoDrop *P = new DebugInfoDrop();

  DebugInfoPerPass DIBeforePass;
  DebugifyCustomPassManager Passes;
  Passes.setDebugInfoBeforePass(DIBeforePass);
  Passes.add(createDebugifyModulePass(DebugifyMode::OriginalDebugInfo, "",
                                      &(Passes.getDebugInfoPerPass())));
  Passes.add(P);
  Passes.add(createCheckDebugifyModulePass(false, "", nullptr,
                                           DebugifyMode::OriginalDebugInfo,
                                           &(Passes.getDebugInfoPerPass())));

  testing::internal::CaptureStderr();
  Passes.run(*M);

  std::string StdOut = testing::internal::GetCapturedStderr();

  std::string ErrorForSP = "ERROR:  dropped DISubprogram of";
  std::string WarningForLoc = "WARNING:  dropped DILocation of";
  std::string FinalResult = "CheckModuleDebugify (original debuginfo): FAIL";

  EXPECT_TRUE(StdOut.find(ErrorForSP) != std::string::npos);
  EXPECT_TRUE(StdOut.find(WarningForLoc) != std::string::npos);
  EXPECT_TRUE(StdOut.find(FinalResult) != std::string::npos);
}

TEST(DebugValueDrop, DropOriginalDebugValues) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "b", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
  )");

  DebugValueDrop *P = new DebugValueDrop();

  DebugInfoPerPass DIBeforePass;
  DebugifyCustomPassManager Passes;
  Passes.setDebugInfoBeforePass(DIBeforePass);
  Passes.add(createDebugifyModulePass(DebugifyMode::OriginalDebugInfo, "",
                                      &(Passes.getDebugInfoPerPass())));
  Passes.add(P);
  Passes.add(createCheckDebugifyModulePass(false, "", nullptr,
                                           DebugifyMode::OriginalDebugInfo,
                                           &(Passes.getDebugInfoPerPass())));

  testing::internal::CaptureStderr();
  Passes.run(*M);

  std::string StdOut = testing::internal::GetCapturedStderr();

  std::string ErrorForSP = "ERROR:  dropped DISubprogram of";
  std::string WarningForLoc = "WARNING:  dropped DILocation of";
  std::string WarningForVars = "WARNING:  drops dbg.value()/dbg.declare() for";
  std::string FinalResult = "CheckModuleDebugify (original debuginfo): FAIL";

  EXPECT_TRUE(StdOut.find(ErrorForSP) == std::string::npos);
  EXPECT_TRUE(StdOut.find(WarningForLoc) == std::string::npos);
  EXPECT_TRUE(StdOut.find(WarningForVars) != std::string::npos);
  EXPECT_TRUE(StdOut.find(FinalResult) != std::string::npos);
}

TEST(DebugInfoDummyAnalysis, PreserveOriginalDebugInfo) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i32 @g(i32 %b) !dbg !6 {
      %c = add i32 %b, 1, !dbg !11
      call void @llvm.dbg.value(metadata i32 %c, metadata !9, metadata !DIExpression()), !dbg !11
      ret i32 1, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "test.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "c", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
  )");

  DebugInfoDummyAnalysis *P = new DebugInfoDummyAnalysis();

  DebugInfoPerPass DIBeforePass;
  DebugifyCustomPassManager Passes;
  Passes.setDebugInfoBeforePass(DIBeforePass);
  Passes.add(createDebugifyModulePass(DebugifyMode::OriginalDebugInfo, "",
                                      &(Passes.getDebugInfoPerPass())));
  Passes.add(P);
  Passes.add(createCheckDebugifyModulePass(false, "", nullptr,
                                           DebugifyMode::OriginalDebugInfo,
                                           &(Passes.getDebugInfoPerPass())));

  testing::internal::CaptureStderr();
  Passes.run(*M);

  std::string StdOut = testing::internal::GetCapturedStderr();

  std::string ErrorForSP = "ERROR:  dropped DISubprogram of";
  std::string WarningForLoc = "WARNING:  dropped DILocation of";
  std::string WarningForVars = "WARNING:  drops dbg.value()/dbg.declare() for";
  std::string FinalResult = "CheckModuleDebugify (original debuginfo): PASS";

  EXPECT_TRUE(StdOut.find(ErrorForSP) == std::string::npos);
  EXPECT_TRUE(StdOut.find(WarningForLoc) == std::string::npos);
  EXPECT_TRUE(StdOut.find(WarningForVars) == std::string::npos);
  EXPECT_TRUE(StdOut.find(FinalResult) != std::string::npos);
}

} // end namespace llvm

INITIALIZE_PASS_BEGIN(DebugInfoDrop, "debuginfodroppass", "debuginfodroppass",
                      false, false)
INITIALIZE_PASS_END(DebugInfoDrop, "debuginfodroppass", "debuginfodroppass", false,
                    false)

INITIALIZE_PASS_BEGIN(DebugInfoDummyAnalysis, "debuginfodummyanalysispass",
                      "debuginfodummyanalysispass", false, false)
INITIALIZE_PASS_END(DebugInfoDummyAnalysis, "debuginfodummyanalysispass",
                    "debuginfodummyanalysispass", false, false)
