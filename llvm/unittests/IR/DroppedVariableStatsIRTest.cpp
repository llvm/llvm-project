//===- unittests/IR/DroppedVariableStatsIRTest.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
namespace llvm {
void initializePassTest1Pass(PassRegistry &);

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AbstractCallSiteTests", errs());
  return Mod;
}
} // namespace llvm

namespace {

// This test ensures that if a #dbg_value and an instruction that exists in the
// same scope as that #dbg_value are both deleted as a result of an optimization
// pass, debug information is considered not dropped.
TEST(DroppedVariableStatsIR, BothDeleted) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(
      ; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9)
      !17 = !DILocation(line: 2, column: 11, scope: !9))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      I.eraseFromParent();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that shares the same scope as the #dbg_value still exists,
// debug information is conisdered dropped.
TEST(DroppedVariableStatsIR, DbgValLost) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(
      ; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9)
      !17 = !DILocation(line: 2, column: 11, scope: !9))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has an unrelated scope as the #dbg_value still
// exists, debug information is conisdered not dropped.
TEST(DroppedVariableStatsIR, UnrelatedScopes) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(
      ; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9)
      !17 = !DILocation(line: 2, column: 11, scope: !18)
      !18 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bari", scope: !10, file: !10, line: 11, type: !11, scopeLine: 1,  unit: !0, retainedNodes: !14))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, debug information is conisdered dropped.
TEST(DroppedVariableStatsIR, ChildScopes) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(
      ; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9)
      !17 = !DILocation(line: 2, column: 11, scope: !18)
      !18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: 28))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, and the #dbg_value is inlined at another location, debug
// information is conisdered not dropped.
TEST(DroppedVariableStatsIR, InlinedAt) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9, inlinedAt: !19)
      !17 = !DILocation(line: 2, column: 11, scope: !18)
      !18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: 28)
      !19 = !DILocation(line: 3, column: 2, scope: !9))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, and the #dbg_value and the instruction are inlined at another
// location, debug information is conisdered dropped.
TEST(DroppedVariableStatsIR, InlinedAtShared) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9, inlinedAt: !19)
      !17 = !DILocation(line: 2, column: 11, scope: !18, inlinedAt: !19)
      !18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: 28)
      !19 = !DILocation(line: 3, column: 2, scope: !9))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, and the instruction is inlined at a location that is the
// #dbg_value's inlined at location, debug information is conisdered dropped.
TEST(DroppedVariableStatsIR, InlinedAtChild) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      R"(; Function Attrs: mustprogress nounwind ssp uwtable(sync)
      define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr #0 !dbg !9 {
      entry:
        #dbg_value(i32 %x, !15, !DIExpression(), !16)
        %add = add nsw i32 %x, 1, !dbg !17
        ret i32 0
      }
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3}
      !llvm.ident = !{!8}
      !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
      !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
      !3 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = !{!"clang"}
      !9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, unit: !0, retainedNodes: !14)
      !10 = !DIFile(filename: "/tmp/code.cpp", directory: "")
      !11 = !DISubroutineType(types: !12)
      !12 = !{!13, !13}
      !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !14 = !{!15}
      !15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
      !16 = !DILocation(line: 0, scope: !9, inlinedAt: !19)
      !17 = !DILocation(line: 2, column: 11, scope: !18, inlinedAt: !20)
      !18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: 28)
      !19 = !DILocation(line: 3, column: 2, scope: !9);
      !20 = !DILocation(line: 4, column: 5, scope: !18, inlinedAt: !19))";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStatsIR Stats(true);
  Stats.runBeforePass("", llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // This loop simulates an IR pass that drops debug information.
  for (auto &F : *M) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())));
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

} // end anonymous namespace
