//===- MemProfilerTest.cpp - MemProfiler unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/MemProfiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern llvm::cl::opt<bool> ClMemProfMatchHotColdNew;

namespace llvm {
namespace memprof {
namespace {

using ::testing::Return;
using ::testing::SizeIs;

struct MemProfilerTest : public ::testing::Test {
  LLVMContext Context;
  std::unique_ptr<Module> M;

  MemProfilerTest() { ClMemProfMatchHotColdNew = true; }

  void parseAssembly(const StringRef IR) {
    SMDiagnostic Error;
    M = parseAssemblyString(IR, Error, Context);
    std::string ErrMsg;
    raw_string_ostream OS(ErrMsg);
    Error.print("", OS);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(OS.str().c_str());
  }
};

// A mock memprof reader we can inject into the function we are testing.
class MockMemProfReader : public IndexedMemProfReader {
public:
  MOCK_METHOD(Expected<MemProfRecord>, getMemProfRecord,
              (const uint64_t FuncNameHash), (const, override));

  // A helper function to create mock records from frames.
  static MemProfRecord makeRecord(ArrayRef<ArrayRef<Frame>> AllocFrames) {
    MemProfRecord Record;
    MemInfoBlock Info;
    // Mimic values which will be below the cold threshold.
    Info.AllocCount = 1, Info.TotalSize = 550;
    Info.TotalLifetime = 1000 * 1000, Info.TotalLifetimeAccessDensity = 1;
    for (const auto &Callstack : AllocFrames) {
      AllocationInfo AI;
      AI.Info = PortableMemInfoBlock(Info, getHotColdSchema());
      AI.CallStack = std::vector(Callstack.begin(), Callstack.end());
      Record.AllocSites.push_back(AI);
    }
    return Record;
  }
};

TEST_F(MemProfilerTest, AnnotatesCall) {
  parseAssembly(R"IR(
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    define void @_Z3foov() !dbg !10 {
    entry:
      %c1 = call {ptr, i64} @__size_returning_new(i64 32), !dbg !13
      %c2 = call {ptr, i64} @__size_returning_new_aligned(i64 32, i64 8), !dbg !14
      %c3 = call {ptr, i64} @__size_returning_new_hot_cold(i64 32, i8 254), !dbg !15
      %c4 = call {ptr, i64} @__size_returning_new_aligned_hot_cold(i64 32, i64 8, i8 254), !dbg !16
      ret void
    }

    declare {ptr, i64} @__size_returning_new(i64)
    declare {ptr, i64} @__size_returning_new_aligned(i64, i64)
    declare {ptr, i64} @__size_returning_new_hot_cold(i64, i8)
    declare {ptr, i64} @__size_returning_new_aligned_hot_cold(i64, i64, i8)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
    !1 = !DIFile(filename: "mock_file.cc", directory: "mock_dir")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, unit: !0, retainedNodes: !12)
    !11 = !DISubroutineType(types: !12)
    !12 = !{}
    !13 = !DILocation(line: 5, column: 10, scope: !10)
    !14 = !DILocation(line: 6, column: 10, scope: !10)
    !15 = !DILocation(line: 7, column: 10, scope: !10)
    !16 = !DILocation(line: 8, column: 10, scope: !10)
  )IR");

  auto *F = M->getFunction("_Z3foov");
  ASSERT_NE(F, nullptr);

  TargetLibraryInfoWrapperPass WrapperPass;
  auto &TLI = WrapperPass.getTLI(*F);

  auto Guid = Function::getGUID("_Z3foov");
  // All the allocation sites are in foo().
  MemProfRecord MockRecord =
      MockMemProfReader::makeRecord({{Frame(Guid, 1, 10, false)},
                                     {Frame(Guid, 2, 10, false)},
                                     {Frame(Guid, 3, 10, false)},
                                     {Frame(Guid, 4, 10, false)}});
  // Set up mocks for the reader.
  MockMemProfReader Reader;
  EXPECT_CALL(Reader, getMemProfRecord(Guid)).WillOnce(Return(MockRecord));

  MemProfUsePass Pass("/unused/profile/path");
  std::map<uint64_t, MemProfUsePass::AllocMatchInfo> Unused;
  Pass.readMemprof(*F, Reader, TLI, Unused);

  // Since we only have a single type of behaviour for each allocation site, we
  // only get function attributes.
  std::vector<llvm::Attribute> CallsiteAttrs;
  for (const auto &BB : *F) {
    for (const auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (!CI->getCalledFunction()->getName().starts_with(
                "__size_returning_new"))
          continue;
        Attribute Attr = CI->getFnAttr("memprof");
        // The attribute will be invalid if it didn't find one named memprof.
        ASSERT_TRUE(Attr.isValid());
        CallsiteAttrs.push_back(Attr);
      }
    }
  }

  // We match all the variants including ones with the hint since we set
  // ClMemProfMatchHotColdNew to true.
  EXPECT_THAT(CallsiteAttrs, SizeIs(4));
}

} // namespace
} // namespace memprof
} // namespace llvm
