//===- MachineStableHashTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class MachineStableHashTest : public testing::Test {
public:
  MachineStableHashTest() {}

protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;

  static void SetUpTestCase() {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override { M = std::make_unique<Module>("Dummy", Context); }

  std::unique_ptr<TargetMachine>
  createTargetMachine(std::string TT, StringRef CPU, StringRef FS) {
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      return nullptr;
    TargetOptions Options;
    return std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TT, CPU, FS, Options, std::nullopt, std::nullopt));
  }

  std::unique_ptr<Module> parseMIR(const TargetMachine &TM, StringRef MIRCode,
                                   MachineModuleInfo &MMI) {
    SMDiagnostic Diagnostic;
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return nullptr;

    std::unique_ptr<Module> Mod = MIR->parseIRModule();
    if (!Mod)
      return nullptr;

    Mod->setDataLayout(TM.createDataLayout());

    if (MIR->parseMachineFunctions(*Mod, MMI)) {
      M.reset();
      return nullptr;
    }

    return Mod;
  }
};

TEST_F(MachineStableHashTest, StableGlobalName) {
  auto TM = createTargetMachine(("aarch64--"), "", "");
  if (!TM)
    GTEST_SKIP();
  StringRef MIRString = R"MIR(
--- |
  define void @f1() { ret void }
  define void @f2() { ret void }
  define void @f3() { ret void }
  define void @f4() { ret void }
  declare void @goo()
  declare void @goo.llvm.123()
  declare void @goo.__uniq.456()
  declare void @goo.invalid.789()
...
---
name:            f1
alignment:       16
tracksRegLiveness: true
frameInfo:
  maxAlignment:    16
machineFunctionInfo: {}
body:             |
  bb.0:
  liveins: $lr
    BL @goo
  RET undef $lr

...
---
name:            f2
body:             |
  bb.0:
  liveins: $lr
    BL @goo.llvm.123
  RET undef $lr
...
---
name:            f3
body:             |
  bb.0:
  liveins: $lr
    BL @goo.__uniq.456
  RET undef $lr
...
---
name:            f4
body:             |
  bb.0:
  liveins: $lr
    BL @goo.invalid.789
  RET undef $lr
...
)MIR";
  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);
  auto *MF1 = MMI.getMachineFunction(*M->getFunction("f1"));
  auto *MF2 = MMI.getMachineFunction(*M->getFunction("f2"));
  auto *MF3 = MMI.getMachineFunction(*M->getFunction("f3"));
  auto *MF4 = MMI.getMachineFunction(*M->getFunction("f4"));

  EXPECT_EQ(stableHashValue(*MF1), stableHashValue(*MF2))
      << "Expect the suffix, `.llvm.{number}` to be ignored.";
  EXPECT_EQ(stableHashValue(*MF1), stableHashValue(*MF3))
      << "Expect the suffix, `.__uniq.{number}` to be ignored.";
  // Do not ignore `.invalid.{number}`.
  EXPECT_NE(stableHashValue(*MF1), stableHashValue(*MF4));
}

TEST_F(MachineStableHashTest, ContentName) {
  auto TM = createTargetMachine(("aarch64--"), "", "");
  if (!TM)
    GTEST_SKIP();
  StringRef MIRString = R"MIR(
--- |
  define void @f1() { ret void }
  define void @f2() { ret void }
  define void @f3() { ret void }
  define void @f4() { ret void }
  declare void @goo()
  declare void @goo.content.123()
  declare void @zoo.content.123()
  declare void @goo.content.456()
...
---
name:            f1
alignment:       16
tracksRegLiveness: true
frameInfo:
  maxAlignment:    16
machineFunctionInfo: {}
body:             |
  bb.0:
  liveins: $lr
    BL @goo
  RET undef $lr
...
---
name:            f2
body:             |
  bb.0:
  liveins: $lr
    BL @goo.content.123
  RET undef $lr
...
---
name:            f3
body:             |
  bb.0:
  liveins: $lr
    BL @zoo.content.123
  RET undef $lr
...
---
name:            f4
body:             |
  bb.0:
  liveins: $lr
    BL @goo.content.456
  RET undef $lr
...
)MIR";
  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);
  auto *MF1 = MMI.getMachineFunction(*M->getFunction("f1"));
  auto *MF2 = MMI.getMachineFunction(*M->getFunction("f2"));
  auto *MF3 = MMI.getMachineFunction(*M->getFunction("f3"));
  auto *MF4 = MMI.getMachineFunction(*M->getFunction("f4"));

  // Do not ignore `.content.{number}`.
  EXPECT_NE(stableHashValue(*MF1), stableHashValue(*MF2));
  EXPECT_EQ(stableHashValue(*MF2), stableHashValue(*MF3))
      << "Expect the same hash for the same suffix, `.content.{number}`";
  // Different suffixes should result in different hashes.
  EXPECT_NE(stableHashValue(*MF2), stableHashValue(*MF4));
  EXPECT_NE(stableHashValue(*MF3), stableHashValue(*MF4));
}
