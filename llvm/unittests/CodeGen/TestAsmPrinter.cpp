//===--- unittests/CodeGen/TestAsmPrinter.cpp -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestAsmPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using ::testing::StrictMock;

// Note: a non-const reference argument cannot be passed through
// testing::StrictMock, thus, we pass a pointer and dereference it here.
MockMCStreamer::MockMCStreamer(MCContext *Ctx) : MCStreamer(*Ctx) {}

MockMCStreamer::~MockMCStreamer() = default;

TestAsmPrinter::TestAsmPrinter() = default;

TestAsmPrinter::~TestAsmPrinter() = default;

llvm::Expected<std::unique_ptr<TestAsmPrinter>>
TestAsmPrinter::create(const std::string &TripleStr, uint16_t DwarfVersion,
                       dwarf::DwarfFormat DwarfFormat) {
  std::string ErrorStr;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleStr, ErrorStr);
  if (!TheTarget)
    return std::unique_ptr<TestAsmPrinter>();

  std::unique_ptr<TestAsmPrinter> TestPrinter(new TestAsmPrinter);
  if (llvm::Error E =
          TestPrinter->init(TheTarget, TripleStr, DwarfVersion, DwarfFormat))
    return std::move(E);

  return std::move(TestPrinter);
}

// Note:: based on dwarfgen::Generator::init() from
// llvm/unittests/DebugInfo/DWARF/DwarfGenerator.cpp
llvm::Error TestAsmPrinter::init(const Target *TheTarget, StringRef TripleName,
                                 uint16_t DwarfVersion,
                                 dwarf::DwarfFormat DwarfFormat) {
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          std::nullopt));
  if (!TM)
    return make_error<StringError>("no target machine for target " + TripleName,
                                   inconvertibleErrorCode());

  Triple TheTriple(TripleName);
  MC.reset(new MCContext(TheTriple, TM->getMCAsmInfo(), TM->getMCRegisterInfo(),
                         TM->getMCSubtargetInfo()));
  TM->getObjFileLowering()->Initialize(*MC, *TM);
  MC->setObjectFileInfo(TM->getObjFileLowering());

  MS = new StrictMock<MockMCStreamer>(MC.get());

  Asm.reset(
      TheTarget->createAsmPrinter(*TM, std::unique_ptr<MockMCStreamer>(MS)));
  if (!Asm)
    return make_error<StringError>("no asm printer for target " + TripleName,
                                   inconvertibleErrorCode());

  // Set the DWARF version correctly on all classes that we use.
  MC->setDwarfVersion(DwarfVersion);
  Asm->setDwarfVersion(DwarfVersion);

  // Set the DWARF format.
  MC->setDwarfFormat(DwarfFormat);

  return Error::success();
}

void TestAsmPrinter::setDwarfUsesRelocationsAcrossSections(bool Enable) {
  Asm->setDwarfUsesRelocationsAcrossSections(Enable);
}
