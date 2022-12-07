//===--- unittests/CodeGen/TestAsmPrinter.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_CODEGEN_TESTASMPRINTER_H
#define LLVM_UNITTESTS_CODEGEN_TESTASMPRINTER_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCStreamer.h"
#include "gmock/gmock.h"

#include <memory>

namespace llvm {
class AsmPrinter;
class MCContext;
class Target;
class TargetMachine;

class MockMCStreamer : public MCStreamer {
public:
  explicit MockMCStreamer(MCContext *Ctx);
  ~MockMCStreamer();

  // These methods are pure virtual in MCStreamer, thus, have to be overridden:

  MOCK_METHOD2(emitSymbolAttribute,
               bool(MCSymbol *Symbol, MCSymbolAttr Attribute));
  MOCK_METHOD3(emitCommonSymbol,
               void(MCSymbol *Symbol, uint64_t Size, unsigned ByteAlignment));
  MOCK_METHOD5(emitZerofill,
               void(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                    unsigned ByteAlignment, SMLoc Loc));

  // The following are mock methods to be used in tests.

  MOCK_METHOD2(emitLabel, void(MCSymbol *Symbol, SMLoc Loc));
  MOCK_METHOD2(emitIntValue, void(uint64_t Value, unsigned Size));
  MOCK_METHOD3(emitValueImpl,
               void(const MCExpr *Value, unsigned Size, SMLoc Loc));
  MOCK_METHOD3(emitAbsoluteSymbolDiff,
               void(const MCSymbol *Hi, const MCSymbol *Lo, unsigned Size));
  MOCK_METHOD2(emitCOFFSecRel32, void(MCSymbol const *Symbol, uint64_t Offset));
};

class TestAsmPrinter {
  std::unique_ptr<MCContext> MC;
  MockMCStreamer *MS = nullptr; // Owned by AsmPrinter
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<AsmPrinter> Asm;

  /// Private constructor; call TestAsmPrinter::create(...)
  /// to create an instance.
  TestAsmPrinter();

  /// Initialize an AsmPrinter instance with a mocked MCStreamer.
  llvm::Error init(const Target *TheTarget, StringRef TripleStr,
                   uint16_t DwarfVersion, dwarf::DwarfFormat DwarfFormat);

public:
  /// Create an AsmPrinter and accompanied objects.
  /// Returns ErrorSuccess() with an empty value if the requested target is not
  /// supported so that the corresponding test can be gracefully skipped.
  static llvm::Expected<std::unique_ptr<TestAsmPrinter>>
  create(const std::string &TripleStr, uint16_t DwarfVersion,
         dwarf::DwarfFormat DwarfFormat);

  ~TestAsmPrinter();

  void setDwarfUsesRelocationsAcrossSections(bool Enable);

  AsmPrinter *getAP() const { return Asm.get(); }
  AsmPrinter *releaseAP() { return Asm.release(); }
  MCContext &getCtx() const { return *MC; }
  MockMCStreamer &getMS() const { return *MS; }
};

} // end namespace llvm

#endif // LLVM_UNITTESTS_CODEGEN_TESTASMPRINTER_H
