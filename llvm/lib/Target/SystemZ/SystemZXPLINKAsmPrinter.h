//===-- SystemZXPLINKAsmPrinter.h - SystemZ XPLINK asm printer --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SystemZXPLINKAsmPrinter class, which owns all
// z/OS XPLINK-specific asm printer behaviour: ADA table management, PPA1/PPA2,
// IDRL, entry-point markers, and XPLINK call/return instruction lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZXPLINKASMPRINTER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZXPLINKASMPRINTER_H

#include "MCTargetDesc/SystemZTargetStreamer.h"
#include "SystemZAsmPrinter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

namespace llvm {
class MCStreamer;
class MCSymbolGOFF;
class MachineFunction;
class MachineInstr;
class MachineOperand;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY SystemZXPLINKAsmPrinter
    : public SystemZAsmPrinter {

  /// Call type information for XPLINK.
  enum class CallType {
    BASR76 = 0,   // b'x000' == BASR  r7,r6
    BRAS7 = 1,    // b'x001' == BRAS  r7,ep
    RESVD_2 = 2,  // b'x010'
    BRASL7 = 3,   // b'x011' == BRASL r7,ep
    RESVD_4 = 4,  // b'x100'
    RESVD_5 = 5,  // b'x101'
    BALR1415 = 6, // b'x110' == BALR  r14,r15
    BASR33 = 7,   // b'x111' == BASR  r3,r3
  };

  // The Associated Data Area (ADA) contains descriptors which help locating
  // external symbols. For each symbol and type, the displacement into the ADA
  // is stored.
  class AssociatedDataAreaTable {
  public:
    using DisplacementTable =
        MapVector<std::pair<const MCSymbol *, unsigned>, uint32_t>;

  private:
    const uint64_t PointerSize;

    /// The mapping of name/slot type pairs to displacements.
    DisplacementTable Displacements;

    /// The next available displacement value. Incremented when new entries into
    /// the ADA are created.
    uint32_t NextDisplacement = 0;

  public:
    AssociatedDataAreaTable(uint64_t PointerSize) : PointerSize(PointerSize) {}

    /// @brief Add a function descriptor to the ADA.
    /// @param MF The function containing the ADA_ENTRY instruction.
    /// @param MO The operand describing the descriptor symbol.
    /// @return The displacement of the descriptor into the ADA.
    uint32_t insert(const MachineFunction &MF, const MachineOperand &MO);

    /// @brief Get the displacement into associated data area (ADA) for a name.
    /// If no displacement is already associated with the name, assign one and
    /// return it.
    /// @param Sym The symbol for which the displacement should be returned.
    /// @param SlotKind The ADA type.
    /// @return The displacement of the descriptor into the ADA.
    uint32_t insert(const MCSymbol *Sym, unsigned SlotKind);

    /// Get the table of GOFF displacements.  This is 'const' since it should
    /// never be modified by anything except the APIs on this class.
    const DisplacementTable &getTable() const { return Displacements; }

    uint32_t getNextDisplacement() const { return NextDisplacement; }
  };

  AssociatedDataAreaTable ADATable;

  // Record a list of GlobalAlias associated with a GlobalObject.
  // This is used for z/OS's extra-label-at-definition aliasing strategy.
  // This is similar to what is done for AIX.
  DenseMap<const GlobalObject *, SmallVector<const GlobalAlias *, 1>>
      GOAliasMap;

  void calculatePPA1();
  void emitPPA2(Module &M);
  void emitADASection();
  void emitIDRLSection(Module &M);
  void emitCallInformation(CallType CT);

  SystemZTargetzOSStreamer *getTargetStreamer() {
    MCTargetStreamer *TS = OutStreamer->getTargetStreamer();
    assert(TS && "do not have a target streamer");
    return static_cast<SystemZTargetzOSStreamer *>(TS);
  }

public:
  SystemZXPLINKAsmPrinter(TargetMachine &TM,
                          std::unique_ptr<MCStreamer> Streamer);

  // Override AsmPrinter.
  void emitInstruction(const MachineInstr *MI) override;
  void emitXXStructorList(const DataLayout &DL, const Constant *List,
                          bool IsCtor) override;
  void emitEndOfAsmFile(Module &M) override;
  bool doInitialization(Module &M) override;
  void emitFunctionEntryLabel() override;
  void emitFunctionBodyEnd() override;
  void emitStartOfAsmFile(Module &M) override;
  void emitGlobalAlias(const Module &M, const GlobalAlias &GA) override;
  const MCExpr *lowerConstant(const Constant *CV,
                              const Constant *BaseCV = nullptr,
                              uint64_t Offset = 0) override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZXPLINKASMPRINTER_H
