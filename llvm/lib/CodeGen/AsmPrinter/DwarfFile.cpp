//===- llvm/CodeGen/DwarfFile.cpp - Dwarf Debug Framework -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfFile.h"
#include "DwarfCompileUnit.h"
#include "DwarfDebug.h"
#include "DwarfUnit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCStreamer.h"
#include <algorithm>
#include <cstdint>

using namespace llvm;

DwarfFile::DwarfFile(AsmPrinter *AP, StringRef Pref, BumpPtrAllocator &DA)
    : Asm(AP), Abbrevs(AbbrevAllocator), StrPool(DA, *Asm, Pref) {}

void DwarfFile::addUnit(std::unique_ptr<DwarfCompileUnit> U) {
  CUs.push_back(std::move(U));
}

// Emit the various dwarf units to the unit section USection with
// the abbreviations going into ASection.
void DwarfFile::emitUnits(bool UseOffsets) {
  for (const auto &TheU : CUs)
    emitUnit(TheU.get(), UseOffsets);
}

void DwarfFile::emitUnit(DwarfUnit *TheU, bool UseOffsets) {
  DIE &Die = TheU->getUnitDie();
  MCSection *USection = TheU->getSection();
  Asm->OutStreamer->SwitchSection(USection);

  TheU->emitHeader(UseOffsets);

  Asm->emitDwarfDIE(Die);
}

// Compute the size and offset for each DIE.
void DwarfFile::computeSizeAndOffsets() {
  // Offset from the first CU in the debug info section is 0 initially.
  unsigned SecOffset = 0;

  // Iterate over each compile unit and set the size and offsets for each
  // DIE within each compile unit. All offsets are CU relative.
  for (const auto &TheU : CUs) {
    TheU->setDebugSectionOffset(SecOffset);
    SecOffset += computeSizeAndOffsetsForUnit(TheU.get());
  }
}

unsigned DwarfFile::computeSizeAndOffsetsForUnit(DwarfUnit *TheU) {
  // CU-relative offset is reset to 0 here.
  unsigned Offset = sizeof(int32_t) +      // Length of Unit Info
                    TheU->getHeaderSize(); // Unit-specific headers

  // The return value here is CU-relative, after laying out
  // all of the CU DIE.
  return computeSizeAndOffset(TheU->getUnitDie(), Offset);
}

// Compute the size and offset of a DIE. The offset is relative to start of the
// CU. It returns the offset after laying out the DIE.
unsigned DwarfFile::computeSizeAndOffset(DIE &Die, unsigned Offset) {
  return Die.computeOffsetsAndAbbrevs(Asm, Abbrevs, Offset);
}

void DwarfFile::emitAbbrevs(MCSection *Section) { Abbrevs.Emit(Asm, Section); }

// Emit strings into a string section.
void DwarfFile::emitStrings(MCSection *StrSection, MCSection *OffsetSection) {
  StrPool.emit(*Asm, StrSection, OffsetSection);
}

bool DwarfFile::addScopeVariable(LexicalScope *LS, DbgVariable *Var) {
  SmallVectorImpl<DbgVariable *> &Vars = ScopeVariables[LS];
  const DILocalVariable *DV = Var->getVariable();
  // Variables with positive arg numbers are parameters.
  if (unsigned ArgNum = DV->getArg()) {
    // Keep all parameters in order at the start of the variable list to ensure
    // function types are correct (no out-of-order parameters)
    //
    // This could be improved by only doing it for optimized builds (unoptimized
    // builds have the right order to begin with), searching from the back (this
    // would catch the unoptimized case quickly), or doing a binary search
    // rather than linear search.
    auto I = Vars.begin();
    while (I != Vars.end()) {
      unsigned CurNum = (*I)->getVariable()->getArg();
      // A local (non-parameter) variable has been found, insert immediately
      // before it.
      if (CurNum == 0)
        break;
      // A later indexed parameter has been found, insert immediately before it.
      if (CurNum > ArgNum)
        break;
      if (CurNum == ArgNum) {
        (*I)->addMMIEntry(*Var);
        return false;
      }
      ++I;
    }
    Vars.insert(I, Var);
    return true;
  }

  Vars.push_back(Var);
  return true;
}
