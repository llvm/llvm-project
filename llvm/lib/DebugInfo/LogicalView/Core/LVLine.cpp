//===-- LVLine.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the LVLine class.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVLine.h"
#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"

using namespace llvm;
using namespace llvm::logicalview;

#define DEBUG_TYPE "Line"

namespace {
const char *const KindBasicBlock = "BasicBlock";
const char *const KindDiscriminator = "Discriminator";
const char *const KindEndSequence = "EndSequence";
const char *const KindEpilogueBegin = "EpilogueBegin";
const char *const KindLineDebug = "Line";
const char *const KindLineSource = "Code";
const char *const KindNewStatement = "NewStatement";
const char *const KindPrologueEnd = "PrologueEnd";
const char *const KindUndefined = "Undefined";
const char *const KindAlwaysStepInto = "AlwaysStepInto"; // CodeView
const char *const KindNeverStepInto = "NeverStepInto";   // CodeView
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Logical line.
//===----------------------------------------------------------------------===//
// Return a string representation for the line kind.
const char *LVLine::kind() const {
  const char *Kind = KindUndefined;
  if (getIsLineDebug())
    Kind = KindLineDebug;
  else if (getIsLineAssembler())
    Kind = KindLineSource;
  return Kind;
}

// String used as padding for printing elements with no line number.
std::string LVLine::noLineAsString(bool ShowZero) const {
  return (ShowZero || options().getAttributeZero()) ? ("    0   ")
                                                    : ("    -   ");
}

void LVLine::print(raw_ostream &OS, bool Full) const {
  if (getReader().doPrintLine(this)) {
    getReaderCompileUnit()->incrementPrintedLines();
    LVElement::print(OS, Full);
    printExtra(OS, Full);
  }
}

//===----------------------------------------------------------------------===//
// DWARF line record.
//===----------------------------------------------------------------------===//
std::string LVLineDebug::statesInfo(bool Formatted) const {
  // Returns the DWARF extra qualifiers.
  std::string String;
  raw_string_ostream Stream(String);

  std::string Separator = Formatted ? " " : "";
  if (getIsNewStatement()) {
    Stream << Separator << "{" << KindNewStatement << "}";
    Separator = " ";
  }
  if (getIsDiscriminator()) {
    Stream << Separator << "{" << KindDiscriminator << "}";
    Separator = " ";
  }
  if (getIsBasicBlock()) {
    Stream << Separator << "{" << KindBasicBlock << "}";
    Separator = " ";
  }
  if (getIsEndSequence()) {
    Stream << Separator << "{" << KindEndSequence << "}";
    Separator = " ";
  }
  if (getIsEpilogueBegin()) {
    Stream << Separator << "{" << KindEpilogueBegin << "}";
    Separator = " ";
  }
  if (getIsPrologueEnd()) {
    Stream << Separator << "{" << KindPrologueEnd << "}";
    Separator = " ";
  }
  if (getIsAlwaysStepInto()) {
    Stream << Separator << "{" << KindAlwaysStepInto << "}";
    Separator = " ";
  }
  if (getIsNeverStepInto()) {
    Stream << Separator << "{" << KindNeverStepInto << "}";
    Separator = " ";
  }

  return String;
}

void LVLineDebug::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind());

  if (options().getAttributeQualifier()) {
    // The qualifier includes the states information and the source filename
    // that contains the line element.
    OS << statesInfo(/*Formatted=*/true);
    OS << " " << formattedName(getPathname());
  }
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// Assembler line extracted from the ELF .text section.
//===----------------------------------------------------------------------===//
void LVLineAssembler::printExtra(raw_ostream &OS, bool Full) const {
  OS << formattedKind(kind());
  OS << " " << formattedName(getName());
  OS << "\n";
}
