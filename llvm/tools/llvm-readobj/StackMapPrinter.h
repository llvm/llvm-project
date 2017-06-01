//===-------- StackMapPrinter.h - Pretty-print stackmaps --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_READOBJ_STACKMAPPRINTER_H
#define LLVM_TOOLS_LLVM_READOBJ_STACKMAPPRINTER_H

#include "llvm/Object/StackMapParser.h"

namespace llvm {

// Pretty print a stackmap to the given ostream.
template <typename OStreamT, typename StackMapParserT>
void prettyPrintStackMap(OStreamT &OS, const StackMapParserT &SMP) {

  OS << "LLVM StackMap Version: " << SMP.getVersion()
     << "\nNum Functions: " << SMP.getNumFunctions();

  // Functions:
  for (const auto &F : SMP.functions())
    OS << "\n  Function address: " << F.getFunctionAddress()
       << ", stack size: " << F.getStackSize()
       << ", callsite record count: " << F.getRecordCount();

  // Constants:
  OS << "\nNum Constants: " << SMP.getNumConstants();
  unsigned ConstantIndex = 0;
  for (const auto &C : SMP.constants())
    OS << "\n  #" << ++ConstantIndex << ": " << C.getValue();

  // Records:
  OS << "\nNum Records: " << SMP.getNumRecords();
  for (const auto &R : SMP.records()) {
    OS << "\n  Record ID: " << R.getID()
       << ", instruction offset: " << R.getInstructionOffset()
       << "\n    " << R.getNumLocations() << " locations:";

    unsigned LocationIndex = 0;
    for (const auto &Loc : R.locations()) {
      OS << "\n      #" << ++LocationIndex << ": ";
      switch (Loc.getKind()) {
      case StackMapParserT::LocationKind::Register:
        OS << "Register R#" << Loc.getDwarfRegNum();
        break;
      case StackMapParserT::LocationKind::Direct:
        OS << "Direct R#" << Loc.getDwarfRegNum() << " + "
           << Loc.getOffset();
        break;
      case StackMapParserT::LocationKind::Indirect:
        OS << "Indirect [R#" << Loc.getDwarfRegNum() << " + "
           << Loc.getOffset() << "]";
        break;
      case StackMapParserT::LocationKind::Constant:
        OS << "Constant " << Loc.getSmallConstant();
        break;
      case StackMapParserT::LocationKind::ConstantIndex:
        OS << "ConstantIndex #" << Loc.getConstantIndex() << " ("
           << SMP.getConstant(Loc.getConstantIndex()).getValue() << ")";
        break;
      }
    }

    OS << "\n    " << R.getNumLiveOuts() << " live-outs: [ ";
    for (const auto &LO : R.liveouts())
      OS << "R#" << LO.getDwarfRegNum() << " ("
         << LO.getSizeInBytes() << "-bytes) ";
    OS << "]\n";
  }

 OS << "\n";

}

}

#endif
