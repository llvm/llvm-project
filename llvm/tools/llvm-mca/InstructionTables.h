//===--------------------- InstructionTables.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements a custom driver to generate instruction tables.
/// See the description of command-line flag -instruction-tables in
/// docs/CommandGuide/lvm-mca.rst
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_INSTRUCTIONTABLES_H
#define LLVM_TOOLS_LLVM_MCA_INSTRUCTIONTABLES_H

#include "View.h"
#include "InstrBuilder.h"
#include "SourceMgr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSchedule.h"

namespace mca {

class InstructionTables {
  const llvm::MCSchedModel &SM;
  InstrBuilder &IB;
  SourceMgr &S;
  llvm::SmallVector<std::unique_ptr<View>, 8> Views;

public:
  InstructionTables(const llvm::MCSchedModel &Model, InstrBuilder &Builder,
                    SourceMgr &Source)
      : SM(Model), IB(Builder), S(Source) {}

  void addView(std::unique_ptr<View> V) {
    Views.emplace_back(std::move(V));
  }

  void run();
  
  void printReport(llvm::raw_ostream &OS) const;
};
} // namespace mca

#endif
