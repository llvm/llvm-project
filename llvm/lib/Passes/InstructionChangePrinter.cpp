//===- InstructionChangePrinter.cpp - Print sparse IR changes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstructionChangePrinter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/FunctionInstructionPrinter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;
using namespace llvm::detail;

namespace {

bool isFunctionSelected(const Function &F) {
  return isFunctionInPrintList("*") || isFunctionInPrintList(F.getName());
}

struct ChangeCounts {
  unsigned AddedBlocks = 0;
  unsigned RemovedBlocks = 0;
  unsigned MovedBlocks = 0;
  unsigned AddedInstructions = 0;
  unsigned RemovedInstructions = 0;
  unsigned ChangedInstructions = 0;
  unsigned MovedInstructions = 0;

  void printSummary(raw_ostream &OS) const {
    OS << formatv("; summary: instructions +{0} -{1} changed {2} moved {3}; "
                  "blocks +{4} -{5} moved {6}\n",
                  AddedInstructions, RemovedInstructions, ChangedInstructions,
                  MovedInstructions, AddedBlocks, RemovedBlocks, MovedBlocks);
  }
};

template <typename RecordT>
DenseMap<uint64_t, const RecordT *>
indexByID(const SmallVectorImpl<RecordT> &Records) {
  DenseMap<uint64_t, const RecordT *> Index;
  for (const RecordT &Record : Records)
    Index[Record.ID] = &Record;
  return Index;
}

const Module *collectFunctions(IRUnitRef IR,
                               SmallVectorImpl<const Function *> &Functions) {
  const Module *M = dyn_cast<Module>(IR);
  if (M) {
    for (const Function &F : *M)
      Functions.push_back(&F);
  } else if (const auto *F = dyn_cast<Function>(IR)) {
    Functions.push_back(F);
  } else if (const auto *C = dyn_cast<LazyCallGraph::SCC>(IR)) {
    for (const LazyCallGraph::Node &N : *C)
      Functions.push_back(&N.getFunction());
  } else if (const auto *L = dyn_cast<Loop>(IR)) {
    Functions.push_back(L->getHeader()->getParent());
  }

  llvm::erase_if(Functions, [](const Function *F) {
    return F->isDeclaration() || !isFunctionSelected(*F);
  });
  if (!M && !Functions.empty())
    M = Functions.front()->getParent();

  if (forcePrintModuleIR() && M) {
    Functions.clear();
    for (const Function &F : *M)
      if (!F.isDeclaration())
        Functions.push_back(&F);
  }
  return M;
}

void printInstructionChange(raw_ostream &OS, char Marker,
                            const InstructionChangeSnapshot &Snapshot,
                            const InstructionChangeRecord &Instruction) {
  OS << Marker << " inst#" << Instruction.ID << " @"
     << Snapshot.Functions[Instruction.FunctionIndex].Name << " block#"
     << Instruction.BlockID << ':' << Instruction.Index << ' '
     << Snapshot.getText(Instruction) << '\n';
}

uint64_t getParentID(const InstructionChangeSnapshot &Snapshot,
                     const BasicBlockChangeRecord &Block) {
  return Snapshot.Functions[Block.FunctionIndex].ID;
}

uint64_t getParentID(const InstructionChangeSnapshot &,
                     const InstructionChangeRecord &Instruction) {
  return Instruction.BlockID;
}

// Rank only records that stay in the same parent so inserting or removing a
// neighbor does not make every following record look moved.
template <typename RecordT>
DenseMap<uint64_t, unsigned>
getRelativeRanks(const InstructionChangeSnapshot &Snapshot,
                 const InstructionChangeSnapshot &Other,
                 const SmallVectorImpl<RecordT> &Records,
                 const DenseMap<uint64_t, const RecordT *> &OtherRecords) {
  DenseMap<uint64_t, unsigned> Ranks;
  DenseMap<uint64_t, unsigned> NextRank;
  for (const RecordT &Record : Records) {
    const RecordT *OtherRecord = OtherRecords.lookup(Record.ID);
    uint64_t ParentID = getParentID(Snapshot, Record);
    if (OtherRecord && ParentID == getParentID(Other, *OtherRecord))
      Ranks[Record.ID] = NextRank[ParentID]++;
  }
  return Ranks;
}

void printBlockChanges(raw_ostream &OS, const InstructionChangeSnapshot &Before,
                       const InstructionChangeSnapshot &After,
                       ChangeCounts &Counts) {
  const auto BeforeBlocks = indexByID(Before.Blocks);
  const auto AfterBlocks = indexByID(After.Blocks);
  const auto BeforeRanks =
      getRelativeRanks(Before, After, Before.Blocks, AfterBlocks);
  const auto AfterRanks =
      getRelativeRanks(After, Before, After.Blocks, BeforeBlocks);
  for (const BasicBlockChangeRecord &Block : Before.Blocks) {
    if (!AfterBlocks.contains(Block.ID)) {
      ++Counts.RemovedBlocks;
      OS << "- block#" << Block.ID << " @"
         << Before.Functions[Block.FunctionIndex].Name << ':' << Block.Index
         << '\n';
    }
  }

  for (const BasicBlockChangeRecord &Block : After.Blocks) {
    const BasicBlockChangeRecord *BeforeBlock = BeforeBlocks.lookup(Block.ID);
    if (!BeforeBlock) {
      ++Counts.AddedBlocks;
      OS << "+ block#" << Block.ID << " @"
         << After.Functions[Block.FunctionIndex].Name << ':' << Block.Index
         << '\n';
      continue;
    }

    if (Before.Functions[BeforeBlock->FunctionIndex].ID !=
            After.Functions[Block.FunctionIndex].ID ||
        BeforeRanks.lookup(Block.ID) != AfterRanks.lookup(Block.ID)) {
      ++Counts.MovedBlocks;
      OS << "> block#" << Block.ID << " @"
         << Before.Functions[BeforeBlock->FunctionIndex].Name << ':'
         << BeforeBlock->Index << " -> @"
         << After.Functions[Block.FunctionIndex].Name << ':' << Block.Index
         << '\n';
    }
  }
}

void printInstructionChanges(raw_ostream &OS,
                             const InstructionChangeSnapshot &Before,
                             const InstructionChangeSnapshot &After,
                             ChangeCounts &Counts) {
  const auto BeforeInstructions = indexByID(Before.Instructions);
  const auto AfterInstructions = indexByID(After.Instructions);
  const auto BeforeRanks =
      getRelativeRanks(Before, After, Before.Instructions, AfterInstructions);
  const auto AfterRanks =
      getRelativeRanks(After, Before, After.Instructions, BeforeInstructions);
  for (const InstructionChangeRecord &Instruction : Before.Instructions) {
    if (!AfterInstructions.contains(Instruction.ID)) {
      ++Counts.RemovedInstructions;
      printInstructionChange(OS, '-', Before, Instruction);
    }
  }

  for (const InstructionChangeRecord &Instruction : After.Instructions) {
    const InstructionChangeRecord *BeforeInstruction =
        BeforeInstructions.lookup(Instruction.ID);
    if (!BeforeInstruction) {
      ++Counts.AddedInstructions;
      printInstructionChange(OS, '+', After, Instruction);
      continue;
    }

    if (Before.getText(*BeforeInstruction) != After.getText(Instruction)) {
      ++Counts.ChangedInstructions;
      printInstructionChange(OS, '-', Before, *BeforeInstruction);
      printInstructionChange(OS, '+', After, Instruction);
      continue;
    }

    if (Before.Functions[BeforeInstruction->FunctionIndex].ID !=
            After.Functions[Instruction.FunctionIndex].ID ||
        BeforeInstruction->BlockID != Instruction.BlockID ||
        BeforeRanks.lookup(Instruction.ID) !=
            AfterRanks.lookup(Instruction.ID)) {
      ++Counts.MovedInstructions;
      OS << "> inst#" << Instruction.ID << " @"
         << Before.Functions[BeforeInstruction->FunctionIndex].Name << " block#"
         << BeforeInstruction->BlockID << ':' << BeforeInstruction->Index
         << " -> @" << After.Functions[Instruction.FunctionIndex].Name
         << " block#" << Instruction.BlockID << ':' << Instruction.Index << ' '
         << After.getText(Instruction) << '\n';
    }
  }
}

ChangeCounts printChanges(raw_ostream &OS,
                          const InstructionChangeSnapshot &Before,
                          const InstructionChangeSnapshot &After) {
  ChangeCounts Counts;
  printBlockChanges(OS, Before, After, Counts);
  printInstructionChanges(OS, Before, After, Counts);
  return Counts;
}

} // namespace

IRChangedPrinter::InstructionChangeReporter::InstructionChangeReporter(
    bool Verbose)
    : ChangeReporter<InstructionChangeSnapshot>(Verbose,
                                                /*ReuseAfterAsBefore=*/true),
      Out(dbgs()) {}

void IRChangedPrinter::InstructionChangeReporter::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  registerRequiredCallbacks(PIC);
}

uint64_t
IRChangedPrinter::InstructionChangeReporter::getValueID(const Value &V) {
  auto [It, Inserted] = ValueIDs.insert({&V, NextValueID});
  if (Inserted)
    ++NextValueID;
  return It->second;
}

uint64_t
IRChangedPrinter::InstructionChangeReporter::getTypeID(const Type &Ty) {
  auto [It, Inserted] = TypeIDs.insert({&Ty, NextTypeID});
  if (Inserted)
    ++NextTypeID;
  return It->second;
}

void IRChangedPrinter::InstructionChangeReporter::generateIRRepresentation(
    IRUnitRef IR, StringRef PassID, InstructionChangeSnapshot &Output) {
  if (isa<MachineFunction>(IR))
    report_fatal_error("instruction-level change printing is only supported "
                       "for LLVM IR under the new pass manager");

  SmallVector<const Function *> Functions;
  const Module *M = collectFunctions(IR, Functions);
  if (!M)
    return;

  ModuleSlotTracker MST(M);
  raw_string_ostream OS(Output.Text);
  for (const Function *F : Functions) {
    unsigned FunctionIndex = Output.Functions.size();
    uint64_t FunctionID = getValueID(*F);
    Output.Functions.push_back(
        {FunctionID, F->hasName()
                         ? F->getName().str()
                         : (Twine("<") + Twine(FunctionID) + ">").str()});

    FunctionInstructionPrinter Printer(
        OS, MST, *F, /*IsForDebug=*/false,
        [this](raw_ostream &OS, const Value &V) {
          OS << (isa<GlobalValue>(V) ? "@<" : "%<") << getValueID(V) << '>';
        },
        [this](raw_ostream &OS, const Type &Ty) {
          OS << "%type<" << getTypeID(Ty) << '>';
        },
        /*PrintCallAttributesInline=*/true);

    unsigned BlockIndex = 0;
    for (const BasicBlock &BB : *F) {
      uint64_t BlockID = getValueID(BB);
      Output.Blocks.push_back({BlockID, FunctionIndex, BlockIndex++});

      unsigned InstructionIndex = 0;
      for (const Instruction &I : BB) {
        uint64_t InstructionID = getValueID(I);
        size_t TextOffset = Output.Text.size();
        Printer.printInstruction(I);
        size_t TextLength = Output.Text.size() - TextOffset;
        Output.Instructions.push_back({InstructionID, BlockID, FunctionIndex,
                                       InstructionIndex++, TextOffset,
                                       TextLength});
      }
    }
  }
}

void IRChangedPrinter::InstructionChangeReporter::handleInitialIR(
    IRUnitRef IR) {
  InstructionChangeSnapshot Snapshot;
  generateIRRepresentation(IR, "Initial IR", Snapshot);

  Out << "*** IR Instruction Snapshot At Start ***\n";
  const InstructionChangeSnapshot Empty;
  printChanges(Out, Empty, Snapshot).printSummary(Out);
}

void IRChangedPrinter::InstructionChangeReporter::omitAfter(StringRef PassID,
                                                            std::string &Name) {
  Out << formatv("*** IR Instruction Changes After {0} on {1} omitted because "
                 "no change ***\n",
                 PassID, Name);
}

void IRChangedPrinter::InstructionChangeReporter::handleInvalidated(
    StringRef PassID) {
  Out << formatv("*** IR Instruction Pass {0} invalidated ***\n", PassID);
}

void IRChangedPrinter::InstructionChangeReporter::handleFiltered(
    StringRef PassID, std::string &Name) {
  Out << formatv(
      "*** IR Instruction Changes After {0} on {1} filtered out ***\n", PassID,
      Name);
}

void IRChangedPrinter::InstructionChangeReporter::handleIgnored(
    StringRef PassID, std::string &Name) {
  Out << formatv("*** IR Instruction Pass {0} on {1} ignored ***\n", PassID,
                 Name);
}

void IRChangedPrinter::InstructionChangeReporter::handleAfter(
    StringRef PassID, std::string &Name,
    const InstructionChangeSnapshot &Before,
    const InstructionChangeSnapshot &After, IRUnitRef) {
  std::string Changes;
  raw_string_ostream ChangeOS(Changes);
  ChangeCounts Counts = printChanges(ChangeOS, Before, After);
  if (Changes.empty()) {
    if (VerboseMode)
      omitAfter(PassID, Name);
    return;
  }

  Out << "*** IR Instruction Changes After " << PassID << " on " << Name
      << " ***\n"
      << Changes;
  Counts.printSummary(Out);
}
