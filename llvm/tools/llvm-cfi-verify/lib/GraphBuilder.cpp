//===- GraphBuilder.cpp -----------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GraphBuilder.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

using Instr = llvm::cfi_verify::FileAnalysis::Instr;

namespace llvm {
namespace cfi_verify {

uint64_t SearchLengthForUndef;
uint64_t SearchLengthForConditionalBranch;

static cl::opt<uint64_t, true> SearchLengthForUndefArg(
    "search-length-undef",
    cl::desc("Specify the maximum amount of instructions "
             "to inspect when searching for an undefined "
             "instruction from a conditional branch."),
    cl::location(SearchLengthForUndef), cl::init(2));

static cl::opt<uint64_t, true> SearchLengthForConditionalBranchArg(
    "search-length-cb",
    cl::desc("Specify the maximum amount of instructions "
             "to inspect when searching for a conditional "
             "branch from an indirect control flow."),
    cl::location(SearchLengthForConditionalBranch), cl::init(20));

std::vector<uint64_t> GraphResult::flattenAddress(uint64_t Address) const {
  std::vector<uint64_t> Addresses;

  auto It = IntermediateNodes.find(Address);
  Addresses.push_back(Address);

  while (It != IntermediateNodes.end()) {
    Addresses.push_back(It->second);
    It = IntermediateNodes.find(It->second);
  }
  return Addresses;
}

GraphResult GraphBuilder::buildFlowGraph(const FileAnalysis &Analysis,
                                         uint64_t Address) {
  GraphResult Result;
  Result.BaseAddress = Address;
  DenseSet<uint64_t> OpenedNodes;

  const auto &IndirectInstructions = Analysis.getIndirectInstructions();

  if (IndirectInstructions.find(Address) == IndirectInstructions.end())
    return Result;

  buildFlowGraphImpl(Analysis, OpenedNodes, Result, Address, 0);
  return Result;
}

void GraphBuilder::buildFlowsToUndefined(const FileAnalysis &Analysis,
                                         GraphResult &Result,
                                         ConditionalBranchNode &BranchNode,
                                         const Instr &BranchInstrMeta) {
  assert(SearchLengthForUndef > 0 &&
         "Search length for undefined flow must be greater than zero.");

  // Start setting up the next node in the block.
  uint64_t NextAddress = 0;
  const Instr *NextMetaPtr;

  // Find out the next instruction in the block and add it to the new
  // node.
  if (BranchNode.Target && !BranchNode.Fallthrough) {
    // We know the target of the branch, find the fallthrough.
    NextMetaPtr = Analysis.getNextInstructionSequential(BranchInstrMeta);
    if (!NextMetaPtr) {
      errs() << "Failed to get next instruction from "
             << format_hex(BranchNode.Address, 2) << ".\n";
      return;
    }

    NextAddress = NextMetaPtr->VMAddress;
    BranchNode.Fallthrough =
        NextMetaPtr->VMAddress; // Add the new node to the branch head.
  } else if (BranchNode.Fallthrough && !BranchNode.Target) {
    // We already know the fallthrough, evaluate the target.
    uint64_t Target;
    if (!Analysis.getMCInstrAnalysis()->evaluateBranch(
            BranchInstrMeta.Instruction, BranchInstrMeta.VMAddress,
            BranchInstrMeta.InstructionSize, Target)) {
      errs() << "Failed to get branch target for conditional branch at address "
             << format_hex(BranchInstrMeta.VMAddress, 2) << ".\n";
      return;
    }

    // Resolve the meta pointer for the target of this branch.
    NextMetaPtr = Analysis.getInstruction(Target);
    if (!NextMetaPtr) {
      errs() << "Failed to find instruction at address "
             << format_hex(Target, 2) << ".\n";
      return;
    }

    NextAddress = Target;
    BranchNode.Target =
        NextMetaPtr->VMAddress; // Add the new node to the branch head.
  } else {
    errs() << "ControlBranchNode supplied to buildFlowsToUndefined should "
              "provide Target xor Fallthrough.\n";
    return;
  }

  uint64_t CurrentAddress = NextAddress;
  const Instr *CurrentMetaPtr = NextMetaPtr;

  // Now the branch head has been set properly, complete the rest of the block.
  for (uint64_t i = 1; i < SearchLengthForUndef; ++i) {
    // Check to see whether the block should die.
    if (Analysis.isCFITrap(*CurrentMetaPtr)) {
      BranchNode.CFIProtection = true;
      return;
    }

    // Find the metadata of the next instruction.
    NextMetaPtr = Analysis.getDefiniteNextInstruction(*CurrentMetaPtr);
    if (!NextMetaPtr)
      return;

    // Setup the next node.
    NextAddress = NextMetaPtr->VMAddress;

    // Add this as an intermediate.
    Result.IntermediateNodes[CurrentAddress] = NextAddress;

    // Move the 'current' pointers to the new tail of the block.
    CurrentMetaPtr = NextMetaPtr;
    CurrentAddress = NextAddress;
  }

  // Final check of the last thing we added to the block.
  if (Analysis.isCFITrap(*CurrentMetaPtr))
    BranchNode.CFIProtection = true;
}

void GraphBuilder::buildFlowGraphImpl(const FileAnalysis &Analysis,
                                      DenseSet<uint64_t> &OpenedNodes,
                                      GraphResult &Result, uint64_t Address,
                                      uint64_t Depth) {
  // If we've exceeded the flow length, terminate.
  if (Depth >= SearchLengthForConditionalBranch) {
    Result.OrphanedNodes.push_back(Address);
    return;
  }

  // Ensure this flow is acyclic.
  if (OpenedNodes.count(Address))
    Result.OrphanedNodes.push_back(Address);

  // If this flow is already explored, stop here.
  if (Result.IntermediateNodes.count(Address))
    return;

  // Get the metadata for the node instruction.
  const auto &InstrMetaPtr = Analysis.getInstruction(Address);
  if (!InstrMetaPtr) {
    errs() << "Failed to build flow graph for instruction at address "
           << format_hex(Address, 2) << ".\n";
    Result.OrphanedNodes.push_back(Address);
    return;
  }
  const auto &ChildMeta = *InstrMetaPtr;

  OpenedNodes.insert(Address);
  std::set<const Instr *> CFCrossRefs =
      Analysis.getDirectControlFlowXRefs(ChildMeta);

  bool HasValidCrossRef = false;

  for (const auto *ParentMetaPtr : CFCrossRefs) {
    assert(ParentMetaPtr && "CFCrossRefs returned nullptr.");
    const auto &ParentMeta = *ParentMetaPtr;
    const auto &ParentDesc =
        Analysis.getMCInstrInfo()->get(ParentMeta.Instruction.getOpcode());

    if (!ParentDesc.mayAffectControlFlow(ParentMeta.Instruction,
                                         *Analysis.getRegisterInfo())) {
      // If this cross reference doesn't affect CF, continue the graph.
      buildFlowGraphImpl(Analysis, OpenedNodes, Result, ParentMeta.VMAddress,
                         Depth + 1);
      Result.IntermediateNodes[ParentMeta.VMAddress] = Address;
      HasValidCrossRef = true;
      continue;
    }

    // Call instructions are not valid in the upwards traversal.
    if (ParentDesc.isCall()) {
      Result.IntermediateNodes[ParentMeta.VMAddress] = Address;
      Result.OrphanedNodes.push_back(ParentMeta.VMAddress);
      continue;
    }

    // Evaluate the branch target to ascertain whether this XRef is the result
    // of a fallthrough or the target of a branch.
    uint64_t BranchTarget;
    if (!Analysis.getMCInstrAnalysis()->evaluateBranch(
            ParentMeta.Instruction, ParentMeta.VMAddress,
            ParentMeta.InstructionSize, BranchTarget)) {
      errs() << "Failed to evaluate branch target for instruction at address "
             << format_hex(ParentMeta.VMAddress, 2) << ".\n";
      Result.IntermediateNodes[ParentMeta.VMAddress] = Address;
      Result.OrphanedNodes.push_back(ParentMeta.VMAddress);
      continue;
    }

    // Allow unconditional branches to be part of the upwards traversal.
    if (ParentDesc.isUnconditionalBranch()) {
      // Ensures that the unconditional branch is actually an XRef to the child.
      if (BranchTarget != Address) {
        errs() << "Control flow to " << format_hex(Address, 2)
               << ", but target resolution of "
               << format_hex(ParentMeta.VMAddress, 2)
               << " is not this address?\n";
        Result.IntermediateNodes[ParentMeta.VMAddress] = Address;
        Result.OrphanedNodes.push_back(ParentMeta.VMAddress);
        continue;
      }

      buildFlowGraphImpl(Analysis, OpenedNodes, Result, ParentMeta.VMAddress,
                         Depth + 1);
      Result.IntermediateNodes[ParentMeta.VMAddress] = Address;
      HasValidCrossRef = true;
      continue;
    }

    // Ensure that any unknown CFs are caught.
    if (!ParentDesc.isConditionalBranch()) {
      errs() << "Unknown control flow encountered when building graph at "
             << format_hex(Address, 2) << "\n.";
      Result.IntermediateNodes[ParentMeta.VMAddress] = Address;
      Result.OrphanedNodes.push_back(ParentMeta.VMAddress);
      continue;
    }

    // Only direct conditional branches should be present at this point. Setup
    // a conditional branch node and build flows to the ud2.
    ConditionalBranchNode BranchNode;
    BranchNode.Address = ParentMeta.VMAddress;
    BranchNode.Target = 0;
    BranchNode.Fallthrough = 0;
    BranchNode.CFIProtection = false;

    if (BranchTarget == Address)
      BranchNode.Target = Address;
    else
      BranchNode.Fallthrough = Address;

    HasValidCrossRef = true;
    buildFlowsToUndefined(Analysis, Result, BranchNode, ParentMeta);
    Result.ConditionalBranchNodes.push_back(BranchNode);
  }

  if (!HasValidCrossRef)
    Result.OrphanedNodes.push_back(Address);

  OpenedNodes.erase(Address);
}

} // namespace cfi_verify
} // namespace llvm
