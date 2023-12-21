//===- SampleProfileProbe.cpp - Pseudo probe Instrumentation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SampleProfileProber transformation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/SampleProfileProbe.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/EHUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/CRC.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <unordered_set>
#include <vector>

using namespace llvm;
#define DEBUG_TYPE "pseudo-probe"

STATISTIC(ArtificialDbgLine,
          "Number of probes that have an artificial debug line");

static cl::opt<bool>
    VerifyPseudoProbe("verify-pseudo-probe", cl::init(false), cl::Hidden,
                      cl::desc("Do pseudo probe verification"));

static cl::list<std::string> VerifyPseudoProbeFuncList(
    "verify-pseudo-probe-funcs", cl::Hidden,
    cl::desc("The option to specify the name of the functions to verify."));

static cl::opt<bool>
    UpdatePseudoProbe("update-pseudo-probe", cl::init(true), cl::Hidden,
                      cl::desc("Update pseudo probe distribution factor"));

static uint64_t getCallStackHash(const DILocation *DIL) {
  uint64_t Hash = 0;
  const DILocation *InlinedAt = DIL ? DIL->getInlinedAt() : nullptr;
  while (InlinedAt) {
    Hash ^= MD5Hash(std::to_string(InlinedAt->getLine()));
    Hash ^= MD5Hash(std::to_string(InlinedAt->getColumn()));
    auto Name = InlinedAt->getSubprogramLinkageName();
    Hash ^= MD5Hash(Name);
    InlinedAt = InlinedAt->getInlinedAt();
  }
  return Hash;
}

static uint64_t computeCallStackHash(const Instruction &Inst) {
  return getCallStackHash(Inst.getDebugLoc());
}

bool PseudoProbeVerifier::shouldVerifyFunction(const Function *F) {
  // Skip function declaration.
  if (F->isDeclaration())
    return false;
  // Skip function that will not be emitted into object file. The prevailing
  // defintion will be verified instead.
  if (F->hasAvailableExternallyLinkage())
    return false;
  // Do a name matching.
  static std::unordered_set<std::string> VerifyFuncNames(
      VerifyPseudoProbeFuncList.begin(), VerifyPseudoProbeFuncList.end());
  return VerifyFuncNames.empty() || VerifyFuncNames.count(F->getName().str());
}

void PseudoProbeVerifier::registerCallbacks(PassInstrumentationCallbacks &PIC) {
  if (VerifyPseudoProbe) {
    PIC.registerAfterPassCallback(
        [this](StringRef P, Any IR, const PreservedAnalyses &) {
          this->runAfterPass(P, IR);
        });
  }
}

// Callback to run after each transformation for the new pass manager.
void PseudoProbeVerifier::runAfterPass(StringRef PassID, Any IR) {
  std::string Banner =
      "\n*** Pseudo Probe Verification After " + PassID.str() + " ***\n";
  dbgs() << Banner;
  if (const auto **M = llvm::any_cast<const Module *>(&IR))
    runAfterPass(*M);
  else if (const auto **F = llvm::any_cast<const Function *>(&IR))
    runAfterPass(*F);
  else if (const auto **C = llvm::any_cast<const LazyCallGraph::SCC *>(&IR))
    runAfterPass(*C);
  else if (const auto **L = llvm::any_cast<const Loop *>(&IR))
    runAfterPass(*L);
  else
    llvm_unreachable("Unknown IR unit");
}

void PseudoProbeVerifier::runAfterPass(const Module *M) {
  for (const Function &F : *M)
    runAfterPass(&F);
}

void PseudoProbeVerifier::runAfterPass(const LazyCallGraph::SCC *C) {
  for (const LazyCallGraph::Node &N : *C)
    runAfterPass(&N.getFunction());
}

void PseudoProbeVerifier::runAfterPass(const Function *F) {
  if (!shouldVerifyFunction(F))
    return;
  ProbeFactorMap ProbeFactors;
  for (const auto &BB : *F)
    collectProbeFactors(&BB, ProbeFactors);
  verifyProbeFactors(F, ProbeFactors);
}

void PseudoProbeVerifier::runAfterPass(const Loop *L) {
  const Function *F = L->getHeader()->getParent();
  runAfterPass(F);
}

void PseudoProbeVerifier::collectProbeFactors(const BasicBlock *Block,
                                              ProbeFactorMap &ProbeFactors) {
  for (const auto &I : *Block) {
    if (std::optional<PseudoProbe> Probe = extractProbe(I)) {
      uint64_t Hash = computeCallStackHash(I);
      ProbeFactors[{Probe->Id, Hash}] += Probe->Factor;
    }
  }
}

void PseudoProbeVerifier::verifyProbeFactors(
    const Function *F, const ProbeFactorMap &ProbeFactors) {
  bool BannerPrinted = false;
  auto &PrevProbeFactors = FunctionProbeFactors[F->getName()];
  for (const auto &I : ProbeFactors) {
    float CurProbeFactor = I.second;
    if (PrevProbeFactors.count(I.first)) {
      float PrevProbeFactor = PrevProbeFactors[I.first];
      if (std::abs(CurProbeFactor - PrevProbeFactor) >
          DistributionFactorVariance) {
        if (!BannerPrinted) {
          dbgs() << "Function " << F->getName() << ":\n";
          BannerPrinted = true;
        }
        dbgs() << "Probe " << I.first.first << "\tprevious factor "
               << format("%0.2f", PrevProbeFactor) << "\tcurrent factor "
               << format("%0.2f", CurProbeFactor) << "\n";
      }
    }

    // Update
    PrevProbeFactors[I.first] = I.second;
  }
}

SampleProfileProber::SampleProfileProber(Function &Func,
                                         const std::string &CurModuleUniqueId)
    : F(&Func), CurModuleUniqueId(CurModuleUniqueId) {
  BlockProbeIds.clear();
  CallProbeIds.clear();
  LastProbeId = (uint32_t)PseudoProbeReservedId::Last;
  computeProbeIdForBlocks();
  computeProbeIdForCallsites();
  computeCFGHash();
}

// Compute Hash value for the CFG: the lower 32 bits are CRC32 of the index
// value of each BB in the CFG. The higher 32 bits record the number of edges
// preceded by the number of indirect calls.
// This is derived from FuncPGOInstrumentation<Edge, BBInfo>::computeCFGHash().
void SampleProfileProber::computeCFGHash() {
  std::vector<uint8_t> Indexes;
  JamCRC JC;
  for (auto &BB : *F) {
    auto *TI = BB.getTerminator();
    for (unsigned I = 0, E = TI->getNumSuccessors(); I != E; ++I) {
      auto *Succ = TI->getSuccessor(I);
      auto Index = getBlockId(Succ);
      for (int J = 0; J < 4; J++)
        Indexes.push_back((uint8_t)(Index >> (J * 8)));
    }
  }

  JC.update(Indexes);

  FunctionHash = (uint64_t)CallProbeIds.size() << 48 |
                 (uint64_t)Indexes.size() << 32 | JC.getCRC();
  // Reserve bit 60-63 for other information purpose.
  FunctionHash &= 0x0FFFFFFFFFFFFFFF;
  assert(FunctionHash && "Function checksum should not be zero");
  LLVM_DEBUG(dbgs() << "\nFunction Hash Computation for " << F->getName()
                    << ":\n"
                    << " CRC = " << JC.getCRC() << ", Edges = "
                    << Indexes.size() << ", ICSites = " << CallProbeIds.size()
                    << ", Hash = " << FunctionHash << "\n");
}

void SampleProfileProber::computeProbeIdForBlocks() {
  DenseSet<BasicBlock *> KnownColdBlocks;
  computeEHOnlyBlocks(*F, KnownColdBlocks);
  // Insert pseudo probe to non-cold blocks only. This will reduce IR size as
  // well as the binary size while retaining the profile quality.
  for (auto &BB : *F) {
    ++LastProbeId;
    if (!KnownColdBlocks.contains(&BB))
      BlockProbeIds[&BB] = LastProbeId;
  }
}

void SampleProfileProber::computeProbeIdForCallsites() {
  LLVMContext &Ctx = F->getContext();
  Module *M = F->getParent();

  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (!isa<CallBase>(I))
        continue;
      if (isa<IntrinsicInst>(&I))
        continue;

      // The current implementation uses the lower 16 bits of the discriminator
      // so anything larger than 0xFFFF will be ignored.
      if (LastProbeId >= 0xFFFF) {
        std::string Msg = "Pseudo instrumentation incomplete for " +
                          std::string(F->getName()) + " because it's too large";
        Ctx.diagnose(
            DiagnosticInfoSampleProfile(M->getName().data(), Msg, DS_Warning));
        return;
      }

      CallProbeIds[&I] = ++LastProbeId;
    }
  }
}

uint32_t SampleProfileProber::getBlockId(const BasicBlock *BB) const {
  auto I = BlockProbeIds.find(const_cast<BasicBlock *>(BB));
  return I == BlockProbeIds.end() ? 0 : I->second;
}

uint32_t SampleProfileProber::getCallsiteId(const Instruction *Call) const {
  auto Iter = CallProbeIds.find(const_cast<Instruction *>(Call));
  return Iter == CallProbeIds.end() ? 0 : Iter->second;
}

void SampleProfileProber::instrumentOneFunc(Function &F, TargetMachine *TM) {
  Module *M = F.getParent();
  MDBuilder MDB(F.getContext());
  // Since the GUID from probe desc and inline stack are computed seperately, we
  // need to make sure their names are consistent, so here also use the name
  // from debug info.
  StringRef FName = F.getName();
  if (auto *SP = F.getSubprogram()) {
    FName = SP->getLinkageName();
    if (FName.empty())
      FName = SP->getName();
  }
  uint64_t Guid = Function::getGUID(FName);

  // Assign an artificial debug line to a probe that doesn't come with a real
  // line. A probe not having a debug line will get an incomplete inline
  // context. This will cause samples collected on the probe to be counted
  // into the base profile instead of a context profile. The line number
  // itself is not important though.
  auto AssignDebugLoc = [&](Instruction *I) {
    assert((isa<PseudoProbeInst>(I) || isa<CallBase>(I)) &&
           "Expecting pseudo probe or call instructions");
    if (!I->getDebugLoc()) {
      if (auto *SP = F.getSubprogram()) {
        auto DIL = DILocation::get(SP->getContext(), 0, 0, SP);
        I->setDebugLoc(DIL);
        ArtificialDbgLine++;
        LLVM_DEBUG({
          dbgs() << "\nIn Function " << F.getName()
                 << " Probe gets an artificial debug line\n";
          I->dump();
        });
      }
    }
  };

  // Probe basic blocks.
  for (auto &I : BlockProbeIds) {
    BasicBlock *BB = I.first;
    uint32_t Index = I.second;
    // Insert a probe before an instruction with a valid debug line number which
    // will be assigned to the probe. The line number will be used later to
    // model the inline context when the probe is inlined into other functions.
    // Debug instructions, phi nodes and lifetime markers do not have an valid
    // line number. Real instructions generated by optimizations may not come
    // with a line number either.
    auto HasValidDbgLine = [](Instruction *J) {
      return !isa<PHINode>(J) && !isa<DbgInfoIntrinsic>(J) &&
             !J->isLifetimeStartOrEnd() && J->getDebugLoc();
    };

    Instruction *J = &*BB->getFirstInsertionPt();
    while (J != BB->getTerminator() && !HasValidDbgLine(J)) {
      J = J->getNextNode();
    }

    IRBuilder<> Builder(J);
    assert(Builder.GetInsertPoint() != BB->end() &&
           "Cannot get the probing point");
    Function *ProbeFn =
        llvm::Intrinsic::getDeclaration(M, Intrinsic::pseudoprobe);
    Value *Args[] = {Builder.getInt64(Guid), Builder.getInt64(Index),
                     Builder.getInt32(0),
                     Builder.getInt64(PseudoProbeFullDistributionFactor)};
    auto *Probe = Builder.CreateCall(ProbeFn, Args);
    AssignDebugLoc(Probe);
    // Reset the dwarf discriminator if the debug location comes with any. The
    // discriminator field may be used by FS-AFDO later in the pipeline.
    if (auto DIL = Probe->getDebugLoc()) {
      if (DIL->getDiscriminator()) {
        DIL = DIL->cloneWithDiscriminator(0);
        Probe->setDebugLoc(DIL);
      }
    }
  }

  // Probe both direct calls and indirect calls. Direct calls are probed so that
  // their probe ID can be used as an call site identifier to represent a
  // calling context.
  for (auto &I : CallProbeIds) {
    auto *Call = I.first;
    uint32_t Index = I.second;
    uint32_t Type = cast<CallBase>(Call)->getCalledFunction()
                        ? (uint32_t)PseudoProbeType::DirectCall
                        : (uint32_t)PseudoProbeType::IndirectCall;
    AssignDebugLoc(Call);
    if (auto DIL = Call->getDebugLoc()) {
      // Levarge the 32-bit discriminator field of debug data to store the ID
      // and type of a callsite probe. This gets rid of the dependency on
      // plumbing a customized metadata through the codegen pipeline.
      uint32_t V = PseudoProbeDwarfDiscriminator::packProbeData(
          Index, Type, 0,
          PseudoProbeDwarfDiscriminator::FullDistributionFactor);
      DIL = DIL->cloneWithDiscriminator(V);
      Call->setDebugLoc(DIL);
    }
  }

  // Create module-level metadata that contains function info necessary to
  // synthesize probe-based sample counts,  which are
  // - FunctionGUID
  // - FunctionHash.
  // - FunctionName
  auto Hash = getFunctionHash();
  auto *MD = MDB.createPseudoProbeDesc(Guid, Hash, FName);
  auto *NMD = M->getNamedMetadata(PseudoProbeDescMetadataName);
  assert(NMD && "llvm.pseudo_probe_desc should be pre-created");
  NMD->addOperand(MD);
}

PreservedAnalyses SampleProfileProbePass::run(Module &M,
                                              ModuleAnalysisManager &AM) {
  auto ModuleId = getUniqueModuleId(&M);
  // Create the pseudo probe desc metadata beforehand.
  // Note that modules with only data but no functions will require this to
  // be set up so that they will be known as probed later.
  M.getOrInsertNamedMetadata(PseudoProbeDescMetadataName);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SampleProfileProber ProbeManager(F, ModuleId);
    ProbeManager.instrumentOneFunc(F, TM);
  }

  return PreservedAnalyses::none();
}

void PseudoProbeUpdatePass::runOnFunction(Function &F,
                                          FunctionAnalysisManager &FAM) {
  BlockFrequencyInfo &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
  auto BBProfileCount = [&BFI](BasicBlock *BB) {
    return BFI.getBlockProfileCount(BB).value_or(0);
  };

  // Collect the sum of execution weight for each probe.
  ProbeFactorMap ProbeFactors;
  for (auto &Block : F) {
    for (auto &I : Block) {
      if (std::optional<PseudoProbe> Probe = extractProbe(I)) {
        uint64_t Hash = computeCallStackHash(I);
        ProbeFactors[{Probe->Id, Hash}] += BBProfileCount(&Block);
      }
    }
  }

  // Fix up over-counted probes.
  for (auto &Block : F) {
    for (auto &I : Block) {
      if (std::optional<PseudoProbe> Probe = extractProbe(I)) {
        uint64_t Hash = computeCallStackHash(I);
        float Sum = ProbeFactors[{Probe->Id, Hash}];
        if (Sum != 0)
          setProbeDistributionFactor(I, BBProfileCount(&Block) / Sum);
      }
    }
  }
}

PreservedAnalyses PseudoProbeUpdatePass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  if (UpdatePseudoProbe) {
    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      FunctionAnalysisManager &FAM =
          AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
      runOnFunction(F, FAM);
    }
  }
  return PreservedAnalyses::none();
}
