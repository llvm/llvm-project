//===-- PGOInstrumentation.cpp - MST-based PGO Instrumentation ------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements PGO instrumentation using a minimum spanning tree based
// on the following paper:
//   [1] Donald E. Knuth, Francis R. Stevenson. Optimal measurement of points
//   for program frequency counts. BIT Numerical Mathematics 1973, Volume 13,
//   Issue 3, pp 313-322
// The idea of the algorithm based on the fact that for each node (except for
// the entry and exit), the sum of incoming edge counts equals the sum of
// outgoing edge counts. The count of edge on spanning tree can be derived from
// those edges not on the spanning tree. Knuth proves this method instruments
// the minimum number of edges.
//
// The minimal spanning tree here is actually a maximum weight tree -- on-tree
// edges have higher frequencies (more likely to execute). The idea is to
// instrument those less frequently executed edges to reduce the runtime
// overhead of instrumented binaries.
//
// This file contains two passes:
// (1) Pass PGOInstrumentationGen which instruments the IR to generate edge
// count profile, and generates the instrumentation for indirect call
// profiling.
// (2) Pass PGOInstrumentationUse which reads the edge count profile and
// annotates the branch weights. It also reads the indirect call value
// profiling records and annotate the indirect call instructions.
//
// To get the precise counter information, These two passes need to invoke at
// the same compilation point (so they see the same IR). For pass
// PGOInstrumentationGen, the real work is done in instrumentOneFunc(). For
// pass PGOInstrumentationUse, the real work in done in class PGOUseFunc and
// the profile is opened in module level and passed to each PGOUseFunc instance.
// The shared code for PGOInstrumentationGen and PGOInstrumentationUse is put
// in class FuncPGOInstrumentation.
//
// Class PGOEdge represents a CFG edge and some auxiliary information. Class
// BBInfo contains auxiliary information for each BB. These two classes are used
// in pass PGOInstrumentationGen. Class PGOUseEdge and UseBBInfo are the derived
// class of PGOEdge and BBInfo, respectively. They contains extra data structure
// used in populating profile counters.
// The MST implementation is in Class CFGMST (CFGMST.h).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/PGOInstrumentation.h"
#include "CFGMST.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/IndirectCallSiteVisitor.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/JamCRC.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-instrumentation"

STATISTIC(NumOfPGOInstrument, "Number of edges instrumented.");
STATISTIC(NumOfPGOSelectInsts, "Number of select instruction instrumented.");
STATISTIC(NumOfPGOEdge, "Number of edges.");
STATISTIC(NumOfPGOBB, "Number of basic-blocks.");
STATISTIC(NumOfPGOSplit, "Number of critical edge splits.");
STATISTIC(NumOfPGOFunc, "Number of functions having valid profile counts.");
STATISTIC(NumOfPGOMismatch, "Number of functions having mismatch profile.");
STATISTIC(NumOfPGOMissing, "Number of functions without profile.");
STATISTIC(NumOfPGOICall, "Number of indirect call value instrumentations.");

// Command line option to specify the file to read profile from. This is
// mainly used for testing.
static cl::opt<std::string>
    PGOTestProfileFile("pgo-test-profile-file", cl::init(""), cl::Hidden,
                       cl::value_desc("filename"),
                       cl::desc("Specify the path of profile data file. This is"
                                "mainly for test purpose."));

// Command line option to disable value profiling. The default is false:
// i.e. value profiling is enabled by default. This is for debug purpose.
static cl::opt<bool> DisableValueProfiling("disable-vp", cl::init(false),
                                           cl::Hidden,
                                           cl::desc("Disable Value Profiling"));

// Command line option to set the maximum number of VP annotations to write to
// the metadata for a single indirect call callsite.
static cl::opt<unsigned> MaxNumAnnotations(
    "icp-max-annotations", cl::init(3), cl::Hidden, cl::ZeroOrMore,
    cl::desc("Max number of annotations for a single indirect "
             "call callsite"));

// Command line option to control appending FunctionHash to the name of a COMDAT
// function. This is to avoid the hash mismatch caused by the preinliner.
static cl::opt<bool> DoComdatRenaming(
    "do-comdat-renaming", cl::init(false), cl::Hidden,
    cl::desc("Append function hash to the name of COMDAT function to avoid "
             "function hash mismatch due to the preinliner"));

// Command line option to enable/disable the warning about missing profile
// information.
static cl::opt<bool>
    PGOWarnMissing("pgo-warn-missing-function", cl::init(false), cl::Hidden,
                   cl::desc("Use this option to turn on/off "
                            "warnings about missing profile data for "
                            "functions."));

// Command line option to enable/disable the warning about a hash mismatch in
// the profile data.
static cl::opt<bool>
    NoPGOWarnMismatch("no-pgo-warn-mismatch", cl::init(false), cl::Hidden,
                      cl::desc("Use this option to turn off/on "
                               "warnings about profile cfg mismatch."));

// Command line option to enable/disable the warning about a hash mismatch in
// the profile data for Comdat functions, which often turns out to be false
// positive due to the pre-instrumentation inline.
static cl::opt<bool>
    NoPGOWarnMismatchComdat("no-pgo-warn-mismatch-comdat", cl::init(true),
                            cl::Hidden,
                            cl::desc("The option is used to turn on/off "
                                     "warnings about hash mismatch for comdat "
                                     "functions."));

// Command line option to enable/disable select instruction instrumentation.
static cl::opt<bool>
    PGOInstrSelect("pgo-instr-select", cl::init(true), cl::Hidden,
                   cl::desc("Use this option to turn on/off SELECT "
                            "instruction instrumentation. "));

// Command line option to turn on CFG dot dump of raw profile counts
static cl::opt<bool>
    PGOViewRawCounts("pgo-view-raw-counts", cl::init(false), cl::Hidden,
                     cl::desc("A boolean option to show CFG dag "
                              "with raw profile counts from "
                              "profile data. See also option "
                              "-pgo-view-counts. To limit graph "
                              "display to only one function, use "
                              "filtering option -view-bfi-func-name."));

// Command line option to turn on CFG dot dump after profile annotation.
// Defined in Analysis/BlockFrequencyInfo.cpp:  -pgo-view-counts
extern cl::opt<bool> PGOViewCounts;

// Command line option to specify the name of the function for CFG dump
// Defined in Analysis/BlockFrequencyInfo.cpp:  -view-bfi-func-name=
extern cl::opt<std::string> ViewBlockFreqFuncName;

namespace {

/// The select instruction visitor plays three roles specified
/// by the mode. In \c VM_counting mode, it simply counts the number of
/// select instructions. In \c VM_instrument mode, it inserts code to count
/// the number times TrueValue of select is taken. In \c VM_annotate mode,
/// it reads the profile data and annotate the select instruction with metadata.
enum VisitMode { VM_counting, VM_instrument, VM_annotate };
class PGOUseFunc;

/// Instruction Visitor class to visit select instructions.
struct SelectInstVisitor : public InstVisitor<SelectInstVisitor> {
  Function &F;
  unsigned NSIs = 0;             // Number of select instructions instrumented.
  VisitMode Mode = VM_counting;  // Visiting mode.
  unsigned *CurCtrIdx = nullptr; // Pointer to current counter index.
  unsigned TotalNumCtrs = 0;     // Total number of counters
  GlobalVariable *FuncNameVar = nullptr;
  uint64_t FuncHash = 0;
  PGOUseFunc *UseFunc = nullptr;

  SelectInstVisitor(Function &Func) : F(Func) {}

  void countSelects(Function &Func) {
    Mode = VM_counting;
    visit(Func);
  }
  // Visit the IR stream and instrument all select instructions. \p
  // Ind is a pointer to the counter index variable; \p TotalNC
  // is the total number of counters; \p FNV is the pointer to the
  // PGO function name var; \p FHash is the function hash.
  void instrumentSelects(Function &Func, unsigned *Ind, unsigned TotalNC,
                         GlobalVariable *FNV, uint64_t FHash) {
    Mode = VM_instrument;
    CurCtrIdx = Ind;
    TotalNumCtrs = TotalNC;
    FuncHash = FHash;
    FuncNameVar = FNV;
    visit(Func);
  }

  // Visit the IR stream and annotate all select instructions.
  void annotateSelects(Function &Func, PGOUseFunc *UF, unsigned *Ind) {
    Mode = VM_annotate;
    UseFunc = UF;
    CurCtrIdx = Ind;
    visit(Func);
  }

  void instrumentOneSelectInst(SelectInst &SI);
  void annotateOneSelectInst(SelectInst &SI);
  // Visit \p SI instruction and perform tasks according to visit mode.
  void visitSelectInst(SelectInst &SI);
  unsigned getNumOfSelectInsts() const { return NSIs; }
};

class PGOInstrumentationGenLegacyPass : public ModulePass {
public:
  static char ID;

  PGOInstrumentationGenLegacyPass() : ModulePass(ID) {
    initializePGOInstrumentationGenLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "PGOInstrumentationGenPass"; }

private:
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
  }
};

class PGOInstrumentationUseLegacyPass : public ModulePass {
public:
  static char ID;

  // Provide the profile filename as the parameter.
  PGOInstrumentationUseLegacyPass(std::string Filename = "")
      : ModulePass(ID), ProfileFileName(std::move(Filename)) {
    if (!PGOTestProfileFile.empty())
      ProfileFileName = PGOTestProfileFile;
    initializePGOInstrumentationUseLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "PGOInstrumentationUsePass"; }

private:
  std::string ProfileFileName;

  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
  }
};

} // end anonymous namespace

char PGOInstrumentationGenLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PGOInstrumentationGenLegacyPass, "pgo-instr-gen",
                      "PGO instrumentation.", false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(PGOInstrumentationGenLegacyPass, "pgo-instr-gen",
                    "PGO instrumentation.", false, false)

ModulePass *llvm::createPGOInstrumentationGenLegacyPass() {
  return new PGOInstrumentationGenLegacyPass();
}

char PGOInstrumentationUseLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(PGOInstrumentationUseLegacyPass, "pgo-instr-use",
                      "Read PGO instrumentation profile.", false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(PGOInstrumentationUseLegacyPass, "pgo-instr-use",
                    "Read PGO instrumentation profile.", false, false)

ModulePass *llvm::createPGOInstrumentationUseLegacyPass(StringRef Filename) {
  return new PGOInstrumentationUseLegacyPass(Filename.str());
}

namespace {
/// \brief An MST based instrumentation for PGO
///
/// Implements a Minimum Spanning Tree (MST) based instrumentation for PGO
/// in the function level.
struct PGOEdge {
  // This class implements the CFG edges. Note the CFG can be a multi-graph.
  // So there might be multiple edges with same SrcBB and DestBB.
  const BasicBlock *SrcBB;
  const BasicBlock *DestBB;
  uint64_t Weight;
  bool InMST;
  bool Removed;
  bool IsCritical;
  PGOEdge(const BasicBlock *Src, const BasicBlock *Dest, unsigned W = 1)
      : SrcBB(Src), DestBB(Dest), Weight(W), InMST(false), Removed(false),
        IsCritical(false) {}
  // Return the information string of an edge.
  const std::string infoString() const {
    return (Twine(Removed ? "-" : " ") + (InMST ? " " : "*") +
            (IsCritical ? "c" : " ") + "  W=" + Twine(Weight)).str();
  }
};

// This class stores the auxiliary information for each BB.
struct BBInfo {
  BBInfo *Group;
  uint32_t Index;
  uint32_t Rank;

  BBInfo(unsigned IX) : Group(this), Index(IX), Rank(0) {}

  // Return the information string of this object.
  const std::string infoString() const {
    return (Twine("Index=") + Twine(Index)).str();
  }
};

// This class implements the CFG edges. Note the CFG can be a multi-graph.
template <class Edge, class BBInfo> class FuncPGOInstrumentation {
private:
  Function &F;
  void computeCFGHash();
  void renameComdatFunction();
  // A map that stores the Comdat group in function F.
  std::unordered_multimap<Comdat *, GlobalValue *> &ComdatMembers;

public:
  std::vector<Instruction *> IndirectCallSites;
  SelectInstVisitor SIVisitor;
  std::string FuncName;
  GlobalVariable *FuncNameVar;
  // CFG hash value for this function.
  uint64_t FunctionHash;

  // The Minimum Spanning Tree of function CFG.
  CFGMST<Edge, BBInfo> MST;

  // Give an edge, find the BB that will be instrumented.
  // Return nullptr if there is no BB to be instrumented.
  BasicBlock *getInstrBB(Edge *E);

  // Return the auxiliary BB information.
  BBInfo &getBBInfo(const BasicBlock *BB) const { return MST.getBBInfo(BB); }

  // Return the auxiliary BB information if available.
  BBInfo *findBBInfo(const BasicBlock *BB) const { return MST.findBBInfo(BB); }

  // Dump edges and BB information.
  void dumpInfo(std::string Str = "") const {
    MST.dumpEdges(dbgs(), Twine("Dump Function ") + FuncName + " Hash: " +
                              Twine(FunctionHash) + "\t" + Str);
  }

  FuncPGOInstrumentation(
      Function &Func,
      std::unordered_multimap<Comdat *, GlobalValue *> &ComdatMembers,
      bool CreateGlobalVar = false, BranchProbabilityInfo *BPI = nullptr,
      BlockFrequencyInfo *BFI = nullptr)
      : F(Func), ComdatMembers(ComdatMembers), SIVisitor(Func), FunctionHash(0),
        MST(F, BPI, BFI) {

    // This should be done before CFG hash computation.
    SIVisitor.countSelects(Func);
    NumOfPGOSelectInsts += SIVisitor.getNumOfSelectInsts();
    IndirectCallSites = findIndirectCallSites(Func);

    FuncName = getPGOFuncName(F);
    computeCFGHash();
    if (ComdatMembers.size())
      renameComdatFunction();
    DEBUG(dumpInfo("after CFGMST"));

    NumOfPGOBB += MST.BBInfos.size();
    for (auto &E : MST.AllEdges) {
      if (E->Removed)
        continue;
      NumOfPGOEdge++;
      if (!E->InMST)
        NumOfPGOInstrument++;
    }

    if (CreateGlobalVar)
      FuncNameVar = createPGOFuncNameVar(F, FuncName);
  }

  // Return the number of profile counters needed for the function.
  unsigned getNumCounters() {
    unsigned NumCounters = 0;
    for (auto &E : this->MST.AllEdges) {
      if (!E->InMST && !E->Removed)
        NumCounters++;
    }
    return NumCounters + SIVisitor.getNumOfSelectInsts();
  }
};

// Compute Hash value for the CFG: the lower 32 bits are CRC32 of the index
// value of each BB in the CFG. The higher 32 bits record the number of edges.
template <class Edge, class BBInfo>
void FuncPGOInstrumentation<Edge, BBInfo>::computeCFGHash() {
  std::vector<char> Indexes;
  JamCRC JC;
  for (auto &BB : F) {
    const TerminatorInst *TI = BB.getTerminator();
    for (unsigned I = 0, E = TI->getNumSuccessors(); I != E; ++I) {
      BasicBlock *Succ = TI->getSuccessor(I);
      auto BI = findBBInfo(Succ);
      if (BI == nullptr)
        continue;
      uint32_t Index = BI->Index;
      for (int J = 0; J < 4; J++)
        Indexes.push_back((char)(Index >> (J * 8)));
    }
  }
  JC.update(Indexes);
  FunctionHash = (uint64_t)SIVisitor.getNumOfSelectInsts() << 56 |
                 (uint64_t)IndirectCallSites.size() << 48 |
                 (uint64_t)MST.AllEdges.size() << 32 | JC.getCRC();
}

// Check if we can safely rename this Comdat function.
static bool canRenameComdat(
    Function &F,
    std::unordered_multimap<Comdat *, GlobalValue *> &ComdatMembers) {
  if (!DoComdatRenaming || !canRenameComdatFunc(F, true))
    return false;

  // FIXME: Current only handle those Comdat groups that only containing one
  // function and function aliases.
  // (1) For a Comdat group containing multiple functions, we need to have a
  // unique postfix based on the hashes for each function. There is a
  // non-trivial code refactoring to do this efficiently.
  // (2) Variables can not be renamed, so we can not rename Comdat function in a
  // group including global vars.
  Comdat *C = F.getComdat();
  for (auto &&CM : make_range(ComdatMembers.equal_range(C))) {
    if (dyn_cast<GlobalAlias>(CM.second))
      continue;
    Function *FM = dyn_cast<Function>(CM.second);
    if (FM != &F)
      return false;
  }
  return true;
}

// Append the CFGHash to the Comdat function name.
template <class Edge, class BBInfo>
void FuncPGOInstrumentation<Edge, BBInfo>::renameComdatFunction() {
  if (!canRenameComdat(F, ComdatMembers))
    return;
  std::string OrigName = F.getName().str();
  std::string NewFuncName =
      Twine(F.getName() + "." + Twine(FunctionHash)).str();
  F.setName(Twine(NewFuncName));
  GlobalAlias::create(GlobalValue::WeakAnyLinkage, OrigName, &F);
  FuncName = Twine(FuncName + "." + Twine(FunctionHash)).str();
  Comdat *NewComdat;
  Module *M = F.getParent();
  // For AvailableExternallyLinkage functions, change the linkage to
  // LinkOnceODR and put them into comdat. This is because after renaming, there
  // is no backup external copy available for the function.
  if (!F.hasComdat()) {
    assert(F.getLinkage() == GlobalValue::AvailableExternallyLinkage);
    NewComdat = M->getOrInsertComdat(StringRef(NewFuncName));
    F.setLinkage(GlobalValue::LinkOnceODRLinkage);
    F.setComdat(NewComdat);
    return;
  }

  // This function belongs to a single function Comdat group.
  Comdat *OrigComdat = F.getComdat();
  std::string NewComdatName =
      Twine(OrigComdat->getName() + "." + Twine(FunctionHash)).str();
  NewComdat = M->getOrInsertComdat(StringRef(NewComdatName));
  NewComdat->setSelectionKind(OrigComdat->getSelectionKind());

  for (auto &&CM : make_range(ComdatMembers.equal_range(OrigComdat))) {
    if (GlobalAlias *GA = dyn_cast<GlobalAlias>(CM.second)) {
      // For aliases, change the name directly.
      assert(dyn_cast<Function>(GA->getAliasee()->stripPointerCasts()) == &F);
      std::string OrigGAName = GA->getName().str();
      GA->setName(Twine(GA->getName() + "." + Twine(FunctionHash)));
      GlobalAlias::create(GlobalValue::WeakAnyLinkage, OrigGAName, GA);
      continue;
    }
    // Must be a function.
    Function *CF = dyn_cast<Function>(CM.second);
    assert(CF);
    CF->setComdat(NewComdat);
  }
}

// Given a CFG E to be instrumented, find which BB to place the instrumented
// code. The function will split the critical edge if necessary.
template <class Edge, class BBInfo>
BasicBlock *FuncPGOInstrumentation<Edge, BBInfo>::getInstrBB(Edge *E) {
  if (E->InMST || E->Removed)
    return nullptr;

  BasicBlock *SrcBB = const_cast<BasicBlock *>(E->SrcBB);
  BasicBlock *DestBB = const_cast<BasicBlock *>(E->DestBB);
  // For a fake edge, instrument the real BB.
  if (SrcBB == nullptr)
    return DestBB;
  if (DestBB == nullptr)
    return SrcBB;

  // Instrument the SrcBB if it has a single successor,
  // otherwise, the DestBB if this is not a critical edge.
  TerminatorInst *TI = SrcBB->getTerminator();
  if (TI->getNumSuccessors() <= 1)
    return SrcBB;
  if (!E->IsCritical)
    return DestBB;

  // For a critical edge, we have to split. Instrument the newly
  // created BB.
  NumOfPGOSplit++;
  DEBUG(dbgs() << "Split critical edge: " << getBBInfo(SrcBB).Index << " --> "
               << getBBInfo(DestBB).Index << "\n");
  unsigned SuccNum = GetSuccessorNumber(SrcBB, DestBB);
  BasicBlock *InstrBB = SplitCriticalEdge(TI, SuccNum);
  assert(InstrBB && "Critical edge is not split");

  E->Removed = true;
  return InstrBB;
}

// Visit all edge and instrument the edges not in MST, and do value profiling.
// Critical edges will be split.
static void instrumentOneFunc(
    Function &F, Module *M, BranchProbabilityInfo *BPI, BlockFrequencyInfo *BFI,
    std::unordered_multimap<Comdat *, GlobalValue *> &ComdatMembers) {
  FuncPGOInstrumentation<PGOEdge, BBInfo> FuncInfo(F, ComdatMembers, true, BPI,
                                                   BFI);
  unsigned NumCounters = FuncInfo.getNumCounters();

  uint32_t I = 0;
  Type *I8PtrTy = Type::getInt8PtrTy(M->getContext());
  for (auto &E : FuncInfo.MST.AllEdges) {
    BasicBlock *InstrBB = FuncInfo.getInstrBB(E.get());
    if (!InstrBB)
      continue;

    IRBuilder<> Builder(InstrBB, InstrBB->getFirstInsertionPt());
    assert(Builder.GetInsertPoint() != InstrBB->end() &&
           "Cannot get the Instrumentation point");
    Builder.CreateCall(
        Intrinsic::getDeclaration(M, Intrinsic::instrprof_increment),
        {llvm::ConstantExpr::getBitCast(FuncInfo.FuncNameVar, I8PtrTy),
         Builder.getInt64(FuncInfo.FunctionHash), Builder.getInt32(NumCounters),
         Builder.getInt32(I++)});
  }

  // Now instrument select instructions:
  FuncInfo.SIVisitor.instrumentSelects(F, &I, NumCounters, FuncInfo.FuncNameVar,
                                       FuncInfo.FunctionHash);
  assert(I == NumCounters);

  if (DisableValueProfiling)
    return;

  unsigned NumIndirectCallSites = 0;
  for (auto &I : FuncInfo.IndirectCallSites) {
    CallSite CS(I);
    Value *Callee = CS.getCalledValue();
    DEBUG(dbgs() << "Instrument one indirect call: CallSite Index = "
                 << NumIndirectCallSites << "\n");
    IRBuilder<> Builder(I);
    assert(Builder.GetInsertPoint() != I->getParent()->end() &&
           "Cannot get the Instrumentation point");
    Builder.CreateCall(
        Intrinsic::getDeclaration(M, Intrinsic::instrprof_value_profile),
        {llvm::ConstantExpr::getBitCast(FuncInfo.FuncNameVar, I8PtrTy),
         Builder.getInt64(FuncInfo.FunctionHash),
         Builder.CreatePtrToInt(Callee, Builder.getInt64Ty()),
         Builder.getInt32(llvm::InstrProfValueKind::IPVK_IndirectCallTarget),
         Builder.getInt32(NumIndirectCallSites++)});
  }
  NumOfPGOICall += NumIndirectCallSites;
}

// This class represents a CFG edge in profile use compilation.
struct PGOUseEdge : public PGOEdge {
  bool CountValid;
  uint64_t CountValue;
  PGOUseEdge(const BasicBlock *Src, const BasicBlock *Dest, unsigned W = 1)
      : PGOEdge(Src, Dest, W), CountValid(false), CountValue(0) {}

  // Set edge count value
  void setEdgeCount(uint64_t Value) {
    CountValue = Value;
    CountValid = true;
  }

  // Return the information string for this object.
  const std::string infoString() const {
    if (!CountValid)
      return PGOEdge::infoString();
    return (Twine(PGOEdge::infoString()) + "  Count=" + Twine(CountValue))
        .str();
  }
};

typedef SmallVector<PGOUseEdge *, 2> DirectEdges;

// This class stores the auxiliary information for each BB.
struct UseBBInfo : public BBInfo {
  uint64_t CountValue;
  bool CountValid;
  int32_t UnknownCountInEdge;
  int32_t UnknownCountOutEdge;
  DirectEdges InEdges;
  DirectEdges OutEdges;
  UseBBInfo(unsigned IX)
      : BBInfo(IX), CountValue(0), CountValid(false), UnknownCountInEdge(0),
        UnknownCountOutEdge(0) {}
  UseBBInfo(unsigned IX, uint64_t C)
      : BBInfo(IX), CountValue(C), CountValid(true), UnknownCountInEdge(0),
        UnknownCountOutEdge(0) {}

  // Set the profile count value for this BB.
  void setBBInfoCount(uint64_t Value) {
    CountValue = Value;
    CountValid = true;
  }

  // Return the information string of this object.
  const std::string infoString() const {
    if (!CountValid)
      return BBInfo::infoString();
    return (Twine(BBInfo::infoString()) + "  Count=" + Twine(CountValue)).str();
  }
};

// Sum up the count values for all the edges.
static uint64_t sumEdgeCount(const ArrayRef<PGOUseEdge *> Edges) {
  uint64_t Total = 0;
  for (auto &E : Edges) {
    if (E->Removed)
      continue;
    Total += E->CountValue;
  }
  return Total;
}

class PGOUseFunc {
public:
  PGOUseFunc(Function &Func, Module *Modu,
             std::unordered_multimap<Comdat *, GlobalValue *> &ComdatMembers,
             BranchProbabilityInfo *BPI = nullptr,
             BlockFrequencyInfo *BFI = nullptr)
      : F(Func), M(Modu), FuncInfo(Func, ComdatMembers, false, BPI, BFI),
        CountPosition(0), ProfileCountSize(0), FreqAttr(FFA_Normal) {}

  // Read counts for the instrumented BB from profile.
  bool readCounters(IndexedInstrProfReader *PGOReader);

  // Populate the counts for all BBs.
  void populateCounters();

  // Set the branch weights based on the count values.
  void setBranchWeights();

  // Annotate the indirect call sites.
  void annotateIndirectCallSites();

  // The hotness of the function from the profile count.
  enum FuncFreqAttr { FFA_Normal, FFA_Cold, FFA_Hot };

  // Return the function hotness from the profile.
  FuncFreqAttr getFuncFreqAttr() const { return FreqAttr; }

  // Return the function hash.
  uint64_t getFuncHash() const { return FuncInfo.FunctionHash; }
  // Return the profile record for this function;
  InstrProfRecord &getProfileRecord() { return ProfileRecord; }

  // Return the auxiliary BB information.
  UseBBInfo &getBBInfo(const BasicBlock *BB) const {
    return FuncInfo.getBBInfo(BB);
  }

  // Return the auxiliary BB information if available.
  UseBBInfo *findBBInfo(const BasicBlock *BB) const {
    return FuncInfo.findBBInfo(BB);
  }

  Function &getFunc() const { return F; }

private:
  Function &F;
  Module *M;
  // This member stores the shared information with class PGOGenFunc.
  FuncPGOInstrumentation<PGOUseEdge, UseBBInfo> FuncInfo;

  // The maximum count value in the profile. This is only used in PGO use
  // compilation.
  uint64_t ProgramMaxCount;

  // Position of counter that remains to be read.
  uint32_t CountPosition;

  // Total size of the profile count for this function.
  uint32_t ProfileCountSize;

  // ProfileRecord for this function.
  InstrProfRecord ProfileRecord;

  // Function hotness info derived from profile.
  FuncFreqAttr FreqAttr;

  // Find the Instrumented BB and set the value.
  void setInstrumentedCounts(const std::vector<uint64_t> &CountFromProfile);

  // Set the edge counter value for the unknown edge -- there should be only
  // one unknown edge.
  void setEdgeCount(DirectEdges &Edges, uint64_t Value);

  // Return FuncName string;
  const std::string getFuncName() const { return FuncInfo.FuncName; }

  // Set the hot/cold inline hints based on the count values.
  // FIXME: This function should be removed once the functionality in
  // the inliner is implemented.
  void markFunctionAttributes(uint64_t EntryCount, uint64_t MaxCount) {
    if (ProgramMaxCount == 0)
      return;
    // Threshold of the hot functions.
    const BranchProbability HotFunctionThreshold(1, 100);
    // Threshold of the cold functions.
    const BranchProbability ColdFunctionThreshold(2, 10000);
    if (EntryCount >= HotFunctionThreshold.scale(ProgramMaxCount))
      FreqAttr = FFA_Hot;
    else if (MaxCount <= ColdFunctionThreshold.scale(ProgramMaxCount))
      FreqAttr = FFA_Cold;
  }
};

// Visit all the edges and assign the count value for the instrumented
// edges and the BB.
void PGOUseFunc::setInstrumentedCounts(
    const std::vector<uint64_t> &CountFromProfile) {

  assert(FuncInfo.getNumCounters() == CountFromProfile.size());
  // Use a worklist as we will update the vector during the iteration.
  std::vector<PGOUseEdge *> WorkList;
  for (auto &E : FuncInfo.MST.AllEdges)
    WorkList.push_back(E.get());

  uint32_t I = 0;
  for (auto &E : WorkList) {
    BasicBlock *InstrBB = FuncInfo.getInstrBB(E);
    if (!InstrBB)
      continue;
    uint64_t CountValue = CountFromProfile[I++];
    if (!E->Removed) {
      getBBInfo(InstrBB).setBBInfoCount(CountValue);
      E->setEdgeCount(CountValue);
      continue;
    }

    // Need to add two new edges.
    BasicBlock *SrcBB = const_cast<BasicBlock *>(E->SrcBB);
    BasicBlock *DestBB = const_cast<BasicBlock *>(E->DestBB);
    // Add new edge of SrcBB->InstrBB.
    PGOUseEdge &NewEdge = FuncInfo.MST.addEdge(SrcBB, InstrBB, 0);
    NewEdge.setEdgeCount(CountValue);
    // Add new edge of InstrBB->DestBB.
    PGOUseEdge &NewEdge1 = FuncInfo.MST.addEdge(InstrBB, DestBB, 0);
    NewEdge1.setEdgeCount(CountValue);
    NewEdge1.InMST = true;
    getBBInfo(InstrBB).setBBInfoCount(CountValue);
  }
  ProfileCountSize =  CountFromProfile.size();
  CountPosition = I;
}

// Set the count value for the unknown edge. There should be one and only one
// unknown edge in Edges vector.
void PGOUseFunc::setEdgeCount(DirectEdges &Edges, uint64_t Value) {
  for (auto &E : Edges) {
    if (E->CountValid)
      continue;
    E->setEdgeCount(Value);

    getBBInfo(E->SrcBB).UnknownCountOutEdge--;
    getBBInfo(E->DestBB).UnknownCountInEdge--;
    return;
  }
  llvm_unreachable("Cannot find the unknown count edge");
}

// Read the profile from ProfileFileName and assign the value to the
// instrumented BB and the edges. This function also updates ProgramMaxCount.
// Return true if the profile are successfully read, and false on errors.
bool PGOUseFunc::readCounters(IndexedInstrProfReader *PGOReader) {
  auto &Ctx = M->getContext();
  Expected<InstrProfRecord> Result =
      PGOReader->getInstrProfRecord(FuncInfo.FuncName, FuncInfo.FunctionHash);
  if (Error E = Result.takeError()) {
    handleAllErrors(std::move(E), [&](const InstrProfError &IPE) {
      auto Err = IPE.get();
      bool SkipWarning = false;
      if (Err == instrprof_error::unknown_function) {
        NumOfPGOMissing++;
        SkipWarning = !PGOWarnMissing;
      } else if (Err == instrprof_error::hash_mismatch ||
                 Err == instrprof_error::malformed) {
        NumOfPGOMismatch++;
        SkipWarning =
            NoPGOWarnMismatch ||
            (NoPGOWarnMismatchComdat &&
             (F.hasComdat() ||
              F.getLinkage() == GlobalValue::AvailableExternallyLinkage));
      }

      if (SkipWarning)
        return;

      std::string Msg = IPE.message() + std::string(" ") + F.getName().str();
      Ctx.diagnose(
          DiagnosticInfoPGOProfile(M->getName().data(), Msg, DS_Warning));
    });
    return false;
  }
  ProfileRecord = std::move(Result.get());
  std::vector<uint64_t> &CountFromProfile = ProfileRecord.Counts;

  NumOfPGOFunc++;
  DEBUG(dbgs() << CountFromProfile.size() << " counts\n");
  uint64_t ValueSum = 0;
  for (unsigned I = 0, S = CountFromProfile.size(); I < S; I++) {
    DEBUG(dbgs() << "  " << I << ": " << CountFromProfile[I] << "\n");
    ValueSum += CountFromProfile[I];
  }

  DEBUG(dbgs() << "SUM =  " << ValueSum << "\n");

  getBBInfo(nullptr).UnknownCountOutEdge = 2;
  getBBInfo(nullptr).UnknownCountInEdge = 2;

  setInstrumentedCounts(CountFromProfile);
  ProgramMaxCount = PGOReader->getMaximumFunctionCount();
  return true;
}

// Populate the counters from instrumented BBs to all BBs.
// In the end of this operation, all BBs should have a valid count value.
void PGOUseFunc::populateCounters() {
  // First set up Count variable for all BBs.
  for (auto &E : FuncInfo.MST.AllEdges) {
    if (E->Removed)
      continue;

    const BasicBlock *SrcBB = E->SrcBB;
    const BasicBlock *DestBB = E->DestBB;
    UseBBInfo &SrcInfo = getBBInfo(SrcBB);
    UseBBInfo &DestInfo = getBBInfo(DestBB);
    SrcInfo.OutEdges.push_back(E.get());
    DestInfo.InEdges.push_back(E.get());
    SrcInfo.UnknownCountOutEdge++;
    DestInfo.UnknownCountInEdge++;

    if (!E->CountValid)
      continue;
    DestInfo.UnknownCountInEdge--;
    SrcInfo.UnknownCountOutEdge--;
  }

  bool Changes = true;
  unsigned NumPasses = 0;
  while (Changes) {
    NumPasses++;
    Changes = false;

    // For efficient traversal, it's better to start from the end as most
    // of the instrumented edges are at the end.
    for (auto &BB : reverse(F)) {
      UseBBInfo *Count = findBBInfo(&BB);
      if (Count == nullptr)
        continue;
      if (!Count->CountValid) {
        if (Count->UnknownCountOutEdge == 0) {
          Count->CountValue = sumEdgeCount(Count->OutEdges);
          Count->CountValid = true;
          Changes = true;
        } else if (Count->UnknownCountInEdge == 0) {
          Count->CountValue = sumEdgeCount(Count->InEdges);
          Count->CountValid = true;
          Changes = true;
        }
      }
      if (Count->CountValid) {
        if (Count->UnknownCountOutEdge == 1) {
          uint64_t Total = 0;
          uint64_t OutSum = sumEdgeCount(Count->OutEdges);
          // If the one of the successor block can early terminate (no-return),
          // we can end up with situation where out edge sum count is larger as
          // the source BB's count is collected by a post-dominated block.
          if (Count->CountValue > OutSum)
            Total = Count->CountValue - OutSum;
          setEdgeCount(Count->OutEdges, Total);
          Changes = true;
        }
        if (Count->UnknownCountInEdge == 1) {
          uint64_t Total = 0;
          uint64_t InSum = sumEdgeCount(Count->InEdges);
          if (Count->CountValue > InSum)
            Total = Count->CountValue - InSum;
          setEdgeCount(Count->InEdges, Total);
          Changes = true;
        }
      }
    }
  }

  DEBUG(dbgs() << "Populate counts in " << NumPasses << " passes.\n");
#ifndef NDEBUG
  // Assert every BB has a valid counter.
  for (auto &BB : F) {
    auto BI = findBBInfo(&BB);
    if (BI == nullptr)
      continue;
    assert(BI->CountValid && "BB count is not valid");
  }
#endif
  uint64_t FuncEntryCount = getBBInfo(&*F.begin()).CountValue;
  F.setEntryCount(FuncEntryCount);
  uint64_t FuncMaxCount = FuncEntryCount;
  for (auto &BB : F) {
    auto BI = findBBInfo(&BB);
    if (BI == nullptr)
      continue;
    FuncMaxCount = std::max(FuncMaxCount, BI->CountValue);
  }
  markFunctionAttributes(FuncEntryCount, FuncMaxCount);

  // Now annotate select instructions
  FuncInfo.SIVisitor.annotateSelects(F, this, &CountPosition);
  assert(CountPosition == ProfileCountSize);

  DEBUG(FuncInfo.dumpInfo("after reading profile."));
}

static void setProfMetadata(Module *M, Instruction *TI,
                            ArrayRef<uint64_t> EdgeCounts, uint64_t MaxCount) {
  MDBuilder MDB(M->getContext());
  assert(MaxCount > 0 && "Bad max count");
  uint64_t Scale = calculateCountScale(MaxCount);
  SmallVector<unsigned, 4> Weights;
  for (const auto &ECI : EdgeCounts)
    Weights.push_back(scaleBranchCount(ECI, Scale));

  DEBUG(dbgs() << "Weight is: ";
        for (const auto &W : Weights) { dbgs() << W << " "; } 
        dbgs() << "\n";);
  TI->setMetadata(llvm::LLVMContext::MD_prof, MDB.createBranchWeights(Weights));
}

// Assign the scaled count values to the BB with multiple out edges.
void PGOUseFunc::setBranchWeights() {
  // Generate MD_prof metadata for every branch instruction.
  DEBUG(dbgs() << "\nSetting branch weights.\n");
  for (auto &BB : F) {
    TerminatorInst *TI = BB.getTerminator();
    if (TI->getNumSuccessors() < 2)
      continue;
    if (!isa<BranchInst>(TI) && !isa<SwitchInst>(TI))
      continue;
    if (getBBInfo(&BB).CountValue == 0)
      continue;

    // We have a non-zero Branch BB.
    const UseBBInfo &BBCountInfo = getBBInfo(&BB);
    unsigned Size = BBCountInfo.OutEdges.size();
    SmallVector<uint64_t, 2> EdgeCounts(Size, 0);
    uint64_t MaxCount = 0;
    for (unsigned s = 0; s < Size; s++) {
      const PGOUseEdge *E = BBCountInfo.OutEdges[s];
      const BasicBlock *SrcBB = E->SrcBB;
      const BasicBlock *DestBB = E->DestBB;
      if (DestBB == nullptr)
        continue;
      unsigned SuccNum = GetSuccessorNumber(SrcBB, DestBB);
      uint64_t EdgeCount = E->CountValue;
      if (EdgeCount > MaxCount)
        MaxCount = EdgeCount;
      EdgeCounts[SuccNum] = EdgeCount;
    }
    setProfMetadata(M, TI, EdgeCounts, MaxCount);
  }
}

void SelectInstVisitor::instrumentOneSelectInst(SelectInst &SI) {
  Module *M = F.getParent();
  IRBuilder<> Builder(&SI);
  Type *Int64Ty = Builder.getInt64Ty();
  Type *I8PtrTy = Builder.getInt8PtrTy();
  auto *Step = Builder.CreateZExt(SI.getCondition(), Int64Ty);
  Builder.CreateCall(
      Intrinsic::getDeclaration(M, Intrinsic::instrprof_increment_step),
      {llvm::ConstantExpr::getBitCast(FuncNameVar, I8PtrTy),
       Builder.getInt64(FuncHash),
       Builder.getInt32(TotalNumCtrs), Builder.getInt32(*CurCtrIdx), Step});
  ++(*CurCtrIdx);
}

void SelectInstVisitor::annotateOneSelectInst(SelectInst &SI) {
  std::vector<uint64_t> &CountFromProfile = UseFunc->getProfileRecord().Counts;
  assert(*CurCtrIdx < CountFromProfile.size() &&
         "Out of bound access of counters");
  uint64_t SCounts[2];
  SCounts[0] = CountFromProfile[*CurCtrIdx]; // True count
  ++(*CurCtrIdx);
  uint64_t TotalCount = 0;
  auto BI = UseFunc->findBBInfo(SI.getParent());
  if (BI != nullptr)
    TotalCount = BI->CountValue;
  // False Count
  SCounts[1] = (TotalCount > SCounts[0] ? TotalCount - SCounts[0] : 0);
  uint64_t MaxCount = std::max(SCounts[0], SCounts[1]);
  if (MaxCount)
    setProfMetadata(F.getParent(), &SI, SCounts, MaxCount);
}

void SelectInstVisitor::visitSelectInst(SelectInst &SI) {
  if (!PGOInstrSelect)
    return;
  // FIXME: do not handle this yet.
  if (SI.getCondition()->getType()->isVectorTy())
    return;

  NSIs++;
  switch (Mode) {
  case VM_counting:
    return;
  case VM_instrument:
    instrumentOneSelectInst(SI);
    return;
  case VM_annotate:
    annotateOneSelectInst(SI);
    return;
  }

  llvm_unreachable("Unknown visiting mode");
}

// Traverse all the indirect callsites and annotate the instructions.
void PGOUseFunc::annotateIndirectCallSites() {
  if (DisableValueProfiling)
    return;

  // Create the PGOFuncName meta data.
  createPGOFuncNameMetadata(F, FuncInfo.FuncName);

  unsigned IndirectCallSiteIndex = 0;
  auto &IndirectCallSites = FuncInfo.IndirectCallSites;
  unsigned NumValueSites =
      ProfileRecord.getNumValueSites(IPVK_IndirectCallTarget);
  if (NumValueSites != IndirectCallSites.size()) {
    std::string Msg =
        std::string("Inconsistent number of indirect call sites: ") +
        F.getName().str();
    auto &Ctx = M->getContext();
    Ctx.diagnose(
        DiagnosticInfoPGOProfile(M->getName().data(), Msg, DS_Warning));
    return;
  }

  for (auto &I : IndirectCallSites) {
    DEBUG(dbgs() << "Read one indirect call instrumentation: Index="
                 << IndirectCallSiteIndex << " out of " << NumValueSites
                 << "\n");
    annotateValueSite(*M, *I, ProfileRecord, IPVK_IndirectCallTarget,
                      IndirectCallSiteIndex, MaxNumAnnotations);
    IndirectCallSiteIndex++;
  }
}
} // end anonymous namespace

// Create a COMDAT variable INSTR_PROF_RAW_VERSION_VAR to make the runtime
// aware this is an ir_level profile so it can set the version flag.
static void createIRLevelProfileFlagVariable(Module &M) {
  Type *IntTy64 = Type::getInt64Ty(M.getContext());
  uint64_t ProfileVersion = (INSTR_PROF_RAW_VERSION | VARIANT_MASK_IR_PROF);
  auto IRLevelVersionVariable = new GlobalVariable(
      M, IntTy64, true, GlobalVariable::ExternalLinkage,
      Constant::getIntegerValue(IntTy64, APInt(64, ProfileVersion)),
      INSTR_PROF_QUOTE(INSTR_PROF_RAW_VERSION_VAR));
  IRLevelVersionVariable->setVisibility(GlobalValue::DefaultVisibility);
  Triple TT(M.getTargetTriple());
  if (!TT.supportsCOMDAT())
    IRLevelVersionVariable->setLinkage(GlobalValue::WeakAnyLinkage);
  else
    IRLevelVersionVariable->setComdat(M.getOrInsertComdat(
        StringRef(INSTR_PROF_QUOTE(INSTR_PROF_RAW_VERSION_VAR))));
}

// Collect the set of members for each Comdat in module M and store
// in ComdatMembers.
static void collectComdatMembers(
    Module &M,
    std::unordered_multimap<Comdat *, GlobalValue *> &ComdatMembers) {
  if (!DoComdatRenaming)
    return;
  for (Function &F : M)
    if (Comdat *C = F.getComdat())
      ComdatMembers.insert(std::make_pair(C, &F));
  for (GlobalVariable &GV : M.globals())
    if (Comdat *C = GV.getComdat())
      ComdatMembers.insert(std::make_pair(C, &GV));
  for (GlobalAlias &GA : M.aliases())
    if (Comdat *C = GA.getComdat())
      ComdatMembers.insert(std::make_pair(C, &GA));
}

static bool InstrumentAllFunctions(
    Module &M, function_ref<BranchProbabilityInfo *(Function &)> LookupBPI,
    function_ref<BlockFrequencyInfo *(Function &)> LookupBFI) {
  createIRLevelProfileFlagVariable(M);
  std::unordered_multimap<Comdat *, GlobalValue *> ComdatMembers;
  collectComdatMembers(M, ComdatMembers);

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    auto *BPI = LookupBPI(F);
    auto *BFI = LookupBFI(F);
    instrumentOneFunc(F, &M, BPI, BFI, ComdatMembers);
  }
  return true;
}

bool PGOInstrumentationGenLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  auto LookupBPI = [this](Function &F) {
    return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
  };
  auto LookupBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };
  return InstrumentAllFunctions(M, LookupBPI, LookupBFI);
}

PreservedAnalyses PGOInstrumentationGen::run(Module &M,
                                             ModuleAnalysisManager &AM) {

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupBPI = [&FAM](Function &F) {
    return &FAM.getResult<BranchProbabilityAnalysis>(F);
  };

  auto LookupBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };

  if (!InstrumentAllFunctions(M, LookupBPI, LookupBFI))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

static bool annotateAllFunctions(
    Module &M, StringRef ProfileFileName,
    function_ref<BranchProbabilityInfo *(Function &)> LookupBPI,
    function_ref<BlockFrequencyInfo *(Function &)> LookupBFI) {
  DEBUG(dbgs() << "Read in profile counters: ");
  auto &Ctx = M.getContext();
  // Read the counter array from file.
  auto ReaderOrErr = IndexedInstrProfReader::create(ProfileFileName);
  if (Error E = ReaderOrErr.takeError()) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
      Ctx.diagnose(
          DiagnosticInfoPGOProfile(ProfileFileName.data(), EI.message()));
    });
    return false;
  }

  std::unique_ptr<IndexedInstrProfReader> PGOReader =
      std::move(ReaderOrErr.get());
  if (!PGOReader) {
    Ctx.diagnose(DiagnosticInfoPGOProfile(ProfileFileName.data(),
                                          StringRef("Cannot get PGOReader")));
    return false;
  }
  // TODO: might need to change the warning once the clang option is finalized.
  if (!PGOReader->isIRLevelProfile()) {
    Ctx.diagnose(DiagnosticInfoPGOProfile(
        ProfileFileName.data(), "Not an IR level instrumentation profile"));
    return false;
  }

  std::unordered_multimap<Comdat *, GlobalValue *> ComdatMembers;
  collectComdatMembers(M, ComdatMembers);
  std::vector<Function *> HotFunctions;
  std::vector<Function *> ColdFunctions;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    auto *BPI = LookupBPI(F);
    auto *BFI = LookupBFI(F);
    PGOUseFunc Func(F, &M, ComdatMembers, BPI, BFI);
    if (!Func.readCounters(PGOReader.get()))
      continue;
    Func.populateCounters();
    Func.setBranchWeights();
    Func.annotateIndirectCallSites();
    PGOUseFunc::FuncFreqAttr FreqAttr = Func.getFuncFreqAttr();
    if (FreqAttr == PGOUseFunc::FFA_Cold)
      ColdFunctions.push_back(&F);
    else if (FreqAttr == PGOUseFunc::FFA_Hot)
      HotFunctions.push_back(&F);
    if (PGOViewCounts && (ViewBlockFreqFuncName.empty() ||
                          F.getName().equals(ViewBlockFreqFuncName))) {
      LoopInfo LI{DominatorTree(F)};
      std::unique_ptr<BranchProbabilityInfo> NewBPI =
          llvm::make_unique<BranchProbabilityInfo>(F, LI);
      std::unique_ptr<BlockFrequencyInfo> NewBFI =
          llvm::make_unique<BlockFrequencyInfo>(F, *NewBPI, LI);

      NewBFI->view();
    }
    if (PGOViewRawCounts && (ViewBlockFreqFuncName.empty() ||
                             F.getName().equals(ViewBlockFreqFuncName))) {
      if (ViewBlockFreqFuncName.empty())
        WriteGraph(&Func, Twine("PGORawCounts_") + Func.getFunc().getName());
      else
        ViewGraph(&Func, Twine("PGORawCounts_") + Func.getFunc().getName());
    }
  }
  M.setProfileSummary(PGOReader->getSummary().getMD(M.getContext()));
  // Set function hotness attribute from the profile.
  // We have to apply these attributes at the end because their presence
  // can affect the BranchProbabilityInfo of any callers, resulting in an
  // inconsistent MST between prof-gen and prof-use.
  for (auto &F : HotFunctions) {
    F->addFnAttr(llvm::Attribute::InlineHint);
    DEBUG(dbgs() << "Set inline attribute to function: " << F->getName()
                 << "\n");
  }
  for (auto &F : ColdFunctions) {
    F->addFnAttr(llvm::Attribute::Cold);
    DEBUG(dbgs() << "Set cold attribute to function: " << F->getName() << "\n");
  }
  return true;
}

PGOInstrumentationUse::PGOInstrumentationUse(std::string Filename)
    : ProfileFileName(std::move(Filename)) {
  if (!PGOTestProfileFile.empty())
    ProfileFileName = PGOTestProfileFile;
}

PreservedAnalyses PGOInstrumentationUse::run(Module &M,
                                             ModuleAnalysisManager &AM) {

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto LookupBPI = [&FAM](Function &F) {
    return &FAM.getResult<BranchProbabilityAnalysis>(F);
  };

  auto LookupBFI = [&FAM](Function &F) {
    return &FAM.getResult<BlockFrequencyAnalysis>(F);
  };

  if (!annotateAllFunctions(M, ProfileFileName, LookupBPI, LookupBFI))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

bool PGOInstrumentationUseLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  auto LookupBPI = [this](Function &F) {
    return &this->getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI();
  };
  auto LookupBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };

  return annotateAllFunctions(M, ProfileFileName, LookupBPI, LookupBFI);
}

namespace llvm {
template <> struct GraphTraits<PGOUseFunc *> {
  typedef const BasicBlock *NodeRef;
  typedef succ_const_iterator ChildIteratorType;
  typedef pointer_iterator<Function::const_iterator> nodes_iterator;

  static NodeRef getEntryNode(const PGOUseFunc *G) {
    return &G->getFunc().front();
  }
  static ChildIteratorType child_begin(const NodeRef N) {
    return succ_begin(N);
  }
  static ChildIteratorType child_end(const NodeRef N) { return succ_end(N); }
  static nodes_iterator nodes_begin(const PGOUseFunc *G) {
    return nodes_iterator(G->getFunc().begin());
  }
  static nodes_iterator nodes_end(const PGOUseFunc *G) {
    return nodes_iterator(G->getFunc().end());
  }
};

static std::string getSimpleNodeName(const BasicBlock *Node) {
  if (!Node->getName().empty())
    return Node->getName();

  std::string SimpleNodeName;
  raw_string_ostream OS(SimpleNodeName);
  Node->printAsOperand(OS, false);
  return OS.str();
}

template <> struct DOTGraphTraits<PGOUseFunc *> : DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple = false)
      : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(const PGOUseFunc *G) {
    return G->getFunc().getName();
  }

  std::string getNodeLabel(const BasicBlock *Node, const PGOUseFunc *Graph) {
    std::string Result;
    raw_string_ostream OS(Result);

    OS << getSimpleNodeName(Node) << ":\\l";
    UseBBInfo *BI = Graph->findBBInfo(Node);
    OS << "Count : ";
    if (BI && BI->CountValid)
      OS << BI->CountValue << "\\l";
    else
      OS << "Unknown\\l";

    if (!PGOInstrSelect)
      return Result;

    for (auto BI = Node->begin(); BI != Node->end(); ++BI) {
      auto *I = &*BI;
      if (!isa<SelectInst>(I))
        continue;
      // Display scaled counts for SELECT instruction:
      OS << "SELECT : { T = ";
      uint64_t TC, FC;
      bool HasProf = I->extractProfMetadata(TC, FC);
      if (!HasProf)
        OS << "Unknown, F = Unknown }\\l";
      else
        OS << TC << ", F = " << FC << " }\\l";
    }
    return Result;
  }
};
}
