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

#include "CFGMST.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JamCRC.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-instrumentation"

STATISTIC(NumOfPGOInstrument, "Number of edges instrumented.");
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

// Command line options to disable value profiling. The default is false:
// i.e. value profiling is enabled by default. This is for debug purpose.
static cl::opt<bool>
DisableValueProfiling("disable-vp", cl::init(false),
                                    cl::Hidden,
                                    cl::desc("Disable Value Profiling"));

namespace {
class PGOInstrumentationGen : public ModulePass {
public:
  static char ID;

  PGOInstrumentationGen() : ModulePass(ID) {
    initializePGOInstrumentationGenPass(*PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override {
    return "PGOInstrumentationGenPass";
  }

private:
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
  }
};

class PGOInstrumentationUse : public ModulePass {
public:
  static char ID;

  // Provide the profile filename as the parameter.
  PGOInstrumentationUse(std::string Filename = "")
      : ModulePass(ID), ProfileFileName(Filename) {
    if (!PGOTestProfileFile.empty())
      ProfileFileName = PGOTestProfileFile;
    initializePGOInstrumentationUsePass(*PassRegistry::getPassRegistry());
  }

  const char *getPassName() const override {
    return "PGOInstrumentationUsePass";
  }

private:
  std::string ProfileFileName;
  std::unique_ptr<IndexedInstrProfReader> PGOReader;
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
  }
};
} // end anonymous namespace

char PGOInstrumentationGen::ID = 0;
INITIALIZE_PASS_BEGIN(PGOInstrumentationGen, "pgo-instr-gen",
                      "PGO instrumentation.", false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(PGOInstrumentationGen, "pgo-instr-gen",
                    "PGO instrumentation.", false, false)

ModulePass *llvm::createPGOInstrumentationGenPass() {
  return new PGOInstrumentationGen();
}

char PGOInstrumentationUse::ID = 0;
INITIALIZE_PASS_BEGIN(PGOInstrumentationUse, "pgo-instr-use",
                      "Read PGO instrumentation profile.", false, false)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(PGOInstrumentationUse, "pgo-instr-use",
                    "Read PGO instrumentation profile.", false, false)

ModulePass *llvm::createPGOInstrumentationUsePass(StringRef Filename) {
  return new PGOInstrumentationUse(Filename.str());
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

public:
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

  // Dump edges and BB information.
  void dumpInfo(std::string Str = "") const {
    MST.dumpEdges(dbgs(), Twine("Dump Function ") + FuncName + " Hash: " +
                              Twine(FunctionHash) + "\t" + Str);
  }

  FuncPGOInstrumentation(Function &Func, bool CreateGlobalVar = false,
                         BranchProbabilityInfo *BPI = nullptr,
                         BlockFrequencyInfo *BFI = nullptr)
      : F(Func), FunctionHash(0), MST(F, BPI, BFI) {
    FuncName = getPGOFuncName(F);
    computeCFGHash();
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
      uint32_t Index = getBBInfo(Succ).Index;
      for (int J = 0; J < 4; J++)
        Indexes.push_back((char)(Index >> (J * 8)));
    }
  }
  JC.update(Indexes);
  FunctionHash = (uint64_t)MST.AllEdges.size() << 32 | JC.getCRC();
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

// Visitor class that finds all indirect call sites.
struct PGOIndirectCallSiteVisitor
    : public InstVisitor<PGOIndirectCallSiteVisitor> {
  std::vector<CallInst *> IndirectCallInsts;
  PGOIndirectCallSiteVisitor() {}

  void visitCallInst(CallInst &I) {
    CallSite CS(&I);
    if (CS.getCalledFunction() || !CS.getCalledValue())
      return;
    IndirectCallInsts.push_back(&I);
  }
};

// Visit all edge and instrument the edges not in MST, and do value profiling.
// Critical edges will be split.
static void instrumentOneFunc(Function &F, Module *M,
                              BranchProbabilityInfo *BPI,
                              BlockFrequencyInfo *BFI) {
  unsigned NumCounters = 0;
  FuncPGOInstrumentation<PGOEdge, BBInfo> FuncInfo(F, true, BPI, BFI);
  for (auto &E : FuncInfo.MST.AllEdges) {
    if (!E->InMST && !E->Removed)
      NumCounters++;
  }

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

  if (DisableValueProfiling)
    return;

  unsigned NumIndirectCallSites = 0;
  PGOIndirectCallSiteVisitor ICV;
  ICV.visit(F);
  for (auto &I : ICV.IndirectCallInsts) {
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
    return (Twine(PGOEdge::infoString()) + "  Count=" + Twine(CountValue)).str();
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
private:
  Function &F;
  Module *M;
  // This member stores the shared information with class PGOGenFunc.
  FuncPGOInstrumentation<PGOUseEdge, UseBBInfo> FuncInfo;

  // Return the auxiliary BB information.
  UseBBInfo &getBBInfo(const BasicBlock *BB) const {
    return FuncInfo.getBBInfo(BB);
  }

  // The maximum count value in the profile. This is only used in PGO use
  // compilation.
  uint64_t ProgramMaxCount;

  // ProfileRecord for this function.
  InstrProfRecord ProfileRecord;

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
  void applyFunctionAttributes(uint64_t EntryCount, uint64_t MaxCount) {
    if (ProgramMaxCount == 0)
      return;
    // Threshold of the hot functions.
    const BranchProbability HotFunctionThreshold(1, 100);
    // Threshold of the cold functions.
    const BranchProbability ColdFunctionThreshold(2, 10000);
    if (EntryCount >= HotFunctionThreshold.scale(ProgramMaxCount))
      F.addFnAttr(llvm::Attribute::InlineHint);
    else if (MaxCount <= ColdFunctionThreshold.scale(ProgramMaxCount))
      F.addFnAttr(llvm::Attribute::Cold);
  }

public:
  PGOUseFunc(Function &Func, Module *Modu, BranchProbabilityInfo *BPI = nullptr,
             BlockFrequencyInfo *BFI = nullptr)
      : F(Func), M(Modu), FuncInfo(Func, false, BPI, BFI) {}

  // Read counts for the instrumented BB from profile.
  bool readCounters(IndexedInstrProfReader *PGOReader);

  // Populate the counts for all BBs.
  void populateCounters();

  // Set the branch weights based on the count values.
  void setBranchWeights();

  // Annotate the indirect call sites.
  void annotateIndirectCallSites();
};

// Visit all the edges and assign the count value for the instrumented
// edges and the BB.
void PGOUseFunc::setInstrumentedCounts(
    const std::vector<uint64_t> &CountFromProfile) {

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
  ErrorOr<InstrProfRecord> Result =
      PGOReader->getInstrProfRecord(FuncInfo.FuncName, FuncInfo.FunctionHash);
  if (std::error_code EC = Result.getError()) {
    if (EC == instrprof_error::unknown_function)
      NumOfPGOMissing++;
    else if (EC == instrprof_error::hash_mismatch ||
             EC == llvm::instrprof_error::malformed)
      NumOfPGOMismatch++;

    std::string Msg = EC.message() + std::string(" ") + F.getName().str();
    Ctx.diagnose(
        DiagnosticInfoPGOProfile(M->getName().data(), Msg, DS_Warning));
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
      UseBBInfo &Count = getBBInfo(&BB);
      if (!Count.CountValid) {
        if (Count.UnknownCountOutEdge == 0) {
          Count.CountValue = sumEdgeCount(Count.OutEdges);
          Count.CountValid = true;
          Changes = true;
        } else if (Count.UnknownCountInEdge == 0) {
          Count.CountValue = sumEdgeCount(Count.InEdges);
          Count.CountValid = true;
          Changes = true;
        }
      }
      if (Count.CountValid) {
        if (Count.UnknownCountOutEdge == 1) {
          uint64_t Total = Count.CountValue - sumEdgeCount(Count.OutEdges);
          setEdgeCount(Count.OutEdges, Total);
          Changes = true;
        }
        if (Count.UnknownCountInEdge == 1) {
          uint64_t Total = Count.CountValue - sumEdgeCount(Count.InEdges);
          setEdgeCount(Count.InEdges, Total);
          Changes = true;
        }
      }
    }
  }

  DEBUG(dbgs() << "Populate counts in " << NumPasses << " passes.\n");
  // Assert every BB has a valid counter.
  uint64_t FuncEntryCount = getBBInfo(&*F.begin()).CountValue;
  uint64_t FuncMaxCount = FuncEntryCount;
  for (auto &BB : F) {
    assert(getBBInfo(&BB).CountValid && "BB count is not valid");
    uint64_t Count = getBBInfo(&BB).CountValue;
    if (Count > FuncMaxCount)
      FuncMaxCount = Count;
  }
  applyFunctionAttributes(FuncEntryCount, FuncMaxCount);

  DEBUG(FuncInfo.dumpInfo("after reading profile."));
}

// Assign the scaled count values to the BB with multiple out edges.
void PGOUseFunc::setBranchWeights() {
  // Generate MD_prof metadata for every branch instruction.
  DEBUG(dbgs() << "\nSetting branch weights.\n");
  MDBuilder MDB(M->getContext());
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
    SmallVector<unsigned, 2> EdgeCounts(Size, 0);
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
    assert(MaxCount > 0 && "Bad max count");
    uint64_t Scale = calculateCountScale(MaxCount);
    SmallVector<unsigned, 4> Weights;
    for (const auto &ECI : EdgeCounts)
      Weights.push_back(scaleBranchCount(ECI, Scale));

    TI->setMetadata(llvm::LLVMContext::MD_prof,
                    MDB.createBranchWeights(Weights));
    DEBUG(dbgs() << "Weight is: ";
          for (const auto &W : Weights) { dbgs() << W << " "; }
          dbgs() << "\n";);
  }
}

// Traverse all the indirect callsites and annotate the instructions.
void PGOUseFunc::annotateIndirectCallSites() {
  if (DisableValueProfiling)
    return;

  unsigned IndirectCallSiteIndex = 0;
  PGOIndirectCallSiteVisitor ICV;
  ICV.visit(F);
  unsigned NumValueSites=
      ProfileRecord.getNumValueSites(IPVK_IndirectCallTarget);
  if (NumValueSites != ICV.IndirectCallInsts.size()) {
    std::string Msg =
        std::string("Inconsistent number of indirect call sites: ") +
            F.getName().str();
    auto &Ctx = M->getContext();
    Ctx.diagnose(
        DiagnosticInfoPGOProfile(M->getName().data(), Msg, DS_Warning));
    return;
  }

  for (auto &I : ICV.IndirectCallInsts) {
    DEBUG(dbgs() << "Read one indirect call instrumentation: Index="
                 << IndirectCallSiteIndex << " out of "
                 << NumValueSites<< "\n");
    annotateValueSite(*M, *I, ProfileRecord, IPVK_IndirectCallTarget,
                      IndirectCallSiteIndex);
    IndirectCallSiteIndex++;
  }
}
} // end anonymous namespace

// Create a COMDAT variable IR_LEVEL_PROF_VARNAME to make the runtime
// aware this is an ir_level profile so it can set the version flag.
static void createIRLevelProfileFlagVariable(Module &M) {
  Type *IntTy64 = Type::getInt64Ty(M.getContext());
  uint64_t ProfileVersion = (INSTR_PROF_RAW_VERSION | VARIANT_MASK_IR_PROF);
  auto IRLevelVersionVariable =
      new GlobalVariable(M, IntTy64, true, GlobalVariable::ExternalLinkage,
                         Constant::getIntegerValue(IntTy64, APInt(64, ProfileVersion)),
                         INSTR_PROF_QUOTE(IR_LEVEL_PROF_VERSION_VAR));
  IRLevelVersionVariable->setVisibility(GlobalValue::DefaultVisibility);
  Triple TT(M.getTargetTriple());
  if (TT.isOSBinFormatMachO())
    IRLevelVersionVariable->setLinkage(GlobalValue::LinkOnceODRLinkage);
  else
    IRLevelVersionVariable->setComdat(
        M.getOrInsertComdat(StringRef(INSTR_PROF_QUOTE(IR_LEVEL_PROF_VERSION_VAR))));
}

bool PGOInstrumentationGen::runOnModule(Module &M) {
  createIRLevelProfileFlagVariable(M);
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    BranchProbabilityInfo *BPI =
        &(getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI());
    BlockFrequencyInfo *BFI =
        &(getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI());
    instrumentOneFunc(F, &M, BPI, BFI);
  }
  return true;
}

static void setPGOCountOnFunc(PGOUseFunc &Func,
                              IndexedInstrProfReader *PGOReader) {
  if (Func.readCounters(PGOReader)) {
    Func.populateCounters();
    Func.setBranchWeights();
    Func.annotateIndirectCallSites();
  }
}

bool PGOInstrumentationUse::runOnModule(Module &M) {
  DEBUG(dbgs() << "Read in profile counters: ");
  auto &Ctx = M.getContext();
  // Read the counter array from file.
  auto ReaderOrErr = IndexedInstrProfReader::create(ProfileFileName);
  if (std::error_code EC = ReaderOrErr.getError()) {
    Ctx.diagnose(
        DiagnosticInfoPGOProfile(ProfileFileName.data(), EC.message()));
    return false;
  }

  PGOReader = std::move(ReaderOrErr.get());
  if (!PGOReader) {
    Ctx.diagnose(DiagnosticInfoPGOProfile(ProfileFileName.data(),
                                          "Cannot get PGOReader"));
    return false;
  }
  // TODO: might need to change the warning once the clang option is finalized.
  if (!PGOReader->isIRLevelProfile()) {
    Ctx.diagnose(DiagnosticInfoPGOProfile(
        ProfileFileName.data(), "Not an IR level instrumentation profile"));
    return false;
  }


  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    BranchProbabilityInfo *BPI =
        &(getAnalysis<BranchProbabilityInfoWrapperPass>(F).getBPI());
    BlockFrequencyInfo *BFI =
        &(getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI());
    PGOUseFunc Func(F, &M, BPI, BFI);
    setPGOCountOnFunc(Func, PGOReader.get());
  }
  return true;
}
