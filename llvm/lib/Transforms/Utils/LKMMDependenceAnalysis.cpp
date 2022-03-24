//===- LKMMDependenceAnalaysis.cpp - LKMM Deps Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements two passes to determine whether data, addr and ctrl
/// dependencies were preserved according to the Linux kernel memory model.
///
/// The first pass annotates relevant dependencies in unoptimized IR and the
/// second pass verifies that the dependenices still hold in optimized IR.
///
/// Linux kernel memory model:
/// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/memory-model/Documentation/explanation.txt
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LKMMDependenceAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

// TODO add clang support for clang -fsanitize=lkmm
// TODO strict and relaxed mode

namespace llvm {
namespace {

#define INTERPROCEDURAL_REC_LIMIT_ANNOTATION 3
#define INTERPROCEDURAL_REC_LIMIT_VERIFICATION 4

// Represents a map of IDs to (potential) dependency halfs.
template <typename T> using DepHalfMap = std::unordered_map<std::string, T>;

// The DepChain type alias reprsents a dependency chain. Even though the term
// 'chain' suggests ordering, an unordered set fits better here as a), the
// algorithm doesn't require the dep chain to be ordered, and b), an unordered
// set allows constant-time lookup on average.
using DepChain = std::unordered_set<Value *>;

// The DepChainPair type alias reprsents the pair (DCUnion, DCInter) of
// dependency chains. It exists for every BB the dependency chains of a given
// potential address dependency beginning visit.
using DepChainPair = std::pair<DepChain, DepChain>;

// The DepChainMap type alias represents the map of BBs to dep chain pairs.
// Such a map exists for every potential addr dep beginning.
using DepChainMap = std::unordered_map<BasicBlock *, DepChainPair>;

using VerIDSet = std::unordered_set<std::string>;

using CallPathStack = std::list<CallInst *>;

using BBtoBBSetMap =
    std::unordered_map<BasicBlock *, std::unordered_set<BasicBlock *>>;

//===----------------------------------------------------------------------===//
// The BFS BB Info Struct
//===----------------------------------------------------------------------===//

struct BFSBBInfo {
  // The BB the two fields below relate to.
  BasicBlock *BB;

  // Denotes the amount of predeceessors which must be visited before the BFS
  // can look at 'BB'.
  unsigned MaxHits;

  // Denotes the amount of predecessors the BFS has already seen (or how many
  // times 'BB' has been 'hit' by an edge from a predecessor).
  unsigned CurrentHits;

  BFSBBInfo(BasicBlock *BB, unsigned MaxHits)
      : BB(BB), MaxHits(MaxHits), CurrentHits(0) {}
};

//===----------------------------------------------------------------------===//
// The Dependency Half Hierarchy
//===----------------------------------------------------------------------===//

class DepHalf {
public:
  enum DepKind {
    DK_AddrBeg,
    DK_CtrlBeg,
    DK_VerAddrBeg,
    DK_VerAddrEnd,
    DK_VerCtrlBeg,
    DK_VerCtrlEnd
  };

  /// Returns the ID of this DepHalf.
  ///
  /// \returns the DepHalf's ID.
  std::string getID() const;

  /// Returns a string representation of the path the annotation pass took
  /// to discover this DepHalf.
  ///
  /// \returns a string representation of the path the annotation pass took
  ///  to discover this DepHalf.
  std::string getPathTo() const { return PathTo; };

  DepKind getKind() const { return Kind; }

protected:
  // Instruction which this potential dependency beginning / ending relates to.
  Instruction *const I;

  // An ID which makes this dependency half unique and is used for annotation /
  // verification of dependencies. IDs are represented by a string
  // representation of the calls the BFS took to reach Inst, including inst, and
  // are assumed to be unique within the BFS.
  const std::string PathTo;

  DepHalf(Instruction *I, std::string PathTo, DepKind Kind)
      : I(I), PathTo(PathTo), Kind(Kind){};

  virtual ~DepHalf() {}

private:
  DepKind Kind;
};

class PotAddrDepBeg : public DepHalf {
public:
  // Copy constructor for copying a returned PotAddrDepBeg into the calling
  // context.
  PotAddrDepBeg(PotAddrDepBeg &ADB, BasicBlock *BB, DepChain DC)
      : PotAddrDepBeg(ADB) {
    DCM.clear();
    DCM.emplace(BB, DepChainPair(DC, DC));
  }

  PotAddrDepBeg(Instruction *I, std::string PathTo, Value *V, bool FDep = true)
      : PotAddrDepBeg(I, PathTo, DepChain{V}, FDep, I->getParent()) {}

  PotAddrDepBeg(Instruction *I, std::string PathTo, DepChain DC, bool FDep,
                BasicBlock *BB)
      : DepHalf(I, PathTo, DK_AddrBeg), DCM(), FDep(FDep) {
    DCM.emplace(BB, DepChainPair{DC, DC});
  }

  /// Checks whether a DepChainPair is currently at a given BB.
  ///
  /// \param BB the BB to be checked.
  ///
  /// \returns true if the PotAddrDepBeg has dep chains at \p BB.
  bool isAt(BasicBlock *BB) const { return DCM.find(BB) != DCM.end(); }

  /// Checks whether this PotAddrDepBeg begins at a given instruction.
  ///
  /// \param I the instruction to be checked.
  ///
  /// \returns true if \p this begins at \p I.
  bool beginsAt(Instruction *I) const { return I == this->I; }

  /// Updates the dep chain map after the BFS has visitied a given BB with a
  /// given succeeding BB.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param SBB one of BB's successors.
  /// \param BEDsForBB the back edge destination map.
  void progressDCPaths(BasicBlock *BB, BasicBlock *SBB,
                       BBtoBBSetMap &BEDsForBB);

  /// Tries to delete DepChains if possible. Needed for a), keeping track of how
  /// many DepChains are still valid, and b), saving space.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param BEDsForBB the back edge destination for \p BB.
  void deleteDCsAt(BasicBlock *BB, std::unordered_set<BasicBlock *> &BEDs);

  /// Tries to add a value to the intersection of all DepChains at \p BB.
  ///
  /// \param BB the BB to whose dep chain intersection \p V should be
  ///  added.
  /// \param V the value to be added.
  void addToDCInter(BasicBlock *BB, Value *V);

  /// Tries to add a value to the union of all dep chains at \p BB.
  ///
  /// \param BB the BB to whose dep chain union \p V should be added.
  /// \param V the value to be added.
  void addToDCUnion(BasicBlock *BB, Value *V);

  /// Checks if a counter-argument for this beginning being a full dependency
  /// has been found yet.
  ///
  /// \returns false if a counter-argument for this beginning being a full
  ///  dependency has been found.
  bool canBeFullDependency() const { return FDep; }

  /// This function is called when the BFS is able to prove that any
  /// instruction it encounters after this call is not able to complete a full
  /// dependency to this beginning. This might be the case when the BFS has just
  /// seen a DepChain running through a back edge.
  void cannotBeFullDependencyAnymore() { FDep = false; }

  /// Tries to continue the DepChain with a new value.
  ///
  /// \param I the instruction which is currently being checked.
  /// \param ValCmp the value which might or might not be part of a DepChain.
  /// \param ValAdd the value to add if \p ValCmp is part of a DepChain.
  bool tryAddValueToDepChains(Instruction *I, Value *VCmp, Value *VAdd);

  /// Checks if a value is part of all dep chains starting at this
  /// PotAddrDepBeg. Can be used for checking whether an address dependency
  /// ending marks a full dependency to this PotAddrDepBeg.
  ///
  /// \param BB the BB the BFS is currently visiting.
  /// \param VCmp the value which might or might not be part of all dep
  ///  chains.
  ///
  /// \returns true if \p VCmp is part of all of the beginning's dep chains.
  bool belongsToAllDepChains(BasicBlock *BB, Value *VCmp) const;

  /// Checks if a value is part of any dep chain of an addr dep beginning.
  ///
  /// \param BB the BB the BFS is currently at.
  /// \param VCmp the value which might or might not be part of a dep chain.
  ///
  /// \returns true if \p VCmp belongs to at least one of the beginning's dep
  ///  chains.
  bool belongsToDepChain(BasicBlock *BB, Value *VCmp) const;

  /// Checks if a value is part of some, but not all, dep chains, starting at
  /// this potential beginning. Can be used for checking whether an address
  /// dependency ending marks a partial dependency to this PotAddrDepBeg.
  ///
  /// \param BB the BB the BFS is currently at.
  /// \param VCmp the value which might or might not be part of all dep
  ///  chains.
  ///
  /// \returns true if \p VCmp belongs to at least one, but not all, of this
  ///  PotAddrDepBeg's DepChains.
  bool belongsToSomeNotAllDepChains(BasicBlock *BB, Value *VCmp) const;

  /// Annotates an address dependency from a given ending to this beginning.
  ///
  /// \param PathTo2 the path the annotation pass took to discover the
  ///  ending.
  /// \param I2 the instruction where the address dependency ends.
  /// \param FDep set to true if this is a full address dependency.
  void addAdrDep(std::string PathTo2, Instruction *I2, bool FDep) const;

  /// Resets the DepChainMap to a new state and potentially alteres the
  /// possibility of this PotAddrDepBeg being the beginning of a full
  /// dependency. This functionality is required for interprocedural analysis,
  /// where a DepChain carries over, but should not be cluttered with values
  /// from previous function(s). In the case where not all DepChains of this
  /// PotAddrDepBeg carry over, this cannot mark the beginning of a full
  /// dependency in the called function anymore.
  ///
  /// \param BB The BB to reset the DepChainMap to.
  /// \param FDep The new \p FDep state.
  /// \param DC A DepChain to be used as initial value for the new DepChainPair
  /// at \p BB. In the interprocedural analysis case, \p DC will contain all
  /// function arguments which are part of a DepChain in the calling function.
  void resetDCMTo(BasicBlock *BB, bool FDep, DepChain &DC) {
    this->FDep = FDep;
    DCM.clear();
    DCM.emplace(BB, DepChainPair(DC, DC));
  }

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_AddrBeg;
  }

private:
  DepChainMap DCM;

  // Denotes whether a matching ending can be annotated as a full dependency. Is
  // set to false if the algorithm encounters something on the way that makes a
  // full dependency impossible, e.g. a backedge.
  bool FDep;

  friend class PotCtrlDepBeg;

  /// Helper function for progressDCPaths(). Used for computing an intersection
  /// of dep chains.
  ///
  /// \param DCs the list of (BasicBlock, DepChain) pairs wheere the DCs might
  ///  all contain \p V
  /// \param V the value to be checked.
  ///
  /// \returns true if \p V is present in all of \p DCs' dep chains.
  bool depChainsShareValue(std::list<std::pair<BasicBlock *, DepChain *>> &DCs,
                           Value *V) const;
};

class PotCtrlDepBeg : public DepHalf {
public:
  // Copy constructor for constructing a PotCtrlDep from a PotAddrDep. As
  // PotCtrlDep require the existence of a DepChain to a READ_ONCE(), marking
  // their beginning, there is no other way of constructing a PotCtrlDepBeg.
  PotCtrlDepBeg(PotAddrDepBeg &ADB, std::string PathToBranch,
                bool Resolvable = true)
      : PotCtrlDepBeg(ADB.I, ADB.PathTo, PathToBranch, Resolvable) {}

  /// Checks whether this PotCtrlDepBeg was recently discovered,
  /// i.e. if it doesn't maintain any paths yet.
  ///
  /// \returns true if this PotCtrlDepBeg was recently discovered.
  bool recentlyDiscovered() const { return CtrlPaths.empty(); }

  /// Updates the PotCtrlDepBeg to be unresolvable. This might be the case when
  /// not all ctrl paths carry over to a called function. Within the context(s)
  /// of the called function (and beyond), the PotCtrlDepBeg cannot be resolved
  /// as some CtrlPaths still reside in the calling function.
  void setCannotResolve() { Resolvable = false; }

  /// Checks if the PotCtrlDepBeg can be resolved right now.
  ///
  /// \returns true if it can be resolved.
  bool canResolve() const { return Resolvable; }

  /// Checks if this PotCtrlDepBeg has ctrl paths whose heads are at \p BB right
  /// now.
  ///
  /// \returns true if a ctrl path is at \p BB.
  bool isAt(BasicBlock *BB) const {
    return CtrlPaths.find(BB) != CtrlPaths.end();
  }

  /// Returns a string representation of the path the annotation pass took
  /// to discover the branch instruction.
  ///
  /// \returns a string representation of the path the annotation pass took
  ///  to discover the branch instruction.
  std::string const &getPathToBranch() const { return PathToBranch; };

  /// Maintains the ctrl paths at this PotCtrlDepBeg.
  ///
  /// \param BB the BB the BFS just ran on.
  /// \param SuccessorsWOBackEdges all successors of BB which are not
  ///  connected through back edges.
  /// \param HasBackEdges denotes whether any back edges start at \p BB.
  ///
  /// \returns true if the scope of this PotCtrlDepBeg has ended.
  bool
  progressCtrlPaths(BasicBlock *BB,
                    std::unordered_set<BasicBlock *> *SuccessorsWOBackEdges,
                    bool HasBackEdges);

  /// Annotates a ctrl dependency from a given ending to this beginning.
  ///
  /// \param PathTo2 the path the annotation pass took to discover \p I2.
  /// \param I2 the instruction where the ctrl dependency ends.
  void addCtrlDep(std::string PathTo2, Instruction *I2) const;

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_CtrlBeg;
  }

private:
  // A string representation of the path the BFS took to discover the branch
  // instruction.
  const std::string PathToBranch;

  // Set to true if this PotCtrlDepBeg cannot be resolved
  // in the current function, e.g. because the branch inst is located in a
  // function which calls the current function.
  bool Resolvable;

  // The set of paths which are induced by the branch instruction. Paths
  // are represented by their 'head', i.e. the BB they are currently at as per
  // the BFS.
  std::unordered_set<BasicBlock *> CtrlPaths;

  PotCtrlDepBeg(Instruction *I, std::string PathTo, std::string PathToBranch,
                bool Resolvable = true)
      : DepHalf(I, PathTo, DK_CtrlBeg), PathToBranch{PathToBranch},
        Resolvable{Resolvable}, CtrlPaths(){};
};

class VerDepHalf : public DepHalf {
public:
  std::string const &getParsedPathTo() const { return ParsedPathTo; }

  Instruction *const &getInst() const { return I; };

  virtual ~VerDepHalf(){};

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() >= DK_VerAddrBeg && VDH->getKind() <= DK_VerCtrlEnd;
  }

  std::string const &getParsedID() const { return ParsedID; }

protected:
  VerDepHalf(Instruction *I, std::string ParsedID, std::string PathTo,
             std::string ParsedPathTo, DepKind Kind)
      : DepHalf(I, PathTo, Kind), ParsedID(ParsedID), ParsedPathTo(PathTo) {}

private:
  // The ID which identifies the two metadata annotations for this dependency.
  const std::string ParsedID;

  // The PathTo which was attached to the metadata annotation, i.e. the
  // path to I in unoptimised IR.
  const std::string ParsedPathTo;
};

class VerAddrDepBeg : public VerDepHalf {
public:
  VerAddrDepBeg(Instruction *I, std::string ParsedID, std::string PathTo,
                std::string ParsedPathTo)
      : VerDepHalf(I, ParsedID, PathTo, ParsedPathTo, DK_VerAddrBeg) {}

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_VerAddrBeg;
  }
};

class VerCtrlDepBeg : public VerDepHalf {
public:
  VerCtrlDepBeg(Instruction *I, std::string ParsedID, std::string PathTo,
                std::string ParsedPathTo, std::string ParsedPathToBranch)
      : VerDepHalf(I, ParsedID, PathTo, ParsedPathTo, DK_VerCtrlBeg),
        ParsedPathToBranch(ParsedPathToBranch) {}

  const std::string &getParsedPathToBranch() const {
    return ParsedPathToBranch;
  }

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_VerCtrlBeg;
  }

private:
  // A string representation of the path the BFS took in unoptimized IR to
  // discover the branch instruction. Is retrived from the metadata annotation.
  const std::string ParsedPathToBranch;
};

class VerAddrDepEnd : public VerDepHalf {
public:
  VerAddrDepEnd(Instruction *I, std::string ParsedID, std::string PathTo,
                std::string ParsedPathTo, bool ParsedFDep)
      : VerDepHalf(I, ParsedID, PathTo, ParsedPathTo, DK_VerAddrEnd),
        ParsedFDep(ParsedFDep) {}

  const bool &getParsedFullDep() const { return ParsedFDep; }

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_VerAddrEnd;
  }

private:
  // Denotes whether the address dependency was annotated as a full dependency
  // or a partial dependency. The value is obtained from the metadata
  // annotation.
  const bool ParsedFDep;
};

class VerCtrlDepEnd : public VerDepHalf {
public:
  VerCtrlDepEnd(Instruction *I, std::string ParsedID, std::string PathTo,
                std::string ParsedPathTo)
      : VerDepHalf(I, ParsedID, PathTo, ParsedPathTo, DK_VerCtrlEnd) {}

  static bool classof(const DepHalf *VDH) {
    return VDH->getKind() == DK_VerCtrlEnd;
  }
};

using InterprocBFSRes =
    std::pair<DepHalfMap<PotAddrDepBeg>, DepHalfMap<PotCtrlDepBeg>>;

struct VerBFSResult {
  // Contains all unverified address dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs;
  // Contains all unverified address dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs;
  // Contains all unverified control dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerCtrlDepBeg>> BrokenCDBs;
  // Contains all unverified control dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerCtrlDepEnd>> BrokenCDEs;

  VerBFSResult(std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs,
               std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs,
               std::shared_ptr<DepHalfMap<VerCtrlDepBeg>> BrokenCDBs,
               std::shared_ptr<DepHalfMap<VerCtrlDepEnd>> BrokenCDEs)
      : BrokenADBs(BrokenADBs), BrokenADEs(BrokenADEs), BrokenCDBs(BrokenCDBs),
        BrokenCDEs(BrokenCDEs) {}
};

//===----------------------------------------------------------------------===//
// The BFS Context Hierarchy
//===----------------------------------------------------------------------===//

// A BFSCtx contains all the info the BFS requires to traverse the CFG.

class BFSCtx : public InstVisitor<BFSCtx> {
public:
  enum CtxKind { CK_Annot, CK_Ver };

  CtxKind getKind() const { return Kind; }

  BFSCtx(BasicBlock *BB, CtxKind CK)
      : BB(BB), ADBs(), CDBs(), ReturnedADBs(), ReturnedCDBs(),
        CallPath(new CallPathStack()), Kind(CK){};

  virtual ~BFSCtx() {
    if (!CallPath->empty())
      CallPath->pop_back();
  }

  /// Runs the BFS algorithm in the given context. This function is called at
  /// the beginning of any function including those which are encountered
  /// through interprocedural analysis.
  void runBFS();

  /// Update all PotAddrDepBegs in the current context after a BasicBlock has
  /// been visited by the BFS. 'Updating' referes to moving the DepChains along
  /// to successors of the BB the BFS just visited.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param SBB one of \p BB's successors
  /// \param BEDsForBB the back edge destination map.
  void progressAddrDepDCPaths(BasicBlock *BB, BasicBlock *SBB,
                              BBtoBBSetMap &BEDsForBB);

  /// Tries to delete unused DepChains for all PotAddrDepBegs in
  /// the current context.
  ///
  /// \param BB the BB the BFS just visited.
  /// \param BEDs the set of back edge destinations for \p BB.
  void deleteAddrDepDCsAt(BasicBlock *BB,
                          std::unordered_set<BasicBlock *> &BEDs);

  /// Updates all PotCtrlDepBegs' paths to reflect the state after the BFS
  /// visited \p BB.
  ///
  /// \param BB the BasicBlock the BFS just visited.
  /// \param SuccessorsWOBackEdges all successors of \p BB which are not
  ///  connected through back edges.
  /// \param HasBackEdges denotes whether any back edges start at \p BB.
  void handleCtrlPaths(BasicBlock *BB,
                       std::unordered_set<BasicBlock *> *SuccessorsWOBackEdges,
                       bool HasBackEdges);

  /// Checks if a function call has arguments which are part of DepChains in the
  /// current context. This function is expected to be called at the beginning
  /// of an interprocedural analysis and might reset DepChains if they don't run
  /// through any of the call's arguments.
  ///
  /// \param CI the function call to be checked.
  /// \param ADBs the current ADBs.
  /// \param ADBsForCall the PotAddrDepBegs which will be
  ///  carried over to the called function. This map is left untouched if none
  ///  of the call's arguments are part of a DepChain.
  void handleDependentFunctionArgs(CallInst *CI, BasicBlock *BB);

  //===--------------------------------------------------------------------===//
  // Visitor Functions
  //===--------------------------------------------------------------------===//

  // In order for the BFS to traverse the CFG easily, BFSCtx implements the
  // InstVisitor pattern with a general instruction case, several concrete
  // cases as well as several excluded cases.

  /// Visits a Basic Block in the BFS. Updates the BB field in the current
  /// BFSCtx.
  ///
  /// \param BB the BasicBlock to be visited.
  void visitBasicBlock(BasicBlock &BB);

  /// Visits any instruction which does not match any concrete or excluded case
  /// below. Checks if any dependency chains run through \p I.
  ///
  /// \param I the instruction which did not match any concrete or excluded
  ///  case.
  void visitInstruction(Instruction &I);

  /// Visits a call instruction. Starts interprocedural analysis if possible.
  ///
  /// \param CallI the call instruction.
  void visitCallInst(CallInst &CallI);

  /// Visits a load instruction. Checks if a DepChain runs through \p LoadI
  /// or if \p LoadI marks a (potential) addr dep beginning / ending.
  ///
  /// \param LoadI the load instruction.
  void visitLoadInst(LoadInst &LoadI);

  /// Visits a store instruction. Checks if a DepChain runs through \p
  /// StoreI, \p StoreI redefines a value in an existing DepChain and if it
  /// marks the ending of a dependency.
  ///
  /// \param StoreI the store instruction.
  void visitStoreInst(StoreInst &StoreI);

  /// Visits a branch instruction. Checks if it marks the beginning of a ctrl
  /// dependency.
  ///
  /// \param BranchI the branch instruction.
  void visitBranchInst(BranchInst &BranchI);

  /// Visits a PHI instruction. Checks if a DepChain runs through the PHI
  /// instruction, and if that's the case, marks it as conditional if not all
  /// incoming values are part of the DepChain.
  ///
  /// \param PhiI the PHI instruction.
  void visitPHINode(PHINode &PhiI);

  /// Visits a switch instruction. Checks if it belongs marks a PotCtrlDepBeg.
  ///
  /// \param SwitchI the branch instruction.
  void visitSwitchInst(SwitchInst &SwitchI);

  /// Visits a return instruction. If visitReturnInst() is called in an
  /// interprocedural context, it handles the returned potential dependency
  /// beginnings. Assumes that only one ReturnInst exists per function.
  ///
  /// \param ReturnI the return instruction.
  void visitReturnInst(ReturnInst &ReturnI);

  // Excluded cases. Instructions of the below types are not handled as they
  // don't affect DepChains.

  /// Excluded.
  void visitAllocaInst(AllocaInst &AllocaI);
  /// Excluded.
  void visitFenceInst(FenceInst &FenceI);
  /// Excluded.
  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &AtomicCmpXchgI);
  /// Excluded.
  void visitAtomicRMWInst(AtomicRMWInst &AtomicRMWI);
  /// Excluded.
  void visitFuncletPadInst(FuncletPadInst &FuncletPadI);
  /// Excluded.
  void visitTerminator(Instruction &TermI);

  // For shared functionality between the visitor functions, BFSCtx
  // provides two internal helper functions.

  /// Shared functionality between visitLoadInstruction() and
  /// visitStoreInstruction().
  ///
  /// \param LoadStoreI the load / store Inst.
  void handleLoadStoreInst(Instruction &LoadStoreI);

  /// Shared functionality between visitBranchInstruction() and
  /// visitSwitchInstruction(), i.e. all instruction which have multiple
  /// successors within a function's CFG.
  ///
  /// \param BranchI the branch / switch instruction.
  /// \param Cond the condition the successors depend on.
  void handleControlFlowInst(Instruction &BranchI, Value *Cond);

protected:
  // The BB the BFS is currently checking.
  BasicBlock *BB;

  // All potential address dependency beginnings (ADBs) which are being tracked.
  DepHalfMap<PotAddrDepBeg> ADBs;

  // All potential control dependency beginnings (CDBs) which are being tracked.
  DepHalfMap<PotCtrlDepBeg> CDBs;

  // Potential beginnings where the return value is part of the DepChain.
  DepHalfMap<PotAddrDepBeg> ReturnedADBs;

  // All PotCtrlDeps which begin in the current function and haven't been
  // resloed when the function returns.
  DepHalfMap<PotCtrlDepBeg> ReturnedCDBs;

  // The path which the BFS took to reach BB.
  std::shared_ptr<CallPathStack> CallPath;

  /// Prepares a newly created BFSCtx for interprocedural analysis.
  ///
  /// \param BB the first BasicBlock in the called function.
  /// \param CallI the call instruction whose called function begins with \p BB.
  void prepareInterproc(BasicBlock *BB, CallInst *CallI);

  /// Spawns an interprocedural BFS from the current context.
  ///
  /// \param FirstBB the first BasicBlock of the called function.
  /// \param CallI the call instructions which calls the function beginning with
  /// \p FirstBB.
  InterprocBFSRes runInterprocBFS(BasicBlock *FirstBB, CallInst *CallI);

  /// Helper function for handleDependentFunctionArgs(). Checks if all arguments
  /// of a function call are part of all of a given PotAddrDepBeg's DepChains
  /// and adds all depenet arguments to a given set. This function is used for
  /// determining whether a PotAddrDepBeg can carry over as potential full
  /// dependency beginning into the interprocedural analysis.
  ///
  /// \param ADB the PotAddrDepBeg in question.
  /// \param CallI the call instruction whose arguments should be checked
  ///  against \p ADB's dep chains.
  /// \param DependentArgs the set which will contain all dependent function
  ///  arguments on return.
  ///
  /// \returns true if all of \p CallI's arguments are part of all of \p ADB's
  ///  DepChains.
  bool areAllFunctionArgsPartOfAllDepChains(
      PotAddrDepBeg &ADB, CallInst *CallI,
      std::unordered_set<Value *> &DependentArgs);

  /// Returns the current limit for interprocedural annotation / verification
  ///
  /// \returns the maximum recursion level
  unsigned currentLimit() const;

  /// Returns string representation of the full path to an instructions, i.e. a
  /// concatenation of the path of calls the BFS took to discover \p I and the
  /// string representation of \p I's location in source code. Such a string is
  /// supposed to uniquely identify an instruction within the BFS.
  ///
  /// \param I the instruction whose full path should be returned.
  ///
  /// \returns a string representation of \p I's full path.
  std::string getFullPath(Instruction *I) {
    return convertPathToing() + getInstLocationString(I);
  }

  /// Converts BFS's call path, i.e. a list of call instructions, to a string.
  ///
  /// \returns a string represenation of \p CallPath.
  std::string convertPathToing() {
    std::string PathStr{""};

    for (auto &CallI : *CallPath)
      PathStr += (getInstLocationString(CallI) + "  ");

    return PathStr;
  }

  /// Returns the depth, i.e. number of calls, of the interprocedural
  /// analysis.
  ///
  /// \returns the number of call instructions the current BFS has already
  ///  followed.
  unsigned recLevel() { return CallPath->size(); }

  /// Checks whether the BFS can visit a given BB and adds it to the BFSQueue if
  /// this is the case.
  ///
  /// \param SBB the successor the BFS wants to visit.
  /// \param BFSInfo the BFS Info for the current function.
  /// \param BEDs the set of back edge destinations for the current BB.
  ///
  /// \returns true if the BFS has seen all of \p SBB's predecessors.
  bool bfsCanVisit(BasicBlock *SBB,
                   std::unordered_map<BasicBlock *, BFSBBInfo> &BFSInfo,
                   std::unordered_set<BasicBlock *> &BEDs);

  /// Parse a dependency annotation string into its individual components.
  ///
  /// \param Annot the dependency annotation string.
  ///
  /// \returns a vector of strings, representing the individual components of
  ///  the annotation string. (ID, type ...)
  void parseDepHalfString(StringRef Annot,
                          SmallVector<std::string, 5> &AnnotData);

  /// Populates a map of BBs to a set of BBs, representing the back edge
  /// destinations.
  ///
  /// \param BEDsForBB the set to be populated.
  /// \param F the current function.
  void buildBackEdgeMap(BBtoBBSetMap *BEDsForBB, Function *F);

  /// Populates a BFSInfo map at the beginning of a function in a BFS.
  ///
  /// \param BFSInfo the map to be populated.
  /// \param BEDsForBB a map of BBs to their back edge
  ///  destinations in \p F.
  /// \param F the current function.
  void buildBFSInfo(std::unordered_map<BasicBlock *, BFSBBInfo> *BFSInfo,
                    BBtoBBSetMap *BEDsForBB, Function *F);

  /// Removes back edges from an unordered set of successors, i.e. BasicBlocks.
  ///
  /// \param BB the BB whose successors this function is supposed to look at.
  /// \param BEDsForBB all successors of BB which are connected
  ///  through backedges.
  /// \param SuccessorsWOBackEdges the set of successors without backedges.
  ///  Assumed to be empty.
  void removeBackEdgesFromSuccessors(
      BasicBlock *BB, std::unordered_set<BasicBlock *> *BEDs,
      std::unordered_set<BasicBlock *> *SuccessorsWOBackEdges) {
    for (auto SBB : successors(BB))
      if (BEDs->find(SBB) == BEDs->end())
        SuccessorsWOBackEdges->insert(SBB);
  }

  /// Returns a string representation of an instruction's location in the form:
  /// <function_name>::<line>:<column>.
  ///
  /// \param I the instruction whose location string should be returned.
  ///
  /// \returns a string represenation \p I's location.
  std::string getInstLocationString(Instruction *I) {
    const llvm::DebugLoc &InstDebugLoc = I->getDebugLoc();

    if (!InstDebugLoc)
      return "no location";

    return I->getFunction()->getName().str() +
           "::" + std::to_string(InstDebugLoc.getLine()) + ":" +
           std::to_string(InstDebugLoc.getCol());
  }

  /// Returns a string representation of how an instruction was inlined in the
  /// form of: <fileN>::<lineN>:<columnN>...<file1>::<line1>:<column1>
  ///
  /// For the algorithm it is important that this representation matches that of
  /// \p convert_path_to_str().
  ///
  /// \param I the instruction whose inline string should be returned.
  ///
  /// \returns a string represenation of how \p I was inlined. The string is
  ///  empty if \p I didn't get inlined.
  std::string buildInlineString(Instruction *I);

private:
  const CtxKind Kind;
};

class AnnotCtx : public BFSCtx {
public:
  static bool classof(const BFSCtx *C) { return C->getKind() == CK_Annot; }

  AnnotCtx(BasicBlock *BB) : BFSCtx(BB, CK_Annot) {}

  // Creates an AnnotCtx for exploring a called function.
  // FIXME Nearly identical to VerCtx's copy constructor. Can we template this?
  AnnotCtx(AnnotCtx &AC, BasicBlock *BB, CallInst *CallI) : AnnotCtx(AC) {
    prepareInterproc(BB, CallI);
  }

private:
  /// Inserts the bugs in the testing functions. Will output to errs() if the
  /// desired annotation can't be found.
  ///
  /// \param F any testing function.
  /// \param IOpCode the type of Instruction whose dependency should be broken.
  ///  Can be Load or Store.
  /// \param AnnotationType the type of annotation to break, i.e. (addr + ctrl)
  ///  dep (beginning + ending).
  void insertBug(Function *F, Instruction::MemoryOps IOpCode,
                 std::string AnnotationType);
};

class VerCtx : public BFSCtx {
public:
  VerCtx(BasicBlock *BB, std::shared_ptr<IDReMap> RemappedIDs,
         std::shared_ptr<VerIDSet> VerifiedIDs)
      : BFSCtx(BB, CK_Ver),
        BrokenADBs(std::make_shared<DepHalfMap<VerAddrDepBeg>>()),
        BrokenADEs(std::make_shared<DepHalfMap<VerAddrDepEnd>>()),
        BrokenCDBs(std::make_shared<DepHalfMap<VerCtrlDepBeg>>()),
        BrokenCDEs(std::make_shared<DepHalfMap<VerCtrlDepEnd>>()),
        RemappedIDs(RemappedIDs), VerifiedIDs(VerifiedIDs) {}

  // Creates a VerCtx for exploring a called function.
  // FIXME Nearly identical to AnnotCtx's copy constructor. Can we template
  // this?
  VerCtx(VerCtx &VC, BasicBlock *BB, CallInst *CallI) : VerCtx(VC) {
    prepareInterproc(BB, CallI);
  }

  /// Responsible for handling an instruction with at least one '!annotation'
  /// type metadata node. Immediately returns if it doesn't find at least one
  /// dependency annotation.
  ///
  /// \param I the instruction which has at least one dependency annotation
  ///  attached.
  /// \param MDAnnotation a pointer to the \p MDNode containing the dependency
  ///  annotation(s).
  void handleDepAnnotations(Instruction *I, MDNode *MDAnnotation);

  /// Returns the result of a Verification BFS, i.e. the dependencies which
  /// couldn't be verified.
  ///
  /// \returns a struct containing DepHalfMaps of unverified address dependency
  /// beginnings / endings and ctrl dependency beginnings / endings.
  VerBFSResult getResult() {
    return VerBFSResult(BrokenADBs, BrokenADEs, BrokenCDBs, BrokenCDEs);
  };

  static bool classof(const BFSCtx *C) { return C->getKind() == CK_Ver; }

private:
  // Contains all unverified address dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepBeg>> BrokenADBs;
  // Contains all unverified address dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerAddrDepEnd>> BrokenADEs;
  // Contains all unverified control dependency beginning annotations.
  std::shared_ptr<DepHalfMap<VerCtrlDepBeg>> BrokenCDBs;
  // Contains all unverified control dependency ending annotations.
  std::shared_ptr<DepHalfMap<VerCtrlDepEnd>> BrokenCDEs;

  // All remapped IDs which were discovered from the current root function.
  std::shared_ptr<IDReMap> RemappedIDs;

  // Contains all IDs which have been verified in the current module.
  std::shared_ptr<VerIDSet> VerifiedIDs;

  /// Responsible for handling a single address dependency ending annotation.
  ///
  /// \param ID the ID of the address dependency.
  /// \param I the instruction the annotation was attached to, i.e. the
  ///  instruction where the address dependency ends.
  /// \param ParsedPathTo the path the annotation pass took to discover
  ///  \p Inst.
  /// \param ParsedFullDep set to true if the address dependency was annotated
  ///  as a full dependency.
  ///
  /// \returns true if the address dependency could be verified.
  bool HandleAddrDepID(std::string &ID, Instruction *I,
                       std::string &ParsedPathTo, bool ParsedFullDep);

  /// Responsible for handling a single control dependency ending annotation.
  ///
  /// \param ID the ID of the ctrl dep.
  /// \param I the instruction the annotation was attached to, i.e. the
  ///  instruction where the ctrl dep ends.
  /// \param ParsedPathTo the CallPath the annotation pass took to discover
  ///  \p I.
  ///
  /// \returns true if the ctrl dep could be verified.
  bool handleCtrlDepID(std::string &ID, Instruction *I,
                       std::string &ParsedPathTo);

  /// Responsible for updating an ID if the verification pass has encountered it
  /// before. Will add the updated ID to \p RemappedIDs.
  ///
  /// \param ID a reference to the ID which should be updated.
  void updateID(std::string &ID) {
    if (RemappedIDs->find(ID) == RemappedIDs->end()) {
      RemappedIDs->emplace(ID, std::unordered_set<std::string>{ID + "-1"});
      ID = ID + "-1";
    }

    else {
      RemappedIDs->at(ID).insert(
          ID + "-" + (std::to_string(RemappedIDs->at(ID).size() + 1)));
      ID = ID + "-" + (std::to_string(RemappedIDs->at(ID).size() + 1));
    }
  }
};

//===----------------------------------------------------------------------===//
// DepHalf Implementations
//===----------------------------------------------------------------------===//

std::string DepHalf::getID() const {
  if (isa<PotAddrDepBeg>(this))
    return PathTo;
  else if (auto PCDB = dyn_cast<PotCtrlDepBeg>(this))
    return PathTo + PCDB->getPathToBranch();
  else if (auto VDH = dyn_cast<VerDepHalf>(this))
    return VDH->getParsedID();
  else
    llvm_unreachable("unhandled case in getID");
}

//===----------------------------------------------------------------------===//
// PotAddrDepBeg Implementations
//===----------------------------------------------------------------------===//

void PotAddrDepBeg::progressDCPaths(BasicBlock *BB, BasicBlock *SBB,
                                    BBtoBBSetMap &BEDsForBB) {
  if (!isAt(BB))
    return;

  if (!isAt(SBB))
    DCM.insert({SBB, DepChainPair{}});

  auto &SDCP = DCM.at(SBB);

  // BB might not be the only predecessor of SBB. Build a list of all
  // preceeding dep chains.
  std::list<std::pair<BasicBlock *, DepChain *>> PDCs;

  // Populate PDCs and DCUnion.
  for (auto Pred : predecessors(SBB)) {
    // If Pred is connected to SBB via a back edge, skip.
    if (BEDsForBB.at(Pred).find(SBB) != BEDsForBB.at(Pred).end())
      continue;

    // If the DepChain don't run through Pred, skip.
    if (!isAt(Pred))
      continue;

    // Previous, i.e. Pred's, DepChainPair.
    auto &PDCP = DCM.at(Pred);

    // Insert preceeding DCunion into succeeding DCUnion.
    SDCP.second.insert(PDCP.second.begin(), PDCP.second.end());

    // Save preceeding DepChain for intersection.
    PDCs.emplace_back(Pred, &PDCP.first);
  }

  // FIXME When this if doesn't fire, depChainsShareValue() will make one
  //  unneccesary loop iteration.

  // If PDCs is empty, we are at the function entry:
  if (PDCs.empty()) {
    // 1. Intiialise PDCs with current DCUnion.
    SDCP.second.insert(DCM.at(BB).second.begin(), DCM.at(BB).second.end());

    // 2. Initialise SDCP's DCUnion with the current DCUnion.
    PDCs.emplace_back(BB, &DCM.at(BB).first);
  }

  // Update DCInter. Only add a value if it's present in every
  // preceeding DepChain.
  DepChain FixedDC = *PDCs.begin()->second;

  // If SDCP's DCInter isn't empty. Intersect succeeding DCInter with
  // current DCInter.
  if (!SDCP.first.empty()) {
    FixedDC = SDCP.first;
    SDCP.first.clear();
  }

  // Compute intersection of all dep chains leading to SBB.
  for (auto &V : FixedDC)
    // Add a value if it is present in all preceeding DepChains.
    if (depChainsShareValue(PDCs, V))
      SDCP.first.insert(V);
}

void PotAddrDepBeg::deleteDCsAt(BasicBlock *BB,
                                std::unordered_set<BasicBlock *> &BEDs) {
  if (!isAt(BB))
    return;

  if (!BEDs.empty() || isa<ReturnInst>(BB->getTerminator())) {
    // Keep the entry in DCM to account for 'dead' DepChain, but clear
    // them to save space.
    DCM.at(BB).first.clear();
    DCM.at(BB).second.clear();
  } else
    // If there's no dead DepChain, erase the DCM entry for the current BB.
    DCM.erase(BB);
}

void PotAddrDepBeg::addToDCInter(BasicBlock *BB, Value *V) {
  if (!isAt(BB))
    return;

  DCM.at(BB).first.insert(V);
}

void PotAddrDepBeg::addToDCUnion(BasicBlock *BB, Value *V) {
  if (!isAt(BB))
    return;

  DCM.at(BB).second.insert(V);
}

bool PotAddrDepBeg::tryAddValueToDepChains(Instruction *I, Value *VCmp,
                                           Value *VAdd) {
  if (!isAt(I->getParent()))
    return false;

  auto ret = false;

  auto &DCP = DCM.at(I->getParent());

  auto &DCInter = DCP.first;
  auto &DCUnion = DCP.second;

  // Add to DCinter and account for redefinition.
  if (DCInter.find(VCmp) != DCInter.end()) {
    DCInter.insert(VAdd);
    ret = true;
  } else if (isa<StoreInst>(I)) {
    auto *PotRedefOp = I->getOperand(1);
    if (DCInter.find(PotRedefOp) != DCInter.end())
      DCInter.erase(PotRedefOp);
  }

  // Add to DCUnion and account for redefinition
  if (DCUnion.find(VCmp) != DCUnion.end()) {
    DCUnion.insert(VAdd);
    ret = true;
  } else if (isa<StoreInst>(I)) {
    auto *PotRedefOp = I->getOperand(1);
    if (DCUnion.find(PotRedefOp) != DCUnion.end())
      DCUnion.erase(PotRedefOp);
  }

  return ret;
}

bool PotAddrDepBeg::belongsToAllDepChains(BasicBlock *BB, Value *VCmp) const {
  if (DCM.find(BB) == DCM.end())
    return false;

  auto &DCInter = DCM.at(BB).first;

  return DCInter.find(VCmp) != DCInter.end() && DCM.size() == 1;
}

bool PotAddrDepBeg::belongsToDepChain(BasicBlock *BB, Value *VCmp) const {
  if (DCM.find(BB) == DCM.end())
    return false;

  auto &DCUnion = DCM.at(BB).second;

  return DCUnion.find(VCmp) != DCUnion.end();
}

bool PotAddrDepBeg::belongsToSomeNotAllDepChains(BasicBlock *BB,
                                                 Value *VCmp) const {
  if (DCM.find(BB) == DCM.end())
    return false;

  return !belongsToAllDepChains(BB, VCmp) && belongsToDepChain(BB, VCmp);
}

void PotAddrDepBeg::addAdrDep(std::string PathTo2, Instruction *I2,
                              bool FDep) const {
  if (beginsAt(I2))
    return;

  auto ID = getID() + PathTo2;

  std::string begin_annotation =
      "DoitLk: address dep begin," + ID + "," + getPathTo() + ";";
  std::string end_annotation = "DoitLk: address dep end," + ID + "," + PathTo2 +
                               "," + std::to_string(FDep) + ";";

  I->addAnnotationMetadata(begin_annotation);
  I2->addAnnotationMetadata(end_annotation);
}

bool PotAddrDepBeg::depChainsShareValue(
    std::list<std::pair<BasicBlock *, DepChain *>> &DCs, Value *V) const {
  for (auto &DCP : DCs)
    if (DCP.second->find(V) == DCP.second->end())
      return false;

  return true;
}

//===----------------------------------------------------------------------===//
// PotCtrlDepBeg Implementations
//===----------------------------------------------------------------------===//

bool PotCtrlDepBeg::progressCtrlPaths(
    BasicBlock *BB, std::unordered_set<BasicBlock *> *SuccessorsWOBackEdges,
    bool HasBackEdges) {

  // Skip beginnigns that lie outside the current BB's function. Necessary
  // for interprocedural analysis.
  if (!canResolve())
    return false;

  // Account for the case where there are no paths at the current BB, but the
  // beginning isn't new.
  if (!recentlyDiscovered() && !isAt(BB))
    return false;

  // If the paths are empty, this dependency begins in BB. Only add
  // successors and continue. Otherwise, continue below.
  if (!recentlyDiscovered()) {
    // Erase paths which run through the current BB.
    // Only erase such paths if the current BB doesn't have a back edge or
    // contains a return. If it does, the path through the back edge /
    // reuturning block doesn't get resolved. Don't erase, only add successors.
    //
    // This check also accounts for 'dead ends', which never get resolved
    // either.
    if (!isa<ReturnInst>(BB->getTerminator()) && !HasBackEdges)
      CtrlPaths.erase({BB});

    // If a control dependency gets resolved, there must be a point where there
    // is only one path left here, which is at the BB resolving the ctrl dep.
    if (CtrlPaths.size() == 1)
      if (SuccessorsWOBackEdges->size() == 1)
        if (*SuccessorsWOBackEdges->begin() == *CtrlPaths.begin())
          return true;
  }

  // Add paths for successors. If a path already is at a successor, the two
  // paths merge.
  for (auto &s : *SuccessorsWOBackEdges)
    CtrlPaths.insert(s);

  return false;
}

void PotCtrlDepBeg::addCtrlDep(std::string PathTo2, Instruction *I2) const {
  auto ID = getID() + PathTo2;

  // begin annotation
  std::string BeginAnnotation = "DoitLk: ctrl dep begin," + ID + "," +
                                getPathTo() + "," + getPathToBranch() + ";";

  std::string EndAnnotation =
      "DoitLk: ctrl dep end," + ID + "," + PathTo2 + ";";

  I->addAnnotationMetadata(BeginAnnotation);
  I2->addAnnotationMetadata(EndAnnotation);
}

//===----------------------------------------------------------------------===//
// BFSCtx Implementations
//===----------------------------------------------------------------------===//

void BFSCtx::runBFS() {
  // Maps a BB to the set of its back edge destinations (BEDs).
  BBtoBBSetMap BEDsForBB;

  buildBackEdgeMap(&BEDsForBB, BB->getParent());

  std::unordered_map<BasicBlock *, BFSBBInfo> BFSInfo;

  buildBFSInfo(&BFSInfo, &BEDsForBB, BB->getParent());

  std::queue<BasicBlock *> BFSQueue = {};

  BFSQueue.push(BB);

  while (!BFSQueue.empty()) {
    auto &BB = BFSQueue.front();

    visit(BB);

    std::unordered_set<BasicBlock *> SuccessorsWOBackEdges{};

    removeBackEdgesFromSuccessors(BB, &BEDsForBB.at(BB),
                                  &SuccessorsWOBackEdges);

    handleCtrlPaths(BB, &SuccessorsWOBackEdges, !BEDsForBB.at(BB).empty());

    for (auto &SBB : SuccessorsWOBackEdges) {
      if (bfsCanVisit(SBB, BFSInfo, BEDsForBB.at(SBB)))
        BFSQueue.push(SBB);

      progressAddrDepDCPaths(BB, SBB, BEDsForBB);
    }

    deleteAddrDepDCsAt(BB, BEDsForBB.at(BB));

    BFSQueue.pop();
  }
}

void BFSCtx::progressAddrDepDCPaths(BasicBlock *BB, BasicBlock *SBB,
                                    BBtoBBSetMap &BEDsForBB) {
  for (auto &ADBP : ADBs)
    ADBP.second.progressDCPaths(BB, SBB, BEDsForBB);
}

void BFSCtx::deleteAddrDepDCsAt(BasicBlock *BB,
                                std::unordered_set<BasicBlock *> &BEDs) {
  for (auto &ADBP : ADBs)
    ADBP.second.deleteDCsAt(BB, BEDs);
}

void BFSCtx::handleCtrlPaths(BasicBlock *BB,
                             std::unordered_set<BasicBlock *> *succ_wo_back,
                             bool HasBackEdges) {
  for (auto cdb_it = CDBs.begin(), cdb_end = CDBs.end(); cdb_it != cdb_end;) {
    auto &CDB = cdb_it->second;
    if (CDB.progressCtrlPaths(BB, succ_wo_back, HasBackEdges))
      cdb_it = CDBs.erase(cdb_it);
    else
      ++cdb_it;
  }
}

void BFSCtx::handleDependentFunctionArgs(CallInst *CallI, BasicBlock *BB) {
  for (auto ADBP = ADBs.begin(), ADBEnd = ADBs.end(); ADBP != ADBEnd;) {
    DepChain DependentArgs;

    auto &ADB = ADBP->second;

    bool FDep = areAllFunctionArgsPartOfAllDepChains(ADB, CallI, DependentArgs);

    if (DependentArgs.empty())
      ADBP = ADBs.erase(ADBP);
    else {
      ADB.resetDCMTo(BB, FDep, DependentArgs);
      ++ADBP;
    }
  }
}

void BFSCtx::prepareInterproc(BasicBlock *BB, CallInst *CallI) {
  handleDependentFunctionArgs(CallI, BB);

  // Every unresolved PotCtrlDepBeg at CallI carries over to the called
  // function. Since the PotCtrlDepBeg is still being tracked, there exists at
  // least one ctrl path whose head is at a different BB than the one the BFS
  // is currently visiting. Therefore the PotCtrlDep cannot be resolved in the
  // called function.
  for (auto &CDBP : CDBs)
    CDBP.second.setCannotResolve();

  CallPath->push_back(CallI);

  this->BB = BB;
}

// FIXME Duplciate code
InterprocBFSRes BFSCtx::runInterprocBFS(BasicBlock *FirstBB, CallInst *CallI) {
  if (auto *AC = dyn_cast<AnnotCtx>(this)) {
    AnnotCtx InterprocCtx = AnnotCtx(*AC, FirstBB, CallI);
    InterprocCtx.runBFS();
    return InterprocBFSRes(std::move(InterprocCtx.ReturnedADBs),
                           std::move(InterprocCtx.ReturnedCDBs));
  } else if (auto *VC = dyn_cast<VerCtx>(this)) {
    VerCtx InterprocCtx = VerCtx(*VC, FirstBB, CallI);
    InterprocCtx.runBFS();
    return InterprocBFSRes(std::move(InterprocCtx.ReturnedADBs),
                           std::move(InterprocCtx.ReturnedCDBs));
  } else
    llvm_unreachable("Called runInterprocBFS() with no BFSCtx child.");
}

unsigned BFSCtx::currentLimit() const {
  if (isa<AnnotCtx>(this))
    return INTERPROCEDURAL_REC_LIMIT_ANNOTATION;
  else if (isa<VerCtx>(this))
    return INTERPROCEDURAL_REC_LIMIT_VERIFICATION;
  else
    llvm_unreachable("called currentLimit with unhandled subclass.");
}

bool BFSCtx::areAllFunctionArgsPartOfAllDepChains(
    PotAddrDepBeg &ADB, CallInst *CallI,
    std::unordered_set<Value *> &DependentArgs) {
  bool FDep = true;

  for (unsigned i = 0; i < CallI->arg_size(); ++i) {
    auto *VCmp = CallI->getArgOperand(i);

    if (!ADB.belongsToDepChain(BB, VCmp))
      continue;

    if (!ADB.belongsToAllDepChains(BB, CallI->getArgOperand(i)))
      FDep = false;

    DependentArgs.insert(CallI->getCalledFunction()->getArg(i));
  }

  return FDep;
}

bool BFSCtx::bfsCanVisit(BasicBlock *SBB,
                         std::unordered_map<BasicBlock *, BFSBBInfo> &BFSInfo,
                         std::unordered_set<BasicBlock *> &BEDs) {
  auto &NextMaxHits{BFSInfo.at(SBB).MaxHits};
  auto &NextCurrentHits{BFSInfo.at(SBB).CurrentHits};

  if (NextMaxHits == 0 || ++NextCurrentHits == NextMaxHits)
    return true;
  else
    return false;
}

void BFSCtx::parseDepHalfString(StringRef Annot,
                                SmallVector<std::string, 5> &AnnotData) {
  if (!Annot.consume_back(";"))
    return;

  while (!Annot.empty()) {
    auto P = Annot.split(",");
    AnnotData.push_back(P.first.str());
    Annot = P.second;
  }
}

void BFSCtx::buildBackEdgeMap(BBtoBBSetMap *BEDsForBB, Function *F) {
  // Initialise backEdges with all BB's and an empty set of back-edge
  // successors.
  for (auto &BB : *F)
    BEDsForBB->insert({&BB, {}});

  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>> backEdgeVector;
  FindFunctionBackedges(*F, backEdgeVector);

  for (auto &backEdge : backEdgeVector) {
    BEDsForBB->at(const_cast<BasicBlock *>(backEdge.first))
        .insert(const_cast<BasicBlock *>(backEdge.second));
  }
}

void BFSCtx::buildBFSInfo(std::unordered_map<BasicBlock *, BFSBBInfo> *BFSInfo,
                          BBtoBBSetMap *BEDsForBB, Function *F) {
  for (auto &BB : *F) {
    unsigned MaxHits{pred_size(&BB)};

    // Every incoming edge which is a back edge, i.e. closes a loop, is not
    // considered in MaxHits.
    for (auto Pred : predecessors(&BB))
      if (BEDsForBB->at(Pred).find(&BB) != BEDsForBB->at(Pred).end())
        --MaxHits;

    BFSInfo->emplace(&BB, BFSBBInfo(&BB, MaxHits));
  }
}

std::string BFSCtx::buildInlineString(Instruction *I) {
  auto InstDebugLoc = I->getDebugLoc();

  if (!InstDebugLoc)
    return "no debug loc when building inline string";

  std::string InlinePath{""};

  auto InlinedAt = InstDebugLoc.getInlinedAt();
  while (InlinedAt) {
    // Column.
    InlinePath = ":" + std::to_string(InlinedAt->getColumn()) + InlinePath;
    // Line.
    InlinePath = "::" + std::to_string(InlinedAt->getLine()) + InlinePath;
    // File name.
    InlinePath = InlinedAt->getFilename().str() + InlinePath;

    // Move to next InlinedAt if it exists.
    InlinedAt = InlinedAt->getInlinedAt();
  }
  return InlinePath;
}

//===----------------------------------------------------------------------===//
// BFSCtx Visitor Functions
//===----------------------------------------------------------------------===//

void BFSCtx::visitBasicBlock(BasicBlock &BB) { this->BB = &BB; }

void BFSCtx::visitInstruction(Instruction &I) {
  for (auto &adb_pair : ADBs)
    for (unsigned i = 0; i < I.getNumOperands(); ++i)
      adb_pair.second.tryAddValueToDepChains(&I, I.getOperand(i),
                                             cast<Value>(&I));
}

void BFSCtx::visitCallInst(CallInst &CallI) {
  auto *CalledF = CallI.getCalledFunction();

  if (!CalledF)
    return;

  if (CalledF->empty() || CalledF->isVarArg())
    return;

  if (recLevel() > currentLimit())
    return;

  InterprocBFSRes ret;

  if (isa<AnnotCtx>(this))
    auto ret = runInterprocBFS(&*CalledF->begin(), &CallI);
  else if (isa<VerCtx>(this))
    auto ret = runInterprocBFS(&*CalledF->begin(), &CallI);

  auto &RADBsFromCall = ret.first;
  auto &RCDBsFromCall = ret.second;

  // Handle returned addr deps.
  for (auto &RADBP : RADBsFromCall) {
    auto &ID = RADBP.first;
    auto &RADB = RADBP.second;

    if (ADBs.find(ID) != ADBs.end()) {
      auto &ADB = ADBs.at(ID);

      ADB.addToDCUnion(BB, cast<Value>(&CallI));

      // If not all dep chains from the beginning got returned, FDep might
      // have changed.
      if (RADB.canBeFullDependency())
        ADB.addToDCInter(BB, cast<Value>(&CallI));
      else
        ADB.cannotBeFullDependencyAnymore();
    } else
      ADBs.emplace(ID, PotAddrDepBeg(RADB, CallI.getParent(),
                                     DepChain{cast<Value>(&CallI)}));
  }

  for (auto &RCDBP : RCDBsFromCall) {
    auto &ID = RCDBP.first;
    auto &RCDB = RCDBP.second;

    CDBs.emplace(ID, RCDB);
    CDBs.at(ID).setCannotResolve();
  }
}

void BFSCtx::visitLoadInst(LoadInst &LoadI) {
  handleLoadStoreInst(LoadI);

  if (!LoadI.isVolatile())
    return;

  if (!isa<AnnotCtx>(this))
    return;

  auto ID = getFullPath(&LoadI);

  if (ADBs.find(ID) == ADBs.end())
    ADBs.emplace(ID, PotAddrDepBeg(&LoadI, ID, cast<Value>(&LoadI)));
}

void BFSCtx::visitStoreInst(StoreInst &StoreI) {
  handleLoadStoreInst(StoreI);

  if (!StoreI.isVolatile())
    return;

  // Check for control depenedency endings.
  if (isa<AnnotCtx>(this))
    for (auto &CDBP : CDBs) {
      auto &CDB = CDBP.second;
      if (CDB.isAt(BB))
        CDB.addCtrlDep(getFullPath(&StoreI), &StoreI);
    }
}

void BFSCtx::visitBranchInst(BranchInst &BranchI) {
  if (BranchI.isConditional())
    handleControlFlowInst(BranchI, BranchI.getCondition());
}

void BFSCtx::visitPHINode(PHINode &PhiI) {
  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;
    for (unsigned i = 0; i < PhiI.getNumIncomingValues(); ++i)
      if (!ADB.tryAddValueToDepChains(&PhiI, PhiI.getIncomingValue(i),
                                      cast<Value>(&PhiI)))
        ADB.cannotBeFullDependencyAnymore();
  }
}

void BFSCtx::visitSwitchInst(SwitchInst &SwitchI) {
  handleControlFlowInst(SwitchI, SwitchI.getCondition());
}

void BFSCtx::visitReturnInst(ReturnInst &RetI) {
  auto *RetVal = RetI.getReturnValue();

  if (!RetVal)
    return;

  if (!recLevel())
    return;

  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;

    if (!ADB.belongsToDepChain(BB, RetVal))
      continue;

    if (ADB.belongsToSomeNotAllDepChains(BB, RetVal))
      ADB.cannotBeFullDependencyAnymore();

    ReturnedADBs.emplace(ADBP.first, ADB);
  }

  for (auto &CDB : CDBs)
    ReturnedCDBs.emplace(CDB.first, CDB.second);
}

// Explicitly excluded cases.
void BFSCtx::visitAllocaInst(AllocaInst &AllocaI) {}
void BFSCtx::visitFenceInst(FenceInst &FenceI) {}
void BFSCtx::visitAtomicCmpXchgInst(AtomicCmpXchgInst &AtomicCmpXchgI) {}
void BFSCtx::visitAtomicRMWInst(AtomicRMWInst &AtomicRMWI) {}
void BFSCtx::visitFuncletPadInst(FuncletPadInst &FuncletPadI) {}
void BFSCtx::visitTerminator(Instruction &TermI) {}

void BFSCtx::handleLoadStoreInst(Instruction &I) {
  if (auto *VC = dyn_cast<VerCtx>(this))
    if (auto *MDAnnotation = I.getMetadata("annotation"))
      VC->handleDepAnnotations(&I, MDAnnotation);

  auto *VAdd = isa<StoreInst>(I) ? I.getOperand(1) : cast<Value>(&I);

  // Operand which can end an address dependency
  auto *VEnd = isa<StoreInst>(I) ? I.getOperand(1) : I.getOperand(0);

  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;

    if (I.isVolatile())
      if (isa<AnnotCtx>(this)) {
        if (ADB.belongsToAllDepChains(BB, VEnd))
          ADB.addAdrDep(getFullPath(&I), &I, true);
        else if (ADB.belongsToSomeNotAllDepChains(BB, VEnd))
          ADB.addAdrDep(getFullPath(&I), &I, false);
      }

    ADB.tryAddValueToDepChains(&I, I.getOperand(0), VAdd);
  }
}

void BFSCtx::handleControlFlowInst(Instruction &BranchI, Value *Cond) {
  for (auto &ADBP : ADBs) {
    auto &ADB = ADBP.second;

    if (ADB.belongsToDepChain(BB, Cond)) {
      auto ID = ADBP.first;

      CDBs.emplace(ID, PotCtrlDepBeg(ADB, getFullPath(&BranchI)));
    }
  }
}

//===----------------------------------------------------------------------===//
// AnnotCtx Implementations
//===----------------------------------------------------------------------===//

void AnnotCtx::insertBug(Function *F, Instruction::MemoryOps IOpCode,
                         std::string AnnotationType) {
  Instruction *InstWithAnnotation = nullptr;

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (auto md = I->getMetadata("annotation")) {
      if (cast<MDString>(md->getOperand(0))
              ->getString()
              .contains(AnnotationType) &&
          (I->getOpcode() == IOpCode)) {
        InstWithAnnotation = &*I;
        break;
      }
    }
  }

  if (!InstWithAnnotation) {
    errs() << "No annotations in testing function " << F->getName()
           << ". No bug was inserted.\n";
    return;
  }

  auto &InstContext = InstWithAnnotation->getContext();

  if (AnnotationType == "dep begin") {
    auto *BugVal1 = InstWithAnnotation->getModule()->getOrInsertGlobal(
        std::string("bugval1_") +
            std::string(InstWithAnnotation->getFunction()->getName()),
        Type::getInt32Ty(F->begin()->begin()->getContext()));

    auto *BugVal2 = InstWithAnnotation->getModule()->getOrInsertGlobal(
        std::string("bugval2_") +
            std::string(InstWithAnnotation->getFunction()->getName()),
        Type::getInt32PtrTy(F->begin()->begin()->getContext()));

    new StoreInst(ConstantInt::get(Type::getInt32Ty(InstContext), 1424242424),
                  cast<Value>(BugVal1), InstWithAnnotation);

    new StoreInst(BugVal1, cast<Value>(BugVal2), InstWithAnnotation);

    for (auto InstIt = InstWithAnnotation->getIterator(),
              InstEnd = InstWithAnnotation->getParent()->end();
         InstIt != InstEnd; ++InstIt)
      if ((InstIt->getOpcode() == Instruction::Store) &&
          (InstIt->getOperand(0) == cast<Value>(InstWithAnnotation))) {
        InstWithAnnotation = &*InstIt;
        break;
      }

    // Replace the source of the store to break the dependency chain.
    InstWithAnnotation->setOperand(0, BugVal2);
  } else {
    auto *GlobalBugVal = InstWithAnnotation->getModule()->getOrInsertGlobal(
        std::string("bugval_") +
            std::string(InstWithAnnotation->getFunction()->getName()),
        Type::getInt32Ty(F->begin()->begin()->getContext()));

    // Store 42 into Global BugVal just before our annotated load.
    new StoreInst(ConstantInt::get(Type::getInt32Ty(InstContext), 1424242424),
                  cast<Value>(GlobalBugVal), InstWithAnnotation);

    if (IOpCode == Instruction::Load) {
      // Update the source of our annotated load to be the global BugVal.
      InstWithAnnotation->setOperand(0, GlobalBugVal);
      // Set a new name.
      InstWithAnnotation->setName("new_ending");
    } else
      InstWithAnnotation->setOperand(1, GlobalBugVal);
  }
}

//===----------------------------------------------------------------------===//
// VerCtx Implementations
//===----------------------------------------------------------------------===//

bool VerCtx::HandleAddrDepID(std::string &ID, Instruction *I,
                             std::string &ParsedPathTo, bool ParsedFullDep) {
  auto *VCmp = isa<StoreInst>(I) ? I->getOperand(1) : I->getOperand(0);

  if (ADBs.find(ID) != ADBs.end()) {
    auto &ADB = ADBs.at(ID);

    if (ParsedFullDep && ADB.belongsToAllDepChains(BB, VCmp)) {
      BrokenADBs->erase(ID);
      return true;
    }

    if (!ParsedFullDep && ADB.belongsToDepChain(BB, VCmp)) {
      BrokenADBs->erase(ID);
      return true;
    }
  }

  BrokenADEs->emplace(
      ID, VerAddrDepEnd(I, ID, getFullPath(I), ParsedPathTo, ParsedFullDep));

  return false;
}

void VerCtx::handleDepAnnotations(Instruction *I, MDNode *MDAnnotation) {
  // For non-greedy verification
  std::unordered_set<int> AddedEndings{};

  for (auto &MDOp : MDAnnotation->operands()) {
    auto CurrentDepHalfStr = cast<MDString>(MDOp.get())->getString();

    if (!CurrentDepHalfStr.contains("DoitLk"))
      continue;

    SmallVector<std::string, 5> AnnotData;

    parseDepHalfString(CurrentDepHalfStr, AnnotData);

    auto &ParsedDepHalfTypeStr = AnnotData[0];
    auto &ParsedID = AnnotData[1];

    if (VerifiedIDs->find(ParsedID) != VerifiedIDs->end())
      continue;

    auto &ParsedPathTo = AnnotData[2];

    // Check if the instruction is the one to end to attached dependency.
    // If it wasn't inlined, this check is straightforward as we can just match
    // the source line and source column. If it was inlined, build the inline
    // string, check if PathTo ends with the inline string. If it doesn't, this
    // is the wrong instruction. Continue.
    auto InlinePath = buildInlineString(I);

    if (!InlinePath.empty() && !ParsedPathTo.empty()) {
      if ((InlinePath.length() > ParsedPathTo.length()) ||
          ParsedPathTo.compare(ParsedPathTo.length() - InlinePath.length(),
                               InlinePath.length(), InlinePath) != 0) {
        continue;
      }
    }

    if (ParsedDepHalfTypeStr.find("begin") != std::string::npos) {
      if (ADBs.find(ParsedID) != ADBs.end())
        updateID(ParsedID);

      // For tracking the dependency chain, always add a PotAddrDepBeg
      // beginning, no matter if the annotation concerns an address dependency
      // or control dependency beginning.
      ADBs.emplace(ParsedID, PotAddrDepBeg(I, getFullPath(I), cast<Value>(I)));

      if (ParsedDepHalfTypeStr.find("address dep") != std::string::npos)
        // Assume broken until proven wrong.
        BrokenADBs->emplace(
            ParsedID, VerAddrDepBeg(I, ParsedID, getFullPath(I), ParsedPathTo));
      else if (ParsedDepHalfTypeStr.find("ctrl dep") != std::string::npos) {
        auto &ParsedPathToBranch = AnnotData[3];

        // Assume broken until proven wrong.
        BrokenCDBs->emplace(ParsedID,
                            VerCtrlDepBeg(I, ParsedID, getFullPath(I),
                                          ParsedPathTo, ParsedPathToBranch));
      }
    } else if (ParsedDepHalfTypeStr.find("end") != std::string::npos) {
      // If we are able to verify one pair in
      // {ORIGINAL_ID} \cup REMAPPED_IDS.at(ORIGINAL_ID) x {ORIGINAL_ID}
      // We consider ORIGINAL_ID verified; there only exists one dependency in
      // unoptimised IR, hence we only look for one dependency in optimised IR.
      if (ParsedDepHalfTypeStr.find("address dep") != std::string::npos) {
        bool ParsedFullDep = std::stoi(AnnotData[3]);

        if (HandleAddrDepID(ParsedID, I, ParsedPathTo, ParsedFullDep)) {
          VerifiedIDs->insert(ParsedID);
          RemappedIDs->erase(ParsedID);
          continue;
        }

        if (RemappedIDs->find(ParsedID) != RemappedIDs->end())
          for (auto RemappedID : RemappedIDs->at(ParsedID))
            if (HandleAddrDepID(RemappedID, I, ParsedPathTo, ParsedFullDep)) {
              VerifiedIDs->insert(ParsedID);
              RemappedIDs->erase(ParsedID);
              continue;
            }
      } else if (ParsedDepHalfTypeStr.find("ctrl dep") != std::string::npos) {
        if (handleCtrlDepID(ParsedID, I, ParsedPathTo)) {
          VerifiedIDs->insert(ParsedID);
          RemappedIDs->erase(ParsedID);
          continue;
        }

        if (RemappedIDs->find(ParsedID) != RemappedIDs->end())
          for (auto RemappedID : RemappedIDs->at(ParsedID))
            if (handleCtrlDepID(RemappedID, I, ParsedPathTo)) {
              VerifiedIDs->insert(ParsedID);
              RemappedIDs->erase(ParsedID);
              continue;
            }
      }
    }
  }
}

bool VerCtx::handleCtrlDepID(std::string &ID, Instruction *I,
                             std::string &ParsedPathTo) {
  if (CDBs.find(ID) != CDBs.end()) {
    BrokenCDBs->erase(ID);
    return true;
  }

  BrokenCDEs->emplace(ID, VerCtrlDepEnd(I, ID, getFullPath(I), ParsedPathTo));

  return false;
}

} // namespace

//===----------------------------------------------------------------------===//
// The Annotation Pass
//===----------------------------------------------------------------------===//

PreservedAnalyses AnnotateLKMMDeps::run(Module &M, ModuleAnalysisManager &AM) {
  bool InsertedBugs = false;

  for (auto &F : M) {
    if (F.empty())
      continue;

    if (F.getParent()->getName() == "lib/modules/dep_chain_tests.c" &&
        F.getName() == "lkm_init") {
      continue;
    }

    // Check for multiple return statements
    // auto NumOfRets = 0;
    // for(auto &BB : F) {
    //   if(isa<ReturnInst>(BB.getTerminator()))
    //     NumOfRets++;
    // }

    // if(NumOfRets > 1) {
    //   errs() << "more than one return in " << F.getName() << "\n";
    //   F.print(errs());
    // }

    // assert(NumOfRets < 2 && "assert failed for more less than one return");

    // Annotate dependencies.
    AnnotCtx(&*F.begin()).runBFS();

    // Insert bugs if the BFS just annotated a testing function.
    // if (F.hasName()) {
    //   auto FName = F.getName();

    //   // Break beginnings.
    //   if (FName.contains("doitlk_rr_addr_dep_begin") ||
    //       FName.contains("doitlk_rw_addr_dep_begin") ||
    //       FName.contains("doitlk_ctrl_dep_begin")) {
    //     insertBug(&F, Instruction::Load, "dep begin");
    //     InsertedBugs = true;
    //   }

    //   // Break read -> read addr dep endings.
    //   else if (FName.contains("doitlk_rr_addr_dep_end")) {
    //     insertBug(&F, Instruction::Load, "dep end");
    //     InsertedBugs = true;
    //   }

    //   // Break read -> write addr dep and ctrl dep endings.
    //   else if (FName.contains("doitlk_rw_addr_dep_end") ||
    //            FName.contains("doitlk_ctrl_dep_end")) {
    //     insertBug(&F, Instruction::Store, "dep end");
    //     InsertedBugs = true;
    //   }
    // }
  }

  return InsertedBugs ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// The Verification Pass
//===----------------------------------------------------------------------===//

PreservedAnalyses VerifyLKMMDeps::run(Module &M, ModuleAnalysisManager &AM) {
  for (auto &F : M) {
    if (F.empty())
      continue;

    if (F.getParent()->getName() == "lib/modules/dep_chain_tests.c" &&
        F.getName() == "init_module")
      continue;

    auto VC = VerCtx(&*F.begin(), RemappedIDs, VerifiedIDs);

    VC.runBFS();

    auto BFSRes = VC.getResult();

    printBrokenDeps(&BFSRes);
  }

  return PreservedAnalyses::all();
}

void VerifyLKMMDeps::printBrokenDeps(VerBFSResult *IBFSRes) {
  auto &BrokenADBs = IBFSRes->BrokenADBs;
  auto &BrokenADEs = IBFSRes->BrokenADEs;
  auto &BrokenCDBs = IBFSRes->BrokenCDBs;
  auto &BrokenCDEs = IBFSRes->BrokenCDEs;

  auto checkDepPair = [this](auto &P, auto &E) {
    auto &ID = P.first;

    auto &VDB = P.second;

    auto VDEP = E->find(ID);

    if (VDEP == E->end())
      return;

    auto &VDE = VDEP->second;

    if (PrintedBrokenIDs.find(ID) != PrintedBrokenIDs.end())
      return;
    else
      PrintedBrokenIDs.insert(ID);

    printBrokenDep(VDB, VDE, ID);
  };

  for (auto &VADBP : *BrokenADBs)
    checkDepPair(VADBP, BrokenADEs);

  for (auto &VCDBP : *BrokenCDBs)
    checkDepPair(VCDBP, BrokenCDEs);
}

void VerifyLKMMDeps::printBrokenDep(VerDepHalf &Beg, VerDepHalf &End,
                                    const std::string &ID) {
  std::string DepKindStr{""};

  if (isa<VerAddrDepBeg>(Beg)) {
    DepKindStr = "Address dependency";
  } else if (isa<VerCtrlDepBeg>(Beg)) {
    DepKindStr = "Control dependency";
  }

  errs() << "==========\n";
  errs() << DepKindStr << " with ID " << ID
         << " couldn't be verified. It might have been broken.\n";
  errs() << "==========\n\n";

  errs() << "Dependency Beginning:\n";
  errs() << "source code path to beginning:\n\t" << Beg.getParsedPathTo()
         << "\n";
  if (auto *VCDB = dyn_cast<VerCtrlDepBeg>(&Beg)) {
    errs() << "source code path to branch:\n\t" << VCDB->getParsedPathToBranch()
           << "\n";
  }

  errs() << "\nDependnecy Ending:\n";
  errs() << "source code path to ending:\n\t" << End.getParsedPathTo() << "\n";
  if (auto *VADE = dyn_cast<VerAddrDepEnd>(&End))
    errs() << "Full dependency: " << (VADE->getParsedFullDep() ? "yes" : "no")
           << "\n";

  errs() << "\nFirst access(es) in optimised IR\n\n";

  errs() << "inst:\n\t";
  Beg.getInst()->print(errs());
  errs() << "\noptimised IR function:\n\t"
         << Beg.getInst()->getFunction()->getName() << "\n";

  errs() << "\n";

  errs() << "\nSecond access(es) in optimised IR\n\n";

  errs() << "inst:\n\t";
  End.getInst()->print(errs());
  errs() << "\noptimised IR function:\n\t"
         << End.getInst()->getFunction()->getName() << "\n";

  errs() << "\n";

  if (PrintedModules.find(Beg.getInst()->getModule()) == PrintedModules.end()) {
    errs() << "Optimised IR module:\n";
    Beg.getInst()->getModule()->print(errs(), nullptr);
    PrintedModules.insert(Beg.getInst()->getModule());
  }
}

} // namespace llvm