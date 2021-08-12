//===- DAGCombiner.cpp - Implement a DAG node combiner --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass combines dag nodes to form fewer, simpler DAG nodes.  It can be run
// both before and after the DAG is legalized.
//
// This pass is not a substitute for the LLVM IR instcombine pass. This pass is
// primarily intended to handle simplification opportunities that are implicit
// in the LLVM IR and exposed by the various codegen lowering phases.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/DAGCombine.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGAddressAnalysis.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "dagcombine"

STATISTIC(NodesCombined   , "Number of dag nodes combined");
STATISTIC(PreIndexedNodes , "Number of pre-indexed nodes created");
STATISTIC(PostIndexedNodes, "Number of post-indexed nodes created");
STATISTIC(OpsNarrowed     , "Number of load/op/store narrowed");
STATISTIC(LdStFP2Int      , "Number of fp load/store pairs transformed to int");
STATISTIC(SlicedLoads, "Number of load sliced");
STATISTIC(NumFPLogicOpsConv, "Number of logic ops converted to fp ops");

static cl::opt<bool>
CombinerGlobalAA("combiner-global-alias-analysis", cl::Hidden,
                 cl::desc("Enable DAG combiner's use of IR alias analysis"));

static cl::opt<bool>
UseTBAA("combiner-use-tbaa", cl::Hidden, cl::init(true),
        cl::desc("Enable DAG combiner's use of TBAA"));

#ifndef NDEBUG
static cl::opt<std::string>
CombinerAAOnlyFunc("combiner-aa-only-func", cl::Hidden,
                   cl::desc("Only use DAG-combiner alias analysis in this"
                            " function"));
#endif

/// Hidden option to stress test load slicing, i.e., when this option
/// is enabled, load slicing bypasses most of its profitability guards.
static cl::opt<bool>
StressLoadSlicing("combiner-stress-load-slicing", cl::Hidden,
                  cl::desc("Bypass the profitability model of load slicing"),
                  cl::init(false));

static cl::opt<bool>
  MaySplitLoadIndex("combiner-split-load-index", cl::Hidden, cl::init(true),
                    cl::desc("DAG combiner may split indexing from loads"));

static cl::opt<bool>
    EnableStoreMerging("combiner-store-merging", cl::Hidden, cl::init(true),
                       cl::desc("DAG combiner enable merging multiple stores "
                                "into a wider store"));

static cl::opt<unsigned> TokenFactorInlineLimit(
    "combiner-tokenfactor-inline-limit", cl::Hidden, cl::init(2048),
    cl::desc("Limit the number of operands to inline for Token Factors"));

static cl::opt<unsigned> StoreMergeDependenceLimit(
    "combiner-store-merge-dependence-limit", cl::Hidden, cl::init(10),
    cl::desc("Limit the number of times for the same StoreNode and RootNode "
             "to bail out in store merging dependence check"));

static cl::opt<bool> EnableReduceLoadOpStoreWidth(
    "combiner-reduce-load-op-store-width", cl::Hidden, cl::init(true),
    cl::desc("DAG cominber enable reducing the width of load/op/store "
             "sequence"));

static cl::opt<bool> EnableShrinkLoadReplaceStoreWithStore(
    "combiner-shrink-load-replace-store-with-store", cl::Hidden, cl::init(true),
    cl::desc("DAG cominber enable load/<replace bytes>/store with "
             "a narrower store"));

namespace {

  class DAGCombiner {
    SelectionDAG &DAG;
    const TargetLowering &TLI;
    const SelectionDAGTargetInfo *STI;
    CombineLevel Level;
    CodeGenOpt::Level OptLevel;
    bool LegalDAG = false;
    bool LegalOperations = false;
    bool LegalTypes = false;
    bool ForCodeSize;
    bool DisableGenericCombines;

    /// Worklist of all of the nodes that need to be simplified.
    ///
    /// This must behave as a stack -- new nodes to process are pushed onto the
    /// back and when processing we pop off of the back.
    ///
    /// The worklist will not contain duplicates but may contain null entries
    /// due to nodes being deleted from the underlying DAG.
    SmallVector<SDNode *, 64> Worklist;

    /// Mapping from an SDNode to its position on the worklist.
    ///
    /// This is used to find and remove nodes from the worklist (by nulling
    /// them) when they are deleted from the underlying DAG. It relies on
    /// stable indices of nodes within the worklist.
    DenseMap<SDNode *, unsigned> WorklistMap;
    /// This records all nodes attempted to add to the worklist since we
    /// considered a new worklist entry. As we keep do not add duplicate nodes
    /// in the worklist, this is different from the tail of the worklist.
    SmallSetVector<SDNode *, 32> PruningList;

    /// Set of nodes which have been combined (at least once).
    ///
    /// This is used to allow us to reliably add any operands of a DAG node
    /// which have not yet been combined to the worklist.
    SmallPtrSet<SDNode *, 32> CombinedNodes;

    /// Map from candidate StoreNode to the pair of RootNode and count.
    /// The count is used to track how many times we have seen the StoreNode
    /// with the same RootNode bail out in dependence check. If we have seen
    /// the bail out for the same pair many times over a limit, we won't
    /// consider the StoreNode with the same RootNode as store merging
    /// candidate again.
    DenseMap<SDNode *, std::pair<SDNode *, unsigned>> StoreRootCountMap;

    // AA - Used for DAG load/store alias analysis.
    AliasAnalysis *AA;

    /// When an instruction is simplified, add all users of the instruction to
    /// the work lists because they might get more simplified now.
    void AddUsersToWorklist(SDNode *N) {
      for (SDNode *Node : N->uses())
        AddToWorklist(Node);
    }

    /// Convenient shorthand to add a node and all of its user to the worklist.
    void AddToWorklistWithUsers(SDNode *N) {
      AddUsersToWorklist(N);
      AddToWorklist(N);
    }

    // Prune potentially dangling nodes. This is called after
    // any visit to a node, but should also be called during a visit after any
    // failed combine which may have created a DAG node.
    void clearAddedDanglingWorklistEntries() {
      // Check any nodes added to the worklist to see if they are prunable.
      while (!PruningList.empty()) {
        auto *N = PruningList.pop_back_val();
        if (N->use_empty())
          recursivelyDeleteUnusedNodes(N);
      }
    }

    SDNode *getNextWorklistEntry() {
      // Before we do any work, remove nodes that are not in use.
      clearAddedDanglingWorklistEntries();
      SDNode *N = nullptr;
      // The Worklist holds the SDNodes in order, but it may contain null
      // entries.
      while (!N && !Worklist.empty()) {
        N = Worklist.pop_back_val();
      }

      if (N) {
        bool GoodWorklistEntry = WorklistMap.erase(N);
        (void)GoodWorklistEntry;
        assert(GoodWorklistEntry &&
               "Found a worklist entry without a corresponding map entry!");
      }
      return N;
    }

    /// Call the node-specific routine that folds each particular type of node.
    SDValue visit(SDNode *N);

  public:
    DAGCombiner(SelectionDAG &D, AliasAnalysis *AA, CodeGenOpt::Level OL)
        : DAG(D), TLI(D.getTargetLoweringInfo()),
          STI(D.getSubtarget().getSelectionDAGInfo()),
          Level(BeforeLegalizeTypes), OptLevel(OL), AA(AA) {
      ForCodeSize = DAG.shouldOptForSize();
      DisableGenericCombines = STI && STI->disableGenericCombines(OptLevel);

      MaximumLegalStoreInBits = 0;
      // We use the minimum store size here, since that's all we can guarantee
      // for the scalable vector types.
      for (MVT VT : MVT::all_valuetypes())
        if (EVT(VT).isSimple() && VT != MVT::Other &&
            TLI.isTypeLegal(EVT(VT)) &&
            VT.getSizeInBits().getKnownMinSize() >= MaximumLegalStoreInBits)
          MaximumLegalStoreInBits = VT.getSizeInBits().getKnownMinSize();
    }

    void ConsiderForPruning(SDNode *N) {
      // Mark this for potential pruning.
      PruningList.insert(N);
    }

    /// Add to the worklist making sure its instance is at the back (next to be
    /// processed.)
    void AddToWorklist(SDNode *N) {
      assert(N->getOpcode() != ISD::DELETED_NODE &&
             "Deleted Node added to Worklist");

      // Skip handle nodes as they can't usefully be combined and confuse the
      // zero-use deletion strategy.
      if (N->getOpcode() == ISD::HANDLENODE)
        return;

      ConsiderForPruning(N);

      if (WorklistMap.insert(std::make_pair(N, Worklist.size())).second)
        Worklist.push_back(N);
    }

    /// Remove all instances of N from the worklist.
    void removeFromWorklist(SDNode *N) {
      CombinedNodes.erase(N);
      PruningList.remove(N);
      StoreRootCountMap.erase(N);

      auto It = WorklistMap.find(N);
      if (It == WorklistMap.end())
        return; // Not in the worklist.

      // Null out the entry rather than erasing it to avoid a linear operation.
      Worklist[It->second] = nullptr;
      WorklistMap.erase(It);
    }

    void deleteAndRecombine(SDNode *N);
    bool recursivelyDeleteUnusedNodes(SDNode *N);

    /// Replaces all uses of the results of one DAG node with new values.
    SDValue CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                      bool AddTo = true);

    /// Replaces all uses of the results of one DAG node with new values.
    SDValue CombineTo(SDNode *N, SDValue Res, bool AddTo = true) {
      return CombineTo(N, &Res, 1, AddTo);
    }

    /// Replaces all uses of the results of one DAG node with new values.
    SDValue CombineTo(SDNode *N, SDValue Res0, SDValue Res1,
                      bool AddTo = true) {
      SDValue To[] = { Res0, Res1 };
      return CombineTo(N, To, 2, AddTo);
    }

    void CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO);

  private:
    unsigned MaximumLegalStoreInBits;

    /// Check the specified integer node value to see if it can be simplified or
    /// if things it uses can be simplified by bit propagation.
    /// If so, return true.
    bool SimplifyDemandedBits(SDValue Op) {
      unsigned BitWidth = Op.getScalarValueSizeInBits();
      APInt DemandedBits = APInt::getAllOnesValue(BitWidth);
      return SimplifyDemandedBits(Op, DemandedBits);
    }

    bool SimplifyDemandedBits(SDValue Op, const APInt &DemandedBits) {
      TargetLowering::TargetLoweringOpt TLO(DAG, LegalTypes, LegalOperations);
      KnownBits Known;
      if (!TLI.SimplifyDemandedBits(Op, DemandedBits, Known, TLO, 0, false))
        return false;

      // Revisit the node.
      AddToWorklist(Op.getNode());

      CommitTargetLoweringOpt(TLO);
      return true;
    }

    /// Check the specified vector node value to see if it can be simplified or
    /// if things it uses can be simplified as it only uses some of the
    /// elements. If so, return true.
    bool SimplifyDemandedVectorElts(SDValue Op) {
      // TODO: For now just pretend it cannot be simplified.
      if (Op.getValueType().isScalableVector())
        return false;

      unsigned NumElts = Op.getValueType().getVectorNumElements();
      APInt DemandedElts = APInt::getAllOnesValue(NumElts);
      return SimplifyDemandedVectorElts(Op, DemandedElts);
    }

    bool SimplifyDemandedBits(SDValue Op, const APInt &DemandedBits,
                              const APInt &DemandedElts,
                              bool AssumeSingleUse = false);
    bool SimplifyDemandedVectorElts(SDValue Op, const APInt &DemandedElts,
                                    bool AssumeSingleUse = false);

    bool CombineToPreIndexedLoadStore(SDNode *N);
    bool CombineToPostIndexedLoadStore(SDNode *N);
    SDValue SplitIndexingFromLoad(LoadSDNode *LD);
    bool SliceUpLoad(SDNode *N);

    // Scalars have size 0 to distinguish from singleton vectors.
    SDValue ForwardStoreValueToDirectLoad(LoadSDNode *LD);
    bool getTruncatedStoreValue(StoreSDNode *ST, SDValue &Val);
    bool extendLoadedValueToExtension(LoadSDNode *LD, SDValue &Val);

    /// Replace an ISD::EXTRACT_VECTOR_ELT of a load with a narrowed
    ///   load.
    ///
    /// \param EVE ISD::EXTRACT_VECTOR_ELT to be replaced.
    /// \param InVecVT type of the input vector to EVE with bitcasts resolved.
    /// \param EltNo index of the vector element to load.
    /// \param OriginalLoad load that EVE came from to be replaced.
    /// \returns EVE on success SDValue() on failure.
    SDValue scalarizeExtractedVectorLoad(SDNode *EVE, EVT InVecVT,
                                         SDValue EltNo,
                                         LoadSDNode *OriginalLoad);
    void ReplaceLoadWithPromotedLoad(SDNode *Load, SDNode *ExtLoad);
    SDValue PromoteOperand(SDValue Op, EVT PVT, bool &Replace);
    SDValue SExtPromoteOperand(SDValue Op, EVT PVT);
    SDValue ZExtPromoteOperand(SDValue Op, EVT PVT);
    SDValue PromoteIntBinOp(SDValue Op);
    SDValue PromoteIntShiftOp(SDValue Op);
    SDValue PromoteExtend(SDValue Op);
    bool PromoteLoad(SDValue Op);

    /// Call the node-specific routine that knows how to fold each
    /// particular type of node. If that doesn't do anything, try the
    /// target-specific DAG combines.
    SDValue combine(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //   SDValue.getNode() == 0 - No change was made
    //   SDValue.getNode() == N - N was replaced, is dead and has been handled.
    //   otherwise              - N should be replaced by the returned Operand.
    //
    SDValue visitTokenFactor(SDNode *N);
    SDValue visitMERGE_VALUES(SDNode *N);
    SDValue visitADD(SDNode *N);
    SDValue visitADDLike(SDNode *N);
    SDValue visitADDLikeCommutative(SDValue N0, SDValue N1, SDNode *LocReference);
    SDValue visitSUB(SDNode *N);
    SDValue visitADDSAT(SDNode *N);
    SDValue visitSUBSAT(SDNode *N);
    SDValue visitADDC(SDNode *N);
    SDValue visitADDO(SDNode *N);
    SDValue visitUADDOLike(SDValue N0, SDValue N1, SDNode *N);
    SDValue visitSUBC(SDNode *N);
    SDValue visitSUBO(SDNode *N);
    SDValue visitADDE(SDNode *N);
    SDValue visitADDCARRY(SDNode *N);
    SDValue visitSADDO_CARRY(SDNode *N);
    SDValue visitADDCARRYLike(SDValue N0, SDValue N1, SDValue CarryIn, SDNode *N);
    SDValue visitSUBE(SDNode *N);
    SDValue visitSUBCARRY(SDNode *N);
    SDValue visitSSUBO_CARRY(SDNode *N);
    SDValue visitMUL(SDNode *N);
    SDValue visitMULFIX(SDNode *N);
    SDValue useDivRem(SDNode *N);
    SDValue visitSDIV(SDNode *N);
    SDValue visitSDIVLike(SDValue N0, SDValue N1, SDNode *N);
    SDValue visitUDIV(SDNode *N);
    SDValue visitUDIVLike(SDValue N0, SDValue N1, SDNode *N);
    SDValue visitREM(SDNode *N);
    SDValue visitMULHU(SDNode *N);
    SDValue visitMULHS(SDNode *N);
    SDValue visitSMUL_LOHI(SDNode *N);
    SDValue visitUMUL_LOHI(SDNode *N);
    SDValue visitMULO(SDNode *N);
    SDValue visitIMINMAX(SDNode *N);
    SDValue visitAND(SDNode *N);
    SDValue visitANDLike(SDValue N0, SDValue N1, SDNode *N);
    SDValue visitOR(SDNode *N);
    SDValue visitORLike(SDValue N0, SDValue N1, SDNode *N);
    SDValue visitXOR(SDNode *N);
    SDValue SimplifyVBinOp(SDNode *N);
    SDValue visitSHL(SDNode *N);
    SDValue visitSRA(SDNode *N);
    SDValue visitSRL(SDNode *N);
    SDValue visitFunnelShift(SDNode *N);
    SDValue visitRotate(SDNode *N);
    SDValue visitABS(SDNode *N);
    SDValue visitBSWAP(SDNode *N);
    SDValue visitBITREVERSE(SDNode *N);
    SDValue visitCTLZ(SDNode *N);
    SDValue visitCTLZ_ZERO_UNDEF(SDNode *N);
    SDValue visitCTTZ(SDNode *N);
    SDValue visitCTTZ_ZERO_UNDEF(SDNode *N);
    SDValue visitCTPOP(SDNode *N);
    SDValue visitSELECT(SDNode *N);
    SDValue visitVSELECT(SDNode *N);
    SDValue visitSELECT_CC(SDNode *N);
    SDValue visitSETCC(SDNode *N);
    SDValue visitSETCCCARRY(SDNode *N);
    SDValue visitSIGN_EXTEND(SDNode *N);
    SDValue visitZERO_EXTEND(SDNode *N);
    SDValue visitANY_EXTEND(SDNode *N);
    SDValue visitAssertExt(SDNode *N);
    SDValue visitAssertAlign(SDNode *N);
    SDValue visitSIGN_EXTEND_INREG(SDNode *N);
    SDValue visitEXTEND_VECTOR_INREG(SDNode *N);
    SDValue visitTRUNCATE(SDNode *N);
    SDValue visitBITCAST(SDNode *N);
    SDValue visitFREEZE(SDNode *N);
    SDValue visitBUILD_PAIR(SDNode *N);
    SDValue visitFADD(SDNode *N);
    SDValue visitSTRICT_FADD(SDNode *N);
    SDValue visitFSUB(SDNode *N);
    SDValue visitFMUL(SDNode *N);
    SDValue visitFMA(SDNode *N);
    SDValue visitFDIV(SDNode *N);
    SDValue visitFREM(SDNode *N);
    SDValue visitFSQRT(SDNode *N);
    SDValue visitFCOPYSIGN(SDNode *N);
    SDValue visitFPOW(SDNode *N);
    SDValue visitSINT_TO_FP(SDNode *N);
    SDValue visitUINT_TO_FP(SDNode *N);
    SDValue visitFP_TO_SINT(SDNode *N);
    SDValue visitFP_TO_UINT(SDNode *N);
    SDValue visitFP_ROUND(SDNode *N);
    SDValue visitFP_EXTEND(SDNode *N);
    SDValue visitFNEG(SDNode *N);
    SDValue visitFABS(SDNode *N);
    SDValue visitFCEIL(SDNode *N);
    SDValue visitFTRUNC(SDNode *N);
    SDValue visitFFLOOR(SDNode *N);
    SDValue visitFMINNUM(SDNode *N);
    SDValue visitFMAXNUM(SDNode *N);
    SDValue visitFMINIMUM(SDNode *N);
    SDValue visitFMAXIMUM(SDNode *N);
    SDValue visitBRCOND(SDNode *N);
    SDValue visitBR_CC(SDNode *N);
    SDValue visitLOAD(SDNode *N);

    SDValue replaceStoreChain(StoreSDNode *ST, SDValue BetterChain);
    SDValue replaceStoreOfFPConstant(StoreSDNode *ST);

    SDValue visitSTORE(SDNode *N);
    SDValue visitLIFETIME_END(SDNode *N);
    SDValue visitINSERT_VECTOR_ELT(SDNode *N);
    SDValue visitEXTRACT_VECTOR_ELT(SDNode *N);
    SDValue visitBUILD_VECTOR(SDNode *N);
    SDValue visitCONCAT_VECTORS(SDNode *N);
    SDValue visitEXTRACT_SUBVECTOR(SDNode *N);
    SDValue visitVECTOR_SHUFFLE(SDNode *N);
    SDValue visitSCALAR_TO_VECTOR(SDNode *N);
    SDValue visitINSERT_SUBVECTOR(SDNode *N);
    SDValue visitMLOAD(SDNode *N);
    SDValue visitMSTORE(SDNode *N);
    SDValue visitMGATHER(SDNode *N);
    SDValue visitMSCATTER(SDNode *N);
    SDValue visitFP_TO_FP16(SDNode *N);
    SDValue visitFP16_TO_FP(SDNode *N);
    SDValue visitVECREDUCE(SDNode *N);

    SDValue visitFADDForFMACombine(SDNode *N);
    SDValue visitFSUBForFMACombine(SDNode *N);
    SDValue visitFMULForFMADistributiveCombine(SDNode *N);

    SDValue XformToShuffleWithZero(SDNode *N);
    bool reassociationCanBreakAddressingModePattern(unsigned Opc,
                                                    const SDLoc &DL, SDValue N0,
                                                    SDValue N1);
    SDValue reassociateOpsCommutative(unsigned Opc, const SDLoc &DL, SDValue N0,
                                      SDValue N1);
    SDValue reassociateOps(unsigned Opc, const SDLoc &DL, SDValue N0,
                           SDValue N1, SDNodeFlags Flags);

    SDValue visitShiftByConstant(SDNode *N);

    SDValue foldSelectOfConstants(SDNode *N);
    SDValue foldVSelectOfConstants(SDNode *N);
    SDValue foldBinOpIntoSelect(SDNode *BO);
    bool SimplifySelectOps(SDNode *SELECT, SDValue LHS, SDValue RHS);
    SDValue hoistLogicOpWithSameOpcodeHands(SDNode *N);
    SDValue SimplifySelect(const SDLoc &DL, SDValue N0, SDValue N1, SDValue N2);
    SDValue SimplifySelectCC(const SDLoc &DL, SDValue N0, SDValue N1,
                             SDValue N2, SDValue N3, ISD::CondCode CC,
                             bool NotExtCompare = false);
    SDValue convertSelectOfFPConstantsToLoadOffset(
        const SDLoc &DL, SDValue N0, SDValue N1, SDValue N2, SDValue N3,
        ISD::CondCode CC);
    SDValue foldSignChangeInBitcast(SDNode *N);
    SDValue foldSelectCCToShiftAnd(const SDLoc &DL, SDValue N0, SDValue N1,
                                   SDValue N2, SDValue N3, ISD::CondCode CC);
    SDValue foldSelectOfBinops(SDNode *N);
    SDValue foldSextSetcc(SDNode *N);
    SDValue foldLogicOfSetCCs(bool IsAnd, SDValue N0, SDValue N1,
                              const SDLoc &DL);
    SDValue foldSubToUSubSat(EVT DstVT, SDNode *N);
    SDValue unfoldMaskedMerge(SDNode *N);
    SDValue unfoldExtremeBitClearingToShifts(SDNode *N);
    SDValue SimplifySetCC(EVT VT, SDValue N0, SDValue N1, ISD::CondCode Cond,
                          const SDLoc &DL, bool foldBooleans);
    SDValue rebuildSetCC(SDValue N);

    bool isSetCCEquivalent(SDValue N, SDValue &LHS, SDValue &RHS,
                           SDValue &CC, bool MatchStrict = false) const;
    bool isOneUseSetCC(SDValue N) const;

    SDValue SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                         unsigned HiOp);
    SDValue CombineConsecutiveLoads(SDNode *N, EVT VT);
    SDValue CombineExtLoad(SDNode *N);
    SDValue CombineZExtLogicopShiftLoad(SDNode *N);
    SDValue combineRepeatedFPDivisors(SDNode *N);
    SDValue combineInsertEltToShuffle(SDNode *N, unsigned InsIndex);
    SDValue ConstantFoldBITCASTofBUILD_VECTOR(SDNode *, EVT);
    SDValue BuildSDIV(SDNode *N);
    SDValue BuildSDIVPow2(SDNode *N);
    SDValue BuildUDIV(SDNode *N);
    SDValue BuildLogBase2(SDValue V, const SDLoc &DL);
    SDValue BuildDivEstimate(SDValue N, SDValue Op, SDNodeFlags Flags);
    SDValue buildRsqrtEstimate(SDValue Op, SDNodeFlags Flags);
    SDValue buildSqrtEstimate(SDValue Op, SDNodeFlags Flags);
    SDValue buildSqrtEstimateImpl(SDValue Op, SDNodeFlags Flags, bool Recip);
    SDValue buildSqrtNROneConst(SDValue Arg, SDValue Est, unsigned Iterations,
                                SDNodeFlags Flags, bool Reciprocal);
    SDValue buildSqrtNRTwoConst(SDValue Arg, SDValue Est, unsigned Iterations,
                                SDNodeFlags Flags, bool Reciprocal);
    SDValue MatchBSwapHWordLow(SDNode *N, SDValue N0, SDValue N1,
                               bool DemandHighBits = true);
    SDValue MatchBSwapHWord(SDNode *N, SDValue N0, SDValue N1);
    SDValue MatchRotatePosNeg(SDValue Shifted, SDValue Pos, SDValue Neg,
                              SDValue InnerPos, SDValue InnerNeg,
                              unsigned PosOpcode, unsigned NegOpcode,
                              const SDLoc &DL);
    SDValue MatchFunnelPosNeg(SDValue N0, SDValue N1, SDValue Pos, SDValue Neg,
                              SDValue InnerPos, SDValue InnerNeg,
                              unsigned PosOpcode, unsigned NegOpcode,
                              const SDLoc &DL);
    SDValue MatchRotate(SDValue LHS, SDValue RHS, const SDLoc &DL);
    SDValue MatchLoadCombine(SDNode *N);
    SDValue mergeTruncStores(StoreSDNode *N);
    SDValue ReduceLoadWidth(SDNode *N);
    SDValue ReduceLoadOpStoreWidth(SDNode *N);
    SDValue splitMergedValStore(StoreSDNode *ST);
    SDValue TransformFPLoadStorePair(SDNode *N);
    SDValue convertBuildVecZextToZext(SDNode *N);
    SDValue reduceBuildVecExtToExtBuildVec(SDNode *N);
    SDValue reduceBuildVecTruncToBitCast(SDNode *N);
    SDValue reduceBuildVecToShuffle(SDNode *N);
    SDValue createBuildVecShuffle(const SDLoc &DL, SDNode *N,
                                  ArrayRef<int> VectorMask, SDValue VecIn1,
                                  SDValue VecIn2, unsigned LeftIdx,
                                  bool DidSplitVec);
    SDValue matchVSelectOpSizesWithSetCC(SDNode *Cast);

    /// Walk up chain skipping non-aliasing memory nodes,
    /// looking for aliasing nodes and adding them to the Aliases vector.
    void GatherAllAliases(SDNode *N, SDValue OriginalChain,
                          SmallVectorImpl<SDValue> &Aliases);

    /// Return true if there is any possibility that the two addresses overlap.
    bool isAlias(SDNode *Op0, SDNode *Op1) const;

    /// Walk up chain skipping non-aliasing memory nodes, looking for a better
    /// chain (aliasing node.)
    SDValue FindBetterChain(SDNode *N, SDValue Chain);

    /// Try to replace a store and any possibly adjacent stores on
    /// consecutive chains with better chains. Return true only if St is
    /// replaced.
    ///
    /// Notice that other chains may still be replaced even if the function
    /// returns false.
    bool findBetterNeighborChains(StoreSDNode *St);

    // Helper for findBetterNeighborChains. Walk up store chain add additional
    // chained stores that do not overlap and can be parallelized.
    bool parallelizeChainedStores(StoreSDNode *St);

    /// Holds a pointer to an LSBaseSDNode as well as information on where it
    /// is located in a sequence of memory operations connected by a chain.
    struct MemOpLink {
      // Ptr to the mem node.
      LSBaseSDNode *MemNode;

      // Offset from the base ptr.
      int64_t OffsetFromBase;

      MemOpLink(LSBaseSDNode *N, int64_t Offset)
          : MemNode(N), OffsetFromBase(Offset) {}
    };

    // Classify the origin of a stored value.
    enum class StoreSource { Unknown, Constant, Extract, Load };
    StoreSource getStoreSource(SDValue StoreVal) {
      switch (StoreVal.getOpcode()) {
      case ISD::Constant:
      case ISD::ConstantFP:
        return StoreSource::Constant;
      case ISD::EXTRACT_VECTOR_ELT:
      case ISD::EXTRACT_SUBVECTOR:
        return StoreSource::Extract;
      case ISD::LOAD:
        return StoreSource::Load;
      default:
        return StoreSource::Unknown;
      }
    }

    /// This is a helper function for visitMUL to check the profitability
    /// of folding (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2).
    /// MulNode is the original multiply, AddNode is (add x, c1),
    /// and ConstNode is c2.
    bool isMulAddWithConstProfitable(SDNode *MulNode,
                                     SDValue &AddNode,
                                     SDValue &ConstNode);

    /// This is a helper function for visitAND and visitZERO_EXTEND.  Returns
    /// true if the (and (load x) c) pattern matches an extload.  ExtVT returns
    /// the type of the loaded value to be extended.
    bool isAndLoadExtLoad(ConstantSDNode *AndC, LoadSDNode *LoadN,
                          EVT LoadResultTy, EVT &ExtVT);

    /// Helper function to calculate whether the given Load/Store can have its
    /// width reduced to ExtVT.
    bool isLegalNarrowLdSt(LSBaseSDNode *LDSTN, ISD::LoadExtType ExtType,
                           EVT &MemVT, unsigned ShAmt = 0);

    /// Used by BackwardsPropagateMask to find suitable loads.
    bool SearchForAndLoads(SDNode *N, SmallVectorImpl<LoadSDNode*> &Loads,
                           SmallPtrSetImpl<SDNode*> &NodesWithConsts,
                           ConstantSDNode *Mask, SDNode *&NodeToMask);
    /// Attempt to propagate a given AND node back to load leaves so that they
    /// can be combined into narrow loads.
    bool BackwardsPropagateMask(SDNode *N);

    /// Helper function for mergeConsecutiveStores which merges the component
    /// store chains.
    SDValue getMergeStoreChains(SmallVectorImpl<MemOpLink> &StoreNodes,
                                unsigned NumStores);

    /// This is a helper function for mergeConsecutiveStores. When the source
    /// elements of the consecutive stores are all constants or all extracted
    /// vector elements, try to merge them into one larger store introducing
    /// bitcasts if necessary.  \return True if a merged store was created.
    bool mergeStoresOfConstantsOrVecElts(SmallVectorImpl<MemOpLink> &StoreNodes,
                                         EVT MemVT, unsigned NumStores,
                                         bool IsConstantSrc, bool UseVector,
                                         bool UseTrunc);

    /// This is a helper function for mergeConsecutiveStores. Stores that
    /// potentially may be merged with St are placed in StoreNodes. RootNode is
    /// a chain predecessor to all store candidates.
    void getStoreMergeCandidates(StoreSDNode *St,
                                 SmallVectorImpl<MemOpLink> &StoreNodes,
                                 SDNode *&Root);

    /// Helper function for mergeConsecutiveStores. Checks if candidate stores
    /// have indirect dependency through their operands. RootNode is the
    /// predecessor to all stores calculated by getStoreMergeCandidates and is
    /// used to prune the dependency check. \return True if safe to merge.
    bool checkMergeStoreCandidatesForDependencies(
        SmallVectorImpl<MemOpLink> &StoreNodes, unsigned NumStores,
        SDNode *RootNode);

    /// This is a helper function for mergeConsecutiveStores. Given a list of
    /// store candidates, find the first N that are consecutive in memory.
    /// Returns 0 if there are not at least 2 consecutive stores to try merging.
    unsigned getConsecutiveStores(SmallVectorImpl<MemOpLink> &StoreNodes,
                                  int64_t ElementSizeBytes) const;

    /// This is a helper function for mergeConsecutiveStores. It is used for
    /// store chains that are composed entirely of constant values.
    bool tryStoreMergeOfConstants(SmallVectorImpl<MemOpLink> &StoreNodes,
                                  unsigned NumConsecutiveStores,
                                  EVT MemVT, SDNode *Root, bool AllowVectors);

    /// This is a helper function for mergeConsecutiveStores. It is used for
    /// store chains that are composed entirely of extracted vector elements.
    /// When extracting multiple vector elements, try to store them in one
    /// vector store rather than a sequence of scalar stores.
    bool tryStoreMergeOfExtracts(SmallVectorImpl<MemOpLink> &StoreNodes,
                                 unsigned NumConsecutiveStores, EVT MemVT,
                                 SDNode *Root);

    /// This is a helper function for mergeConsecutiveStores. It is used for
    /// store chains that are composed entirely of loaded values.
    bool tryStoreMergeOfLoads(SmallVectorImpl<MemOpLink> &StoreNodes,
                              unsigned NumConsecutiveStores, EVT MemVT,
                              SDNode *Root, bool AllowVectors,
                              bool IsNonTemporalStore, bool IsNonTemporalLoad);

    /// Merge consecutive store operations into a wide store.
    /// This optimization uses wide integers or vectors when possible.
    /// \return true if stores were merged.
    bool mergeConsecutiveStores(StoreSDNode *St);

    /// Try to transform a truncation where C is a constant:
    ///     (trunc (and X, C)) -> (and (trunc X), (trunc C))
    ///
    /// \p N needs to be a truncation and its first operand an AND. Other
    /// requirements are checked by the function (e.g. that trunc is
    /// single-use) and if missed an empty SDValue is returned.
    SDValue distributeTruncateThroughAnd(SDNode *N);

    /// Helper function to determine whether the target supports operation
    /// given by \p Opcode for type \p VT, that is, whether the operation
    /// is legal or custom before legalizing operations, and whether is
    /// legal (but not custom) after legalization.
    bool hasOperation(unsigned Opcode, EVT VT) {
      return TLI.isOperationLegalOrCustom(Opcode, VT, LegalOperations);
    }

  public:
    /// Runs the dag combiner on all nodes in the work list
    void Run(CombineLevel AtLevel);

    SelectionDAG &getDAG() const { return DAG; }

    /// Returns a type large enough to hold any valid shift amount - before type
    /// legalization these can be huge.
    EVT getShiftAmountTy(EVT LHSTy) {
      assert(LHSTy.isInteger() && "Shift amount is not an integer type!");
      return TLI.getShiftAmountTy(LHSTy, DAG.getDataLayout(), LegalTypes);
    }

    /// This method returns true if we are running before type legalization or
    /// if the specified VT is legal.
    bool isTypeLegal(const EVT &VT) {
      if (!LegalTypes) return true;
      return TLI.isTypeLegal(VT);
    }

    /// Convenience wrapper around TargetLowering::getSetCCResultType
    EVT getSetCCResultType(EVT VT) const {
      return TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
    }

    void ExtendSetCCUses(const SmallVectorImpl<SDNode *> &SetCCs,
                         SDValue OrigLoad, SDValue ExtLoad,
                         ISD::NodeType ExtType);
  };

/// This class is a DAGUpdateListener that removes any deleted
/// nodes from the worklist.
class WorklistRemover : public SelectionDAG::DAGUpdateListener {
  DAGCombiner &DC;

public:
  explicit WorklistRemover(DAGCombiner &dc)
    : SelectionDAG::DAGUpdateListener(dc.getDAG()), DC(dc) {}

  void NodeDeleted(SDNode *N, SDNode *E) override {
    DC.removeFromWorklist(N);
  }
};

class WorklistInserter : public SelectionDAG::DAGUpdateListener {
  DAGCombiner &DC;

public:
  explicit WorklistInserter(DAGCombiner &dc)
      : SelectionDAG::DAGUpdateListener(dc.getDAG()), DC(dc) {}

  // FIXME: Ideally we could add N to the worklist, but this causes exponential
  //        compile time costs in large DAGs, e.g. Halide.
  void NodeInserted(SDNode *N) override { DC.ConsiderForPruning(N); }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//  TargetLowering::DAGCombinerInfo implementation
//===----------------------------------------------------------------------===//

void TargetLowering::DAGCombinerInfo::AddToWorklist(SDNode *N) {
  ((DAGCombiner*)DC)->AddToWorklist(N);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, ArrayRef<SDValue> To, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, &To[0], To.size(), AddTo);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res, AddTo);
}

SDValue TargetLowering::DAGCombinerInfo::
CombineTo(SDNode *N, SDValue Res0, SDValue Res1, bool AddTo) {
  return ((DAGCombiner*)DC)->CombineTo(N, Res0, Res1, AddTo);
}

bool TargetLowering::DAGCombinerInfo::
recursivelyDeleteUnusedNodes(SDNode *N) {
  return ((DAGCombiner*)DC)->recursivelyDeleteUnusedNodes(N);
}

void TargetLowering::DAGCombinerInfo::
CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO) {
  return ((DAGCombiner*)DC)->CommitTargetLoweringOpt(TLO);
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

void DAGCombiner::deleteAndRecombine(SDNode *N) {
  removeFromWorklist(N);

  // If the operands of this node are only used by the node, they will now be
  // dead. Make sure to re-visit them and recursively delete dead nodes.
  for (const SDValue &Op : N->ops())
    // For an operand generating multiple values, one of the values may
    // become dead allowing further simplification (e.g. split index
    // arithmetic from an indexed load).
    if (Op->hasOneUse() || Op->getNumValues() > 1)
      AddToWorklist(Op.getNode());

  DAG.DeleteNode(N);
}

// APInts must be the same size for most operations, this helper
// function zero extends the shorter of the pair so that they match.
// We provide an Offset so that we can create bitwidths that won't overflow.
static void zeroExtendToMatch(APInt &LHS, APInt &RHS, unsigned Offset = 0) {
  unsigned Bits = Offset + std::max(LHS.getBitWidth(), RHS.getBitWidth());
  LHS = LHS.zextOrSelf(Bits);
  RHS = RHS.zextOrSelf(Bits);
}

// Return true if this node is a setcc, or is a select_cc
// that selects between the target values used for true and false, making it
// equivalent to a setcc. Also, set the incoming LHS, RHS, and CC references to
// the appropriate nodes based on the type of node we are checking. This
// simplifies life a bit for the callers.
bool DAGCombiner::isSetCCEquivalent(SDValue N, SDValue &LHS, SDValue &RHS,
                                    SDValue &CC, bool MatchStrict) const {
  if (N.getOpcode() == ISD::SETCC) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(2);
    return true;
  }

  if (MatchStrict &&
      (N.getOpcode() == ISD::STRICT_FSETCC ||
       N.getOpcode() == ISD::STRICT_FSETCCS)) {
    LHS = N.getOperand(1);
    RHS = N.getOperand(2);
    CC  = N.getOperand(3);
    return true;
  }

  if (N.getOpcode() != ISD::SELECT_CC ||
      !TLI.isConstTrueVal(N.getOperand(2).getNode()) ||
      !TLI.isConstFalseVal(N.getOperand(3).getNode()))
    return false;

  if (TLI.getBooleanContents(N.getValueType()) ==
      TargetLowering::UndefinedBooleanContent)
    return false;

  LHS = N.getOperand(0);
  RHS = N.getOperand(1);
  CC  = N.getOperand(4);
  return true;
}

/// Return true if this is a SetCC-equivalent operation with only one use.
/// If this is true, it allows the users to invert the operation for free when
/// it is profitable to do so.
bool DAGCombiner::isOneUseSetCC(SDValue N) const {
  SDValue N0, N1, N2;
  if (isSetCCEquivalent(N, N0, N1, N2) && N.getNode()->hasOneUse())
    return true;
  return false;
}

static bool isConstantSplatVectorMaskForType(SDNode *N, EVT ScalarTy) {
  if (!ScalarTy.isSimple())
    return false;

  uint64_t MaskForTy = 0ULL;
  switch (ScalarTy.getSimpleVT().SimpleTy) {
  case MVT::i8:
    MaskForTy = 0xFFULL;
    break;
  case MVT::i16:
    MaskForTy = 0xFFFFULL;
    break;
  case MVT::i32:
    MaskForTy = 0xFFFFFFFFULL;
    break;
  default:
    return false;
    break;
  }

  APInt Val;
  if (ISD::isConstantSplatVector(N, Val))
    return Val.getLimitedValue() == MaskForTy;

  return false;
}

// Determines if it is a constant integer or a splat/build vector of constant
// integers (and undefs).
// Do not permit build vector implicit truncation.
static bool isConstantOrConstantVector(SDValue N, bool NoOpaques = false) {
  if (ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N))
    return !(Const->isOpaque() && NoOpaques);
  if (N.getOpcode() != ISD::BUILD_VECTOR && N.getOpcode() != ISD::SPLAT_VECTOR)
    return false;
  unsigned BitWidth = N.getScalarValueSizeInBits();
  for (const SDValue &Op : N->op_values()) {
    if (Op.isUndef())
      continue;
    ConstantSDNode *Const = dyn_cast<ConstantSDNode>(Op);
    if (!Const || Const->getAPIntValue().getBitWidth() != BitWidth ||
        (Const->isOpaque() && NoOpaques))
      return false;
  }
  return true;
}

// Determines if a BUILD_VECTOR is composed of all-constants possibly mixed with
// undef's.
static bool isAnyConstantBuildVector(SDValue V, bool NoOpaques = false) {
  if (V.getOpcode() != ISD::BUILD_VECTOR)
    return false;
  return isConstantOrConstantVector(V, NoOpaques) ||
         ISD::isBuildVectorOfConstantFPSDNodes(V.getNode());
}

// Determine if this an indexed load with an opaque target constant index.
static bool canSplitIdx(LoadSDNode *LD) {
  return MaySplitLoadIndex &&
         (LD->getOperand(2).getOpcode() != ISD::TargetConstant ||
          !cast<ConstantSDNode>(LD->getOperand(2))->isOpaque());
}

bool DAGCombiner::reassociationCanBreakAddressingModePattern(unsigned Opc,
                                                             const SDLoc &DL,
                                                             SDValue N0,
                                                             SDValue N1) {
  // Currently this only tries to ensure we don't undo the GEP splits done by
  // CodeGenPrepare when shouldConsiderGEPOffsetSplit is true. To ensure this,
  // we check if the following transformation would be problematic:
  // (load/store (add, (add, x, offset1), offset2)) ->
  // (load/store (add, x, offset1+offset2)).

  if (Opc != ISD::ADD || N0.getOpcode() != ISD::ADD)
    return false;

  if (N0.hasOneUse())
    return false;

  auto *C1 = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  auto *C2 = dyn_cast<ConstantSDNode>(N1);
  if (!C1 || !C2)
    return false;

  const APInt &C1APIntVal = C1->getAPIntValue();
  const APInt &C2APIntVal = C2->getAPIntValue();
  if (C1APIntVal.getBitWidth() > 64 || C2APIntVal.getBitWidth() > 64)
    return false;

  const APInt CombinedValueIntVal = C1APIntVal + C2APIntVal;
  if (CombinedValueIntVal.getBitWidth() > 64)
    return false;
  const int64_t CombinedValue = CombinedValueIntVal.getSExtValue();

  for (SDNode *Node : N0->uses()) {
    auto LoadStore = dyn_cast<MemSDNode>(Node);
    if (LoadStore) {
      // Is x[offset2] already not a legal addressing mode? If so then
      // reassociating the constants breaks nothing (we test offset2 because
      // that's the one we hope to fold into the load or store).
      TargetLoweringBase::AddrMode AM;
      AM.HasBaseReg = true;
      AM.BaseOffs = C2APIntVal.getSExtValue();
      EVT VT = LoadStore->getMemoryVT();
      unsigned AS = LoadStore->getAddressSpace();
      Type *AccessTy = VT.getTypeForEVT(*DAG.getContext());
      if (!TLI.isLegalAddressingMode(DAG.getDataLayout(), AM, AccessTy, AS))
        continue;

      // Would x[offset1+offset2] still be a legal addressing mode?
      AM.BaseOffs = CombinedValue;
      if (!TLI.isLegalAddressingMode(DAG.getDataLayout(), AM, AccessTy, AS))
        return true;
    }
  }

  return false;
}

// Helper for DAGCombiner::reassociateOps. Try to reassociate an expression
// such as (Opc N0, N1), if \p N0 is the same kind of operation as \p Opc.
SDValue DAGCombiner::reassociateOpsCommutative(unsigned Opc, const SDLoc &DL,
                                               SDValue N0, SDValue N1) {
  EVT VT = N0.getValueType();

  if (N0.getOpcode() != Opc)
    return SDValue();

  if (DAG.isConstantIntBuildVectorOrConstantInt(N0.getOperand(1))) {
    if (DAG.isConstantIntBuildVectorOrConstantInt(N1)) {
      // Reassociate: (op (op x, c1), c2) -> (op x, (op c1, c2))
      if (SDValue OpNode =
              DAG.FoldConstantArithmetic(Opc, DL, VT, {N0.getOperand(1), N1}))
        return DAG.getNode(Opc, DL, VT, N0.getOperand(0), OpNode);
      return SDValue();
    }
    if (N0.hasOneUse()) {
      // Reassociate: (op (op x, c1), y) -> (op (op x, y), c1)
      //              iff (op x, c1) has one use
      SDValue OpNode = DAG.getNode(Opc, SDLoc(N0), VT, N0.getOperand(0), N1);
      if (!OpNode.getNode())
        return SDValue();
      return DAG.getNode(Opc, DL, VT, OpNode, N0.getOperand(1));
    }
  }
  return SDValue();
}

// Try to reassociate commutative binops.
SDValue DAGCombiner::reassociateOps(unsigned Opc, const SDLoc &DL, SDValue N0,
                                    SDValue N1, SDNodeFlags Flags) {
  assert(TLI.isCommutativeBinOp(Opc) && "Operation not commutative.");

  // Floating-point reassociation is not allowed without loose FP math.
  if (N0.getValueType().isFloatingPoint() ||
      N1.getValueType().isFloatingPoint())
    if (!Flags.hasAllowReassociation() || !Flags.hasNoSignedZeros())
      return SDValue();

  if (SDValue Combined = reassociateOpsCommutative(Opc, DL, N0, N1))
    return Combined;
  if (SDValue Combined = reassociateOpsCommutative(Opc, DL, N1, N0))
    return Combined;
  return SDValue();
}

SDValue DAGCombiner::CombineTo(SDNode *N, const SDValue *To, unsigned NumTo,
                               bool AddTo) {
  assert(N->getNumValues() == NumTo && "Broken CombineTo call!");
  ++NodesCombined;
  LLVM_DEBUG(dbgs() << "\nReplacing.1 "; N->dump(&DAG); dbgs() << "\nWith: ";
             To[0].getNode()->dump(&DAG);
             dbgs() << " and " << NumTo - 1 << " other values\n");
  for (unsigned i = 0, e = NumTo; i != e; ++i)
    assert((!To[i].getNode() ||
            N->getValueType(i) == To[i].getValueType()) &&
           "Cannot combine value to value of different type!");

  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesWith(N, To);
  if (AddTo) {
    // Push the new nodes and any users onto the worklist
    for (unsigned i = 0, e = NumTo; i != e; ++i) {
      if (To[i].getNode()) {
        AddToWorklist(To[i].getNode());
        AddUsersToWorklist(To[i].getNode());
      }
    }
  }

  // Finally, if the node is now dead, remove it from the graph.  The node
  // may not be dead if the replacement process recursively simplified to
  // something else needing this node.
  if (N->use_empty())
    deleteAndRecombine(N);
  return SDValue(N, 0);
}

void DAGCombiner::
CommitTargetLoweringOpt(const TargetLowering::TargetLoweringOpt &TLO) {
  // Replace the old value with the new one.
  ++NodesCombined;
  LLVM_DEBUG(dbgs() << "\nReplacing.2 "; TLO.Old.getNode()->dump(&DAG);
             dbgs() << "\nWith: "; TLO.New.getNode()->dump(&DAG);
             dbgs() << '\n');

  // Replace all uses.  If any nodes become isomorphic to other nodes and
  // are deleted, make sure to remove them from our worklist.
  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(TLO.Old, TLO.New);

  // Push the new node and any (possibly new) users onto the worklist.
  AddToWorklistWithUsers(TLO.New.getNode());

  // Finally, if the node is now dead, remove it from the graph.  The node
  // may not be dead if the replacement process recursively simplified to
  // something else needing this node.
  if (TLO.Old.getNode()->use_empty())
    deleteAndRecombine(TLO.Old.getNode());
}

/// Check the specified integer node value to see if it can be simplified or if
/// things it uses can be simplified by bit propagation. If so, return true.
bool DAGCombiner::SimplifyDemandedBits(SDValue Op, const APInt &DemandedBits,
                                       const APInt &DemandedElts,
                                       bool AssumeSingleUse) {
  TargetLowering::TargetLoweringOpt TLO(DAG, LegalTypes, LegalOperations);
  KnownBits Known;
  if (!TLI.SimplifyDemandedBits(Op, DemandedBits, DemandedElts, Known, TLO, 0,
                                AssumeSingleUse))
    return false;

  // Revisit the node.
  AddToWorklist(Op.getNode());

  CommitTargetLoweringOpt(TLO);
  return true;
}

/// Check the specified vector node value to see if it can be simplified or
/// if things it uses can be simplified as it only uses some of the elements.
/// If so, return true.
bool DAGCombiner::SimplifyDemandedVectorElts(SDValue Op,
                                             const APInt &DemandedElts,
                                             bool AssumeSingleUse) {
  TargetLowering::TargetLoweringOpt TLO(DAG, LegalTypes, LegalOperations);
  APInt KnownUndef, KnownZero;
  if (!TLI.SimplifyDemandedVectorElts(Op, DemandedElts, KnownUndef, KnownZero,
                                      TLO, 0, AssumeSingleUse))
    return false;

  // Revisit the node.
  AddToWorklist(Op.getNode());

  CommitTargetLoweringOpt(TLO);
  return true;
}

void DAGCombiner::ReplaceLoadWithPromotedLoad(SDNode *Load, SDNode *ExtLoad) {
  SDLoc DL(Load);
  EVT VT = Load->getValueType(0);
  SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, VT, SDValue(ExtLoad, 0));

  LLVM_DEBUG(dbgs() << "\nReplacing.9 "; Load->dump(&DAG); dbgs() << "\nWith: ";
             Trunc.getNode()->dump(&DAG); dbgs() << '\n');
  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 0), Trunc);
  DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 1), SDValue(ExtLoad, 1));
  deleteAndRecombine(Load);
  AddToWorklist(Trunc.getNode());
}

SDValue DAGCombiner::PromoteOperand(SDValue Op, EVT PVT, bool &Replace) {
  Replace = false;
  SDLoc DL(Op);
  if (ISD::isUNINDEXEDLoad(Op.getNode())) {
    LoadSDNode *LD = cast<LoadSDNode>(Op);
    EVT MemVT = LD->getMemoryVT();
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(LD) ? ISD::EXTLOAD
                                                      : LD->getExtensionType();
    Replace = true;
    return DAG.getExtLoad(ExtType, DL, PVT,
                          LD->getChain(), LD->getBasePtr(),
                          MemVT, LD->getMemOperand());
  }

  unsigned Opc = Op.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::AssertSext:
    if (SDValue Op0 = SExtPromoteOperand(Op.getOperand(0), PVT))
      return DAG.getNode(ISD::AssertSext, DL, PVT, Op0, Op.getOperand(1));
    break;
  case ISD::AssertZext:
    if (SDValue Op0 = ZExtPromoteOperand(Op.getOperand(0), PVT))
      return DAG.getNode(ISD::AssertZext, DL, PVT, Op0, Op.getOperand(1));
    break;
  case ISD::Constant: {
    unsigned ExtOpc =
      Op.getValueType().isByteSized() ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
    return DAG.getNode(ExtOpc, DL, PVT, Op);
  }
  }

  if (!TLI.isOperationLegal(ISD::ANY_EXTEND, PVT))
    return SDValue();
  return DAG.getNode(ISD::ANY_EXTEND, DL, PVT, Op);
}

SDValue DAGCombiner::SExtPromoteOperand(SDValue Op, EVT PVT) {
  if (!TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG, PVT))
    return SDValue();
  EVT OldVT = Op.getValueType();
  SDLoc DL(Op);
  bool Replace = false;
  SDValue NewOp = PromoteOperand(Op, PVT, Replace);
  if (!NewOp.getNode())
    return SDValue();
  AddToWorklist(NewOp.getNode());

  if (Replace)
    ReplaceLoadWithPromotedLoad(Op.getNode(), NewOp.getNode());
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, NewOp.getValueType(), NewOp,
                     DAG.getValueType(OldVT));
}

SDValue DAGCombiner::ZExtPromoteOperand(SDValue Op, EVT PVT) {
  EVT OldVT = Op.getValueType();
  SDLoc DL(Op);
  bool Replace = false;
  SDValue NewOp = PromoteOperand(Op, PVT, Replace);
  if (!NewOp.getNode())
    return SDValue();
  AddToWorklist(NewOp.getNode());

  if (Replace)
    ReplaceLoadWithPromotedLoad(Op.getNode(), NewOp.getNode());
  return DAG.getZeroExtendInReg(NewOp, DL, OldVT);
}

/// Promote the specified integer binary operation if the target indicates it is
/// beneficial. e.g. On x86, it's usually better to promote i16 operations to
/// i32 since i16 instructions are longer.
SDValue DAGCombiner::PromoteIntBinOp(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    LLVM_DEBUG(dbgs() << "\nPromoting "; Op.getNode()->dump(&DAG));

    bool Replace0 = false;
    SDValue N0 = Op.getOperand(0);
    SDValue NN0 = PromoteOperand(N0, PVT, Replace0);

    bool Replace1 = false;
    SDValue N1 = Op.getOperand(1);
    SDValue NN1 = PromoteOperand(N1, PVT, Replace1);
    SDLoc DL(Op);

    SDValue RV =
        DAG.getNode(ISD::TRUNCATE, DL, VT, DAG.getNode(Opc, DL, PVT, NN0, NN1));

    // We are always replacing N0/N1's use in N and only need additional
    // replacements if there are additional uses.
    // Note: We are checking uses of the *nodes* (SDNode) rather than values
    //       (SDValue) here because the node may reference multiple values
    //       (for example, the chain value of a load node).
    Replace0 &= !N0->hasOneUse();
    Replace1 &= (N0 != N1) && !N1->hasOneUse();

    // Combine Op here so it is preserved past replacements.
    CombineTo(Op.getNode(), RV);

    // If operands have a use ordering, make sure we deal with
    // predecessor first.
    if (Replace0 && Replace1 && N0.getNode()->isPredecessorOf(N1.getNode())) {
      std::swap(N0, N1);
      std::swap(NN0, NN1);
    }

    if (Replace0) {
      AddToWorklist(NN0.getNode());
      ReplaceLoadWithPromotedLoad(N0.getNode(), NN0.getNode());
    }
    if (Replace1) {
      AddToWorklist(NN1.getNode());
      ReplaceLoadWithPromotedLoad(N1.getNode(), NN1.getNode());
    }
    return Op;
  }
  return SDValue();
}

/// Promote the specified integer shift operation if the target indicates it is
/// beneficial. e.g. On x86, it's usually better to promote i16 operations to
/// i32 since i16 instructions are longer.
SDValue DAGCombiner::PromoteIntShiftOp(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    LLVM_DEBUG(dbgs() << "\nPromoting "; Op.getNode()->dump(&DAG));

    bool Replace = false;
    SDValue N0 = Op.getOperand(0);
    SDValue N1 = Op.getOperand(1);
    if (Opc == ISD::SRA)
      N0 = SExtPromoteOperand(N0, PVT);
    else if (Opc == ISD::SRL)
      N0 = ZExtPromoteOperand(N0, PVT);
    else
      N0 = PromoteOperand(N0, PVT, Replace);

    if (!N0.getNode())
      return SDValue();

    SDLoc DL(Op);
    SDValue RV =
        DAG.getNode(ISD::TRUNCATE, DL, VT, DAG.getNode(Opc, DL, PVT, N0, N1));

    if (Replace)
      ReplaceLoadWithPromotedLoad(Op.getOperand(0).getNode(), N0.getNode());

    // Deal with Op being deleted.
    if (Op && Op.getOpcode() != ISD::DELETED_NODE)
      return RV;
  }
  return SDValue();
}

SDValue DAGCombiner::PromoteExtend(SDValue Op) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return SDValue();

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");
    // fold (aext (aext x)) -> (aext x)
    // fold (aext (zext x)) -> (zext x)
    // fold (aext (sext x)) -> (sext x)
    LLVM_DEBUG(dbgs() << "\nPromoting "; Op.getNode()->dump(&DAG));
    return DAG.getNode(Op.getOpcode(), SDLoc(Op), VT, Op.getOperand(0));
  }
  return SDValue();
}

bool DAGCombiner::PromoteLoad(SDValue Op) {
  if (!LegalOperations)
    return false;

  if (!ISD::isUNINDEXEDLoad(Op.getNode()))
    return false;

  EVT VT = Op.getValueType();
  if (VT.isVector() || !VT.isInteger())
    return false;

  // If operation type is 'undesirable', e.g. i16 on x86, consider
  // promoting it.
  unsigned Opc = Op.getOpcode();
  if (TLI.isTypeDesirableForOp(Opc, VT))
    return false;

  EVT PVT = VT;
  // Consult target whether it is a good idea to promote this operation and
  // what's the right type to promote it to.
  if (TLI.IsDesirableToPromoteOp(Op, PVT)) {
    assert(PVT != VT && "Don't know what type to promote to!");

    SDLoc DL(Op);
    SDNode *N = Op.getNode();
    LoadSDNode *LD = cast<LoadSDNode>(N);
    EVT MemVT = LD->getMemoryVT();
    ISD::LoadExtType ExtType = ISD::isNON_EXTLoad(LD) ? ISD::EXTLOAD
                                                      : LD->getExtensionType();
    SDValue NewLD = DAG.getExtLoad(ExtType, DL, PVT,
                                   LD->getChain(), LD->getBasePtr(),
                                   MemVT, LD->getMemOperand());
    SDValue Result = DAG.getNode(ISD::TRUNCATE, DL, VT, NewLD);

    LLVM_DEBUG(dbgs() << "\nPromoting "; N->dump(&DAG); dbgs() << "\nTo: ";
               Result.getNode()->dump(&DAG); dbgs() << '\n');
    WorklistRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), NewLD.getValue(1));
    deleteAndRecombine(N);
    AddToWorklist(Result.getNode());
    return true;
  }
  return false;
}

/// Recursively delete a node which has no uses and any operands for
/// which it is the only use.
///
/// Note that this both deletes the nodes and removes them from the worklist.
/// It also adds any nodes who have had a user deleted to the worklist as they
/// may now have only one use and subject to other combines.
bool DAGCombiner::recursivelyDeleteUnusedNodes(SDNode *N) {
  if (!N->use_empty())
    return false;

  SmallSetVector<SDNode *, 16> Nodes;
  Nodes.insert(N);
  do {
    N = Nodes.pop_back_val();
    if (!N)
      continue;

    if (N->use_empty()) {
      for (const SDValue &ChildN : N->op_values())
        Nodes.insert(ChildN.getNode());

      removeFromWorklist(N);
      DAG.DeleteNode(N);
    } else {
      AddToWorklist(N);
    }
  } while (!Nodes.empty());
  return true;
}

//===----------------------------------------------------------------------===//
//  Main DAG Combiner implementation
//===----------------------------------------------------------------------===//

void DAGCombiner::Run(CombineLevel AtLevel) {
  // set the instance variables, so that the various visit routines may use it.
  Level = AtLevel;
  LegalDAG = Level >= AfterLegalizeDAG;
  LegalOperations = Level >= AfterLegalizeVectorOps;
  LegalTypes = Level >= AfterLegalizeTypes;

  WorklistInserter AddNodes(*this);

  // Add all the dag nodes to the worklist.
  for (SDNode &Node : DAG.allnodes())
    AddToWorklist(&Node);

  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());

  // While we have a valid worklist entry node, try to combine it.
  while (SDNode *N = getNextWorklistEntry()) {
    // If N has no uses, it is dead.  Make sure to revisit all N's operands once
    // N is deleted from the DAG, since they too may now be dead or may have a
    // reduced number of uses, allowing other xforms.
    if (recursivelyDeleteUnusedNodes(N))
      continue;

    WorklistRemover DeadNodes(*this);

    // If this combine is running after legalizing the DAG, re-legalize any
    // nodes pulled off the worklist.
    if (LegalDAG) {
      SmallSetVector<SDNode *, 16> UpdatedNodes;
      bool NIsValid = DAG.LegalizeOp(N, UpdatedNodes);

      for (SDNode *LN : UpdatedNodes)
        AddToWorklistWithUsers(LN);

      if (!NIsValid)
        continue;
    }

    LLVM_DEBUG(dbgs() << "\nCombining: "; N->dump(&DAG));

    // Add any operands of the new node which have not yet been combined to the
    // worklist as well. Because the worklist uniques things already, this
    // won't repeatedly process the same operand.
    CombinedNodes.insert(N);
    for (const SDValue &ChildN : N->op_values())
      if (!CombinedNodes.count(ChildN.getNode()))
        AddToWorklist(ChildN.getNode());

    SDValue RV = combine(N);

    if (!RV.getNode())
      continue;

    ++NodesCombined;

    // If we get back the same node we passed in, rather than a new node or
    // zero, we know that the node must have defined multiple values and
    // CombineTo was used.  Since CombineTo takes care of the worklist
    // mechanics for us, we have no work to do in this case.
    if (RV.getNode() == N)
      continue;

    assert(N->getOpcode() != ISD::DELETED_NODE &&
           RV.getOpcode() != ISD::DELETED_NODE &&
           "Node was deleted but visit returned new node!");

    LLVM_DEBUG(dbgs() << " ... into: "; RV.getNode()->dump(&DAG));

    if (N->getNumValues() == RV.getNode()->getNumValues())
      DAG.ReplaceAllUsesWith(N, RV.getNode());
    else {
      assert(N->getValueType(0) == RV.getValueType() &&
             N->getNumValues() == 1 && "Type mismatch");
      DAG.ReplaceAllUsesWith(N, &RV);
    }

    // Push the new node and any users onto the worklist.  Omit this if the
    // new node is the EntryToken (e.g. if a store managed to get optimized
    // out), because re-visiting the EntryToken and its users will not uncover
    // any additional opportunities, but there may be a large number of such
    // users, potentially causing compile time explosion.
    if (RV.getOpcode() != ISD::EntryToken) {
      AddToWorklist(RV.getNode());
      AddUsersToWorklist(RV.getNode());
    }

    // Finally, if the node is now dead, remove it from the graph.  The node
    // may not be dead if the replacement process recursively simplified to
    // something else needing this node. This will also take care of adding any
    // operands which have lost a user to the worklist.
    recursivelyDeleteUnusedNodes(N);
  }

  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());
  DAG.RemoveDeadNodes();
}

SDValue DAGCombiner::visit(SDNode *N) {
  switch (N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::MERGE_VALUES:       return visitMERGE_VALUES(N);
  case ISD::ADD:                return visitADD(N);
  case ISD::SUB:                return visitSUB(N);
  case ISD::SADDSAT:
  case ISD::UADDSAT:            return visitADDSAT(N);
  case ISD::SSUBSAT:
  case ISD::USUBSAT:            return visitSUBSAT(N);
  case ISD::ADDC:               return visitADDC(N);
  case ISD::SADDO:
  case ISD::UADDO:              return visitADDO(N);
  case ISD::SUBC:               return visitSUBC(N);
  case ISD::SSUBO:
  case ISD::USUBO:              return visitSUBO(N);
  case ISD::ADDE:               return visitADDE(N);
  case ISD::ADDCARRY:           return visitADDCARRY(N);
  case ISD::SADDO_CARRY:        return visitSADDO_CARRY(N);
  case ISD::SUBE:               return visitSUBE(N);
  case ISD::SUBCARRY:           return visitSUBCARRY(N);
  case ISD::SSUBO_CARRY:        return visitSSUBO_CARRY(N);
  case ISD::SMULFIX:
  case ISD::SMULFIXSAT:
  case ISD::UMULFIX:
  case ISD::UMULFIXSAT:         return visitMULFIX(N);
  case ISD::MUL:                return visitMUL(N);
  case ISD::SDIV:               return visitSDIV(N);
  case ISD::UDIV:               return visitUDIV(N);
  case ISD::SREM:
  case ISD::UREM:               return visitREM(N);
  case ISD::MULHU:              return visitMULHU(N);
  case ISD::MULHS:              return visitMULHS(N);
  case ISD::SMUL_LOHI:          return visitSMUL_LOHI(N);
  case ISD::UMUL_LOHI:          return visitUMUL_LOHI(N);
  case ISD::SMULO:
  case ISD::UMULO:              return visitMULO(N);
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:               return visitIMINMAX(N);
  case ISD::AND:                return visitAND(N);
  case ISD::OR:                 return visitOR(N);
  case ISD::XOR:                return visitXOR(N);
  case ISD::SHL:                return visitSHL(N);
  case ISD::SRA:                return visitSRA(N);
  case ISD::SRL:                return visitSRL(N);
  case ISD::ROTR:
  case ISD::ROTL:               return visitRotate(N);
  case ISD::FSHL:
  case ISD::FSHR:               return visitFunnelShift(N);
  case ISD::ABS:                return visitABS(N);
  case ISD::BSWAP:              return visitBSWAP(N);
  case ISD::BITREVERSE:         return visitBITREVERSE(N);
  case ISD::CTLZ:               return visitCTLZ(N);
  case ISD::CTLZ_ZERO_UNDEF:    return visitCTLZ_ZERO_UNDEF(N);
  case ISD::CTTZ:               return visitCTTZ(N);
  case ISD::CTTZ_ZERO_UNDEF:    return visitCTTZ_ZERO_UNDEF(N);
  case ISD::CTPOP:              return visitCTPOP(N);
  case ISD::SELECT:             return visitSELECT(N);
  case ISD::VSELECT:            return visitVSELECT(N);
  case ISD::SELECT_CC:          return visitSELECT_CC(N);
  case ISD::SETCC:              return visitSETCC(N);
  case ISD::SETCCCARRY:         return visitSETCCCARRY(N);
  case ISD::SIGN_EXTEND:        return visitSIGN_EXTEND(N);
  case ISD::ZERO_EXTEND:        return visitZERO_EXTEND(N);
  case ISD::ANY_EXTEND:         return visitANY_EXTEND(N);
  case ISD::AssertSext:
  case ISD::AssertZext:         return visitAssertExt(N);
  case ISD::AssertAlign:        return visitAssertAlign(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSIGN_EXTEND_INREG(N);
  case ISD::SIGN_EXTEND_VECTOR_INREG:
  case ISD::ZERO_EXTEND_VECTOR_INREG: return visitEXTEND_VECTOR_INREG(N);
  case ISD::TRUNCATE:           return visitTRUNCATE(N);
  case ISD::BITCAST:            return visitBITCAST(N);
  case ISD::BUILD_PAIR:         return visitBUILD_PAIR(N);
  case ISD::FADD:               return visitFADD(N);
  case ISD::STRICT_FADD:        return visitSTRICT_FADD(N);
  case ISD::FSUB:               return visitFSUB(N);
  case ISD::FMUL:               return visitFMUL(N);
  case ISD::FMA:                return visitFMA(N);
  case ISD::FDIV:               return visitFDIV(N);
  case ISD::FREM:               return visitFREM(N);
  case ISD::FSQRT:              return visitFSQRT(N);
  case ISD::FCOPYSIGN:          return visitFCOPYSIGN(N);
  case ISD::FPOW:               return visitFPOW(N);
  case ISD::SINT_TO_FP:         return visitSINT_TO_FP(N);
  case ISD::UINT_TO_FP:         return visitUINT_TO_FP(N);
  case ISD::FP_TO_SINT:         return visitFP_TO_SINT(N);
  case ISD::FP_TO_UINT:         return visitFP_TO_UINT(N);
  case ISD::FP_ROUND:           return visitFP_ROUND(N);
  case ISD::FP_EXTEND:          return visitFP_EXTEND(N);
  case ISD::FNEG:               return visitFNEG(N);
  case ISD::FABS:               return visitFABS(N);
  case ISD::FFLOOR:             return visitFFLOOR(N);
  case ISD::FMINNUM:            return visitFMINNUM(N);
  case ISD::FMAXNUM:            return visitFMAXNUM(N);
  case ISD::FMINIMUM:           return visitFMINIMUM(N);
  case ISD::FMAXIMUM:           return visitFMAXIMUM(N);
  case ISD::FCEIL:              return visitFCEIL(N);
  case ISD::FTRUNC:             return visitFTRUNC(N);
  case ISD::BRCOND:             return visitBRCOND(N);
  case ISD::BR_CC:              return visitBR_CC(N);
  case ISD::LOAD:               return visitLOAD(N);
  case ISD::STORE:              return visitSTORE(N);
  case ISD::INSERT_VECTOR_ELT:  return visitINSERT_VECTOR_ELT(N);
  case ISD::EXTRACT_VECTOR_ELT: return visitEXTRACT_VECTOR_ELT(N);
  case ISD::BUILD_VECTOR:       return visitBUILD_VECTOR(N);
  case ISD::CONCAT_VECTORS:     return visitCONCAT_VECTORS(N);
  case ISD::EXTRACT_SUBVECTOR:  return visitEXTRACT_SUBVECTOR(N);
  case ISD::VECTOR_SHUFFLE:     return visitVECTOR_SHUFFLE(N);
  case ISD::SCALAR_TO_VECTOR:   return visitSCALAR_TO_VECTOR(N);
  case ISD::INSERT_SUBVECTOR:   return visitINSERT_SUBVECTOR(N);
  case ISD::MGATHER:            return visitMGATHER(N);
  case ISD::MLOAD:              return visitMLOAD(N);
  case ISD::MSCATTER:           return visitMSCATTER(N);
  case ISD::MSTORE:             return visitMSTORE(N);
  case ISD::LIFETIME_END:       return visitLIFETIME_END(N);
  case ISD::FP_TO_FP16:         return visitFP_TO_FP16(N);
  case ISD::FP16_TO_FP:         return visitFP16_TO_FP(N);
  case ISD::FREEZE:             return visitFREEZE(N);
  case ISD::VECREDUCE_FADD:
  case ISD::VECREDUCE_FMUL:
  case ISD::VECREDUCE_ADD:
  case ISD::VECREDUCE_MUL:
  case ISD::VECREDUCE_AND:
  case ISD::VECREDUCE_OR:
  case ISD::VECREDUCE_XOR:
  case ISD::VECREDUCE_SMAX:
  case ISD::VECREDUCE_SMIN:
  case ISD::VECREDUCE_UMAX:
  case ISD::VECREDUCE_UMIN:
  case ISD::VECREDUCE_FMAX:
  case ISD::VECREDUCE_FMIN:     return visitVECREDUCE(N);
  }
  return SDValue();
}

SDValue DAGCombiner::combine(SDNode *N) {
  SDValue RV;
  if (!DisableGenericCombines)
    RV = visit(N);

  // If nothing happened, try a target-specific DAG combine.
  if (!RV.getNode()) {
    assert(N->getOpcode() != ISD::DELETED_NODE &&
           "Node was deleted but visit returned NULL!");

    if (N->getOpcode() >= ISD::BUILTIN_OP_END ||
        TLI.hasTargetDAGCombine((ISD::NodeType)N->getOpcode())) {

      // Expose the DAG combiner to the target combiner impls.
      TargetLowering::DAGCombinerInfo
        DagCombineInfo(DAG, Level, false, this);

      RV = TLI.PerformDAGCombine(N, DagCombineInfo);
    }
  }

  // If nothing happened still, try promoting the operation.
  if (!RV.getNode()) {
    switch (N->getOpcode()) {
    default: break;
    case ISD::ADD:
    case ISD::SUB:
    case ISD::MUL:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
      RV = PromoteIntBinOp(SDValue(N, 0));
      break;
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
      RV = PromoteIntShiftOp(SDValue(N, 0));
      break;
    case ISD::SIGN_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::ANY_EXTEND:
      RV = PromoteExtend(SDValue(N, 0));
      break;
    case ISD::LOAD:
      if (PromoteLoad(SDValue(N, 0)))
        RV = SDValue(N, 0);
      break;
    }
  }

  // If N is a commutative binary node, try to eliminate it if the commuted
  // version is already present in the DAG.
  if (!RV.getNode() && TLI.isCommutativeBinOp(N->getOpcode()) &&
      N->getNumValues() == 1) {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);

    // Constant operands are canonicalized to RHS.
    if (N0 != N1 && (isa<ConstantSDNode>(N0) || !isa<ConstantSDNode>(N1))) {
      SDValue Ops[] = {N1, N0};
      SDNode *CSENode = DAG.getNodeIfExists(N->getOpcode(), N->getVTList(), Ops,
                                            N->getFlags());
      if (CSENode)
        return SDValue(CSENode, 0);
    }
  }

  return RV;
}

/// Given a node, return its input chain if it has one, otherwise return a null
/// sd operand.
static SDValue getInputChainForNode(SDNode *N) {
  if (unsigned NumOps = N->getNumOperands()) {
    if (N->getOperand(0).getValueType() == MVT::Other)
      return N->getOperand(0);
    if (N->getOperand(NumOps-1).getValueType() == MVT::Other)
      return N->getOperand(NumOps-1);
    for (unsigned i = 1; i < NumOps-1; ++i)
      if (N->getOperand(i).getValueType() == MVT::Other)
        return N->getOperand(i);
  }
  return SDValue();
}

SDValue DAGCombiner::visitTokenFactor(SDNode *N) {
  // If N has two operands, where one has an input chain equal to the other,
  // the 'other' chain is redundant.
  if (N->getNumOperands() == 2) {
    if (getInputChainForNode(N->getOperand(0).getNode()) == N->getOperand(1))
      return N->getOperand(0);
    if (getInputChainForNode(N->getOperand(1).getNode()) == N->getOperand(0))
      return N->getOperand(1);
  }

  // Don't simplify token factors if optnone.
  if (OptLevel == CodeGenOpt::None)
    return SDValue();

  // Don't simplify the token factor if the node itself has too many operands.
  if (N->getNumOperands() > TokenFactorInlineLimit)
    return SDValue();

  // If the sole user is a token factor, we should make sure we have a
  // chance to merge them together. This prevents TF chains from inhibiting
  // optimizations.
  if (N->hasOneUse() && N->use_begin()->getOpcode() == ISD::TokenFactor)
    AddToWorklist(*(N->use_begin()));

  SmallVector<SDNode *, 8> TFs;     // List of token factors to visit.
  SmallVector<SDValue, 8> Ops;      // Ops for replacing token factor.
  SmallPtrSet<SDNode*, 16> SeenOps;
  bool Changed = false;             // If we should replace this token factor.

  // Start out with this token factor.
  TFs.push_back(N);

  // Iterate through token factors.  The TFs grows when new token factors are
  // encountered.
  for (unsigned i = 0; i < TFs.size(); ++i) {
    // Limit number of nodes to inline, to avoid quadratic compile times.
    // We have to add the outstanding Token Factors to Ops, otherwise we might
    // drop Ops from the resulting Token Factors.
    if (Ops.size() > TokenFactorInlineLimit) {
      for (unsigned j = i; j < TFs.size(); j++)
        Ops.emplace_back(TFs[j], 0);
      // Drop unprocessed Token Factors from TFs, so we do not add them to the
      // combiner worklist later.
      TFs.resize(i);
      break;
    }

    SDNode *TF = TFs[i];
    // Check each of the operands.
    for (const SDValue &Op : TF->op_values()) {
      switch (Op.getOpcode()) {
      case ISD::EntryToken:
        // Entry tokens don't need to be added to the list. They are
        // redundant.
        Changed = true;
        break;

      case ISD::TokenFactor:
        if (Op.hasOneUse() && !is_contained(TFs, Op.getNode())) {
          // Queue up for processing.
          TFs.push_back(Op.getNode());
          Changed = true;
          break;
        }
        LLVM_FALLTHROUGH;

      default:
        // Only add if it isn't already in the list.
        if (SeenOps.insert(Op.getNode()).second)
          Ops.push_back(Op);
        else
          Changed = true;
        break;
      }
    }
  }

  // Re-visit inlined Token Factors, to clean them up in case they have been
  // removed. Skip the first Token Factor, as this is the current node.
  for (unsigned i = 1, e = TFs.size(); i < e; i++)
    AddToWorklist(TFs[i]);

  // Remove Nodes that are chained to another node in the list. Do so
  // by walking up chains breath-first stopping when we've seen
  // another operand. In general we must climb to the EntryNode, but we can exit
  // early if we find all remaining work is associated with just one operand as
  // no further pruning is possible.

  // List of nodes to search through and original Ops from which they originate.
  SmallVector<std::pair<SDNode *, unsigned>, 8> Worklist;
  SmallVector<unsigned, 8> OpWorkCount; // Count of work for each Op.
  SmallPtrSet<SDNode *, 16> SeenChains;
  bool DidPruneOps = false;

  unsigned NumLeftToConsider = 0;
  for (const SDValue &Op : Ops) {
    Worklist.push_back(std::make_pair(Op.getNode(), NumLeftToConsider++));
    OpWorkCount.push_back(1);
  }

  auto AddToWorklist = [&](unsigned CurIdx, SDNode *Op, unsigned OpNumber) {
    // If this is an Op, we can remove the op from the list. Remark any
    // search associated with it as from the current OpNumber.
    if (SeenOps.contains(Op)) {
      Changed = true;
      DidPruneOps = true;
      unsigned OrigOpNumber = 0;
      while (OrigOpNumber < Ops.size() && Ops[OrigOpNumber].getNode() != Op)
        OrigOpNumber++;
      assert((OrigOpNumber != Ops.size()) &&
             "expected to find TokenFactor Operand");
      // Re-mark worklist from OrigOpNumber to OpNumber
      for (unsigned i = CurIdx + 1; i < Worklist.size(); ++i) {
        if (Worklist[i].second == OrigOpNumber) {
          Worklist[i].second = OpNumber;
        }
      }
      OpWorkCount[OpNumber] += OpWorkCount[OrigOpNumber];
      OpWorkCount[OrigOpNumber] = 0;
      NumLeftToConsider--;
    }
    // Add if it's a new chain
    if (SeenChains.insert(Op).second) {
      OpWorkCount[OpNumber]++;
      Worklist.push_back(std::make_pair(Op, OpNumber));
    }
  };

  for (unsigned i = 0; i < Worklist.size() && i < 1024; ++i) {
    // We need at least be consider at least 2 Ops to prune.
    if (NumLeftToConsider <= 1)
      break;
    auto CurNode = Worklist[i].first;
    auto CurOpNumber = Worklist[i].second;
    assert((OpWorkCount[CurOpNumber] > 0) &&
           "Node should not appear in worklist");
    switch (CurNode->getOpcode()) {
    case ISD::EntryToken:
      // Hitting EntryToken is the only way for the search to terminate without
      // hitting
      // another operand's search. Prevent us from marking this operand
      // considered.
      NumLeftToConsider++;
      break;
    case ISD::TokenFactor:
      for (const SDValue &Op : CurNode->op_values())
        AddToWorklist(i, Op.getNode(), CurOpNumber);
      break;
    case ISD::LIFETIME_START:
    case ISD::LIFETIME_END:
    case ISD::CopyFromReg:
    case ISD::CopyToReg:
      AddToWorklist(i, CurNode->getOperand(0).getNode(), CurOpNumber);
      break;
    default:
      if (auto *MemNode = dyn_cast<MemSDNode>(CurNode))
        AddToWorklist(i, MemNode->getChain().getNode(), CurOpNumber);
      break;
    }
    OpWorkCount[CurOpNumber]--;
    if (OpWorkCount[CurOpNumber] == 0)
      NumLeftToConsider--;
  }

  // If we've changed things around then replace token factor.
  if (Changed) {
    SDValue Result;
    if (Ops.empty()) {
      // The entry token is the only possible outcome.
      Result = DAG.getEntryNode();
    } else {
      if (DidPruneOps) {
        SmallVector<SDValue, 8> PrunedOps;
        //
        for (const SDValue &Op : Ops) {
          if (SeenChains.count(Op.getNode()) == 0)
            PrunedOps.push_back(Op);
        }
        Result = DAG.getTokenFactor(SDLoc(N), PrunedOps);
      } else {
        Result = DAG.getTokenFactor(SDLoc(N), Ops);
      }
    }
    return Result;
  }
  return SDValue();
}

/// MERGE_VALUES can always be eliminated.
SDValue DAGCombiner::visitMERGE_VALUES(SDNode *N) {
  WorklistRemover DeadNodes(*this);
  // Replacing results may cause a different MERGE_VALUES to suddenly
  // be CSE'd with N, and carry its uses with it. Iterate until no
  // uses remain, to ensure that the node can be safely deleted.
  // First add the users of this node to the work list so that they
  // can be tried again once they have new operands.
  AddUsersToWorklist(N);
  do {
    // Do as a single replacement to avoid rewalking use lists.
    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
      Ops.push_back(N->getOperand(i));
    DAG.ReplaceAllUsesWith(N, Ops.data());
  } while (!N->use_empty());
  deleteAndRecombine(N);
  return SDValue(N, 0);   // Return N so it doesn't get rechecked!
}

/// If \p N is a ConstantSDNode with isOpaque() == false return it casted to a
/// ConstantSDNode pointer else nullptr.
static ConstantSDNode *getAsNonOpaqueConstant(SDValue N) {
  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(N);
  return Const != nullptr && !Const->isOpaque() ? Const : nullptr;
}

/// Return true if 'Use' is a load or a store that uses N as its base pointer
/// and that N may be folded in the load / store addressing mode.
static bool canFoldInAddressingMode(SDNode *N, SDNode *Use, SelectionDAG &DAG,
                                    const TargetLowering &TLI) {
  EVT VT;
  unsigned AS;

  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(Use)) {
    if (LD->isIndexed() || LD->getBasePtr().getNode() != N)
      return false;
    VT = LD->getMemoryVT();
    AS = LD->getAddressSpace();
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(Use)) {
    if (ST->isIndexed() || ST->getBasePtr().getNode() != N)
      return false;
    VT = ST->getMemoryVT();
    AS = ST->getAddressSpace();
  } else if (MaskedLoadSDNode *LD = dyn_cast<MaskedLoadSDNode>(Use)) {
    if (LD->isIndexed() || LD->getBasePtr().getNode() != N)
      return false;
    VT = LD->getMemoryVT();
    AS = LD->getAddressSpace();
  } else if (MaskedStoreSDNode *ST = dyn_cast<MaskedStoreSDNode>(Use)) {
    if (ST->isIndexed() || ST->getBasePtr().getNode() != N)
      return false;
    VT = ST->getMemoryVT();
    AS = ST->getAddressSpace();
  } else
    return false;

  TargetLowering::AddrMode AM;
  if (N->getOpcode() == ISD::ADD) {
    AM.HasBaseReg = true;
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (Offset)
      // [reg +/- imm]
      AM.BaseOffs = Offset->getSExtValue();
    else
      // [reg +/- reg]
      AM.Scale = 1;
  } else if (N->getOpcode() == ISD::SUB) {
    AM.HasBaseReg = true;
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (Offset)
      // [reg +/- imm]
      AM.BaseOffs = -Offset->getSExtValue();
    else
      // [reg +/- reg]
      AM.Scale = 1;
  } else
    return false;

  return TLI.isLegalAddressingMode(DAG.getDataLayout(), AM,
                                   VT.getTypeForEVT(*DAG.getContext()), AS);
}

SDValue DAGCombiner::foldBinOpIntoSelect(SDNode *BO) {
  assert(TLI.isBinOp(BO->getOpcode()) && BO->getNumValues() == 1 &&
         "Unexpected binary operator");

  // Don't do this unless the old select is going away. We want to eliminate the
  // binary operator, not replace a binop with a select.
  // TODO: Handle ISD::SELECT_CC.
  unsigned SelOpNo = 0;
  SDValue Sel = BO->getOperand(0);
  if (Sel.getOpcode() != ISD::SELECT || !Sel.hasOneUse()) {
    SelOpNo = 1;
    Sel = BO->getOperand(1);
  }

  if (Sel.getOpcode() != ISD::SELECT || !Sel.hasOneUse())
    return SDValue();

  SDValue CT = Sel.getOperand(1);
  if (!isConstantOrConstantVector(CT, true) &&
      !DAG.isConstantFPBuildVectorOrConstantFP(CT))
    return SDValue();

  SDValue CF = Sel.getOperand(2);
  if (!isConstantOrConstantVector(CF, true) &&
      !DAG.isConstantFPBuildVectorOrConstantFP(CF))
    return SDValue();

  // Bail out if any constants are opaque because we can't constant fold those.
  // The exception is "and" and "or" with either 0 or -1 in which case we can
  // propagate non constant operands into select. I.e.:
  // and (select Cond, 0, -1), X --> select Cond, 0, X
  // or X, (select Cond, -1, 0) --> select Cond, -1, X
  auto BinOpcode = BO->getOpcode();
  bool CanFoldNonConst =
      (BinOpcode == ISD::AND || BinOpcode == ISD::OR) &&
      (isNullOrNullSplat(CT) || isAllOnesOrAllOnesSplat(CT)) &&
      (isNullOrNullSplat(CF) || isAllOnesOrAllOnesSplat(CF));

  SDValue CBO = BO->getOperand(SelOpNo ^ 1);
  if (!CanFoldNonConst &&
      !isConstantOrConstantVector(CBO, true) &&
      !DAG.isConstantFPBuildVectorOrConstantFP(CBO))
    return SDValue();

  EVT VT = BO->getValueType(0);

  // We have a select-of-constants followed by a binary operator with a
  // constant. Eliminate the binop by pulling the constant math into the select.
  // Example: add (select Cond, CT, CF), CBO --> select Cond, CT + CBO, CF + CBO
  SDLoc DL(Sel);
  SDValue NewCT = SelOpNo ? DAG.getNode(BinOpcode, DL, VT, CBO, CT)
                          : DAG.getNode(BinOpcode, DL, VT, CT, CBO);
  if (!CanFoldNonConst && !NewCT.isUndef() &&
      !isConstantOrConstantVector(NewCT, true) &&
      !DAG.isConstantFPBuildVectorOrConstantFP(NewCT))
    return SDValue();

  SDValue NewCF = SelOpNo ? DAG.getNode(BinOpcode, DL, VT, CBO, CF)
                          : DAG.getNode(BinOpcode, DL, VT, CF, CBO);
  if (!CanFoldNonConst && !NewCF.isUndef() &&
      !isConstantOrConstantVector(NewCF, true) &&
      !DAG.isConstantFPBuildVectorOrConstantFP(NewCF))
    return SDValue();

  SDValue SelectOp = DAG.getSelect(DL, VT, Sel.getOperand(0), NewCT, NewCF);
  SelectOp->setFlags(BO->getFlags());
  return SelectOp;
}

static SDValue foldAddSubBoolOfMaskedVal(SDNode *N, SelectionDAG &DAG) {
  assert((N->getOpcode() == ISD::ADD || N->getOpcode() == ISD::SUB) &&
         "Expecting add or sub");

  // Match a constant operand and a zext operand for the math instruction:
  // add Z, C
  // sub C, Z
  bool IsAdd = N->getOpcode() == ISD::ADD;
  SDValue C = IsAdd ? N->getOperand(1) : N->getOperand(0);
  SDValue Z = IsAdd ? N->getOperand(0) : N->getOperand(1);
  auto *CN = dyn_cast<ConstantSDNode>(C);
  if (!CN || Z.getOpcode() != ISD::ZERO_EXTEND)
    return SDValue();

  // Match the zext operand as a setcc of a boolean.
  if (Z.getOperand(0).getOpcode() != ISD::SETCC ||
      Z.getOperand(0).getValueType() != MVT::i1)
    return SDValue();

  // Match the compare as: setcc (X & 1), 0, eq.
  SDValue SetCC = Z.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(SetCC->getOperand(2))->get();
  if (CC != ISD::SETEQ || !isNullConstant(SetCC.getOperand(1)) ||
      SetCC.getOperand(0).getOpcode() != ISD::AND ||
      !isOneConstant(SetCC.getOperand(0).getOperand(1)))
    return SDValue();

  // We are adding/subtracting a constant and an inverted low bit. Turn that
  // into a subtract/add of the low bit with incremented/decremented constant:
  // add (zext i1 (seteq (X & 1), 0)), C --> sub C+1, (zext (X & 1))
  // sub C, (zext i1 (seteq (X & 1), 0)) --> add C-1, (zext (X & 1))
  EVT VT = C.getValueType();
  SDLoc DL(N);
  SDValue LowBit = DAG.getZExtOrTrunc(SetCC.getOperand(0), DL, VT);
  SDValue C1 = IsAdd ? DAG.getConstant(CN->getAPIntValue() + 1, DL, VT) :
                       DAG.getConstant(CN->getAPIntValue() - 1, DL, VT);
  return DAG.getNode(IsAdd ? ISD::SUB : ISD::ADD, DL, VT, C1, LowBit);
}

/// Try to fold a 'not' shifted sign-bit with add/sub with constant operand into
/// a shift and add with a different constant.
static SDValue foldAddSubOfSignBit(SDNode *N, SelectionDAG &DAG) {
  assert((N->getOpcode() == ISD::ADD || N->getOpcode() == ISD::SUB) &&
         "Expecting add or sub");

  // We need a constant operand for the add/sub, and the other operand is a
  // logical shift right: add (srl), C or sub C, (srl).
  bool IsAdd = N->getOpcode() == ISD::ADD;
  SDValue ConstantOp = IsAdd ? N->getOperand(1) : N->getOperand(0);
  SDValue ShiftOp = IsAdd ? N->getOperand(0) : N->getOperand(1);
  if (!DAG.isConstantIntBuildVectorOrConstantInt(ConstantOp) ||
      ShiftOp.getOpcode() != ISD::SRL)
    return SDValue();

  // The shift must be of a 'not' value.
  SDValue Not = ShiftOp.getOperand(0);
  if (!Not.hasOneUse() || !isBitwiseNot(Not))
    return SDValue();

  // The shift must be moving the sign bit to the least-significant-bit.
  EVT VT = ShiftOp.getValueType();
  SDValue ShAmt = ShiftOp.getOperand(1);
  ConstantSDNode *ShAmtC = isConstOrConstSplat(ShAmt);
  if (!ShAmtC || ShAmtC->getAPIntValue() != (VT.getScalarSizeInBits() - 1))
    return SDValue();

  // Eliminate the 'not' by adjusting the shift and add/sub constant:
  // add (srl (not X), 31), C --> add (sra X, 31), (C + 1)
  // sub C, (srl (not X), 31) --> add (srl X, 31), (C - 1)
  SDLoc DL(N);
  auto ShOpcode = IsAdd ? ISD::SRA : ISD::SRL;
  SDValue NewShift = DAG.getNode(ShOpcode, DL, VT, Not.getOperand(0), ShAmt);
  if (SDValue NewC =
          DAG.FoldConstantArithmetic(IsAdd ? ISD::ADD : ISD::SUB, DL, VT,
                                     {ConstantOp, DAG.getConstant(1, DL, VT)}))
    return DAG.getNode(ISD::ADD, DL, VT, NewShift, NewC);
  return SDValue();
}

/// Try to fold a node that behaves like an ADD (note that N isn't necessarily
/// an ISD::ADD here, it could for example be an ISD::OR if we know that there
/// are no common bits set in the operands).
SDValue DAGCombiner::visitADDLike(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (add x, 0) -> x, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return N0;
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()))
      return N1;
  }

  // fold (add x, undef) -> undef
  if (N0.isUndef())
    return N0;

  if (N1.isUndef())
    return N1;

  if (DAG.isConstantIntBuildVectorOrConstantInt(N0)) {
    // canonicalize constant to RHS
    if (!DAG.isConstantIntBuildVectorOrConstantInt(N1))
      return DAG.getNode(ISD::ADD, DL, VT, N1, N0);
    // fold (add c1, c2) -> c1+c2
    return DAG.FoldConstantArithmetic(ISD::ADD, DL, VT, {N0, N1});
  }

  // fold (add x, 0) -> x
  if (isNullConstant(N1))
    return N0;

  if (isConstantOrConstantVector(N1, /* NoOpaque */ true)) {
    // fold ((A-c1)+c2) -> (A+(c2-c1))
    if (N0.getOpcode() == ISD::SUB &&
        isConstantOrConstantVector(N0.getOperand(1), /* NoOpaque */ true)) {
      SDValue Sub =
          DAG.FoldConstantArithmetic(ISD::SUB, DL, VT, {N1, N0.getOperand(1)});
      assert(Sub && "Constant folding failed");
      return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0), Sub);
    }

    // fold ((c1-A)+c2) -> (c1+c2)-A
    if (N0.getOpcode() == ISD::SUB &&
        isConstantOrConstantVector(N0.getOperand(0), /* NoOpaque */ true)) {
      SDValue Add =
          DAG.FoldConstantArithmetic(ISD::ADD, DL, VT, {N1, N0.getOperand(0)});
      assert(Add && "Constant folding failed");
      return DAG.getNode(ISD::SUB, DL, VT, Add, N0.getOperand(1));
    }

    // add (sext i1 X), 1 -> zext (not i1 X)
    // We don't transform this pattern:
    //   add (zext i1 X), -1 -> sext (not i1 X)
    // because most (?) targets generate better code for the zext form.
    if (N0.getOpcode() == ISD::SIGN_EXTEND && N0.hasOneUse() &&
        isOneOrOneSplat(N1)) {
      SDValue X = N0.getOperand(0);
      if ((!LegalOperations ||
           (TLI.isOperationLegal(ISD::XOR, X.getValueType()) &&
            TLI.isOperationLegal(ISD::ZERO_EXTEND, VT))) &&
          X.getScalarValueSizeInBits() == 1) {
        SDValue Not = DAG.getNOT(DL, X, X.getValueType());
        return DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Not);
      }
    }

    // Fold (add (or x, c0), c1) -> (add x, (c0 + c1)) if (or x, c0) is
    // equivalent to (add x, c0).
    if (N0.getOpcode() == ISD::OR &&
        isConstantOrConstantVector(N0.getOperand(1), /* NoOpaque */ true) &&
        DAG.haveNoCommonBitsSet(N0.getOperand(0), N0.getOperand(1))) {
      if (SDValue Add0 = DAG.FoldConstantArithmetic(ISD::ADD, DL, VT,
                                                    {N1, N0.getOperand(1)}))
        return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0), Add0);
    }
  }

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // reassociate add
  if (!reassociationCanBreakAddressingModePattern(ISD::ADD, DL, N0, N1)) {
    if (SDValue RADD = reassociateOps(ISD::ADD, DL, N0, N1, N->getFlags()))
      return RADD;

    // Reassociate (add (or x, c), y) -> (add add(x, y), c)) if (or x, c) is
    // equivalent to (add x, c).
    auto ReassociateAddOr = [&](SDValue N0, SDValue N1) {
      if (N0.getOpcode() == ISD::OR && N0.hasOneUse() &&
          isConstantOrConstantVector(N0.getOperand(1), /* NoOpaque */ true) &&
          DAG.haveNoCommonBitsSet(N0.getOperand(0), N0.getOperand(1))) {
        return DAG.getNode(ISD::ADD, DL, VT,
                           DAG.getNode(ISD::ADD, DL, VT, N1, N0.getOperand(0)),
                           N0.getOperand(1));
      }
      return SDValue();
    };
    if (SDValue Add = ReassociateAddOr(N0, N1))
      return Add;
    if (SDValue Add = ReassociateAddOr(N1, N0))
      return Add;
  }
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isNullOrNullSplat(N0.getOperand(0)))
    return DAG.getNode(ISD::SUB, DL, VT, N1, N0.getOperand(1));

  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isNullOrNullSplat(N1.getOperand(0)))
    return DAG.getNode(ISD::SUB, DL, VT, N0, N1.getOperand(1));

  // fold (A+(B-A)) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1))
    return N1.getOperand(0);

  // fold ((B-A)+A) -> B
  if (N0.getOpcode() == ISD::SUB && N1 == N0.getOperand(1))
    return N0.getOperand(0);

  // fold ((A-B)+(C-A)) -> (C-B)
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB &&
      N0.getOperand(0) == N1.getOperand(1))
    return DAG.getNode(ISD::SUB, DL, VT, N1.getOperand(0),
                       N0.getOperand(1));

  // fold ((A-B)+(B-C)) -> (A-C)
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB &&
      N0.getOperand(1) == N1.getOperand(0))
    return DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(0),
                       N1.getOperand(1));

  // fold (A+(B-(A+C))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(0))
    return DAG.getNode(ISD::SUB, DL, VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(1));

  // fold (A+(B-(C+A))) to (B-C)
  if (N1.getOpcode() == ISD::SUB && N1.getOperand(1).getOpcode() == ISD::ADD &&
      N0 == N1.getOperand(1).getOperand(1))
    return DAG.getNode(ISD::SUB, DL, VT, N1.getOperand(0),
                       N1.getOperand(1).getOperand(0));

  // fold (A+((B-A)+or-C)) to (B+or-C)
  if ((N1.getOpcode() == ISD::SUB || N1.getOpcode() == ISD::ADD) &&
      N1.getOperand(0).getOpcode() == ISD::SUB &&
      N0 == N1.getOperand(0).getOperand(1))
    return DAG.getNode(N1.getOpcode(), DL, VT, N1.getOperand(0).getOperand(0),
                       N1.getOperand(1));

  // fold (A-B)+(C-D) to (A+C)-(B+D) when A or C is constant
  if (N0.getOpcode() == ISD::SUB && N1.getOpcode() == ISD::SUB) {
    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);
    SDValue N10 = N1.getOperand(0);
    SDValue N11 = N1.getOperand(1);

    if (isConstantOrConstantVector(N00) || isConstantOrConstantVector(N10))
      return DAG.getNode(ISD::SUB, DL, VT,
                         DAG.getNode(ISD::ADD, SDLoc(N0), VT, N00, N10),
                         DAG.getNode(ISD::ADD, SDLoc(N1), VT, N01, N11));
  }

  // fold (add (umax X, C), -C) --> (usubsat X, C)
  if (N0.getOpcode() == ISD::UMAX && hasOperation(ISD::USUBSAT, VT)) {
    auto MatchUSUBSAT = [](ConstantSDNode *Max, ConstantSDNode *Op) {
      return (!Max && !Op) ||
             (Max && Op && Max->getAPIntValue() == (-Op->getAPIntValue()));
    };
    if (ISD::matchBinaryPredicate(N0.getOperand(1), N1, MatchUSUBSAT,
                                  /*AllowUndefs*/ true))
      return DAG.getNode(ISD::USUBSAT, DL, VT, N0.getOperand(0),
                         N0.getOperand(1));
  }

  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  if (isOneOrOneSplat(N1)) {
    // fold (add (xor a, -1), 1) -> (sub 0, a)
    if (isBitwiseNot(N0))
      return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT),
                         N0.getOperand(0));

    // fold (add (add (xor a, -1), b), 1) -> (sub b, a)
    if (N0.getOpcode() == ISD::ADD) {
      SDValue A, Xor;

      if (isBitwiseNot(N0.getOperand(0))) {
        A = N0.getOperand(1);
        Xor = N0.getOperand(0);
      } else if (isBitwiseNot(N0.getOperand(1))) {
        A = N0.getOperand(0);
        Xor = N0.getOperand(1);
      }

      if (Xor)
        return DAG.getNode(ISD::SUB, DL, VT, A, Xor.getOperand(0));
    }

    // Look for:
    //   add (add x, y), 1
    // And if the target does not like this form then turn into:
    //   sub y, (xor x, -1)
    if (!TLI.preferIncOfAddToSubOfNot(VT) && N0.hasOneUse() &&
        N0.getOpcode() == ISD::ADD) {
      SDValue Not = DAG.getNode(ISD::XOR, DL, VT, N0.getOperand(0),
                                DAG.getAllOnesConstant(DL, VT));
      return DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(1), Not);
    }
  }

  // (x - y) + -1  ->  add (xor y, -1), x
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SUB &&
      isAllOnesOrAllOnesSplat(N1)) {
    SDValue Xor = DAG.getNode(ISD::XOR, DL, VT, N0.getOperand(1), N1);
    return DAG.getNode(ISD::ADD, DL, VT, Xor, N0.getOperand(0));
  }

  if (SDValue Combined = visitADDLikeCommutative(N0, N1, N))
    return Combined;

  if (SDValue Combined = visitADDLikeCommutative(N1, N0, N))
    return Combined;

  return SDValue();
}

SDValue DAGCombiner::visitADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  if (SDValue Combined = visitADDLike(N))
    return Combined;

  if (SDValue V = foldAddSubBoolOfMaskedVal(N, DAG))
    return V;

  if (SDValue V = foldAddSubOfSignBit(N, DAG))
    return V;

  // fold (a+b) -> (a|b) iff a and b share no bits.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::OR, VT)) &&
      DAG.haveNoCommonBitsSet(N0, N1))
    return DAG.getNode(ISD::OR, DL, VT, N0, N1);

  // Fold (add (vscale * C0), (vscale * C1)) to (vscale * (C0 + C1)).
  if (N0.getOpcode() == ISD::VSCALE && N1.getOpcode() == ISD::VSCALE) {
    const APInt &C0 = N0->getConstantOperandAPInt(0);
    const APInt &C1 = N1->getConstantOperandAPInt(0);
    return DAG.getVScale(DL, VT, C0 + C1);
  }

  // fold a+vscale(c1)+vscale(c2) -> a+vscale(c1+c2)
  if ((N0.getOpcode() == ISD::ADD) &&
      (N0.getOperand(1).getOpcode() == ISD::VSCALE) &&
      (N1.getOpcode() == ISD::VSCALE)) {
    const APInt &VS0 = N0.getOperand(1)->getConstantOperandAPInt(0);
    const APInt &VS1 = N1->getConstantOperandAPInt(0);
    SDValue VS = DAG.getVScale(DL, VT, VS0 + VS1);
    return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0), VS);
  }

  // Fold (add step_vector(c1), step_vector(c2)  to step_vector(c1+c2))
  if (N0.getOpcode() == ISD::STEP_VECTOR &&
      N1.getOpcode() == ISD::STEP_VECTOR) {
    const APInt &C0 = N0->getConstantOperandAPInt(0);
    const APInt &C1 = N1->getConstantOperandAPInt(0);
    APInt NewStep = C0 + C1;
    return DAG.getStepVector(DL, VT, NewStep);
  }

  // Fold a + step_vector(c1) + step_vector(c2) to a + step_vector(c1+c2)
  if ((N0.getOpcode() == ISD::ADD) &&
      (N0.getOperand(1).getOpcode() == ISD::STEP_VECTOR) &&
      (N1.getOpcode() == ISD::STEP_VECTOR)) {
    const APInt &SV0 = N0.getOperand(1)->getConstantOperandAPInt(0);
    const APInt &SV1 = N1->getConstantOperandAPInt(0);
    APInt NewStep = SV0 + SV1;
    SDValue SV = DAG.getStepVector(DL, VT, NewStep);
    return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0), SV);
  }

  return SDValue();
}

SDValue DAGCombiner::visitADDSAT(SDNode *N) {
  unsigned Opcode = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  // fold vector ops
  if (VT.isVector()) {
    // TODO SimplifyVBinOp

    // fold (add_sat x, 0) -> x, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return N0;
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()))
      return N1;
  }

  // fold (add_sat x, undef) -> -1
  if (N0.isUndef() || N1.isUndef())
    return DAG.getAllOnesConstant(DL, VT);

  if (DAG.isConstantIntBuildVectorOrConstantInt(N0)) {
    // canonicalize constant to RHS
    if (!DAG.isConstantIntBuildVectorOrConstantInt(N1))
      return DAG.getNode(Opcode, DL, VT, N1, N0);
    // fold (add_sat c1, c2) -> c3
    return DAG.FoldConstantArithmetic(Opcode, DL, VT, {N0, N1});
  }

  // fold (add_sat x, 0) -> x
  if (isNullConstant(N1))
    return N0;

  // If it cannot overflow, transform into an add.
  if (Opcode == ISD::UADDSAT)
    if (DAG.computeOverflowKind(N0, N1) == SelectionDAG::OFK_Never)
      return DAG.getNode(ISD::ADD, DL, VT, N0, N1);

  return SDValue();
}

static SDValue getAsCarry(const TargetLowering &TLI, SDValue V) {
  bool Masked = false;

  // First, peel away TRUNCATE/ZERO_EXTEND/AND nodes due to legalization.
  while (true) {
    if (V.getOpcode() == ISD::TRUNCATE || V.getOpcode() == ISD::ZERO_EXTEND) {
      V = V.getOperand(0);
      continue;
    }

    if (V.getOpcode() == ISD::AND && isOneConstant(V.getOperand(1))) {
      Masked = true;
      V = V.getOperand(0);
      continue;
    }

    break;
  }

  // If this is not a carry, return.
  if (V.getResNo() != 1)
    return SDValue();

  if (V.getOpcode() != ISD::ADDCARRY && V.getOpcode() != ISD::SUBCARRY &&
      V.getOpcode() != ISD::UADDO && V.getOpcode() != ISD::USUBO)
    return SDValue();

  EVT VT = V.getNode()->getValueType(0);
  if (!TLI.isOperationLegalOrCustom(V.getOpcode(), VT))
    return SDValue();

  // If the result is masked, then no matter what kind of bool it is we can
  // return. If it isn't, then we need to make sure the bool type is either 0 or
  // 1 and not other values.
  if (Masked ||
      TLI.getBooleanContents(V.getValueType()) ==
          TargetLoweringBase::ZeroOrOneBooleanContent)
    return V;

  return SDValue();
}

/// Given the operands of an add/sub operation, see if the 2nd operand is a
/// masked 0/1 whose source operand is actually known to be 0/-1. If so, invert
/// the opcode and bypass the mask operation.
static SDValue foldAddSubMasked1(bool IsAdd, SDValue N0, SDValue N1,
                                 SelectionDAG &DAG, const SDLoc &DL) {
  if (N1.getOpcode() != ISD::AND || !isOneOrOneSplat(N1->getOperand(1)))
    return SDValue();

  EVT VT = N0.getValueType();
  if (DAG.ComputeNumSignBits(N1.getOperand(0)) != VT.getScalarSizeInBits())
    return SDValue();

  // add N0, (and (AssertSext X, i1), 1) --> sub N0, X
  // sub N0, (and (AssertSext X, i1), 1) --> add N0, X
  return DAG.getNode(IsAdd ? ISD::SUB : ISD::ADD, DL, VT, N0, N1.getOperand(0));
}

/// Helper for doing combines based on N0 and N1 being added to each other.
SDValue DAGCombiner::visitADDLikeCommutative(SDValue N0, SDValue N1,
                                          SDNode *LocReference) {
  EVT VT = N0.getValueType();
  SDLoc DL(LocReference);

  // fold (add x, shl(0 - y, n)) -> sub(x, shl(y, n))
  if (N1.getOpcode() == ISD::SHL && N1.getOperand(0).getOpcode() == ISD::SUB &&
      isNullOrNullSplat(N1.getOperand(0).getOperand(0)))
    return DAG.getNode(ISD::SUB, DL, VT, N0,
                       DAG.getNode(ISD::SHL, DL, VT,
                                   N1.getOperand(0).getOperand(1),
                                   N1.getOperand(1)));

  if (SDValue V = foldAddSubMasked1(true, N0, N1, DAG, DL))
    return V;

  // Look for:
  //   add (add x, 1), y
  // And if the target does not like this form then turn into:
  //   sub y, (xor x, -1)
  if (!TLI.preferIncOfAddToSubOfNot(VT) && N0.hasOneUse() &&
      N0.getOpcode() == ISD::ADD && isOneOrOneSplat(N0.getOperand(1))) {
    SDValue Not = DAG.getNode(ISD::XOR, DL, VT, N0.getOperand(0),
                              DAG.getAllOnesConstant(DL, VT));
    return DAG.getNode(ISD::SUB, DL, VT, N1, Not);
  }

  // Hoist one-use subtraction by non-opaque constant:
  //   (x - C) + y  ->  (x + y) - C
  // This is necessary because SUB(X,C) -> ADD(X,-C) doesn't work for vectors.
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SUB &&
      isConstantOrConstantVector(N0.getOperand(1), /*NoOpaques=*/true)) {
    SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0), N1);
    return DAG.getNode(ISD::SUB, DL, VT, Add, N0.getOperand(1));
  }
  // Hoist one-use subtraction from non-opaque constant:
  //   (C - x) + y  ->  (y - x) + C
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SUB &&
      isConstantOrConstantVector(N0.getOperand(0), /*NoOpaques=*/true)) {
    SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N1, N0.getOperand(1));
    return DAG.getNode(ISD::ADD, DL, VT, Sub, N0.getOperand(0));
  }

  // If the target's bool is represented as 0/1, prefer to make this 'sub 0/1'
  // rather than 'add 0/-1' (the zext should get folded).
  // add (sext i1 Y), X --> sub X, (zext i1 Y)
  if (N0.getOpcode() == ISD::SIGN_EXTEND &&
      N0.getOperand(0).getScalarValueSizeInBits() == 1 &&
      TLI.getBooleanContents(VT) == TargetLowering::ZeroOrOneBooleanContent) {
    SDValue ZExt = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0));
    return DAG.getNode(ISD::SUB, DL, VT, N1, ZExt);
  }

  // add X, (sextinreg Y i1) -> sub X, (and Y 1)
  if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    VTSDNode *TN = cast<VTSDNode>(N1.getOperand(1));
    if (TN->getVT() == MVT::i1) {
      SDValue ZExt = DAG.getNode(ISD::AND, DL, VT, N1.getOperand(0),
                                 DAG.getConstant(1, DL, VT));
      return DAG.getNode(ISD::SUB, DL, VT, N0, ZExt);
    }
  }

  // (add X, (addcarry Y, 0, Carry)) -> (addcarry X, Y, Carry)
  if (N1.getOpcode() == ISD::ADDCARRY && isNullConstant(N1.getOperand(1)) &&
      N1.getResNo() == 0)
    return DAG.getNode(ISD::ADDCARRY, DL, N1->getVTList(),
                       N0, N1.getOperand(0), N1.getOperand(2));

  // (add X, Carry) -> (addcarry X, 0, Carry)
  if (TLI.isOperationLegalOrCustom(ISD::ADDCARRY, VT))
    if (SDValue Carry = getAsCarry(TLI, N1))
      return DAG.getNode(ISD::ADDCARRY, DL,
                         DAG.getVTList(VT, Carry.getValueType()), N0,
                         DAG.getConstant(0, DL, VT), Carry);

  return SDValue();
}

SDValue DAGCombiner::visitADDC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  // If the flag result is dead, turn this into an ADD.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::ADD, DL, VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE, DL, MVT::Glue));

  // canonicalize constant to RHS.
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDC, DL, N->getVTList(), N1, N0);

  // fold (addc x, 0) -> x + no carry out
  if (isNullConstant(N1))
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE,
                                        DL, MVT::Glue));

  // If it cannot overflow, transform into an add.
  if (DAG.computeOverflowKind(N0, N1) == SelectionDAG::OFK_Never)
    return CombineTo(N, DAG.getNode(ISD::ADD, DL, VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE, DL, MVT::Glue));

  return SDValue();
}

/**
 * Flips a boolean if it is cheaper to compute. If the Force parameters is set,
 * then the flip also occurs if computing the inverse is the same cost.
 * This function returns an empty SDValue in case it cannot flip the boolean
 * without increasing the cost of the computation. If you want to flip a boolean
 * no matter what, use DAG.getLogicalNOT.
 */
static SDValue extractBooleanFlip(SDValue V, SelectionDAG &DAG,
                                  const TargetLowering &TLI,
                                  bool Force) {
  if (Force && isa<ConstantSDNode>(V))
    return DAG.getLogicalNOT(SDLoc(V), V, V.getValueType());

  if (V.getOpcode() != ISD::XOR)
    return SDValue();

  ConstantSDNode *Const = isConstOrConstSplat(V.getOperand(1), false);
  if (!Const)
    return SDValue();

  EVT VT = V.getValueType();

  bool IsFlip = false;
  switch(TLI.getBooleanContents(VT)) {
    case TargetLowering::ZeroOrOneBooleanContent:
      IsFlip = Const->isOne();
      break;
    case TargetLowering::ZeroOrNegativeOneBooleanContent:
      IsFlip = Const->isAllOnesValue();
      break;
    case TargetLowering::UndefinedBooleanContent:
      IsFlip = (Const->getAPIntValue() & 0x01) == 1;
      break;
  }

  if (IsFlip)
    return V.getOperand(0);
  if (Force)
    return DAG.getLogicalNOT(SDLoc(V), V, V.getValueType());
  return SDValue();
}

SDValue DAGCombiner::visitADDO(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  bool IsSigned = (ISD::SADDO == N->getOpcode());

  EVT CarryVT = N->getValueType(1);
  SDLoc DL(N);

  // If the flag result is dead, turn this into an ADD.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::ADD, DL, VT, N0, N1),
                     DAG.getUNDEF(CarryVT));

  // canonicalize constant to RHS.
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(N->getOpcode(), DL, N->getVTList(), N1, N0);

  // fold (addo x, 0) -> x + no carry out
  if (isNullOrNullSplat(N1))
    return CombineTo(N, N0, DAG.getConstant(0, DL, CarryVT));

  if (!IsSigned) {
    // If it cannot overflow, transform into an add.
    if (DAG.computeOverflowKind(N0, N1) == SelectionDAG::OFK_Never)
      return CombineTo(N, DAG.getNode(ISD::ADD, DL, VT, N0, N1),
                       DAG.getConstant(0, DL, CarryVT));

    // fold (uaddo (xor a, -1), 1) -> (usub 0, a) and flip carry.
    if (isBitwiseNot(N0) && isOneOrOneSplat(N1)) {
      SDValue Sub = DAG.getNode(ISD::USUBO, DL, N->getVTList(),
                                DAG.getConstant(0, DL, VT), N0.getOperand(0));
      return CombineTo(
          N, Sub, DAG.getLogicalNOT(DL, Sub.getValue(1), Sub->getValueType(1)));
    }

    if (SDValue Combined = visitUADDOLike(N0, N1, N))
      return Combined;

    if (SDValue Combined = visitUADDOLike(N1, N0, N))
      return Combined;
  }

  return SDValue();
}

SDValue DAGCombiner::visitUADDOLike(SDValue N0, SDValue N1, SDNode *N) {
  EVT VT = N0.getValueType();
  if (VT.isVector())
    return SDValue();

  // (uaddo X, (addcarry Y, 0, Carry)) -> (addcarry X, Y, Carry)
  // If Y + 1 cannot overflow.
  if (N1.getOpcode() == ISD::ADDCARRY && isNullConstant(N1.getOperand(1))) {
    SDValue Y = N1.getOperand(0);
    SDValue One = DAG.getConstant(1, SDLoc(N), Y.getValueType());
    if (DAG.computeOverflowKind(Y, One) == SelectionDAG::OFK_Never)
      return DAG.getNode(ISD::ADDCARRY, SDLoc(N), N->getVTList(), N0, Y,
                         N1.getOperand(2));
  }

  // (uaddo X, Carry) -> (addcarry X, 0, Carry)
  if (TLI.isOperationLegalOrCustom(ISD::ADDCARRY, VT))
    if (SDValue Carry = getAsCarry(TLI, N1))
      return DAG.getNode(ISD::ADDCARRY, SDLoc(N), N->getVTList(), N0,
                         DAG.getConstant(0, SDLoc(N), VT), Carry);

  return SDValue();
}

SDValue DAGCombiner::visitADDE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // canonicalize constant to RHS
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDE, SDLoc(N), N->getVTList(),
                       N1, N0, CarryIn);

  // fold (adde x, y, false) -> (addc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::ADDC, SDLoc(N), N->getVTList(), N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitADDCARRY(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);
  SDLoc DL(N);

  // canonicalize constant to RHS
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADDCARRY, DL, N->getVTList(), N1, N0, CarryIn);

  // fold (addcarry x, y, false) -> (uaddo x, y)
  if (isNullConstant(CarryIn)) {
    if (!LegalOperations ||
        TLI.isOperationLegalOrCustom(ISD::UADDO, N->getValueType(0)))
      return DAG.getNode(ISD::UADDO, DL, N->getVTList(), N0, N1);
  }

  // fold (addcarry 0, 0, X) -> (and (ext/trunc X), 1) and no carry.
  if (isNullConstant(N0) && isNullConstant(N1)) {
    EVT VT = N0.getValueType();
    EVT CarryVT = CarryIn.getValueType();
    SDValue CarryExt = DAG.getBoolExtOrTrunc(CarryIn, DL, VT, CarryVT);
    AddToWorklist(CarryExt.getNode());
    return CombineTo(N, DAG.getNode(ISD::AND, DL, VT, CarryExt,
                                    DAG.getConstant(1, DL, VT)),
                     DAG.getConstant(0, DL, CarryVT));
  }

  if (SDValue Combined = visitADDCARRYLike(N0, N1, CarryIn, N))
    return Combined;

  if (SDValue Combined = visitADDCARRYLike(N1, N0, CarryIn, N))
    return Combined;

  return SDValue();
}

SDValue DAGCombiner::visitSADDO_CARRY(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);
  SDLoc DL(N);

  // canonicalize constant to RHS
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (N0C && !N1C)
    return DAG.getNode(ISD::SADDO_CARRY, DL, N->getVTList(), N1, N0, CarryIn);

  // fold (saddo_carry x, y, false) -> (saddo x, y)
  if (isNullConstant(CarryIn)) {
    if (!LegalOperations ||
        TLI.isOperationLegalOrCustom(ISD::SADDO, N->getValueType(0)))
      return DAG.getNode(ISD::SADDO, DL, N->getVTList(), N0, N1);
  }

  return SDValue();
}

/**
 * If we are facing some sort of diamond carry propapagtion pattern try to
 * break it up to generate something like:
 *   (addcarry X, 0, (addcarry A, B, Z):Carry)
 *
 * The end result is usually an increase in operation required, but because the
 * carry is now linearized, other tranforms can kick in and optimize the DAG.
 *
 * Patterns typically look something like
 *            (uaddo A, B)
 *             /       \
 *          Carry      Sum
 *            |          \
 *            | (addcarry *, 0, Z)
 *            |       /
 *             \   Carry
 *              |   /
 * (addcarry X, *, *)
 *
 * But numerous variation exist. Our goal is to identify A, B, X and Z and
 * produce a combine with a single path for carry propagation.
 */
static SDValue combineADDCARRYDiamond(DAGCombiner &Combiner, SelectionDAG &DAG,
                                      SDValue X, SDValue Carry0, SDValue Carry1,
                                      SDNode *N) {
  if (Carry1.getResNo() != 1 || Carry0.getResNo() != 1)
    return SDValue();
  if (Carry1.getOpcode() != ISD::UADDO)
    return SDValue();

  SDValue Z;

  /**
   * First look for a suitable Z. It will present itself in the form of
   * (addcarry Y, 0, Z) or its equivalent (uaddo Y, 1) for Z=true
   */
  if (Carry0.getOpcode() == ISD::ADDCARRY &&
      isNullConstant(Carry0.getOperand(1))) {
    Z = Carry0.getOperand(2);
  } else if (Carry0.getOpcode() == ISD::UADDO &&
             isOneConstant(Carry0.getOperand(1))) {
    EVT VT = Combiner.getSetCCResultType(Carry0.getValueType());
    Z = DAG.getConstant(1, SDLoc(Carry0.getOperand(1)), VT);
  } else {
    // We couldn't find a suitable Z.
    return SDValue();
  }


  auto cancelDiamond = [&](SDValue A,SDValue B) {
    SDLoc DL(N);
    SDValue NewY = DAG.getNode(ISD::ADDCARRY, DL, Carry0->getVTList(), A, B, Z);
    Combiner.AddToWorklist(NewY.getNode());
    return DAG.getNode(ISD::ADDCARRY, DL, N->getVTList(), X,
                       DAG.getConstant(0, DL, X.getValueType()),
                       NewY.getValue(1));
  };

  /**
   *      (uaddo A, B)
   *           |
   *          Sum
   *           |
   * (addcarry *, 0, Z)
   */
  if (Carry0.getOperand(0) == Carry1.getValue(0)) {
    return cancelDiamond(Carry1.getOperand(0), Carry1.getOperand(1));
  }

  /**
   * (addcarry A, 0, Z)
   *         |
   *        Sum
   *         |
   *  (uaddo *, B)
   */
  if (Carry1.getOperand(0) == Carry0.getValue(0)) {
    return cancelDiamond(Carry0.getOperand(0), Carry1.getOperand(1));
  }

  if (Carry1.getOperand(1) == Carry0.getValue(0)) {
    return cancelDiamond(Carry1.getOperand(0), Carry0.getOperand(0));
  }

  return SDValue();
}

// If we are facing some sort of diamond carry/borrow in/out pattern try to
// match patterns like:
//
//          (uaddo A, B)            CarryIn
//            |  \                     |
//            |   \                    |
//    PartialSum   PartialCarryOutX   /
//            |        |             /
//            |    ____|____________/
//            |   /    |
//     (uaddo *, *)    \________
//       |  \                   \
//       |   \                   |
//       |    PartialCarryOutY   |
//       |        \              |
//       |         \            /
//   AddCarrySum    |    ______/
//                  |   /
//   CarryOut = (or *, *)
//
// And generate ADDCARRY (or SUBCARRY) with two result values:
//
//    {AddCarrySum, CarryOut} = (addcarry A, B, CarryIn)
//
// Our goal is to identify A, B, and CarryIn and produce ADDCARRY/SUBCARRY with
// a single path for carry/borrow out propagation:
static SDValue combineCarryDiamond(DAGCombiner &Combiner, SelectionDAG &DAG,
                                   const TargetLowering &TLI, SDValue Carry0,
                                   SDValue Carry1, SDNode *N) {
  if (Carry0.getResNo() != 1 || Carry1.getResNo() != 1)
    return SDValue();
  unsigned Opcode = Carry0.getOpcode();
  if (Opcode != Carry1.getOpcode())
    return SDValue();
  if (Opcode != ISD::UADDO && Opcode != ISD::USUBO)
    return SDValue();

  // Canonicalize the add/sub of A and B as Carry0 and the add/sub of the
  // carry/borrow in as Carry1. (The top and middle uaddo nodes respectively in
  // the above ASCII art.)
  if (Carry1.getOperand(0) != Carry0.getValue(0) &&
      Carry1.getOperand(1) != Carry0.getValue(0))
    std::swap(Carry0, Carry1);
  if (Carry1.getOperand(0) != Carry0.getValue(0) &&
      Carry1.getOperand(1) != Carry0.getValue(0))
    return SDValue();

  // The carry in value must be on the righthand side for subtraction.
  unsigned CarryInOperandNum =
      Carry1.getOperand(0) == Carry0.getValue(0) ? 1 : 0;
  if (Opcode == ISD::USUBO && CarryInOperandNum != 1)
    return SDValue();
  SDValue CarryIn = Carry1.getOperand(CarryInOperandNum);

  unsigned NewOp = Opcode == ISD::UADDO ? ISD::ADDCARRY : ISD::SUBCARRY;
  if (!TLI.isOperationLegalOrCustom(NewOp, Carry0.getValue(0).getValueType()))
    return SDValue();

  // Verify that the carry/borrow in is plausibly a carry/borrow bit.
  // TODO: make getAsCarry() aware of how partial carries are merged.
  if (CarryIn.getOpcode() != ISD::ZERO_EXTEND)
    return SDValue();
  CarryIn = CarryIn.getOperand(0);
  if (CarryIn.getValueType() != MVT::i1)
    return SDValue();

  SDLoc DL(N);
  SDValue Merged =
      DAG.getNode(NewOp, DL, Carry1->getVTList(), Carry0.getOperand(0),
                  Carry0.getOperand(1), CarryIn);

  // Please note that because we have proven that the result of the UADDO/USUBO
  // of A and B feeds into the UADDO/USUBO that does the carry/borrow in, we can
  // therefore prove that if the first UADDO/USUBO overflows, the second
  // UADDO/USUBO cannot. For example consider 8-bit numbers where 0xFF is the
  // maximum value.
  //
  //   0xFF + 0xFF == 0xFE with carry but 0xFE + 1 does not carry
  //   0x00 - 0xFF == 1 with a carry/borrow but 1 - 1 == 0 (no carry/borrow)
  //
  // This is important because it means that OR and XOR can be used to merge
  // carry flags; and that AND can return a constant zero.
  //
  // TODO: match other operations that can merge flags (ADD, etc)
  DAG.ReplaceAllUsesOfValueWith(Carry1.getValue(0), Merged.getValue(0));
  if (N->getOpcode() == ISD::AND)
    return DAG.getConstant(0, DL, MVT::i1);
  return Merged.getValue(1);
}

SDValue DAGCombiner::visitADDCARRYLike(SDValue N0, SDValue N1, SDValue CarryIn,
                                       SDNode *N) {
  // fold (addcarry (xor a, -1), b, c) -> (subcarry b, a, !c) and flip carry.
  if (isBitwiseNot(N0))
    if (SDValue NotC = extractBooleanFlip(CarryIn, DAG, TLI, true)) {
      SDLoc DL(N);
      SDValue Sub = DAG.getNode(ISD::SUBCARRY, DL, N->getVTList(), N1,
                                N0.getOperand(0), NotC);
      return CombineTo(
          N, Sub, DAG.getLogicalNOT(DL, Sub.getValue(1), Sub->getValueType(1)));
    }

  // Iff the flag result is dead:
  // (addcarry (add|uaddo X, Y), 0, Carry) -> (addcarry X, Y, Carry)
  // Don't do this if the Carry comes from the uaddo. It won't remove the uaddo
  // or the dependency between the instructions.
  if ((N0.getOpcode() == ISD::ADD ||
       (N0.getOpcode() == ISD::UADDO && N0.getResNo() == 0 &&
        N0.getValue(1) != CarryIn)) &&
      isNullConstant(N1) && !N->hasAnyUseOfValue(1))
    return DAG.getNode(ISD::ADDCARRY, SDLoc(N), N->getVTList(),
                       N0.getOperand(0), N0.getOperand(1), CarryIn);

  /**
   * When one of the addcarry argument is itself a carry, we may be facing
   * a diamond carry propagation. In which case we try to transform the DAG
   * to ensure linear carry propagation if that is possible.
   */
  if (auto Y = getAsCarry(TLI, N1)) {
    // Because both are carries, Y and Z can be swapped.
    if (auto R = combineADDCARRYDiamond(*this, DAG, N0, Y, CarryIn, N))
      return R;
    if (auto R = combineADDCARRYDiamond(*this, DAG, N0, CarryIn, Y, N))
      return R;
  }

  return SDValue();
}

// Attempt to create a USUBSAT(LHS, RHS) node with DstVT, performing a
// clamp/truncation if necessary.
static SDValue getTruncatedUSUBSAT(EVT DstVT, EVT SrcVT, SDValue LHS,
                                   SDValue RHS, SelectionDAG &DAG,
                                   const SDLoc &DL) {
  assert(DstVT.getScalarSizeInBits() <= SrcVT.getScalarSizeInBits() &&
         "Illegal truncation");

  if (DstVT == SrcVT)
    return DAG.getNode(ISD::USUBSAT, DL, DstVT, LHS, RHS);

  // If the LHS is zero-extended then we can perform the USUBSAT as DstVT by
  // clamping RHS.
  APInt UpperBits = APInt::getBitsSetFrom(SrcVT.getScalarSizeInBits(),
                                          DstVT.getScalarSizeInBits());
  if (!DAG.MaskedValueIsZero(LHS, UpperBits))
    return SDValue();

  SDValue SatLimit =
      DAG.getConstant(APInt::getLowBitsSet(SrcVT.getScalarSizeInBits(),
                                           DstVT.getScalarSizeInBits()),
                      DL, SrcVT);
  RHS = DAG.getNode(ISD::UMIN, DL, SrcVT, RHS, SatLimit);
  RHS = DAG.getNode(ISD::TRUNCATE, DL, DstVT, RHS);
  LHS = DAG.getNode(ISD::TRUNCATE, DL, DstVT, LHS);
  return DAG.getNode(ISD::USUBSAT, DL, DstVT, LHS, RHS);
}

// Try to find umax(a,b) - b or a - umin(a,b) patterns that may be converted to
// usubsat(a,b), optionally as a truncated type.
SDValue DAGCombiner::foldSubToUSubSat(EVT DstVT, SDNode *N) {
  if (N->getOpcode() != ISD::SUB ||
      !(!LegalOperations || hasOperation(ISD::USUBSAT, DstVT)))
    return SDValue();

  EVT SubVT = N->getValueType(0);
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);

  // Try to find umax(a,b) - b or a - umin(a,b) patterns
  // they may be converted to usubsat(a,b).
  if (Op0.getOpcode() == ISD::UMAX && Op0.hasOneUse()) {
    SDValue MaxLHS = Op0.getOperand(0);
    SDValue MaxRHS = Op0.getOperand(1);
    if (MaxLHS == Op1)
      return getTruncatedUSUBSAT(DstVT, SubVT, MaxRHS, Op1, DAG, SDLoc(N));
    if (MaxRHS == Op1)
      return getTruncatedUSUBSAT(DstVT, SubVT, MaxLHS, Op1, DAG, SDLoc(N));
  }

  if (Op1.getOpcode() == ISD::UMIN && Op1.hasOneUse()) {
    SDValue MinLHS = Op1.getOperand(0);
    SDValue MinRHS = Op1.getOperand(1);
    if (MinLHS == Op0)
      return getTruncatedUSUBSAT(DstVT, SubVT, Op0, MinRHS, DAG, SDLoc(N));
    if (MinRHS == Op0)
      return getTruncatedUSUBSAT(DstVT, SubVT, Op0, MinLHS, DAG, SDLoc(N));
  }

  // sub(a,trunc(umin(zext(a),b))) -> usubsat(a,trunc(umin(b,SatLimit)))
  if (Op1.getOpcode() == ISD::TRUNCATE &&
      Op1.getOperand(0).getOpcode() == ISD::UMIN &&
      Op1.getOperand(0).hasOneUse()) {
    SDValue MinLHS = Op1.getOperand(0).getOperand(0);
    SDValue MinRHS = Op1.getOperand(0).getOperand(1);
    if (MinLHS.getOpcode() == ISD::ZERO_EXTEND && MinLHS.getOperand(0) == Op0)
      return getTruncatedUSUBSAT(DstVT, MinLHS.getValueType(), MinLHS, MinRHS,
                                 DAG, SDLoc(N));
    if (MinRHS.getOpcode() == ISD::ZERO_EXTEND && MinRHS.getOperand(0) == Op0)
      return getTruncatedUSUBSAT(DstVT, MinLHS.getValueType(), MinRHS, MinLHS,
                                 DAG, SDLoc(N));
  }

  return SDValue();
}

// Since it may not be valid to emit a fold to zero for vector initializers
// check if we can before folding.
static SDValue tryFoldToZero(const SDLoc &DL, const TargetLowering &TLI, EVT VT,
                             SelectionDAG &DAG, bool LegalOperations) {
  if (!VT.isVector())
    return DAG.getConstant(0, DL, VT);
  if (!LegalOperations || TLI.isOperationLegal(ISD::BUILD_VECTOR, VT))
    return DAG.getConstant(0, DL, VT);
  return SDValue();
}

SDValue DAGCombiner::visitSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (sub x, 0) -> x, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return N0;
  }

  // fold (sub x, x) -> 0
  // FIXME: Refactor this and xor and other similar operations together.
  if (N0 == N1)
    return tryFoldToZero(DL, TLI, VT, DAG, LegalOperations);

  // fold (sub c1, c2) -> c3
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::SUB, DL, VT, {N0, N1}))
    return C;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  ConstantSDNode *N1C = getAsNonOpaqueConstant(N1);

  // fold (sub x, c) -> (add x, -c)
  if (N1C) {
    return DAG.getNode(ISD::ADD, DL, VT, N0,
                       DAG.getConstant(-N1C->getAPIntValue(), DL, VT));
  }

  if (isNullOrNullSplat(N0)) {
    unsigned BitWidth = VT.getScalarSizeInBits();
    // Right-shifting everything out but the sign bit followed by negation is
    // the same as flipping arithmetic/logical shift type without the negation:
    // -(X >>u 31) -> (X >>s 31)
    // -(X >>s 31) -> (X >>u 31)
    if (N1->getOpcode() == ISD::SRA || N1->getOpcode() == ISD::SRL) {
      ConstantSDNode *ShiftAmt = isConstOrConstSplat(N1.getOperand(1));
      if (ShiftAmt && ShiftAmt->getAPIntValue() == (BitWidth - 1)) {
        auto NewSh = N1->getOpcode() == ISD::SRA ? ISD::SRL : ISD::SRA;
        if (!LegalOperations || TLI.isOperationLegal(NewSh, VT))
          return DAG.getNode(NewSh, DL, VT, N1.getOperand(0), N1.getOperand(1));
      }
    }

    // 0 - X --> 0 if the sub is NUW.
    if (N->getFlags().hasNoUnsignedWrap())
      return N0;

    if (DAG.MaskedValueIsZero(N1, ~APInt::getSignMask(BitWidth))) {
      // N1 is either 0 or the minimum signed value. If the sub is NSW, then
      // N1 must be 0 because negating the minimum signed value is undefined.
      if (N->getFlags().hasNoSignedWrap())
        return N0;

      // 0 - X --> X if X is 0 or the minimum signed value.
      return N1;
    }

    // Convert 0 - abs(x).
    SDValue Result;
    if (N1->getOpcode() == ISD::ABS &&
        !TLI.isOperationLegalOrCustom(ISD::ABS, VT) &&
        TLI.expandABS(N1.getNode(), Result, DAG, true))
      return Result;

    // Fold neg(splat(neg(x)) -> splat(x)
    if (VT.isVector()) {
      SDValue N1S = DAG.getSplatValue(N1, true);
      if (N1S && N1S.getOpcode() == ISD::SUB &&
          isNullConstant(N1S.getOperand(0))) {
        if (VT.isScalableVector())
          return DAG.getSplatVector(VT, DL, N1S.getOperand(1));
        return DAG.getSplatBuildVector(VT, DL, N1S.getOperand(1));
      }
    }
  }

  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1)
  if (isAllOnesOrAllOnesSplat(N0))
    return DAG.getNode(ISD::XOR, DL, VT, N1, N0);

  // fold (A - (0-B)) -> A+B
  if (N1.getOpcode() == ISD::SUB && isNullOrNullSplat(N1.getOperand(0)))
    return DAG.getNode(ISD::ADD, DL, VT, N0, N1.getOperand(1));

  // fold A-(A-B) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(0))
    return N1.getOperand(1);

  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1)
    return N0.getOperand(1);

  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1)
    return N0.getOperand(0);

  // fold (A+C1)-C2 -> A+(C1-C2)
  if (N0.getOpcode() == ISD::ADD &&
      isConstantOrConstantVector(N1, /* NoOpaques */ true) &&
      isConstantOrConstantVector(N0.getOperand(1), /* NoOpaques */ true)) {
    SDValue NewC =
        DAG.FoldConstantArithmetic(ISD::SUB, DL, VT, {N0.getOperand(1), N1});
    assert(NewC && "Constant folding failed");
    return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0), NewC);
  }

  // fold C2-(A+C1) -> (C2-C1)-A
  if (N1.getOpcode() == ISD::ADD) {
    SDValue N11 = N1.getOperand(1);
    if (isConstantOrConstantVector(N0, /* NoOpaques */ true) &&
        isConstantOrConstantVector(N11, /* NoOpaques */ true)) {
      SDValue NewC = DAG.FoldConstantArithmetic(ISD::SUB, DL, VT, {N0, N11});
      assert(NewC && "Constant folding failed");
      return DAG.getNode(ISD::SUB, DL, VT, NewC, N1.getOperand(0));
    }
  }

  // fold (A-C1)-C2 -> A-(C1+C2)
  if (N0.getOpcode() == ISD::SUB &&
      isConstantOrConstantVector(N1, /* NoOpaques */ true) &&
      isConstantOrConstantVector(N0.getOperand(1), /* NoOpaques */ true)) {
    SDValue NewC =
        DAG.FoldConstantArithmetic(ISD::ADD, DL, VT, {N0.getOperand(1), N1});
    assert(NewC && "Constant folding failed");
    return DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(0), NewC);
  }

  // fold (c1-A)-c2 -> (c1-c2)-A
  if (N0.getOpcode() == ISD::SUB &&
      isConstantOrConstantVector(N1, /* NoOpaques */ true) &&
      isConstantOrConstantVector(N0.getOperand(0), /* NoOpaques */ true)) {
    SDValue NewC =
        DAG.FoldConstantArithmetic(ISD::SUB, DL, VT, {N0.getOperand(0), N1});
    assert(NewC && "Constant folding failed");
    return DAG.getNode(ISD::SUB, DL, VT, NewC, N0.getOperand(1));
  }

  // fold ((A+(B+or-C))-B) -> A+or-C
  if (N0.getOpcode() == ISD::ADD &&
      (N0.getOperand(1).getOpcode() == ISD::SUB ||
       N0.getOperand(1).getOpcode() == ISD::ADD) &&
      N0.getOperand(1).getOperand(0) == N1)
    return DAG.getNode(N0.getOperand(1).getOpcode(), DL, VT, N0.getOperand(0),
                       N0.getOperand(1).getOperand(1));

  // fold ((A+(C+B))-B) -> A+C
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1).getOpcode() == ISD::ADD &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(0),
                       N0.getOperand(1).getOperand(0));

  // fold ((A-(B-C))-C) -> A-B
  if (N0.getOpcode() == ISD::SUB && N0.getOperand(1).getOpcode() == ISD::SUB &&
      N0.getOperand(1).getOperand(1) == N1)
    return DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(0),
                       N0.getOperand(1).getOperand(0));

  // fold (A-(B-C)) -> A+(C-B)
  if (N1.getOpcode() == ISD::SUB && N1.hasOneUse())
    return DAG.getNode(ISD::ADD, DL, VT, N0,
                       DAG.getNode(ISD::SUB, DL, VT, N1.getOperand(1),
                                   N1.getOperand(0)));

  // A - (A & B)  ->  A & (~B)
  if (N1.getOpcode() == ISD::AND) {
    SDValue A = N1.getOperand(0);
    SDValue B = N1.getOperand(1);
    if (A != N0)
      std::swap(A, B);
    if (A == N0 &&
        (N1.hasOneUse() || isConstantOrConstantVector(B, /*NoOpaques=*/true))) {
      SDValue InvB =
          DAG.getNode(ISD::XOR, DL, VT, B, DAG.getAllOnesConstant(DL, VT));
      return DAG.getNode(ISD::AND, DL, VT, A, InvB);
    }
  }

  // fold (X - (-Y * Z)) -> (X + (Y * Z))
  if (N1.getOpcode() == ISD::MUL && N1.hasOneUse()) {
    if (N1.getOperand(0).getOpcode() == ISD::SUB &&
        isNullOrNullSplat(N1.getOperand(0).getOperand(0))) {
      SDValue Mul = DAG.getNode(ISD::MUL, DL, VT,
                                N1.getOperand(0).getOperand(1),
                                N1.getOperand(1));
      return DAG.getNode(ISD::ADD, DL, VT, N0, Mul);
    }
    if (N1.getOperand(1).getOpcode() == ISD::SUB &&
        isNullOrNullSplat(N1.getOperand(1).getOperand(0))) {
      SDValue Mul = DAG.getNode(ISD::MUL, DL, VT,
                                N1.getOperand(0),
                                N1.getOperand(1).getOperand(1));
      return DAG.getNode(ISD::ADD, DL, VT, N0, Mul);
    }
  }

  // If either operand of a sub is undef, the result is undef
  if (N0.isUndef())
    return N0;
  if (N1.isUndef())
    return N1;

  if (SDValue V = foldAddSubBoolOfMaskedVal(N, DAG))
    return V;

  if (SDValue V = foldAddSubOfSignBit(N, DAG))
    return V;

  if (SDValue V = foldAddSubMasked1(false, N0, N1, DAG, SDLoc(N)))
    return V;

  if (SDValue V = foldSubToUSubSat(VT, N))
    return V;

  // (x - y) - 1  ->  add (xor y, -1), x
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SUB && isOneOrOneSplat(N1)) {
    SDValue Xor = DAG.getNode(ISD::XOR, DL, VT, N0.getOperand(1),
                              DAG.getAllOnesConstant(DL, VT));
    return DAG.getNode(ISD::ADD, DL, VT, Xor, N0.getOperand(0));
  }

  // Look for:
  //   sub y, (xor x, -1)
  // And if the target does not like this form then turn into:
  //   add (add x, y), 1
  if (TLI.preferIncOfAddToSubOfNot(VT) && N1.hasOneUse() && isBitwiseNot(N1)) {
    SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N0, N1.getOperand(0));
    return DAG.getNode(ISD::ADD, DL, VT, Add, DAG.getConstant(1, DL, VT));
  }

  // Hoist one-use addition by non-opaque constant:
  //   (x + C) - y  ->  (x - y) + C
  if (N0.hasOneUse() && N0.getOpcode() == ISD::ADD &&
      isConstantOrConstantVector(N0.getOperand(1), /*NoOpaques=*/true)) {
    SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(0), N1);
    return DAG.getNode(ISD::ADD, DL, VT, Sub, N0.getOperand(1));
  }
  // y - (x + C)  ->  (y - x) - C
  if (N1.hasOneUse() && N1.getOpcode() == ISD::ADD &&
      isConstantOrConstantVector(N1.getOperand(1), /*NoOpaques=*/true)) {
    SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N0, N1.getOperand(0));
    return DAG.getNode(ISD::SUB, DL, VT, Sub, N1.getOperand(1));
  }
  // (x - C) - y  ->  (x - y) - C
  // This is necessary because SUB(X,C) -> ADD(X,-C) doesn't work for vectors.
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SUB &&
      isConstantOrConstantVector(N0.getOperand(1), /*NoOpaques=*/true)) {
    SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(0), N1);
    return DAG.getNode(ISD::SUB, DL, VT, Sub, N0.getOperand(1));
  }
  // (C - x) - y  ->  C - (x + y)
  if (N0.hasOneUse() && N0.getOpcode() == ISD::SUB &&
      isConstantOrConstantVector(N0.getOperand(0), /*NoOpaques=*/true)) {
    SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(1), N1);
    return DAG.getNode(ISD::SUB, DL, VT, N0.getOperand(0), Add);
  }

  // If the target's bool is represented as 0/-1, prefer to make this 'add 0/-1'
  // rather than 'sub 0/1' (the sext should get folded).
  // sub X, (zext i1 Y) --> add X, (sext i1 Y)
  if (N1.getOpcode() == ISD::ZERO_EXTEND &&
      N1.getOperand(0).getScalarValueSizeInBits() == 1 &&
      TLI.getBooleanContents(VT) ==
          TargetLowering::ZeroOrNegativeOneBooleanContent) {
    SDValue SExt = DAG.getNode(ISD::SIGN_EXTEND, DL, VT, N1.getOperand(0));
    return DAG.getNode(ISD::ADD, DL, VT, N0, SExt);
  }

  // fold Y = sra (X, size(X)-1); sub (xor (X, Y), Y) -> (abs X)
  if (TLI.isOperationLegalOrCustom(ISD::ABS, VT)) {
    if (N0.getOpcode() == ISD::XOR && N1.getOpcode() == ISD::SRA) {
      SDValue X0 = N0.getOperand(0), X1 = N0.getOperand(1);
      SDValue S0 = N1.getOperand(0);
      if ((X0 == S0 && X1 == N1) || (X0 == N1 && X1 == S0))
        if (ConstantSDNode *C = isConstOrConstSplat(N1.getOperand(1)))
          if (C->getAPIntValue() == (VT.getScalarSizeInBits() - 1))
            return DAG.getNode(ISD::ABS, SDLoc(N), VT, S0);
    }
  }

  // If the relocation model supports it, consider symbol offsets.
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(N0))
    if (!LegalOperations && TLI.isOffsetFoldingLegal(GA)) {
      // fold (sub Sym, c) -> Sym-c
      if (N1C && GA->getOpcode() == ISD::GlobalAddress)
        return DAG.getGlobalAddress(GA->getGlobal(), SDLoc(N1C), VT,
                                    GA->getOffset() -
                                        (uint64_t)N1C->getSExtValue());
      // fold (sub Sym+c1, Sym+c2) -> c1-c2
      if (GlobalAddressSDNode *GB = dyn_cast<GlobalAddressSDNode>(N1))
        if (GA->getGlobal() == GB->getGlobal())
          return DAG.getConstant((uint64_t)GA->getOffset() - GB->getOffset(),
                                 DL, VT);
    }

  // sub X, (sextinreg Y i1) -> add X, (and Y 1)
  if (N1.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    VTSDNode *TN = cast<VTSDNode>(N1.getOperand(1));
    if (TN->getVT() == MVT::i1) {
      SDValue ZExt = DAG.getNode(ISD::AND, DL, VT, N1.getOperand(0),
                                 DAG.getConstant(1, DL, VT));
      return DAG.getNode(ISD::ADD, DL, VT, N0, ZExt);
    }
  }

  // canonicalize (sub X, (vscale * C)) to (add X, (vscale * -C))
  if (N1.getOpcode() == ISD::VSCALE) {
    const APInt &IntVal = N1.getConstantOperandAPInt(0);
    return DAG.getNode(ISD::ADD, DL, VT, N0, DAG.getVScale(DL, VT, -IntVal));
  }

  // canonicalize (sub X, step_vector(C)) to (add X, step_vector(-C))
  if (N1.getOpcode() == ISD::STEP_VECTOR && N1.hasOneUse()) {
    APInt NewStep = -N1.getConstantOperandAPInt(0);
    return DAG.getNode(ISD::ADD, DL, VT, N0,
                       DAG.getStepVector(DL, VT, NewStep));
  }

  // Prefer an add for more folding potential and possibly better codegen:
  // sub N0, (lshr N10, width-1) --> add N0, (ashr N10, width-1)
  if (!LegalOperations && N1.getOpcode() == ISD::SRL && N1.hasOneUse()) {
    SDValue ShAmt = N1.getOperand(1);
    ConstantSDNode *ShAmtC = isConstOrConstSplat(ShAmt);
    if (ShAmtC &&
        ShAmtC->getAPIntValue() == (N1.getScalarValueSizeInBits() - 1)) {
      SDValue SRA = DAG.getNode(ISD::SRA, DL, VT, N1.getOperand(0), ShAmt);
      return DAG.getNode(ISD::ADD, DL, VT, N0, SRA);
    }
  }

  if (TLI.isOperationLegalOrCustom(ISD::ADDCARRY, VT)) {
    // (sub Carry, X)  ->  (addcarry (sub 0, X), 0, Carry)
    if (SDValue Carry = getAsCarry(TLI, N0)) {
      SDValue X = N1;
      SDValue Zero = DAG.getConstant(0, DL, VT);
      SDValue NegX = DAG.getNode(ISD::SUB, DL, VT, Zero, X);
      return DAG.getNode(ISD::ADDCARRY, DL,
                         DAG.getVTList(VT, Carry.getValueType()), NegX, Zero,
                         Carry);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitSUBSAT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  // fold vector ops
  if (VT.isVector()) {
    // TODO SimplifyVBinOp

    // fold (sub_sat x, 0) -> x, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return N0;
  }

  // fold (sub_sat x, undef) -> 0
  if (N0.isUndef() || N1.isUndef())
    return DAG.getConstant(0, DL, VT);

  // fold (sub_sat x, x) -> 0
  if (N0 == N1)
    return DAG.getConstant(0, DL, VT);

  // fold (sub_sat c1, c2) -> c3
  if (SDValue C = DAG.FoldConstantArithmetic(N->getOpcode(), DL, VT, {N0, N1}))
    return C;

  // fold (sub_sat x, 0) -> x
  if (isNullConstant(N1))
    return N0;

  return SDValue();
}

SDValue DAGCombiner::visitSUBC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  SDLoc DL(N);

  // If the flag result is dead, turn this into an SUB.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::SUB, DL, VT, N0, N1),
                     DAG.getNode(ISD::CARRY_FALSE, DL, MVT::Glue));

  // fold (subc x, x) -> 0 + no borrow
  if (N0 == N1)
    return CombineTo(N, DAG.getConstant(0, DL, VT),
                     DAG.getNode(ISD::CARRY_FALSE, DL, MVT::Glue));

  // fold (subc x, 0) -> x + no borrow
  if (isNullConstant(N1))
    return CombineTo(N, N0, DAG.getNode(ISD::CARRY_FALSE, DL, MVT::Glue));

  // Canonicalize (sub -1, x) -> ~x, i.e. (xor x, -1) + no borrow
  if (isAllOnesConstant(N0))
    return CombineTo(N, DAG.getNode(ISD::XOR, DL, VT, N1, N0),
                     DAG.getNode(ISD::CARRY_FALSE, DL, MVT::Glue));

  return SDValue();
}

SDValue DAGCombiner::visitSUBO(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  bool IsSigned = (ISD::SSUBO == N->getOpcode());

  EVT CarryVT = N->getValueType(1);
  SDLoc DL(N);

  // If the flag result is dead, turn this into an SUB.
  if (!N->hasAnyUseOfValue(1))
    return CombineTo(N, DAG.getNode(ISD::SUB, DL, VT, N0, N1),
                     DAG.getUNDEF(CarryVT));

  // fold (subo x, x) -> 0 + no borrow
  if (N0 == N1)
    return CombineTo(N, DAG.getConstant(0, DL, VT),
                     DAG.getConstant(0, DL, CarryVT));

  ConstantSDNode *N1C = getAsNonOpaqueConstant(N1);

  // fold (subox, c) -> (addo x, -c)
  if (IsSigned && N1C && !N1C->getAPIntValue().isMinSignedValue()) {
    return DAG.getNode(ISD::SADDO, DL, N->getVTList(), N0,
                       DAG.getConstant(-N1C->getAPIntValue(), DL, VT));
  }

  // fold (subo x, 0) -> x + no borrow
  if (isNullOrNullSplat(N1))
    return CombineTo(N, N0, DAG.getConstant(0, DL, CarryVT));

  // Canonicalize (usubo -1, x) -> ~x, i.e. (xor x, -1) + no borrow
  if (!IsSigned && isAllOnesOrAllOnesSplat(N0))
    return CombineTo(N, DAG.getNode(ISD::XOR, DL, VT, N1, N0),
                     DAG.getConstant(0, DL, CarryVT));

  return SDValue();
}

SDValue DAGCombiner::visitSUBE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // fold (sube x, y, false) -> (subc x, y)
  if (CarryIn.getOpcode() == ISD::CARRY_FALSE)
    return DAG.getNode(ISD::SUBC, SDLoc(N), N->getVTList(), N0, N1);

  return SDValue();
}

SDValue DAGCombiner::visitSUBCARRY(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // fold (subcarry x, y, false) -> (usubo x, y)
  if (isNullConstant(CarryIn)) {
    if (!LegalOperations ||
        TLI.isOperationLegalOrCustom(ISD::USUBO, N->getValueType(0)))
      return DAG.getNode(ISD::USUBO, SDLoc(N), N->getVTList(), N0, N1);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSSUBO_CARRY(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue CarryIn = N->getOperand(2);

  // fold (ssubo_carry x, y, false) -> (ssubo x, y)
  if (isNullConstant(CarryIn)) {
    if (!LegalOperations ||
        TLI.isOperationLegalOrCustom(ISD::SSUBO, N->getValueType(0)))
      return DAG.getNode(ISD::SSUBO, SDLoc(N), N->getVTList(), N0, N1);
  }

  return SDValue();
}

// Notice that "mulfix" can be any of SMULFIX, SMULFIXSAT, UMULFIX and
// UMULFIXSAT here.
SDValue DAGCombiner::visitMULFIX(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue Scale = N->getOperand(2);
  EVT VT = N0.getValueType();

  // fold (mulfix x, undef, scale) -> 0
  if (N0.isUndef() || N1.isUndef())
    return DAG.getConstant(0, SDLoc(N), VT);

  // Canonicalize constant to RHS (vector doesn't have to splat)
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
     !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(N->getOpcode(), SDLoc(N), VT, N1, N0, Scale);

  // fold (mulfix x, 0, scale) -> 0
  if (isNullConstant(N1))
    return DAG.getConstant(0, SDLoc(N), VT);

  return SDValue();
}

SDValue DAGCombiner::visitMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold (mul x, undef) -> 0
  if (N0.isUndef() || N1.isUndef())
    return DAG.getConstant(0, SDLoc(N), VT);

  bool N1IsConst = false;
  bool N1IsOpaqueConst = false;
  APInt ConstValue1;

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    N1IsConst = ISD::isConstantSplatVector(N1.getNode(), ConstValue1);
    assert((!N1IsConst ||
            ConstValue1.getBitWidth() == VT.getScalarSizeInBits()) &&
           "Splat APInt should be element width");
  } else {
    N1IsConst = isa<ConstantSDNode>(N1);
    if (N1IsConst) {
      ConstValue1 = cast<ConstantSDNode>(N1)->getAPIntValue();
      N1IsOpaqueConst = cast<ConstantSDNode>(N1)->isOpaque();
    }
  }

  // fold (mul c1, c2) -> c1*c2
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::MUL, SDLoc(N), VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS (vector doesn't have to splat)
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
     !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::MUL, SDLoc(N), VT, N1, N0);

  // fold (mul x, 0) -> 0
  if (N1IsConst && ConstValue1.isNullValue())
    return N1;

  // fold (mul x, 1) -> x
  if (N1IsConst && ConstValue1.isOneValue())
    return N0;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // fold (mul x, -1) -> 0-x
  if (N1IsConst && ConstValue1.isAllOnesValue()) {
    SDLoc DL(N);
    return DAG.getNode(ISD::SUB, DL, VT,
                       DAG.getConstant(0, DL, VT), N0);
  }

  // fold (mul x, (1 << c)) -> x << c
  if (isConstantOrConstantVector(N1, /*NoOpaques*/ true) &&
      DAG.isKnownToBeAPowerOfTwo(N1) &&
      (!VT.isVector() || Level <= AfterLegalizeVectorOps)) {
    SDLoc DL(N);
    SDValue LogBase2 = BuildLogBase2(N1, DL);
    EVT ShiftVT = getShiftAmountTy(N0.getValueType());
    SDValue Trunc = DAG.getZExtOrTrunc(LogBase2, DL, ShiftVT);
    return DAG.getNode(ISD::SHL, DL, VT, N0, Trunc);
  }

  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1IsConst && !N1IsOpaqueConst && (-ConstValue1).isPowerOf2()) {
    unsigned Log2Val = (-ConstValue1).logBase2();
    SDLoc DL(N);
    // FIXME: If the input is something that is easily negated (e.g. a
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, DL, VT,
                       DAG.getConstant(0, DL, VT),
                       DAG.getNode(ISD::SHL, DL, VT, N0,
                            DAG.getConstant(Log2Val, DL,
                                      getShiftAmountTy(N0.getValueType()))));
  }

  // Try to transform:
  // (1) multiply-by-(power-of-2 +/- 1) into shift and add/sub.
  // mul x, (2^N + 1) --> add (shl x, N), x
  // mul x, (2^N - 1) --> sub (shl x, N), x
  // Examples: x * 33 --> (x << 5) + x
  //           x * 15 --> (x << 4) - x
  //           x * -33 --> -((x << 5) + x)
  //           x * -15 --> -((x << 4) - x) ; this reduces --> x - (x << 4)
  // (2) multiply-by-(power-of-2 +/- power-of-2) into shifts and add/sub.
  // mul x, (2^N + 2^M) --> (add (shl x, N), (shl x, M))
  // mul x, (2^N - 2^M) --> (sub (shl x, N), (shl x, M))
  // Examples: x * 0x8800 --> (x << 15) + (x << 11)
  //           x * 0xf800 --> (x << 16) - (x << 11)
  //           x * -0x8800 --> -((x << 15) + (x << 11))
  //           x * -0xf800 --> -((x << 16) - (x << 11)) ; (x << 11) - (x << 16)
  if (N1IsConst && TLI.decomposeMulByConstant(*DAG.getContext(), VT, N1)) {
    // TODO: We could handle more general decomposition of any constant by
    //       having the target set a limit on number of ops and making a
    //       callback to determine that sequence (similar to sqrt expansion).
    unsigned MathOp = ISD::DELETED_NODE;
    APInt MulC = ConstValue1.abs();
    // The constant `2` should be treated as (2^0 + 1).
    unsigned TZeros = MulC == 2 ? 0 : MulC.countTrailingZeros();
    MulC.lshrInPlace(TZeros);
    if ((MulC - 1).isPowerOf2())
      MathOp = ISD::ADD;
    else if ((MulC + 1).isPowerOf2())
      MathOp = ISD::SUB;

    if (MathOp != ISD::DELETED_NODE) {
      unsigned ShAmt =
          MathOp == ISD::ADD ? (MulC - 1).logBase2() : (MulC + 1).logBase2();
      ShAmt += TZeros;
      assert(ShAmt < VT.getScalarSizeInBits() &&
             "multiply-by-constant generated out of bounds shift");
      SDLoc DL(N);
      SDValue Shl =
          DAG.getNode(ISD::SHL, DL, VT, N0, DAG.getConstant(ShAmt, DL, VT));
      SDValue R =
          TZeros ? DAG.getNode(MathOp, DL, VT, Shl,
                               DAG.getNode(ISD::SHL, DL, VT, N0,
                                           DAG.getConstant(TZeros, DL, VT)))
                 : DAG.getNode(MathOp, DL, VT, Shl, N0);
      if (ConstValue1.isNegative())
        R = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), R);
      return R;
    }
  }

  // (mul (shl X, c1), c2) -> (mul X, c2 << c1)
  if (N0.getOpcode() == ISD::SHL &&
      isConstantOrConstantVector(N1, /* NoOpaques */ true) &&
      isConstantOrConstantVector(N0.getOperand(1), /* NoOpaques */ true)) {
    SDValue C3 = DAG.getNode(ISD::SHL, SDLoc(N), VT, N1, N0.getOperand(1));
    if (isConstantOrConstantVector(C3))
      return DAG.getNode(ISD::MUL, SDLoc(N), VT, N0.getOperand(0), C3);
  }

  // Change (mul (shl X, C), Y) -> (shl (mul X, Y), C) when the shift has one
  // use.
  {
    SDValue Sh(nullptr, 0), Y(nullptr, 0);

    // Check for both (mul (shl X, C), Y)  and  (mul Y, (shl X, C)).
    if (N0.getOpcode() == ISD::SHL &&
        isConstantOrConstantVector(N0.getOperand(1)) &&
        N0.getNode()->hasOneUse()) {
      Sh = N0; Y = N1;
    } else if (N1.getOpcode() == ISD::SHL &&
               isConstantOrConstantVector(N1.getOperand(1)) &&
               N1.getNode()->hasOneUse()) {
      Sh = N1; Y = N0;
    }

    if (Sh.getNode()) {
      SDValue Mul = DAG.getNode(ISD::MUL, SDLoc(N), VT, Sh.getOperand(0), Y);
      return DAG.getNode(ISD::SHL, SDLoc(N), VT, Mul, Sh.getOperand(1));
    }
  }

  // fold (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2)
  if (DAG.isConstantIntBuildVectorOrConstantInt(N1) &&
      N0.getOpcode() == ISD::ADD &&
      DAG.isConstantIntBuildVectorOrConstantInt(N0.getOperand(1)) &&
      isMulAddWithConstProfitable(N, N0, N1))
      return DAG.getNode(ISD::ADD, SDLoc(N), VT,
                         DAG.getNode(ISD::MUL, SDLoc(N0), VT,
                                     N0.getOperand(0), N1),
                         DAG.getNode(ISD::MUL, SDLoc(N1), VT,
                                     N0.getOperand(1), N1));

  // Fold (mul (vscale * C0), C1) to (vscale * (C0 * C1)).
  if (N0.getOpcode() == ISD::VSCALE)
    if (ConstantSDNode *NC1 = isConstOrConstSplat(N1)) {
      const APInt &C0 = N0.getConstantOperandAPInt(0);
      const APInt &C1 = NC1->getAPIntValue();
      return DAG.getVScale(SDLoc(N), VT, C0 * C1);
    }

  // Fold (mul step_vector(C0), C1) to (step_vector(C0 * C1)).
  APInt MulVal;
  if (N0.getOpcode() == ISD::STEP_VECTOR)
    if (ISD::isConstantSplatVector(N1.getNode(), MulVal)) {
      const APInt &C0 = N0.getConstantOperandAPInt(0);
      APInt NewStep = C0 * MulVal;
      return DAG.getStepVector(SDLoc(N), VT, NewStep);
    }

  // Fold ((mul x, 0/undef) -> 0,
  //       (mul x, 1) -> x) -> x)
  // -> and(x, mask)
  // We can replace vectors with '0' and '1' factors with a clearing mask.
  if (VT.isFixedLengthVector()) {
    unsigned NumElts = VT.getVectorNumElements();
    SmallBitVector ClearMask;
    ClearMask.reserve(NumElts);
    auto IsClearMask = [&ClearMask](ConstantSDNode *V) {
      if (!V || V->isNullValue()) {
        ClearMask.push_back(true);
        return true;
      }
      ClearMask.push_back(false);
      return V->isOne();
    };
    if ((!LegalOperations || TLI.isOperationLegalOrCustom(ISD::AND, VT)) &&
        ISD::matchUnaryPredicate(N1, IsClearMask, /*AllowUndefs*/ true)) {
      assert(N1.getOpcode() == ISD::BUILD_VECTOR && "Unknown constant vector");
      SDLoc DL(N);
      EVT LegalSVT = N1.getOperand(0).getValueType();
      SDValue Zero = DAG.getConstant(0, DL, LegalSVT);
      SDValue AllOnes = DAG.getAllOnesConstant(DL, LegalSVT);
      SmallVector<SDValue, 16> Mask(NumElts, AllOnes);
      for (unsigned I = 0; I != NumElts; ++I)
        if (ClearMask[I])
          Mask[I] = Zero;
      return DAG.getNode(ISD::AND, DL, VT, N0, DAG.getBuildVector(VT, DL, Mask));
    }
  }

  // reassociate mul
  if (SDValue RMUL = reassociateOps(ISD::MUL, SDLoc(N), N0, N1, N->getFlags()))
    return RMUL;

  return SDValue();
}

/// Return true if divmod libcall is available.
static bool isDivRemLibcallAvailable(SDNode *Node, bool isSigned,
                                     const TargetLowering &TLI) {
  RTLIB::Libcall LC;
  EVT NodeType = Node->getValueType(0);
  if (!NodeType.isSimple())
    return false;
  switch (NodeType.getSimpleVT().SimpleTy) {
  default: return false; // No libcall for vector types.
  case MVT::i8:   LC= isSigned ? RTLIB::SDIVREM_I8  : RTLIB::UDIVREM_I8;  break;
  case MVT::i16:  LC= isSigned ? RTLIB::SDIVREM_I16 : RTLIB::UDIVREM_I16; break;
  case MVT::i32:  LC= isSigned ? RTLIB::SDIVREM_I32 : RTLIB::UDIVREM_I32; break;
  case MVT::i64:  LC= isSigned ? RTLIB::SDIVREM_I64 : RTLIB::UDIVREM_I64; break;
  case MVT::i128: LC= isSigned ? RTLIB::SDIVREM_I128:RTLIB::UDIVREM_I128; break;
  }

  return TLI.getLibcallName(LC) != nullptr;
}

/// Issue divrem if both quotient and remainder are needed.
SDValue DAGCombiner::useDivRem(SDNode *Node) {
  if (Node->use_empty())
    return SDValue(); // This is a dead node, leave it alone.

  unsigned Opcode = Node->getOpcode();
  bool isSigned = (Opcode == ISD::SDIV) || (Opcode == ISD::SREM);
  unsigned DivRemOpc = isSigned ? ISD::SDIVREM : ISD::UDIVREM;

  // DivMod lib calls can still work on non-legal types if using lib-calls.
  EVT VT = Node->getValueType(0);
  if (VT.isVector() || !VT.isInteger())
    return SDValue();

  if (!TLI.isTypeLegal(VT) && !TLI.isOperationCustom(DivRemOpc, VT))
    return SDValue();

  // If DIVREM is going to get expanded into a libcall,
  // but there is no libcall available, then don't combine.
  if (!TLI.isOperationLegalOrCustom(DivRemOpc, VT) &&
      !isDivRemLibcallAvailable(Node, isSigned, TLI))
    return SDValue();

  // If div is legal, it's better to do the normal expansion
  unsigned OtherOpcode = 0;
  if ((Opcode == ISD::SDIV) || (Opcode == ISD::UDIV)) {
    OtherOpcode = isSigned ? ISD::SREM : ISD::UREM;
    if (TLI.isOperationLegalOrCustom(Opcode, VT))
      return SDValue();
  } else {
    OtherOpcode = isSigned ? ISD::SDIV : ISD::UDIV;
    if (TLI.isOperationLegalOrCustom(OtherOpcode, VT))
      return SDValue();
  }

  SDValue Op0 = Node->getOperand(0);
  SDValue Op1 = Node->getOperand(1);
  SDValue combined;
  for (SDNode::use_iterator UI = Op0.getNode()->use_begin(),
         UE = Op0.getNode()->use_end(); UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User == Node || User->getOpcode() == ISD::DELETED_NODE ||
        User->use_empty())
      continue;
    // Convert the other matching node(s), too;
    // otherwise, the DIVREM may get target-legalized into something
    // target-specific that we won't be able to recognize.
    unsigned UserOpc = User->getOpcode();
    if ((UserOpc == Opcode || UserOpc == OtherOpcode || UserOpc == DivRemOpc) &&
        User->getOperand(0) == Op0 &&
        User->getOperand(1) == Op1) {
      if (!combined) {
        if (UserOpc == OtherOpcode) {
          SDVTList VTs = DAG.getVTList(VT, VT);
          combined = DAG.getNode(DivRemOpc, SDLoc(Node), VTs, Op0, Op1);
        } else if (UserOpc == DivRemOpc) {
          combined = SDValue(User, 0);
        } else {
          assert(UserOpc == Opcode);
          continue;
        }
      }
      if (UserOpc == ISD::SDIV || UserOpc == ISD::UDIV)
        CombineTo(User, combined);
      else if (UserOpc == ISD::SREM || UserOpc == ISD::UREM)
        CombineTo(User, combined.getValue(1));
    }
  }
  return combined;
}

static SDValue simplifyDivRem(SDNode *N, SelectionDAG &DAG) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  unsigned Opc = N->getOpcode();
  bool IsDiv = (ISD::SDIV == Opc) || (ISD::UDIV == Opc);
  ConstantSDNode *N1C = isConstOrConstSplat(N1);

  // X / undef -> undef
  // X % undef -> undef
  // X / 0 -> undef
  // X % 0 -> undef
  // NOTE: This includes vectors where any divisor element is zero/undef.
  if (DAG.isUndef(Opc, {N0, N1}))
    return DAG.getUNDEF(VT);

  // undef / X -> 0
  // undef % X -> 0
  if (N0.isUndef())
    return DAG.getConstant(0, DL, VT);

  // 0 / X -> 0
  // 0 % X -> 0
  ConstantSDNode *N0C = isConstOrConstSplat(N0);
  if (N0C && N0C->isNullValue())
    return N0;

  // X / X -> 1
  // X % X -> 0
  if (N0 == N1)
    return DAG.getConstant(IsDiv ? 1 : 0, DL, VT);

  // X / 1 -> X
  // X % 1 -> 0
  // If this is a boolean op (single-bit element type), we can't have
  // division-by-zero or remainder-by-zero, so assume the divisor is 1.
  // TODO: Similarly, if we're zero-extending a boolean divisor, then assume
  // it's a 1.
  if ((N1C && N1C->isOne()) || (VT.getScalarType() == MVT::i1))
    return IsDiv ? N0 : DAG.getConstant(0, DL, VT);

  return SDValue();
}

SDValue DAGCombiner::visitSDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT CCVT = getSetCCResultType(VT);

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  SDLoc DL(N);

  // fold (sdiv c1, c2) -> c1/c2
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::SDIV, DL, VT, {N0, N1}))
    return C;

  // fold (sdiv X, -1) -> 0-X
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), N0);

  // fold (sdiv X, MIN_SIGNED) -> select(X == MIN_SIGNED, 1, 0)
  if (N1C && N1C->getAPIntValue().isMinSignedValue())
    return DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, N0, N1, ISD::SETEQ),
                         DAG.getConstant(1, DL, VT),
                         DAG.getConstant(0, DL, VT));

  if (SDValue V = simplifyDivRem(N, DAG))
    return V;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::UDIV, DL, N1.getValueType(), N0, N1);

  if (SDValue V = visitSDIVLike(N0, N1, N)) {
    // If the corresponding remainder node exists, update its users with
    // (Dividend - (Quotient * Divisor).
    if (SDNode *RemNode = DAG.getNodeIfExists(ISD::SREM, N->getVTList(),
                                              { N0, N1 })) {
      SDValue Mul = DAG.getNode(ISD::MUL, DL, VT, V, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N0, Mul);
      AddToWorklist(Mul.getNode());
      AddToWorklist(Sub.getNode());
      CombineTo(RemNode, Sub);
    }
    return V;
  }

  // sdiv, srem -> sdivrem
  // If the divisor is constant, then return DIVREM only if isIntDivCheap() is
  // true.  Otherwise, we break the simplification logic in visitREM().
  AttributeList Attr = DAG.getMachineFunction().getFunction().getAttributes();
  if (!N1C || TLI.isIntDivCheap(N->getValueType(0), Attr))
    if (SDValue DivRem = useDivRem(N))
        return DivRem;

  return SDValue();
}

SDValue DAGCombiner::visitSDIVLike(SDValue N0, SDValue N1, SDNode *N) {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  EVT CCVT = getSetCCResultType(VT);
  unsigned BitWidth = VT.getScalarSizeInBits();

  // Helper for determining whether a value is a power-2 constant scalar or a
  // vector of such elements.
  auto IsPowerOfTwo = [](ConstantSDNode *C) {
    if (C->isNullValue() || C->isOpaque())
      return false;
    if (C->getAPIntValue().isPowerOf2())
      return true;
    if ((-C->getAPIntValue()).isPowerOf2())
      return true;
    return false;
  };

  // fold (sdiv X, pow2) -> simple ops after legalize
  // FIXME: We check for the exact bit here because the generic lowering gives
  // better results in that case. The target-specific lowering should learn how
  // to handle exact sdivs efficiently.
  if (!N->getFlags().hasExact() && ISD::matchUnaryPredicate(N1, IsPowerOfTwo)) {
    // Target-specific implementation of sdiv x, pow2.
    if (SDValue Res = BuildSDIVPow2(N))
      return Res;

    // Create constants that are functions of the shift amount value.
    EVT ShiftAmtTy = getShiftAmountTy(N0.getValueType());
    SDValue Bits = DAG.getConstant(BitWidth, DL, ShiftAmtTy);
    SDValue C1 = DAG.getNode(ISD::CTTZ, DL, VT, N1);
    C1 = DAG.getZExtOrTrunc(C1, DL, ShiftAmtTy);
    SDValue Inexact = DAG.getNode(ISD::SUB, DL, ShiftAmtTy, Bits, C1);
    if (!isConstantOrConstantVector(Inexact))
      return SDValue();

    // Splat the sign bit into the register
    SDValue Sign = DAG.getNode(ISD::SRA, DL, VT, N0,
                               DAG.getConstant(BitWidth - 1, DL, ShiftAmtTy));
    AddToWorklist(Sign.getNode());

    // Add (N0 < 0) ? abs2 - 1 : 0;
    SDValue Srl = DAG.getNode(ISD::SRL, DL, VT, Sign, Inexact);
    AddToWorklist(Srl.getNode());
    SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N0, Srl);
    AddToWorklist(Add.getNode());
    SDValue Sra = DAG.getNode(ISD::SRA, DL, VT, Add, C1);
    AddToWorklist(Sra.getNode());

    // Special case: (sdiv X, 1) -> X
    // Special Case: (sdiv X, -1) -> 0-X
    SDValue One = DAG.getConstant(1, DL, VT);
    SDValue AllOnes = DAG.getAllOnesConstant(DL, VT);
    SDValue IsOne = DAG.getSetCC(DL, CCVT, N1, One, ISD::SETEQ);
    SDValue IsAllOnes = DAG.getSetCC(DL, CCVT, N1, AllOnes, ISD::SETEQ);
    SDValue IsOneOrAllOnes = DAG.getNode(ISD::OR, DL, CCVT, IsOne, IsAllOnes);
    Sra = DAG.getSelect(DL, VT, IsOneOrAllOnes, N0, Sra);

    // If dividing by a positive value, we're done. Otherwise, the result must
    // be negated.
    SDValue Zero = DAG.getConstant(0, DL, VT);
    SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, Zero, Sra);

    // FIXME: Use SELECT_CC once we improve SELECT_CC constant-folding.
    SDValue IsNeg = DAG.getSetCC(DL, CCVT, N1, Zero, ISD::SETLT);
    SDValue Res = DAG.getSelect(DL, VT, IsNeg, Sub, Sra);
    return Res;
  }

  // If integer divide is expensive and we satisfy the requirements, emit an
  // alternate sequence.  Targets may check function attributes for size/speed
  // trade-offs.
  AttributeList Attr = DAG.getMachineFunction().getFunction().getAttributes();
  if (isConstantOrConstantVector(N1) &&
      !TLI.isIntDivCheap(N->getValueType(0), Attr))
    if (SDValue Op = BuildSDIV(N))
      return Op;

  return SDValue();
}

SDValue DAGCombiner::visitUDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT CCVT = getSetCCResultType(VT);

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  SDLoc DL(N);

  // fold (udiv c1, c2) -> c1/c2
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::UDIV, DL, VT, {N0, N1}))
    return C;

  // fold (udiv X, -1) -> select(X == -1, 1, 0)
  if (N1C && N1C->getAPIntValue().isAllOnesValue())
    return DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, N0, N1, ISD::SETEQ),
                         DAG.getConstant(1, DL, VT),
                         DAG.getConstant(0, DL, VT));

  if (SDValue V = simplifyDivRem(N, DAG))
    return V;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  if (SDValue V = visitUDIVLike(N0, N1, N)) {
    // If the corresponding remainder node exists, update its users with
    // (Dividend - (Quotient * Divisor).
    if (SDNode *RemNode = DAG.getNodeIfExists(ISD::UREM, N->getVTList(),
                                              { N0, N1 })) {
      SDValue Mul = DAG.getNode(ISD::MUL, DL, VT, V, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N0, Mul);
      AddToWorklist(Mul.getNode());
      AddToWorklist(Sub.getNode());
      CombineTo(RemNode, Sub);
    }
    return V;
  }

  // sdiv, srem -> sdivrem
  // If the divisor is constant, then return DIVREM only if isIntDivCheap() is
  // true.  Otherwise, we break the simplification logic in visitREM().
  AttributeList Attr = DAG.getMachineFunction().getFunction().getAttributes();
  if (!N1C || TLI.isIntDivCheap(N->getValueType(0), Attr))
    if (SDValue DivRem = useDivRem(N))
        return DivRem;

  return SDValue();
}

SDValue DAGCombiner::visitUDIVLike(SDValue N0, SDValue N1, SDNode *N) {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // fold (udiv x, (1 << c)) -> x >>u c
  if (isConstantOrConstantVector(N1, /*NoOpaques*/ true) &&
      DAG.isKnownToBeAPowerOfTwo(N1)) {
    SDValue LogBase2 = BuildLogBase2(N1, DL);
    AddToWorklist(LogBase2.getNode());

    EVT ShiftVT = getShiftAmountTy(N0.getValueType());
    SDValue Trunc = DAG.getZExtOrTrunc(LogBase2, DL, ShiftVT);
    AddToWorklist(Trunc.getNode());
    return DAG.getNode(ISD::SRL, DL, VT, N0, Trunc);
  }

  // fold (udiv x, (shl c, y)) -> x >>u (log2(c)+y) iff c is power of 2
  if (N1.getOpcode() == ISD::SHL) {
    SDValue N10 = N1.getOperand(0);
    if (isConstantOrConstantVector(N10, /*NoOpaques*/ true) &&
        DAG.isKnownToBeAPowerOfTwo(N10)) {
      SDValue LogBase2 = BuildLogBase2(N10, DL);
      AddToWorklist(LogBase2.getNode());

      EVT ADDVT = N1.getOperand(1).getValueType();
      SDValue Trunc = DAG.getZExtOrTrunc(LogBase2, DL, ADDVT);
      AddToWorklist(Trunc.getNode());
      SDValue Add = DAG.getNode(ISD::ADD, DL, ADDVT, N1.getOperand(1), Trunc);
      AddToWorklist(Add.getNode());
      return DAG.getNode(ISD::SRL, DL, VT, N0, Add);
    }
  }

  // fold (udiv x, c) -> alternate
  AttributeList Attr = DAG.getMachineFunction().getFunction().getAttributes();
  if (isConstantOrConstantVector(N1) &&
      !TLI.isIntDivCheap(N->getValueType(0), Attr))
    if (SDValue Op = BuildUDIV(N))
      return Op;

  return SDValue();
}

// handles ISD::SREM and ISD::UREM
SDValue DAGCombiner::visitREM(SDNode *N) {
  unsigned Opcode = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT CCVT = getSetCCResultType(VT);

  bool isSigned = (Opcode == ISD::SREM);
  SDLoc DL(N);

  // fold (rem c1, c2) -> c1%c2
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (SDValue C = DAG.FoldConstantArithmetic(Opcode, DL, VT, {N0, N1}))
    return C;

  // fold (urem X, -1) -> select(X == -1, 0, x)
  if (!isSigned && N1C && N1C->getAPIntValue().isAllOnesValue())
    return DAG.getSelect(DL, VT, DAG.getSetCC(DL, CCVT, N0, N1, ISD::SETEQ),
                         DAG.getConstant(0, DL, VT), N0);

  if (SDValue V = simplifyDivRem(N, DAG))
    return V;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  if (isSigned) {
    // If we know the sign bits of both operands are zero, strength reduce to a
    // urem instead.  Handles (X & 0x0FFFFFFF) %s 16 -> X&15
    if (DAG.SignBitIsZero(N1) && DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UREM, DL, VT, N0, N1);
  } else {
    if (DAG.isKnownToBeAPowerOfTwo(N1)) {
      // fold (urem x, pow2) -> (and x, pow2-1)
      SDValue NegOne = DAG.getAllOnesConstant(DL, VT);
      SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N1, NegOne);
      AddToWorklist(Add.getNode());
      return DAG.getNode(ISD::AND, DL, VT, N0, Add);
    }
    if (N1.getOpcode() == ISD::SHL &&
        DAG.isKnownToBeAPowerOfTwo(N1.getOperand(0))) {
      // fold (urem x, (shl pow2, y)) -> (and x, (add (shl pow2, y), -1))
      SDValue NegOne = DAG.getAllOnesConstant(DL, VT);
      SDValue Add = DAG.getNode(ISD::ADD, DL, VT, N1, NegOne);
      AddToWorklist(Add.getNode());
      return DAG.getNode(ISD::AND, DL, VT, N0, Add);
    }
  }

  AttributeList Attr = DAG.getMachineFunction().getFunction().getAttributes();

  // If X/C can be simplified by the division-by-constant logic, lower
  // X%C to the equivalent of X-X/C*C.
  // Reuse the SDIVLike/UDIVLike combines - to avoid mangling nodes, the
  // speculative DIV must not cause a DIVREM conversion.  We guard against this
  // by skipping the simplification if isIntDivCheap().  When div is not cheap,
  // combine will not return a DIVREM.  Regardless, checking cheapness here
  // makes sense since the simplification results in fatter code.
  if (DAG.isKnownNeverZero(N1) && !TLI.isIntDivCheap(VT, Attr)) {
    SDValue OptimizedDiv =
        isSigned ? visitSDIVLike(N0, N1, N) : visitUDIVLike(N0, N1, N);
    if (OptimizedDiv.getNode()) {
      // If the equivalent Div node also exists, update its users.
      unsigned DivOpcode = isSigned ? ISD::SDIV : ISD::UDIV;
      if (SDNode *DivNode = DAG.getNodeIfExists(DivOpcode, N->getVTList(),
                                                { N0, N1 }))
        CombineTo(DivNode, OptimizedDiv);
      SDValue Mul = DAG.getNode(ISD::MUL, DL, VT, OptimizedDiv, N1);
      SDValue Sub = DAG.getNode(ISD::SUB, DL, VT, N0, Mul);
      AddToWorklist(OptimizedDiv.getNode());
      AddToWorklist(Mul.getNode());
      return Sub;
    }
  }

  // sdiv, srem -> sdivrem
  if (SDValue DivRem = useDivRem(N))
    return DivRem.getValue(1);

  return SDValue();
}

SDValue DAGCombiner::visitMULHS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  if (VT.isVector()) {
    // fold (mulhs x, 0) -> 0
    // do not return N0/N1, because undef node may exist.
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()) ||
        ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return DAG.getConstant(0, DL, VT);
  }

  // fold (mulhs c1, c2)
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::MULHS, DL, VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS.
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::MULHS, DL, N->getVTList(), N1, N0);

  // fold (mulhs x, 0) -> 0
  if (isNullConstant(N1))
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (isOneConstant(N1))
    return DAG.getNode(ISD::SRA, DL, N0.getValueType(), N0,
                       DAG.getConstant(N0.getScalarValueSizeInBits() - 1, DL,
                                       getShiftAmountTy(N0.getValueType())));

  // fold (mulhs x, undef) -> 0
  if (N0.isUndef() || N1.isUndef())
    return DAG.getConstant(0, DL, VT);

  // If the type twice as wide is legal, transform the mulhs to a wider multiply
  // plus a shift.
  if (!TLI.isOperationLegalOrCustom(ISD::MULHS, VT) && VT.isSimple() &&
      !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      N0 = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N0);
      N1 = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N1);
      N1 = DAG.getNode(ISD::MUL, DL, NewVT, N0, N1);
      N1 = DAG.getNode(ISD::SRL, DL, NewVT, N1,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(N1.getValueType())));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, N1);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitMULHU(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  if (VT.isVector()) {
    // fold (mulhu x, 0) -> 0
    // do not return N0/N1, because undef node may exist.
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()) ||
        ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return DAG.getConstant(0, DL, VT);
  }

  // fold (mulhu c1, c2)
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::MULHU, DL, VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS.
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::MULHU, DL, N->getVTList(), N1, N0);

  // fold (mulhu x, 0) -> 0
  if (isNullConstant(N1))
    return N1;
  // fold (mulhu x, 1) -> 0
  if (isOneConstant(N1))
    return DAG.getConstant(0, DL, N0.getValueType());
  // fold (mulhu x, undef) -> 0
  if (N0.isUndef() || N1.isUndef())
    return DAG.getConstant(0, DL, VT);

  // fold (mulhu x, (1 << c)) -> x >> (bitwidth - c)
  if (isConstantOrConstantVector(N1, /*NoOpaques*/ true) &&
      DAG.isKnownToBeAPowerOfTwo(N1) && hasOperation(ISD::SRL, VT)) {
    unsigned NumEltBits = VT.getScalarSizeInBits();
    SDValue LogBase2 = BuildLogBase2(N1, DL);
    SDValue SRLAmt = DAG.getNode(
        ISD::SUB, DL, VT, DAG.getConstant(NumEltBits, DL, VT), LogBase2);
    EVT ShiftVT = getShiftAmountTy(N0.getValueType());
    SDValue Trunc = DAG.getZExtOrTrunc(SRLAmt, DL, ShiftVT);
    return DAG.getNode(ISD::SRL, DL, VT, N0, Trunc);
  }

  // If the type twice as wide is legal, transform the mulhu to a wider multiply
  // plus a shift.
  if (!TLI.isOperationLegalOrCustom(ISD::MULHU, VT) && VT.isSimple() &&
      !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      N0 = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N0);
      N1 = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N1);
      N1 = DAG.getNode(ISD::MUL, DL, NewVT, N0, N1);
      N1 = DAG.getNode(ISD::SRL, DL, NewVT, N1,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(N1.getValueType())));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, N1);
    }
  }

  // Simplify the operands using demanded-bits information.
  // We don't have demanded bits support for MULHU so this just enables constant
  // folding based on known bits.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

/// Perform optimizations common to nodes that compute two values. LoOp and HiOp
/// give the opcodes for the two computations that are being performed. Return
/// true if a simplification was made.
SDValue DAGCombiner::SimplifyNodeWithTwoResults(SDNode *N, unsigned LoOp,
                                                unsigned HiOp) {
  // If the high half is not needed, just compute the low half.
  bool HiExists = N->hasAnyUseOfValue(1);
  if (!HiExists && (!LegalOperations ||
                    TLI.isOperationLegalOrCustom(LoOp, N->getValueType(0)))) {
    SDValue Res = DAG.getNode(LoOp, SDLoc(N), N->getValueType(0), N->ops());
    return CombineTo(N, Res, Res);
  }

  // If the low half is not needed, just compute the high half.
  bool LoExists = N->hasAnyUseOfValue(0);
  if (!LoExists && (!LegalOperations ||
                    TLI.isOperationLegalOrCustom(HiOp, N->getValueType(1)))) {
    SDValue Res = DAG.getNode(HiOp, SDLoc(N), N->getValueType(1), N->ops());
    return CombineTo(N, Res, Res);
  }

  // If both halves are used, return as it is.
  if (LoExists && HiExists)
    return SDValue();

  // If the two computed results can be simplified separately, separate them.
  if (LoExists) {
    SDValue Lo = DAG.getNode(LoOp, SDLoc(N), N->getValueType(0), N->ops());
    AddToWorklist(Lo.getNode());
    SDValue LoOpt = combine(Lo.getNode());
    if (LoOpt.getNode() && LoOpt.getNode() != Lo.getNode() &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(LoOpt.getOpcode(), LoOpt.getValueType())))
      return CombineTo(N, LoOpt, LoOpt);
  }

  if (HiExists) {
    SDValue Hi = DAG.getNode(HiOp, SDLoc(N), N->getValueType(1), N->ops());
    AddToWorklist(Hi.getNode());
    SDValue HiOpt = combine(Hi.getNode());
    if (HiOpt.getNode() && HiOpt != Hi &&
        (!LegalOperations ||
         TLI.isOperationLegalOrCustom(HiOpt.getOpcode(), HiOpt.getValueType())))
      return CombineTo(N, HiOpt, HiOpt);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSMUL_LOHI(SDNode *N) {
  if (SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHS))
    return Res;

  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // If the type is twice as wide is legal, transform the mulhu to a wider
  // multiply plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      SDValue Lo = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N->getOperand(0));
      SDValue Hi = DAG.getNode(ISD::SIGN_EXTEND, DL, NewVT, N->getOperand(1));
      Lo = DAG.getNode(ISD::MUL, DL, NewVT, Lo, Hi);
      // Compute the high part as N1.
      Hi = DAG.getNode(ISD::SRL, DL, NewVT, Lo,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(Lo.getValueType())));
      Hi = DAG.getNode(ISD::TRUNCATE, DL, VT, Hi);
      // Compute the low part as N0.
      Lo = DAG.getNode(ISD::TRUNCATE, DL, VT, Lo);
      return CombineTo(N, Lo, Hi);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitUMUL_LOHI(SDNode *N) {
  if (SDValue Res = SimplifyNodeWithTwoResults(N, ISD::MUL, ISD::MULHU))
    return Res;

  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  // (umul_lohi N0, 0) -> (0, 0)
  if (isNullConstant(N->getOperand(1))) {
    SDValue Zero = DAG.getConstant(0, DL, VT);
    return CombineTo(N, Zero, Zero);
  }

  // (umul_lohi N0, 1) -> (N0, 0)
  if (isOneConstant(N->getOperand(1))) {
    SDValue Zero = DAG.getConstant(0, DL, VT);
    return CombineTo(N, N->getOperand(0), Zero);
  }

  // If the type is twice as wide is legal, transform the mulhu to a wider
  // multiply plus a shift.
  if (VT.isSimple() && !VT.isVector()) {
    MVT Simple = VT.getSimpleVT();
    unsigned SimpleSize = Simple.getSizeInBits();
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), SimpleSize*2);
    if (TLI.isOperationLegal(ISD::MUL, NewVT)) {
      SDValue Lo = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N->getOperand(0));
      SDValue Hi = DAG.getNode(ISD::ZERO_EXTEND, DL, NewVT, N->getOperand(1));
      Lo = DAG.getNode(ISD::MUL, DL, NewVT, Lo, Hi);
      // Compute the high part as N1.
      Hi = DAG.getNode(ISD::SRL, DL, NewVT, Lo,
            DAG.getConstant(SimpleSize, DL,
                            getShiftAmountTy(Lo.getValueType())));
      Hi = DAG.getNode(ISD::TRUNCATE, DL, VT, Hi);
      // Compute the low part as N0.
      Lo = DAG.getNode(ISD::TRUNCATE, DL, VT, Lo);
      return CombineTo(N, Lo, Hi);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitMULO(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  bool IsSigned = (ISD::SMULO == N->getOpcode());

  EVT CarryVT = N->getValueType(1);
  SDLoc DL(N);

  ConstantSDNode *N0C = isConstOrConstSplat(N0);
  ConstantSDNode *N1C = isConstOrConstSplat(N1);

  // fold operation with constant operands.
  // TODO: Move this to FoldConstantArithmetic when it supports nodes with
  // multiple results.
  if (N0C && N1C) {
    bool Overflow;
    APInt Result =
        IsSigned ? N0C->getAPIntValue().smul_ov(N1C->getAPIntValue(), Overflow)
                 : N0C->getAPIntValue().umul_ov(N1C->getAPIntValue(), Overflow);
    return CombineTo(N, DAG.getConstant(Result, DL, VT),
                     DAG.getBoolConstant(Overflow, DL, CarryVT, CarryVT));
  }

  // canonicalize constant to RHS.
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(N->getOpcode(), DL, N->getVTList(), N1, N0);

  // fold (mulo x, 0) -> 0 + no carry out
  if (isNullOrNullSplat(N1))
    return CombineTo(N, DAG.getConstant(0, DL, VT),
                     DAG.getConstant(0, DL, CarryVT));

  // (mulo x, 2) -> (addo x, x)
  if (N1C && N1C->getAPIntValue() == 2)
    return DAG.getNode(IsSigned ? ISD::SADDO : ISD::UADDO, DL,
                       N->getVTList(), N0, N0);

  if (IsSigned) {
    // A 1 bit SMULO overflows if both inputs are 1.
    if (VT.getScalarSizeInBits() == 1) {
      SDValue And = DAG.getNode(ISD::AND, DL, VT, N0, N1);
      return CombineTo(N, And,
                       DAG.getSetCC(DL, CarryVT, And,
                                    DAG.getConstant(0, DL, VT), ISD::SETNE));
    }

    // Multiplying n * m significant bits yields a result of n + m significant
    // bits. If the total number of significant bits does not exceed the
    // result bit width (minus 1), there is no overflow.
    unsigned SignBits = DAG.ComputeNumSignBits(N0);
    if (SignBits > 1)
      SignBits += DAG.ComputeNumSignBits(N1);
    if (SignBits > VT.getScalarSizeInBits() + 1)
      return CombineTo(N, DAG.getNode(ISD::MUL, DL, VT, N0, N1),
                       DAG.getConstant(0, DL, CarryVT));
  } else {
    KnownBits N1Known = DAG.computeKnownBits(N1);
    KnownBits N0Known = DAG.computeKnownBits(N0);
    bool Overflow;
    (void)N0Known.getMaxValue().umul_ov(N1Known.getMaxValue(), Overflow);
    if (!Overflow)
      return CombineTo(N, DAG.getNode(ISD::MUL, DL, VT, N0, N1),
                       DAG.getConstant(0, DL, CarryVT));
  }

  return SDValue();
}

SDValue DAGCombiner::visitIMINMAX(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  unsigned Opcode = N->getOpcode();

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold operation with constant operands.
  if (SDValue C = DAG.FoldConstantArithmetic(Opcode, SDLoc(N), VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(N->getOpcode(), SDLoc(N), VT, N1, N0);

  // Is sign bits are zero, flip between UMIN/UMAX and SMIN/SMAX.
  // Only do this if the current op isn't legal and the flipped is.
  if (!TLI.isOperationLegal(Opcode, VT) &&
      (N0.isUndef() || DAG.SignBitIsZero(N0)) &&
      (N1.isUndef() || DAG.SignBitIsZero(N1))) {
    unsigned AltOpcode;
    switch (Opcode) {
    case ISD::SMIN: AltOpcode = ISD::UMIN; break;
    case ISD::SMAX: AltOpcode = ISD::UMAX; break;
    case ISD::UMIN: AltOpcode = ISD::SMIN; break;
    case ISD::UMAX: AltOpcode = ISD::SMAX; break;
    default: llvm_unreachable("Unknown MINMAX opcode");
    }
    if (TLI.isOperationLegal(AltOpcode, VT))
      return DAG.getNode(AltOpcode, SDLoc(N), VT, N0, N1);
  }

  // Simplify the operands using demanded-bits information.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

/// If this is a bitwise logic instruction and both operands have the same
/// opcode, try to sink the other opcode after the logic instruction.
SDValue DAGCombiner::hoistLogicOpWithSameOpcodeHands(SDNode *N) {
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
  EVT VT = N0.getValueType();
  unsigned LogicOpcode = N->getOpcode();
  unsigned HandOpcode = N0.getOpcode();
  assert((LogicOpcode == ISD::AND || LogicOpcode == ISD::OR ||
          LogicOpcode == ISD::XOR) && "Expected logic opcode");
  assert(HandOpcode == N1.getOpcode() && "Bad input!");

  // Bail early if none of these transforms apply.
  if (N0.getNumOperands() == 0)
    return SDValue();

  // FIXME: We should check number of uses of the operands to not increase
  //        the instruction count for all transforms.

  // Handle size-changing casts.
  SDValue X = N0.getOperand(0);
  SDValue Y = N1.getOperand(0);
  EVT XVT = X.getValueType();
  SDLoc DL(N);
  if (HandOpcode == ISD::ANY_EXTEND || HandOpcode == ISD::ZERO_EXTEND ||
      HandOpcode == ISD::SIGN_EXTEND) {
    // If both operands have other uses, this transform would create extra
    // instructions without eliminating anything.
    if (!N0.hasOneUse() && !N1.hasOneUse())
      return SDValue();
    // We need matching integer source types.
    if (XVT != Y.getValueType())
      return SDValue();
    // Don't create an illegal op during or after legalization. Don't ever
    // create an unsupported vector op.
    if ((VT.isVector() || LegalOperations) &&
        !TLI.isOperationLegalOrCustom(LogicOpcode, XVT))
      return SDValue();
    // Avoid infinite looping with PromoteIntBinOp.
    // TODO: Should we apply desirable/legal constraints to all opcodes?
    if (HandOpcode == ISD::ANY_EXTEND && LegalTypes &&
        !TLI.isTypeDesirableForOp(LogicOpcode, XVT))
      return SDValue();
    // logic_op (hand_op X), (hand_op Y) --> hand_op (logic_op X, Y)
    SDValue Logic = DAG.getNode(LogicOpcode, DL, XVT, X, Y);
    return DAG.getNode(HandOpcode, DL, VT, Logic);
  }

  // logic_op (truncate x), (truncate y) --> truncate (logic_op x, y)
  if (HandOpcode == ISD::TRUNCATE) {
    // If both operands have other uses, this transform would create extra
    // instructions without eliminating anything.
    if (!N0.hasOneUse() && !N1.hasOneUse())
      return SDValue();
    // We need matching source types.
    if (XVT != Y.getValueType())
      return SDValue();
    // Don't create an illegal op during or after legalization.
    if (LegalOperations && !TLI.isOperationLegal(LogicOpcode, XVT))
      return SDValue();
    // Be extra careful sinking truncate. If it's free, there's no benefit in
    // widening a binop. Also, don't create a logic op on an illegal type.
    if (TLI.isZExtFree(VT, XVT) && TLI.isTruncateFree(XVT, VT))
      return SDValue();
    if (!TLI.isTypeLegal(XVT))
      return SDValue();
    SDValue Logic = DAG.getNode(LogicOpcode, DL, XVT, X, Y);
    return DAG.getNode(HandOpcode, DL, VT, Logic);
  }

  // For binops SHL/SRL/SRA/AND:
  //   logic_op (OP x, z), (OP y, z) --> OP (logic_op x, y), z
  if ((HandOpcode == ISD::SHL || HandOpcode == ISD::SRL ||
       HandOpcode == ISD::SRA || HandOpcode == ISD::AND) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    // If either operand has other uses, this transform is not an improvement.
    if (!N0.hasOneUse() || !N1.hasOneUse())
      return SDValue();
    SDValue Logic = DAG.getNode(LogicOpcode, DL, XVT, X, Y);
    return DAG.getNode(HandOpcode, DL, VT, Logic, N0.getOperand(1));
  }

  // Unary ops: logic_op (bswap x), (bswap y) --> bswap (logic_op x, y)
  if (HandOpcode == ISD::BSWAP) {
    // If either operand has other uses, this transform is not an improvement.
    if (!N0.hasOneUse() || !N1.hasOneUse())
      return SDValue();
    SDValue Logic = DAG.getNode(LogicOpcode, DL, XVT, X, Y);
    return DAG.getNode(HandOpcode, DL, VT, Logic);
  }

  // Simplify xor/and/or (bitcast(A), bitcast(B)) -> bitcast(op (A,B))
  // Only perform this optimization up until type legalization, before
  // LegalizeVectorOprs. LegalizeVectorOprs promotes vector operations by
  // adding bitcasts. For example (xor v4i32) is promoted to (v2i64), and
  // we don't want to undo this promotion.
  // We also handle SCALAR_TO_VECTOR because xor/or/and operations are cheaper
  // on scalars.
  if ((HandOpcode == ISD::BITCAST || HandOpcode == ISD::SCALAR_TO_VECTOR) &&
       Level <= AfterLegalizeTypes) {
    // Input types must be integer and the same.
    if (XVT.isInteger() && XVT == Y.getValueType() &&
        !(VT.isVector() && TLI.isTypeLegal(VT) &&
          !XVT.isVector() && !TLI.isTypeLegal(XVT))) {
      SDValue Logic = DAG.getNode(LogicOpcode, DL, XVT, X, Y);
      return DAG.getNode(HandOpcode, DL, VT, Logic);
    }
  }

  // Xor/and/or are indifferent to the swizzle operation (shuffle of one value).
  // Simplify xor/and/or (shuff(A), shuff(B)) -> shuff(op (A,B))
  // If both shuffles use the same mask, and both shuffle within a single
  // vector, then it is worthwhile to move the swizzle after the operation.
  // The type-legalizer generates this pattern when loading illegal
  // vector types from memory. In many cases this allows additional shuffle
  // optimizations.
  // There are other cases where moving the shuffle after the xor/and/or
  // is profitable even if shuffles don't perform a swizzle.
  // If both shuffles use the same mask, and both shuffles have the same first
  // or second operand, then it might still be profitable to move the shuffle
  // after the xor/and/or operation.
  if (HandOpcode == ISD::VECTOR_SHUFFLE && Level < AfterLegalizeDAG) {
    auto *SVN0 = cast<ShuffleVectorSDNode>(N0);
    auto *SVN1 = cast<ShuffleVectorSDNode>(N1);
    assert(X.getValueType() == Y.getValueType() &&
           "Inputs to shuffles are not the same type");

    // Check that both shuffles use the same mask. The masks are known to be of
    // the same length because the result vector type is the same.
    // Check also that shuffles have only one use to avoid introducing extra
    // instructions.
    if (!SVN0->hasOneUse() || !SVN1->hasOneUse() ||
        !SVN0->getMask().equals(SVN1->getMask()))
      return SDValue();

    // Don't try to fold this node if it requires introducing a
    // build vector of all zeros that might be illegal at this stage.
    SDValue ShOp = N0.getOperand(1);
    if (LogicOpcode == ISD::XOR && !ShOp.isUndef())
      ShOp = tryFoldToZero(DL, TLI, VT, DAG, LegalOperations);

    // (logic_op (shuf (A, C), shuf (B, C))) --> shuf (logic_op (A, B), C)
    if (N0.getOperand(1) == N1.getOperand(1) && ShOp.getNode()) {
      SDValue Logic = DAG.getNode(LogicOpcode, DL, VT,
                                  N0.getOperand(0), N1.getOperand(0));
      return DAG.getVectorShuffle(VT, DL, Logic, ShOp, SVN0->getMask());
    }

    // Don't try to fold this node if it requires introducing a
    // build vector of all zeros that might be illegal at this stage.
    ShOp = N0.getOperand(0);
    if (LogicOpcode == ISD::XOR && !ShOp.isUndef())
      ShOp = tryFoldToZero(DL, TLI, VT, DAG, LegalOperations);

    // (logic_op (shuf (C, A), shuf (C, B))) --> shuf (C, logic_op (A, B))
    if (N0.getOperand(0) == N1.getOperand(0) && ShOp.getNode()) {
      SDValue Logic = DAG.getNode(LogicOpcode, DL, VT, N0.getOperand(1),
                                  N1.getOperand(1));
      return DAG.getVectorShuffle(VT, DL, ShOp, Logic, SVN0->getMask());
    }
  }

  return SDValue();
}

/// Try to make (and/or setcc (LL, LR), setcc (RL, RR)) more efficient.
SDValue DAGCombiner::foldLogicOfSetCCs(bool IsAnd, SDValue N0, SDValue N1,
                                       const SDLoc &DL) {
  SDValue LL, LR, RL, RR, N0CC, N1CC;
  if (!isSetCCEquivalent(N0, LL, LR, N0CC) ||
      !isSetCCEquivalent(N1, RL, RR, N1CC))
    return SDValue();

  assert(N0.getValueType() == N1.getValueType() &&
         "Unexpected operand types for bitwise logic op");
  assert(LL.getValueType() == LR.getValueType() &&
         RL.getValueType() == RR.getValueType() &&
         "Unexpected operand types for setcc");

  // If we're here post-legalization or the logic op type is not i1, the logic
  // op type must match a setcc result type. Also, all folds require new
  // operations on the left and right operands, so those types must match.
  EVT VT = N0.getValueType();
  EVT OpVT = LL.getValueType();
  if (LegalOperations || VT.getScalarType() != MVT::i1)
    if (VT != getSetCCResultType(OpVT))
      return SDValue();
  if (OpVT != RL.getValueType())
    return SDValue();

  ISD::CondCode CC0 = cast<CondCodeSDNode>(N0CC)->get();
  ISD::CondCode CC1 = cast<CondCodeSDNode>(N1CC)->get();
  bool IsInteger = OpVT.isInteger();
  if (LR == RR && CC0 == CC1 && IsInteger) {
    bool IsZero = isNullOrNullSplat(LR);
    bool IsNeg1 = isAllOnesOrAllOnesSplat(LR);

    // All bits clear?
    bool AndEqZero = IsAnd && CC1 == ISD::SETEQ && IsZero;
    // All sign bits clear?
    bool AndGtNeg1 = IsAnd && CC1 == ISD::SETGT && IsNeg1;
    // Any bits set?
    bool OrNeZero = !IsAnd && CC1 == ISD::SETNE && IsZero;
    // Any sign bits set?
    bool OrLtZero = !IsAnd && CC1 == ISD::SETLT && IsZero;

    // (and (seteq X,  0), (seteq Y,  0)) --> (seteq (or X, Y),  0)
    // (and (setgt X, -1), (setgt Y, -1)) --> (setgt (or X, Y), -1)
    // (or  (setne X,  0), (setne Y,  0)) --> (setne (or X, Y),  0)
    // (or  (setlt X,  0), (setlt Y,  0)) --> (setlt (or X, Y),  0)
    if (AndEqZero || AndGtNeg1 || OrNeZero || OrLtZero) {
      SDValue Or = DAG.getNode(ISD::OR, SDLoc(N0), OpVT, LL, RL);
      AddToWorklist(Or.getNode());
      return DAG.getSetCC(DL, VT, Or, LR, CC1);
    }

    // All bits set?
    bool AndEqNeg1 = IsAnd && CC1 == ISD::SETEQ && IsNeg1;
    // All sign bits set?
    bool AndLtZero = IsAnd && CC1 == ISD::SETLT && IsZero;
    // Any bits clear?
    bool OrNeNeg1 = !IsAnd && CC1 == ISD::SETNE && IsNeg1;
    // Any sign bits clear?
    bool OrGtNeg1 = !IsAnd && CC1 == ISD::SETGT && IsNeg1;

    // (and (seteq X, -1), (seteq Y, -1)) --> (seteq (and X, Y), -1)
    // (and (setlt X,  0), (setlt Y,  0)) --> (setlt (and X, Y),  0)
    // (or  (setne X, -1), (setne Y, -1)) --> (setne (and X, Y), -1)
    // (or  (setgt X, -1), (setgt Y  -1)) --> (setgt (and X, Y), -1)
    if (AndEqNeg1 || AndLtZero || OrNeNeg1 || OrGtNeg1) {
      SDValue And = DAG.getNode(ISD::AND, SDLoc(N0), OpVT, LL, RL);
      AddToWorklist(And.getNode());
      return DAG.getSetCC(DL, VT, And, LR, CC1);
    }
  }

  // TODO: What is the 'or' equivalent of this fold?
  // (and (setne X, 0), (setne X, -1)) --> (setuge (add X, 1), 2)
  if (IsAnd && LL == RL && CC0 == CC1 && OpVT.getScalarSizeInBits() > 1 &&
      IsInteger && CC0 == ISD::SETNE &&
      ((isNullConstant(LR) && isAllOnesConstant(RR)) ||
       (isAllOnesConstant(LR) && isNullConstant(RR)))) {
    SDValue One = DAG.getConstant(1, DL, OpVT);
    SDValue Two = DAG.getConstant(2, DL, OpVT);
    SDValue Add = DAG.getNode(ISD::ADD, SDLoc(N0), OpVT, LL, One);
    AddToWorklist(Add.getNode());
    return DAG.getSetCC(DL, VT, Add, Two, ISD::SETUGE);
  }

  // Try more general transforms if the predicates match and the only user of
  // the compares is the 'and' or 'or'.
  if (IsInteger && TLI.convertSetCCLogicToBitwiseLogic(OpVT) && CC0 == CC1 &&
      N0.hasOneUse() && N1.hasOneUse()) {
    // and (seteq A, B), (seteq C, D) --> seteq (or (xor A, B), (xor C, D)), 0
    // or  (setne A, B), (setne C, D) --> setne (or (xor A, B), (xor C, D)), 0
    if ((IsAnd && CC1 == ISD::SETEQ) || (!IsAnd && CC1 == ISD::SETNE)) {
      SDValue XorL = DAG.getNode(ISD::XOR, SDLoc(N0), OpVT, LL, LR);
      SDValue XorR = DAG.getNode(ISD::XOR, SDLoc(N1), OpVT, RL, RR);
      SDValue Or = DAG.getNode(ISD::OR, DL, OpVT, XorL, XorR);
      SDValue Zero = DAG.getConstant(0, DL, OpVT);
      return DAG.getSetCC(DL, VT, Or, Zero, CC1);
    }

    // Turn compare of constants whose difference is 1 bit into add+and+setcc.
    // TODO - support non-uniform vector amounts.
    if ((IsAnd && CC1 == ISD::SETNE) || (!IsAnd && CC1 == ISD::SETEQ)) {
      // Match a shared variable operand and 2 non-opaque constant operands.
      ConstantSDNode *C0 = isConstOrConstSplat(LR);
      ConstantSDNode *C1 = isConstOrConstSplat(RR);
      if (LL == RL && C0 && C1 && !C0->isOpaque() && !C1->isOpaque()) {
        const APInt &CMax =
            APIntOps::umax(C0->getAPIntValue(), C1->getAPIntValue());
        const APInt &CMin =
            APIntOps::umin(C0->getAPIntValue(), C1->getAPIntValue());
        // The difference of the constants must be a single bit.
        if ((CMax - CMin).isPowerOf2()) {
          // and/or (setcc X, CMax, ne), (setcc X, CMin, ne/eq) -->
          // setcc ((sub X, CMin), ~(CMax - CMin)), 0, ne/eq
          SDValue Max = DAG.getNode(ISD::UMAX, DL, OpVT, LR, RR);
          SDValue Min = DAG.getNode(ISD::UMIN, DL, OpVT, LR, RR);
          SDValue Offset = DAG.getNode(ISD::SUB, DL, OpVT, LL, Min);
          SDValue Diff = DAG.getNode(ISD::SUB, DL, OpVT, Max, Min);
          SDValue Mask = DAG.getNOT(DL, Diff, OpVT);
          SDValue And = DAG.getNode(ISD::AND, DL, OpVT, Offset, Mask);
          SDValue Zero = DAG.getConstant(0, DL, OpVT);
          return DAG.getSetCC(DL, VT, And, Zero, CC0);
        }
      }
    }
  }

  // Canonicalize equivalent operands to LL == RL.
  if (LL == RR && LR == RL) {
    CC1 = ISD::getSetCCSwappedOperands(CC1);
    std::swap(RL, RR);
  }

  // (and (setcc X, Y, CC0), (setcc X, Y, CC1)) --> (setcc X, Y, NewCC)
  // (or  (setcc X, Y, CC0), (setcc X, Y, CC1)) --> (setcc X, Y, NewCC)
  if (LL == RL && LR == RR) {
    ISD::CondCode NewCC = IsAnd ? ISD::getSetCCAndOperation(CC0, CC1, OpVT)
                                : ISD::getSetCCOrOperation(CC0, CC1, OpVT);
    if (NewCC != ISD::SETCC_INVALID &&
        (!LegalOperations ||
         (TLI.isCondCodeLegal(NewCC, LL.getSimpleValueType()) &&
          TLI.isOperationLegal(ISD::SETCC, OpVT))))
      return DAG.getSetCC(DL, VT, LL, LR, NewCC);
  }

  return SDValue();
}

/// This contains all DAGCombine rules which reduce two values combined by
/// an And operation to a single value. This makes them reusable in the context
/// of visitSELECT(). Rules involving constants are not included as
/// visitSELECT() already handles those cases.
SDValue DAGCombiner::visitANDLike(SDValue N0, SDValue N1, SDNode *N) {
  EVT VT = N1.getValueType();
  SDLoc DL(N);

  // fold (and x, undef) -> 0
  if (N0.isUndef() || N1.isUndef())
    return DAG.getConstant(0, DL, VT);

  if (SDValue V = foldLogicOfSetCCs(true, N0, N1, DL))
    return V;

  if (N0.getOpcode() == ISD::ADD && N1.getOpcode() == ISD::SRL &&
      VT.getSizeInBits() <= 64) {
    if (ConstantSDNode *ADDI = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      if (ConstantSDNode *SRLI = dyn_cast<ConstantSDNode>(N1.getOperand(1))) {
        // Look for (and (add x, c1), (lshr y, c2)). If C1 wasn't a legal
        // immediate for an add, but it is legal if its top c2 bits are set,
        // transform the ADD so the immediate doesn't need to be materialized
        // in a register.
        APInt ADDC = ADDI->getAPIntValue();
        APInt SRLC = SRLI->getAPIntValue();
        if (ADDC.getMinSignedBits() <= 64 &&
            SRLC.ult(VT.getSizeInBits()) &&
            !TLI.isLegalAddImmediate(ADDC.getSExtValue())) {
          APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                             SRLC.getZExtValue());
          if (DAG.MaskedValueIsZero(N0.getOperand(1), Mask)) {
            ADDC |= Mask;
            if (TLI.isLegalAddImmediate(ADDC.getSExtValue())) {
              SDLoc DL0(N0);
              SDValue NewAdd =
                DAG.getNode(ISD::ADD, DL0, VT,
                            N0.getOperand(0), DAG.getConstant(ADDC, DL, VT));
              CombineTo(N0.getNode(), NewAdd);
              // Return N so it doesn't get rechecked!
              return SDValue(N, 0);
            }
          }
        }
      }
    }
  }

  // Reduce bit extract of low half of an integer to the narrower type.
  // (and (srl i64:x, K), KMask) ->
  //   (i64 zero_extend (and (srl (i32 (trunc i64:x)), K)), KMask)
  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    if (ConstantSDNode *CAnd = dyn_cast<ConstantSDNode>(N1)) {
      if (ConstantSDNode *CShift = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
        unsigned Size = VT.getSizeInBits();
        const APInt &AndMask = CAnd->getAPIntValue();
        unsigned ShiftBits = CShift->getZExtValue();

        // Bail out, this node will probably disappear anyway.
        if (ShiftBits == 0)
          return SDValue();

        unsigned MaskBits = AndMask.countTrailingOnes();
        EVT HalfVT = EVT::getIntegerVT(*DAG.getContext(), Size / 2);

        if (AndMask.isMask() &&
            // Required bits must not span the two halves of the integer and
            // must fit in the half size type.
            (ShiftBits + MaskBits <= Size / 2) &&
            TLI.isNarrowingProfitable(VT, HalfVT) &&
            TLI.isTypeDesirableForOp(ISD::AND, HalfVT) &&
            TLI.isTypeDesirableForOp(ISD::SRL, HalfVT) &&
            TLI.isTruncateFree(VT, HalfVT) &&
            TLI.isZExtFree(HalfVT, VT)) {
          // The isNarrowingProfitable is to avoid regressions on PPC and
          // AArch64 which match a few 64-bit bit insert / bit extract patterns
          // on downstream users of this. Those patterns could probably be
          // extended to handle extensions mixed in.

          SDValue SL(N0);
          assert(MaskBits <= Size);

          // Extracting the highest bit of the low half.
          EVT ShiftVT = TLI.getShiftAmountTy(HalfVT, DAG.getDataLayout());
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SL, HalfVT,
                                      N0.getOperand(0));

          SDValue NewMask = DAG.getConstant(AndMask.trunc(Size / 2), SL, HalfVT);
          SDValue ShiftK = DAG.getConstant(ShiftBits, SL, ShiftVT);
          SDValue Shift = DAG.getNode(ISD::SRL, SL, HalfVT, Trunc, ShiftK);
          SDValue And = DAG.getNode(ISD::AND, SL, HalfVT, Shift, NewMask);
          return DAG.getNode(ISD::ZERO_EXTEND, SL, VT, And);
        }
      }
    }
  }

  return SDValue();
}

bool DAGCombiner::isAndLoadExtLoad(ConstantSDNode *AndC, LoadSDNode *LoadN,
                                   EVT LoadResultTy, EVT &ExtVT) {
  if (!AndC->getAPIntValue().isMask())
    return false;

  unsigned ActiveBits = AndC->getAPIntValue().countTrailingOnes();

  ExtVT = EVT::getIntegerVT(*DAG.getContext(), ActiveBits);
  EVT LoadedVT = LoadN->getMemoryVT();

  if (ExtVT == LoadedVT &&
      (!LegalOperations ||
       TLI.isLoadExtLegal(ISD::ZEXTLOAD, LoadResultTy, ExtVT))) {
    // ZEXTLOAD will match without needing to change the size of the value being
    // loaded.
    return true;
  }

  // Do not change the width of a volatile or atomic loads.
  if (!LoadN->isSimple())
    return false;

  // Do not generate loads of non-round integer types since these can
  // be expensive (and would be wrong if the type is not byte sized).
  if (!LoadedVT.bitsGT(ExtVT) || !ExtVT.isRound())
    return false;

  if (LegalOperations &&
      !TLI.isLoadExtLegal(ISD::ZEXTLOAD, LoadResultTy, ExtVT))
    return false;

  if (!TLI.shouldReduceLoadWidth(LoadN, ISD::ZEXTLOAD, ExtVT))
    return false;

  return true;
}

bool DAGCombiner::isLegalNarrowLdSt(LSBaseSDNode *LDST,
                                    ISD::LoadExtType ExtType, EVT &MemVT,
                                    unsigned ShAmt) {
  if (!LDST)
    return false;
  // Only allow byte offsets.
  if (ShAmt % 8)
    return false;

  // Do not generate loads of non-round integer types since these can
  // be expensive (and would be wrong if the type is not byte sized).
  if (!MemVT.isRound())
    return false;

  // Don't change the width of a volatile or atomic loads.
  if (!LDST->isSimple())
    return false;

  EVT LdStMemVT = LDST->getMemoryVT();

  // Bail out when changing the scalable property, since we can't be sure that
  // we're actually narrowing here.
  if (LdStMemVT.isScalableVector() != MemVT.isScalableVector())
    return false;

  // Verify that we are actually reducing a load width here.
  if (LdStMemVT.bitsLT(MemVT))
    return false;

  // Ensure that this isn't going to produce an unsupported memory access.
  if (ShAmt) {
    assert(ShAmt % 8 == 0 && "ShAmt is byte offset");
    const unsigned ByteShAmt = ShAmt / 8;
    const Align LDSTAlign = LDST->getAlign();
    const Align NarrowAlign = commonAlignment(LDSTAlign, ByteShAmt);
    if (!TLI.allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), MemVT,
                                LDST->getAddressSpace(), NarrowAlign,
                                LDST->getMemOperand()->getFlags()))
      return false;
  }

  // It's not possible to generate a constant of extended or untyped type.
  EVT PtrType = LDST->getBasePtr().getValueType();
  if (PtrType == MVT::Untyped || PtrType.isExtended())
    return false;

  if (isa<LoadSDNode>(LDST)) {
    LoadSDNode *Load = cast<LoadSDNode>(LDST);
    // Don't transform one with multiple uses, this would require adding a new
    // load.
    if (!SDValue(Load, 0).hasOneUse())
      return false;

    if (LegalOperations &&
        !TLI.isLoadExtLegal(ExtType, Load->getValueType(0), MemVT))
      return false;

    // For the transform to be legal, the load must produce only two values
    // (the value loaded and the chain).  Don't transform a pre-increment
    // load, for example, which produces an extra value.  Otherwise the
    // transformation is not equivalent, and the downstream logic to replace
    // uses gets things wrong.
    if (Load->getNumValues() > 2)
      return false;

    // If the load that we're shrinking is an extload and we're not just
    // discarding the extension we can't simply shrink the load. Bail.
    // TODO: It would be possible to merge the extensions in some cases.
    if (Load->getExtensionType() != ISD::NON_EXTLOAD &&
        Load->getMemoryVT().getSizeInBits() < MemVT.getSizeInBits() + ShAmt)
      return false;

    if (!TLI.shouldReduceLoadWidth(Load, ExtType, MemVT))
      return false;
  } else {
    assert(isa<StoreSDNode>(LDST) && "It is not a Load nor a Store SDNode");
    StoreSDNode *Store = cast<StoreSDNode>(LDST);
    // Can't write outside the original store
    if (Store->getMemoryVT().getSizeInBits() < MemVT.getSizeInBits() + ShAmt)
      return false;

    if (LegalOperations &&
        !TLI.isTruncStoreLegal(Store->getValue().getValueType(), MemVT))
      return false;
  }
  return true;
}

bool DAGCombiner::SearchForAndLoads(SDNode *N,
                                    SmallVectorImpl<LoadSDNode*> &Loads,
                                    SmallPtrSetImpl<SDNode*> &NodesWithConsts,
                                    ConstantSDNode *Mask,
                                    SDNode *&NodeToMask) {
  // Recursively search for the operands, looking for loads which can be
  // narrowed.
  for (SDValue Op : N->op_values()) {
    if (Op.getValueType().isVector())
      return false;

    // Some constants may need fixing up later if they are too large.
    if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
      if ((N->getOpcode() == ISD::OR || N->getOpcode() == ISD::XOR) &&
          (Mask->getAPIntValue() & C->getAPIntValue()) != C->getAPIntValue())
        NodesWithConsts.insert(N);
      continue;
    }

    if (!Op.hasOneUse())
      return false;

    switch(Op.getOpcode()) {
    case ISD::LOAD: {
      auto *Load = cast<LoadSDNode>(Op);
      EVT ExtVT;
      if (isAndLoadExtLoad(Mask, Load, Load->getValueType(0), ExtVT) &&
          isLegalNarrowLdSt(Load, ISD::ZEXTLOAD, ExtVT)) {

        // ZEXTLOAD is already small enough.
        if (Load->getExtensionType() == ISD::ZEXTLOAD &&
            ExtVT.bitsGE(Load->getMemoryVT()))
          continue;

        // Use LE to convert equal sized loads to zext.
        if (ExtVT.bitsLE(Load->getMemoryVT()))
          Loads.push_back(Load);

        continue;
      }
      return false;
    }
    case ISD::ZERO_EXTEND:
    case ISD::AssertZext: {
      unsigned ActiveBits = Mask->getAPIntValue().countTrailingOnes();
      EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), ActiveBits);
      EVT VT = Op.getOpcode() == ISD::AssertZext ?
        cast<VTSDNode>(Op.getOperand(1))->getVT() :
        Op.getOperand(0).getValueType();

      // We can accept extending nodes if the mask is wider or an equal
      // width to the original type.
      if (ExtVT.bitsGE(VT))
        continue;
      break;
    }
    case ISD::OR:
    case ISD::XOR:
    case ISD::AND:
      if (!SearchForAndLoads(Op.getNode(), Loads, NodesWithConsts, Mask,
                             NodeToMask))
        return false;
      continue;
    }

    // Allow one node which will masked along with any loads found.
    if (NodeToMask)
      return false;

    // Also ensure that the node to be masked only produces one data result.
    NodeToMask = Op.getNode();
    if (NodeToMask->getNumValues() > 1) {
      bool HasValue = false;
      for (unsigned i = 0, e = NodeToMask->getNumValues(); i < e; ++i) {
        MVT VT = SDValue(NodeToMask, i).getSimpleValueType();
        if (VT != MVT::Glue && VT != MVT::Other) {
          if (HasValue) {
            NodeToMask = nullptr;
            return false;
          }
          HasValue = true;
        }
      }
      assert(HasValue && "Node to be masked has no data result?");
    }
  }
  return true;
}

bool DAGCombiner::BackwardsPropagateMask(SDNode *N) {
  auto *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!Mask)
    return false;

  if (!Mask->getAPIntValue().isMask())
    return false;

  // No need to do anything if the and directly uses a load.
  if (isa<LoadSDNode>(N->getOperand(0)))
    return false;

  SmallVector<LoadSDNode*, 8> Loads;
  SmallPtrSet<SDNode*, 2> NodesWithConsts;
  SDNode *FixupNode = nullptr;
  if (SearchForAndLoads(N, Loads, NodesWithConsts, Mask, FixupNode)) {
    if (Loads.size() == 0)
      return false;

    LLVM_DEBUG(dbgs() << "Backwards propagate AND: "; N->dump());
    SDValue MaskOp = N->getOperand(1);

    // If it exists, fixup the single node we allow in the tree that needs
    // masking.
    if (FixupNode) {
      LLVM_DEBUG(dbgs() << "First, need to fix up: "; FixupNode->dump());
      SDValue And = DAG.getNode(ISD::AND, SDLoc(FixupNode),
                                FixupNode->getValueType(0),
                                SDValue(FixupNode, 0), MaskOp);
      DAG.ReplaceAllUsesOfValueWith(SDValue(FixupNode, 0), And);
      if (And.getOpcode() == ISD ::AND)
        DAG.UpdateNodeOperands(And.getNode(), SDValue(FixupNode, 0), MaskOp);
    }

    // Narrow any constants that need it.
    for (auto *LogicN : NodesWithConsts) {
      SDValue Op0 = LogicN->getOperand(0);
      SDValue Op1 = LogicN->getOperand(1);

      if (isa<ConstantSDNode>(Op0))
          std::swap(Op0, Op1);

      SDValue And = DAG.getNode(ISD::AND, SDLoc(Op1), Op1.getValueType(),
                                Op1, MaskOp);

      DAG.UpdateNodeOperands(LogicN, Op0, And);
    }

    // Create narrow loads.
    for (auto *Load : Loads) {
      LLVM_DEBUG(dbgs() << "Propagate AND back to: "; Load->dump());
      SDValue And = DAG.getNode(ISD::AND, SDLoc(Load), Load->getValueType(0),
                                SDValue(Load, 0), MaskOp);
      DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 0), And);
      if (And.getOpcode() == ISD ::AND)
        And = SDValue(
            DAG.UpdateNodeOperands(And.getNode(), SDValue(Load, 0), MaskOp), 0);
      SDValue NewLoad = ReduceLoadWidth(And.getNode());
      assert(NewLoad &&
             "Shouldn't be masking the load if it can't be narrowed");
      CombineTo(Load, NewLoad, NewLoad.getValue(1));
    }
    DAG.ReplaceAllUsesWith(N, N->getOperand(0).getNode());
    return true;
  }
  return false;
}

// Unfold
//    x &  (-1 'logical shift' y)
// To
//    (x 'opposite logical shift' y) 'logical shift' y
// if it is better for performance.
SDValue DAGCombiner::unfoldExtremeBitClearingToShifts(SDNode *N) {
  assert(N->getOpcode() == ISD::AND);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // Do we actually prefer shifts over mask?
  if (!TLI.shouldFoldMaskToVariableShiftPair(N0))
    return SDValue();

  // Try to match  (-1 '[outer] logical shift' y)
  unsigned OuterShift;
  unsigned InnerShift; // The opposite direction to the OuterShift.
  SDValue Y;           // Shift amount.
  auto matchMask = [&OuterShift, &InnerShift, &Y](SDValue M) -> bool {
    if (!M.hasOneUse())
      return false;
    OuterShift = M->getOpcode();
    if (OuterShift == ISD::SHL)
      InnerShift = ISD::SRL;
    else if (OuterShift == ISD::SRL)
      InnerShift = ISD::SHL;
    else
      return false;
    if (!isAllOnesConstant(M->getOperand(0)))
      return false;
    Y = M->getOperand(1);
    return true;
  };

  SDValue X;
  if (matchMask(N1))
    X = N0;
  else if (matchMask(N0))
    X = N1;
  else
    return SDValue();

  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  //     tmp = x   'opposite logical shift' y
  SDValue T0 = DAG.getNode(InnerShift, DL, VT, X, Y);
  //     ret = tmp 'logical shift' y
  SDValue T1 = DAG.getNode(OuterShift, DL, VT, T0, Y);

  return T1;
}

/// Try to replace shift/logic that tests if a bit is clear with mask + setcc.
/// For a target with a bit test, this is expected to become test + set and save
/// at least 1 instruction.
static SDValue combineShiftAnd1ToBitTest(SDNode *And, SelectionDAG &DAG) {
  assert(And->getOpcode() == ISD::AND && "Expected an 'and' op");

  // This is probably not worthwhile without a supported type.
  EVT VT = And->getValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!TLI.isTypeLegal(VT))
    return SDValue();

  // Look through an optional extension and find a 'not'.
  // TODO: Should we favor test+set even without the 'not' op?
  SDValue Not = And->getOperand(0), And1 = And->getOperand(1);
  if (Not.getOpcode() == ISD::ANY_EXTEND)
    Not = Not.getOperand(0);
  if (!isBitwiseNot(Not) || !Not.hasOneUse() || !isOneConstant(And1))
    return SDValue();

  // Look though an optional truncation. The source operand may not be the same
  // type as the original 'and', but that is ok because we are masking off
  // everything but the low bit.
  SDValue Srl = Not.getOperand(0);
  if (Srl.getOpcode() == ISD::TRUNCATE)
    Srl = Srl.getOperand(0);

  // Match a shift-right by constant.
  if (Srl.getOpcode() != ISD::SRL || !Srl.hasOneUse() ||
      !isa<ConstantSDNode>(Srl.getOperand(1)))
    return SDValue();

  // We might have looked through casts that make this transform invalid.
  // TODO: If the source type is wider than the result type, do the mask and
  //       compare in the source type.
  const APInt &ShiftAmt = Srl.getConstantOperandAPInt(1);
  unsigned VTBitWidth = VT.getSizeInBits();
  if (ShiftAmt.uge(VTBitWidth))
    return SDValue();

  // Turn this into a bit-test pattern using mask op + setcc:
  // and (not (srl X, C)), 1 --> (and X, 1<<C) == 0
  SDLoc DL(And);
  SDValue X = DAG.getZExtOrTrunc(Srl.getOperand(0), DL, VT);
  EVT CCVT = TLI.getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);
  SDValue Mask = DAG.getConstant(
      APInt::getOneBitSet(VTBitWidth, ShiftAmt.getZExtValue()), DL, VT);
  SDValue NewAnd = DAG.getNode(ISD::AND, DL, VT, X, Mask);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue Setcc = DAG.getSetCC(DL, CCVT, NewAnd, Zero, ISD::SETEQ);
  return DAG.getZExtOrTrunc(Setcc, DL, VT);
}

SDValue DAGCombiner::visitAND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N1.getValueType();

  // x & x --> x
  if (N0 == N1)
    return N0;

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (and x, 0) -> 0, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()))
      // do not return N0, because undef node may exist in N0
      return DAG.getConstant(APInt::getNullValue(N0.getScalarValueSizeInBits()),
                             SDLoc(N), N0.getValueType());
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      // do not return N1, because undef node may exist in N1
      return DAG.getConstant(APInt::getNullValue(N1.getScalarValueSizeInBits()),
                             SDLoc(N), N1.getValueType());

    // fold (and x, -1) -> x, vector edition
    if (ISD::isConstantSplatVectorAllOnes(N0.getNode()))
      return N1;
    if (ISD::isConstantSplatVectorAllOnes(N1.getNode()))
      return N0;

    // fold (and (masked_load) (build_vec (x, ...))) to zext_masked_load
    auto *MLoad = dyn_cast<MaskedLoadSDNode>(N0);
    auto *BVec = dyn_cast<BuildVectorSDNode>(N1);
    if (MLoad && BVec && MLoad->getExtensionType() == ISD::EXTLOAD &&
        N0.hasOneUse() && N1.hasOneUse()) {
      EVT LoadVT = MLoad->getMemoryVT();
      EVT ExtVT = VT;
      if (TLI.isLoadExtLegal(ISD::ZEXTLOAD, ExtVT, LoadVT)) {
        // For this AND to be a zero extension of the masked load the elements
        // of the BuildVec must mask the bottom bits of the extended element
        // type
        if (ConstantSDNode *Splat = BVec->getConstantSplatNode()) {
          uint64_t ElementSize =
              LoadVT.getVectorElementType().getScalarSizeInBits();
          if (Splat->getAPIntValue().isMask(ElementSize)) {
            return DAG.getMaskedLoad(
                ExtVT, SDLoc(N), MLoad->getChain(), MLoad->getBasePtr(),
                MLoad->getOffset(), MLoad->getMask(), MLoad->getPassThru(),
                LoadVT, MLoad->getMemOperand(), MLoad->getAddressingMode(),
                ISD::ZEXTLOAD, MLoad->isExpandingLoad());
          }
        }
      }
    }
  }

  // fold (and c1, c2) -> c1&c2
  ConstantSDNode *N1C = isConstOrConstSplat(N1);
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::AND, SDLoc(N), VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::AND, SDLoc(N), VT, N1, N0);

  // fold (and x, -1) -> x
  if (isAllOnesConstant(N1))
    return N0;

  // if (and x, c) is known to be zero, return 0
  unsigned BitWidth = VT.getScalarSizeInBits();
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(BitWidth)))
    return DAG.getConstant(0, SDLoc(N), VT);

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // reassociate and
  if (SDValue RAND = reassociateOps(ISD::AND, SDLoc(N), N0, N1, N->getFlags()))
    return RAND;

  // Try to convert a constant mask AND into a shuffle clear mask.
  if (VT.isVector())
    if (SDValue Shuffle = XformToShuffleWithZero(N))
      return Shuffle;

  if (SDValue Combined = combineCarryDiamond(*this, DAG, TLI, N0, N1, N))
    return Combined;

  // fold (and (or x, C), D) -> D if (C & D) == D
  auto MatchSubset = [](ConstantSDNode *LHS, ConstantSDNode *RHS) {
    return RHS->getAPIntValue().isSubsetOf(LHS->getAPIntValue());
  };
  if (N0.getOpcode() == ISD::OR &&
      ISD::matchBinaryPredicate(N0.getOperand(1), N1, MatchSubset))
    return N1;
  // fold (and (any_ext V), c) -> (zero_ext V) if 'and' only clears top bits.
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N0Op0 = N0.getOperand(0);
    APInt Mask = ~N1C->getAPIntValue();
    Mask = Mask.trunc(N0Op0.getScalarValueSizeInBits());
    if (DAG.MaskedValueIsZero(N0Op0, Mask)) {
      SDValue Zext = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N),
                                 N0.getValueType(), N0Op0);

      // Replace uses of the AND with uses of the Zero extend node.
      CombineTo(N, Zext);

      // We actually want to replace all uses of the any_extend with the
      // zero_extend, to avoid duplicating things.  This will later cause this
      // AND to be folded.
      CombineTo(N0.getNode(), Zext);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // similarly fold (and (X (load ([non_ext|any_ext|zero_ext] V))), c) ->
  // (X (load ([non_ext|zero_ext] V))) if 'and' only clears top bits which must
  // already be zero by virtue of the width of the base type of the load.
  //
  // the 'X' node here can either be nothing or an extract_vector_elt to catch
  // more cases.
  if ((N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
       N0.getValueSizeInBits() == N0.getOperand(0).getScalarValueSizeInBits() &&
       N0.getOperand(0).getOpcode() == ISD::LOAD &&
       N0.getOperand(0).getResNo() == 0) ||
      (N0.getOpcode() == ISD::LOAD && N0.getResNo() == 0)) {
    LoadSDNode *Load = cast<LoadSDNode>( (N0.getOpcode() == ISD::LOAD) ?
                                         N0 : N0.getOperand(0) );

    // Get the constant (if applicable) the zero'th operand is being ANDed with.
    // This can be a pure constant or a vector splat, in which case we treat the
    // vector as a scalar and use the splat value.
    APInt Constant = APInt::getNullValue(1);
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N1)) {
      Constant = C->getAPIntValue();
    } else if (BuildVectorSDNode *Vector = dyn_cast<BuildVectorSDNode>(N1)) {
      APInt SplatValue, SplatUndef;
      unsigned SplatBitSize;
      bool HasAnyUndefs;
      bool IsSplat = Vector->isConstantSplat(SplatValue, SplatUndef,
                                             SplatBitSize, HasAnyUndefs);
      if (IsSplat) {
        // Undef bits can contribute to a possible optimisation if set, so
        // set them.
        SplatValue |= SplatUndef;

        // The splat value may be something like "0x00FFFFFF", which means 0 for
        // the first vector value and FF for the rest, repeating. We need a mask
        // that will apply equally to all members of the vector, so AND all the
        // lanes of the constant together.
        unsigned EltBitWidth = Vector->getValueType(0).getScalarSizeInBits();

        // If the splat value has been compressed to a bitlength lower
        // than the size of the vector lane, we need to re-expand it to
        // the lane size.
        if (EltBitWidth > SplatBitSize)
          for (SplatValue = SplatValue.zextOrTrunc(EltBitWidth);
               SplatBitSize < EltBitWidth; SplatBitSize = SplatBitSize * 2)
            SplatValue |= SplatValue.shl(SplatBitSize);

        // Make sure that variable 'Constant' is only set if 'SplatBitSize' is a
        // multiple of 'BitWidth'. Otherwise, we could propagate a wrong value.
        if ((SplatBitSize % EltBitWidth) == 0) {
          Constant = APInt::getAllOnesValue(EltBitWidth);
          for (unsigned i = 0, n = (SplatBitSize / EltBitWidth); i < n; ++i)
            Constant &= SplatValue.extractBits(EltBitWidth, i * EltBitWidth);
        }
      }
    }

    // If we want to change an EXTLOAD to a ZEXTLOAD, ensure a ZEXTLOAD is
    // actually legal and isn't going to get expanded, else this is a false
    // optimisation.
    bool CanZextLoadProfitably = TLI.isLoadExtLegal(ISD::ZEXTLOAD,
                                                    Load->getValueType(0),
                                                    Load->getMemoryVT());

    // Resize the constant to the same size as the original memory access before
    // extension. If it is still the AllOnesValue then this AND is completely
    // unneeded.
    Constant = Constant.zextOrTrunc(Load->getMemoryVT().getScalarSizeInBits());

    bool B;
    switch (Load->getExtensionType()) {
    default: B = false; break;
    case ISD::EXTLOAD: B = CanZextLoadProfitably; break;
    case ISD::ZEXTLOAD:
    case ISD::NON_EXTLOAD: B = true; break;
    }

    if (B && Constant.isAllOnesValue()) {
      // If the load type was an EXTLOAD, convert to ZEXTLOAD in order to
      // preserve semantics once we get rid of the AND.
      SDValue NewLoad(Load, 0);

      // Fold the AND away. NewLoad may get replaced immediately.
      CombineTo(N, (N0.getNode() == Load) ? NewLoad : N0);

      if (Load->getExtensionType() == ISD::EXTLOAD) {
        NewLoad = DAG.getLoad(Load->getAddressingMode(), ISD::ZEXTLOAD,
                              Load->getValueType(0), SDLoc(Load),
                              Load->getChain(), Load->getBasePtr(),
                              Load->getOffset(), Load->getMemoryVT(),
                              Load->getMemOperand());
        // Replace uses of the EXTLOAD with the new ZEXTLOAD.
        if (Load->getNumValues() == 3) {
          // PRE/POST_INC loads have 3 values.
          SDValue To[] = { NewLoad.getValue(0), NewLoad.getValue(1),
                           NewLoad.getValue(2) };
          CombineTo(Load, To, 3, true);
        } else {
          CombineTo(Load, NewLoad.getValue(0), NewLoad.getValue(1));
        }
      }

      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }
  }

  // fold (and (masked_gather x)) -> (zext_masked_gather x)
  if (auto *GN0 = dyn_cast<MaskedGatherSDNode>(N0)) {
    EVT MemVT = GN0->getMemoryVT();
    EVT ScalarVT = MemVT.getScalarType();

    if (SDValue(GN0, 0).hasOneUse() &&
        isConstantSplatVectorMaskForType(N1.getNode(), ScalarVT) &&
        TLI.isVectorLoadExtDesirable(SDValue(SDValue(GN0, 0)))) {
      SDValue Ops[] = {GN0->getChain(),   GN0->getPassThru(), GN0->getMask(),
                       GN0->getBasePtr(), GN0->getIndex(),    GN0->getScale()};

      SDValue ZExtLoad = DAG.getMaskedGather(
          DAG.getVTList(VT, MVT::Other), MemVT, SDLoc(N), Ops,
          GN0->getMemOperand(), GN0->getIndexType(), ISD::ZEXTLOAD);

      CombineTo(N, ZExtLoad);
      AddToWorklist(ZExtLoad.getNode());
      // Avoid recheck of N.
      return SDValue(N, 0);
    }
  }

  // fold (and (load x), 255) -> (zextload x, i8)
  // fold (and (extload x, i16), 255) -> (zextload x, i8)
  // fold (and (any_ext (extload x, i16)), 255) -> (zextload x, i8)
  if (!VT.isVector() && N1C && (N0.getOpcode() == ISD::LOAD ||
                                (N0.getOpcode() == ISD::ANY_EXTEND &&
                                 N0.getOperand(0).getOpcode() == ISD::LOAD))) {
    if (SDValue Res = ReduceLoadWidth(N)) {
      LoadSDNode *LN0 = N0->getOpcode() == ISD::ANY_EXTEND
        ? cast<LoadSDNode>(N0.getOperand(0)) : cast<LoadSDNode>(N0);
      AddToWorklist(N);
      DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 0), Res);
      return SDValue(N, 0);
    }
  }

  if (LegalTypes) {
    // Attempt to propagate the AND back up to the leaves which, if they're
    // loads, can be combined to narrow loads and the AND node can be removed.
    // Perform after legalization so that extend nodes will already be
    // combined into the loads.
    if (BackwardsPropagateMask(N))
      return SDValue(N, 0);
  }

  if (SDValue Combined = visitANDLike(N0, N1, N))
    return Combined;

  // Simplify: (and (op x...), (op y...))  -> (op (and x, y))
  if (N0.getOpcode() == N1.getOpcode())
    if (SDValue V = hoistLogicOpWithSameOpcodeHands(N))
      return V;

  // Masking the negated extension of a boolean is just the zero-extended
  // boolean:
  // and (sub 0, zext(bool X)), 1 --> zext(bool X)
  // and (sub 0, sext(bool X)), 1 --> zext(bool X)
  //
  // Note: the SimplifyDemandedBits fold below can make an information-losing
  // transform, and then we have no way to find this better fold.
  if (N1C && N1C->isOne() && N0.getOpcode() == ISD::SUB) {
    if (isNullOrNullSplat(N0.getOperand(0))) {
      SDValue SubRHS = N0.getOperand(1);
      if (SubRHS.getOpcode() == ISD::ZERO_EXTEND &&
          SubRHS.getOperand(0).getScalarValueSizeInBits() == 1)
        return SubRHS;
      if (SubRHS.getOpcode() == ISD::SIGN_EXTEND &&
          SubRHS.getOperand(0).getScalarValueSizeInBits() == 1)
        return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, SubRHS.getOperand(0));
    }
  }

  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  // fold (and (sra)) -> (and (srl)) when possible.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (zext_inreg (extload x)) -> (zextload x)
  // fold (zext_inreg (sextload x)) -> (zextload x) iff load has one use
  if (ISD::isUNINDEXEDLoad(N0.getNode()) &&
      (ISD::isEXTLoad(N0.getNode()) ||
       (ISD::isSEXTLoad(N0.getNode()) && N0.hasOneUse()))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    EVT MemVT = LN0->getMemoryVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    unsigned ExtBitSize = N1.getScalarValueSizeInBits();
    unsigned MemBitSize = MemVT.getScalarSizeInBits();
    APInt ExtBits = APInt::getHighBitsSet(ExtBitSize, ExtBitSize - MemBitSize);
    if (DAG.MaskedValueIsZero(N1, ExtBits) &&
        ((!LegalOperations && LN0->isSimple()) ||
         TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, MemVT))) {
      SDValue ExtLoad =
          DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(N0), VT, LN0->getChain(),
                         LN0->getBasePtr(), MemVT, LN0->getMemOperand());
      AddToWorklist(N);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }
  }

  // fold (and (or (srl N, 8), (shl N, 8)), 0xffff) -> (srl (bswap N), const)
  if (N1C && N1C->getAPIntValue() == 0xffff && N0.getOpcode() == ISD::OR) {
    if (SDValue BSwap = MatchBSwapHWordLow(N0.getNode(), N0.getOperand(0),
                                           N0.getOperand(1), false))
      return BSwap;
  }

  if (SDValue Shifts = unfoldExtremeBitClearingToShifts(N))
    return Shifts;

  if (TLI.hasBitTest(N0, N1))
    if (SDValue V = combineShiftAnd1ToBitTest(N, DAG))
      return V;

  // Recognize the following pattern:
  //
  // AndVT = (and (sign_extend NarrowVT to AndVT) #bitmask)
  //
  // where bitmask is a mask that clears the upper bits of AndVT. The
  // number of bits in bitmask must be a power of two.
  auto IsAndZeroExtMask = [](SDValue LHS, SDValue RHS) {
    if (LHS->getOpcode() != ISD::SIGN_EXTEND)
      return false;

    auto *C = dyn_cast<ConstantSDNode>(RHS);
    if (!C)
      return false;

    if (!C->getAPIntValue().isMask(
            LHS.getOperand(0).getValueType().getFixedSizeInBits()))
      return false;

    return true;
  };

  // Replace (and (sign_extend ...) #bitmask) with (zero_extend ...).
  if (IsAndZeroExtMask(N0, N1))
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT, N0.getOperand(0));

  return SDValue();
}

/// Match (a >> 8) | (a << 8) as (bswap a) >> 16.
SDValue DAGCombiner::MatchBSwapHWordLow(SDNode *N, SDValue N0, SDValue N1,
                                        bool DemandHighBits) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i64 && VT != MVT::i32 && VT != MVT::i16)
    return SDValue();
  if (!TLI.isOperationLegalOrCustom(ISD::BSWAP, VT))
    return SDValue();

  // Recognize (and (shl a, 8), 0xff00), (and (srl a, 8), 0xff)
  bool LookPassAnd0 = false;
  bool LookPassAnd1 = false;
  if (N0.getOpcode() == ISD::AND && N0.getOperand(0).getOpcode() == ISD::SRL)
      std::swap(N0, N1);
  if (N1.getOpcode() == ISD::AND && N1.getOperand(0).getOpcode() == ISD::SHL)
      std::swap(N0, N1);
  if (N0.getOpcode() == ISD::AND) {
    if (!N0.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    // Also handle 0xffff since the LHS is guaranteed to have zeros there.
    // This is needed for X86.
    if (!N01C || (N01C->getZExtValue() != 0xFF00 &&
                  N01C->getZExtValue() != 0xFFFF))
      return SDValue();
    N0 = N0.getOperand(0);
    LookPassAnd0 = true;
  }

  if (N1.getOpcode() == ISD::AND) {
    if (!N1.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
    if (!N11C || N11C->getZExtValue() != 0xFF)
      return SDValue();
    N1 = N1.getOperand(0);
    LookPassAnd1 = true;
  }

  if (N0.getOpcode() == ISD::SRL && N1.getOpcode() == ISD::SHL)
    std::swap(N0, N1);
  if (N0.getOpcode() != ISD::SHL || N1.getOpcode() != ISD::SRL)
    return SDValue();
  if (!N0.getNode()->hasOneUse() || !N1.getNode()->hasOneUse())
    return SDValue();

  ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  ConstantSDNode *N11C = dyn_cast<ConstantSDNode>(N1.getOperand(1));
  if (!N01C || !N11C)
    return SDValue();
  if (N01C->getZExtValue() != 8 || N11C->getZExtValue() != 8)
    return SDValue();

  // Look for (shl (and a, 0xff), 8), (srl (and a, 0xff00), 8)
  SDValue N00 = N0->getOperand(0);
  if (!LookPassAnd0 && N00.getOpcode() == ISD::AND) {
    if (!N00.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N001C = dyn_cast<ConstantSDNode>(N00.getOperand(1));
    if (!N001C || N001C->getZExtValue() != 0xFF)
      return SDValue();
    N00 = N00.getOperand(0);
    LookPassAnd0 = true;
  }

  SDValue N10 = N1->getOperand(0);
  if (!LookPassAnd1 && N10.getOpcode() == ISD::AND) {
    if (!N10.getNode()->hasOneUse())
      return SDValue();
    ConstantSDNode *N101C = dyn_cast<ConstantSDNode>(N10.getOperand(1));
    // Also allow 0xFFFF since the bits will be shifted out. This is needed
    // for X86.
    if (!N101C || (N101C->getZExtValue() != 0xFF00 &&
                   N101C->getZExtValue() != 0xFFFF))
      return SDValue();
    N10 = N10.getOperand(0);
    LookPassAnd1 = true;
  }

  if (N00 != N10)
    return SDValue();

  // Make sure everything beyond the low halfword gets set to zero since the SRL
  // 16 will clear the top bits.
  unsigned OpSizeInBits = VT.getSizeInBits();
  if (DemandHighBits && OpSizeInBits > 16) {
    // If the left-shift isn't masked out then the only way this is a bswap is
    // if all bits beyond the low 8 are 0. In that case the entire pattern
    // reduces to a left shift anyway: leave it for other parts of the combiner.
    if (!LookPassAnd0)
      return SDValue();

    // However, if the right shift isn't masked out then it might be because
    // it's not needed. See if we can spot that too.
    if (!LookPassAnd1 &&
        !DAG.MaskedValueIsZero(
            N10, APInt::getHighBitsSet(OpSizeInBits, OpSizeInBits - 16)))
      return SDValue();
  }

  SDValue Res = DAG.getNode(ISD::BSWAP, SDLoc(N), VT, N00);
  if (OpSizeInBits > 16) {
    SDLoc DL(N);
    Res = DAG.getNode(ISD::SRL, DL, VT, Res,
                      DAG.getConstant(OpSizeInBits - 16, DL,
                                      getShiftAmountTy(VT)));
  }
  return Res;
}

/// Return true if the specified node is an element that makes up a 32-bit
/// packed halfword byteswap.
/// ((x & 0x000000ff) << 8) |
/// ((x & 0x0000ff00) >> 8) |
/// ((x & 0x00ff0000) << 8) |
/// ((x & 0xff000000) >> 8)
static bool isBSwapHWordElement(SDValue N, MutableArrayRef<SDNode *> Parts) {
  if (!N.getNode()->hasOneUse())
    return false;

  unsigned Opc = N.getOpcode();
  if (Opc != ISD::AND && Opc != ISD::SHL && Opc != ISD::SRL)
    return false;

  SDValue N0 = N.getOperand(0);
  unsigned Opc0 = N0.getOpcode();
  if (Opc0 != ISD::AND && Opc0 != ISD::SHL && Opc0 != ISD::SRL)
    return false;

  ConstantSDNode *N1C = nullptr;
  // SHL or SRL: look upstream for AND mask operand
  if (Opc == ISD::AND)
    N1C = dyn_cast<ConstantSDNode>(N.getOperand(1));
  else if (Opc0 == ISD::AND)
    N1C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  if (!N1C)
    return false;

  unsigned MaskByteOffset;
  switch (N1C->getZExtValue()) {
  default:
    return false;
  case 0xFF:       MaskByteOffset = 0; break;
  case 0xFF00:     MaskByteOffset = 1; break;
  case 0xFFFF:
    // In case demanded bits didn't clear the bits that will be shifted out.
    // This is needed for X86.
    if (Opc == ISD::SRL || (Opc == ISD::AND && Opc0 == ISD::SHL)) {
      MaskByteOffset = 1;
      break;
    }
    return false;
  case 0xFF0000:   MaskByteOffset = 2; break;
  case 0xFF000000: MaskByteOffset = 3; break;
  }

  // Look for (x & 0xff) << 8 as well as ((x << 8) & 0xff00).
  if (Opc == ISD::AND) {
    if (MaskByteOffset == 0 || MaskByteOffset == 2) {
      // (x >> 8) & 0xff
      // (x >> 8) & 0xff0000
      if (Opc0 != ISD::SRL)
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C || C->getZExtValue() != 8)
        return false;
    } else {
      // (x << 8) & 0xff00
      // (x << 8) & 0xff000000
      if (Opc0 != ISD::SHL)
        return false;
      ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
      if (!C || C->getZExtValue() != 8)
        return false;
    }
  } else if (Opc == ISD::SHL) {
    // (x & 0xff) << 8
    // (x & 0xff0000) << 8
    if (MaskByteOffset != 0 && MaskByteOffset != 2)
      return false;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!C || C->getZExtValue() != 8)
      return false;
  } else { // Opc == ISD::SRL
    // (x & 0xff00) >> 8
    // (x & 0xff000000) >> 8
    if (MaskByteOffset != 1 && MaskByteOffset != 3)
      return false;
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!C || C->getZExtValue() != 8)
      return false;
  }

  if (Parts[MaskByteOffset])
    return false;

  Parts[MaskByteOffset] = N0.getOperand(0).getNode();
  return true;
}

// Match 2 elements of a packed halfword bswap.
static bool isBSwapHWordPair(SDValue N, MutableArrayRef<SDNode *> Parts) {
  if (N.getOpcode() == ISD::OR)
    return isBSwapHWordElement(N.getOperand(0), Parts) &&
           isBSwapHWordElement(N.getOperand(1), Parts);

  if (N.getOpcode() == ISD::SRL && N.getOperand(0).getOpcode() == ISD::BSWAP) {
    ConstantSDNode *C = isConstOrConstSplat(N.getOperand(1));
    if (!C || C->getAPIntValue() != 16)
      return false;
    Parts[0] = Parts[1] = N.getOperand(0).getOperand(0).getNode();
    return true;
  }

  return false;
}

// Match this pattern:
//   (or (and (shl (A, 8)), 0xff00ff00), (and (srl (A, 8)), 0x00ff00ff))
// And rewrite this to:
//   (rotr (bswap A), 16)
static SDValue matchBSwapHWordOrAndAnd(const TargetLowering &TLI,
                                       SelectionDAG &DAG, SDNode *N, SDValue N0,
                                       SDValue N1, EVT VT, EVT ShiftAmountTy) {
  assert(N->getOpcode() == ISD::OR && VT == MVT::i32 &&
         "MatchBSwapHWordOrAndAnd: expecting i32");
  if (!TLI.isOperationLegalOrCustom(ISD::ROTR, VT))
    return SDValue();
  if (N0.getOpcode() != ISD::AND || N1.getOpcode() != ISD::AND)
    return SDValue();
  // TODO: this is too restrictive; lifting this restriction requires more tests
  if (!N0->hasOneUse() || !N1->hasOneUse())
    return SDValue();
  ConstantSDNode *Mask0 = isConstOrConstSplat(N0.getOperand(1));
  ConstantSDNode *Mask1 = isConstOrConstSplat(N1.getOperand(1));
  if (!Mask0 || !Mask1)
    return SDValue();
  if (Mask0->getAPIntValue() != 0xff00ff00 ||
      Mask1->getAPIntValue() != 0x00ff00ff)
    return SDValue();
  SDValue Shift0 = N0.getOperand(0);
  SDValue Shift1 = N1.getOperand(0);
  if (Shift0.getOpcode() != ISD::SHL || Shift1.getOpcode() != ISD::SRL)
    return SDValue();
  ConstantSDNode *ShiftAmt0 = isConstOrConstSplat(Shift0.getOperand(1));
  ConstantSDNode *ShiftAmt1 = isConstOrConstSplat(Shift1.getOperand(1));
  if (!ShiftAmt0 || !ShiftAmt1)
    return SDValue();
  if (ShiftAmt0->getAPIntValue() != 8 || ShiftAmt1->getAPIntValue() != 8)
    return SDValue();
  if (Shift0.getOperand(0) != Shift1.getOperand(0))
    return SDValue();

  SDLoc DL(N);
  SDValue BSwap = DAG.getNode(ISD::BSWAP, DL, VT, Shift0.getOperand(0));
  SDValue ShAmt = DAG.getConstant(16, DL, ShiftAmountTy);
  return DAG.getNode(ISD::ROTR, DL, VT, BSwap, ShAmt);
}

/// Match a 32-bit packed halfword bswap. That is
/// ((x & 0x000000ff) << 8) |
/// ((x & 0x0000ff00) >> 8) |
/// ((x & 0x00ff0000) << 8) |
/// ((x & 0xff000000) >> 8)
/// => (rotl (bswap x), 16)
SDValue DAGCombiner::MatchBSwapHWord(SDNode *N, SDValue N0, SDValue N1) {
  if (!LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i32)
    return SDValue();
  if (!TLI.isOperationLegalOrCustom(ISD::BSWAP, VT))
    return SDValue();

  if (SDValue BSwap = matchBSwapHWordOrAndAnd(TLI, DAG, N, N0, N1, VT,
                                              getShiftAmountTy(VT)))
  return BSwap;

  // Try again with commuted operands.
  if (SDValue BSwap = matchBSwapHWordOrAndAnd(TLI, DAG, N, N1, N0, VT,
                                              getShiftAmountTy(VT)))
  return BSwap;


  // Look for either
  // (or (bswaphpair), (bswaphpair))
  // (or (or (bswaphpair), (and)), (and))
  // (or (or (and), (bswaphpair)), (and))
  SDNode *Parts[4] = {};

  if (isBSwapHWordPair(N0, Parts)) {
    // (or (or (and), (and)), (or (and), (and)))
    if (!isBSwapHWordPair(N1, Parts))
      return SDValue();
  } else if (N0.getOpcode() == ISD::OR) {
    // (or (or (or (and), (and)), (and)), (and))
    if (!isBSwapHWordElement(N1, Parts))
      return SDValue();
    SDValue N00 = N0.getOperand(0);
    SDValue N01 = N0.getOperand(1);
    if (!(isBSwapHWordElement(N01, Parts) && isBSwapHWordPair(N00, Parts)) &&
        !(isBSwapHWordElement(N00, Parts) && isBSwapHWordPair(N01, Parts)))
      return SDValue();
  } else
    return SDValue();

  // Make sure the parts are all coming from the same node.
  if (Parts[0] != Parts[1] || Parts[0] != Parts[2] || Parts[0] != Parts[3])
    return SDValue();

  SDLoc DL(N);
  SDValue BSwap = DAG.getNode(ISD::BSWAP, DL, VT,
                              SDValue(Parts[0], 0));

  // Result of the bswap should be rotated by 16. If it's not legal, then
  // do  (x << 16) | (x >> 16).
  SDValue ShAmt = DAG.getConstant(16, DL, getShiftAmountTy(VT));
  if (TLI.isOperationLegalOrCustom(ISD::ROTL, VT))
    return DAG.getNode(ISD::ROTL, DL, VT, BSwap, ShAmt);
  if (TLI.isOperationLegalOrCustom(ISD::ROTR, VT))
    return DAG.getNode(ISD::ROTR, DL, VT, BSwap, ShAmt);
  return DAG.getNode(ISD::OR, DL, VT,
                     DAG.getNode(ISD::SHL, DL, VT, BSwap, ShAmt),
                     DAG.getNode(ISD::SRL, DL, VT, BSwap, ShAmt));
}

/// This contains all DAGCombine rules which reduce two values combined by
/// an Or operation to a single value \see visitANDLike().
SDValue DAGCombiner::visitORLike(SDValue N0, SDValue N1, SDNode *N) {
  EVT VT = N1.getValueType();
  SDLoc DL(N);

  // fold (or x, undef) -> -1
  if (!LegalOperations && (N0.isUndef() || N1.isUndef()))
    return DAG.getAllOnesConstant(DL, VT);

  if (SDValue V = foldLogicOfSetCCs(false, N0, N1, DL))
    return V;

  // (or (and X, C1), (and Y, C2))  -> (and (or X, Y), C3) if possible.
  if (N0.getOpcode() == ISD::AND && N1.getOpcode() == ISD::AND &&
      // Don't increase # computations.
      (N0.getNode()->hasOneUse() || N1.getNode()->hasOneUse())) {
    // We can only do this xform if we know that bits from X that are set in C2
    // but not in C1 are already zero.  Likewise for Y.
    if (const ConstantSDNode *N0O1C =
        getAsNonOpaqueConstant(N0.getOperand(1))) {
      if (const ConstantSDNode *N1O1C =
          getAsNonOpaqueConstant(N1.getOperand(1))) {
        // We can only do this xform if we know that bits from X that are set in
        // C2 but not in C1 are already zero.  Likewise for Y.
        const APInt &LHSMask = N0O1C->getAPIntValue();
        const APInt &RHSMask = N1O1C->getAPIntValue();

        if (DAG.MaskedValueIsZero(N0.getOperand(0), RHSMask&~LHSMask) &&
            DAG.MaskedValueIsZero(N1.getOperand(0), LHSMask&~RHSMask)) {
          SDValue X = DAG.getNode(ISD::OR, SDLoc(N0), VT,
                                  N0.getOperand(0), N1.getOperand(0));
          return DAG.getNode(ISD::AND, DL, VT, X,
                             DAG.getConstant(LHSMask | RHSMask, DL, VT));
        }
      }
    }
  }

  // (or (and X, M), (and X, N)) -> (and X, (or M, N))
  if (N0.getOpcode() == ISD::AND &&
      N1.getOpcode() == ISD::AND &&
      N0.getOperand(0) == N1.getOperand(0) &&
      // Don't increase # computations.
      (N0.getNode()->hasOneUse() || N1.getNode()->hasOneUse())) {
    SDValue X = DAG.getNode(ISD::OR, SDLoc(N0), VT,
                            N0.getOperand(1), N1.getOperand(1));
    return DAG.getNode(ISD::AND, DL, VT, N0.getOperand(0), X);
  }

  return SDValue();
}

/// OR combines for which the commuted variant will be tried as well.
static SDValue visitORCommutative(
    SelectionDAG &DAG, SDValue N0, SDValue N1, SDNode *N) {
  EVT VT = N0.getValueType();
  if (N0.getOpcode() == ISD::AND) {
    // fold (or (and X, (xor Y, -1)), Y) -> (or X, Y)
    if (isBitwiseNot(N0.getOperand(1)) && N0.getOperand(1).getOperand(0) == N1)
      return DAG.getNode(ISD::OR, SDLoc(N), VT, N0.getOperand(0), N1);

    // fold (or (and (xor Y, -1), X), Y) -> (or X, Y)
    if (isBitwiseNot(N0.getOperand(0)) && N0.getOperand(0).getOperand(0) == N1)
      return DAG.getNode(ISD::OR, SDLoc(N), VT, N0.getOperand(1), N1);
  }

  return SDValue();
}

SDValue DAGCombiner::visitOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N1.getValueType();

  // x | x --> x
  if (N0 == N1)
    return N0;

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (or x, 0) -> x, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()))
      return N1;
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return N0;

    // fold (or x, -1) -> -1, vector edition
    if (ISD::isConstantSplatVectorAllOnes(N0.getNode()))
      // do not return N0, because undef node may exist in N0
      return DAG.getAllOnesConstant(SDLoc(N), N0.getValueType());
    if (ISD::isConstantSplatVectorAllOnes(N1.getNode()))
      // do not return N1, because undef node may exist in N1
      return DAG.getAllOnesConstant(SDLoc(N), N1.getValueType());

    // fold (or (shuf A, V_0, MA), (shuf B, V_0, MB)) -> (shuf A, B, Mask)
    // Do this only if the resulting shuffle is legal.
    if (isa<ShuffleVectorSDNode>(N0) &&
        isa<ShuffleVectorSDNode>(N1) &&
        // Avoid folding a node with illegal type.
        TLI.isTypeLegal(VT)) {
      bool ZeroN00 = ISD::isBuildVectorAllZeros(N0.getOperand(0).getNode());
      bool ZeroN01 = ISD::isBuildVectorAllZeros(N0.getOperand(1).getNode());
      bool ZeroN10 = ISD::isBuildVectorAllZeros(N1.getOperand(0).getNode());
      bool ZeroN11 = ISD::isBuildVectorAllZeros(N1.getOperand(1).getNode());
      // Ensure both shuffles have a zero input.
      if ((ZeroN00 != ZeroN01) && (ZeroN10 != ZeroN11)) {
        assert((!ZeroN00 || !ZeroN01) && "Both inputs zero!");
        assert((!ZeroN10 || !ZeroN11) && "Both inputs zero!");
        const ShuffleVectorSDNode *SV0 = cast<ShuffleVectorSDNode>(N0);
        const ShuffleVectorSDNode *SV1 = cast<ShuffleVectorSDNode>(N1);
        bool CanFold = true;
        int NumElts = VT.getVectorNumElements();
        SmallVector<int, 4> Mask(NumElts);

        for (int i = 0; i != NumElts; ++i) {
          int M0 = SV0->getMaskElt(i);
          int M1 = SV1->getMaskElt(i);

          // Determine if either index is pointing to a zero vector.
          bool M0Zero = M0 < 0 || (ZeroN00 == (M0 < NumElts));
          bool M1Zero = M1 < 0 || (ZeroN10 == (M1 < NumElts));

          // If one element is zero and the otherside is undef, keep undef.
          // This also handles the case that both are undef.
          if ((M0Zero && M1 < 0) || (M1Zero && M0 < 0)) {
            Mask[i] = -1;
            continue;
          }

          // Make sure only one of the elements is zero.
          if (M0Zero == M1Zero) {
            CanFold = false;
            break;
          }

          assert((M0 >= 0 || M1 >= 0) && "Undef index!");

          // We have a zero and non-zero element. If the non-zero came from
          // SV0 make the index a LHS index. If it came from SV1, make it
          // a RHS index. We need to mod by NumElts because we don't care
          // which operand it came from in the original shuffles.
          Mask[i] = M1Zero ? M0 % NumElts : (M1 % NumElts) + NumElts;
        }

        if (CanFold) {
          SDValue NewLHS = ZeroN00 ? N0.getOperand(1) : N0.getOperand(0);
          SDValue NewRHS = ZeroN10 ? N1.getOperand(1) : N1.getOperand(0);

          SDValue LegalShuffle =
              TLI.buildLegalVectorShuffle(VT, SDLoc(N), NewLHS, NewRHS,
                                          Mask, DAG);
          if (LegalShuffle)
            return LegalShuffle;
        }
      }
    }
  }

  // fold (or c1, c2) -> c1|c2
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::OR, SDLoc(N), VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
     !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::OR, SDLoc(N), VT, N1, N0);

  // fold (or x, 0) -> x
  if (isNullConstant(N1))
    return N0;

  // fold (or x, -1) -> -1
  if (isAllOnesConstant(N1))
    return N1;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // fold (or x, c) -> c iff (x & ~c) == 0
  if (N1C && DAG.MaskedValueIsZero(N0, ~N1C->getAPIntValue()))
    return N1;

  if (SDValue Combined = visitORLike(N0, N1, N))
    return Combined;

  if (SDValue Combined = combineCarryDiamond(*this, DAG, TLI, N0, N1, N))
    return Combined;

  // Recognize halfword bswaps as (bswap + rotl 16) or (bswap + shl 16)
  if (SDValue BSwap = MatchBSwapHWord(N, N0, N1))
    return BSwap;
  if (SDValue BSwap = MatchBSwapHWordLow(N, N0, N1))
    return BSwap;

  // reassociate or
  if (SDValue ROR = reassociateOps(ISD::OR, SDLoc(N), N0, N1, N->getFlags()))
    return ROR;

  // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
  // iff (c1 & c2) != 0 or c1/c2 are undef.
  auto MatchIntersect = [](ConstantSDNode *C1, ConstantSDNode *C2) {
    return !C1 || !C2 || C1->getAPIntValue().intersects(C2->getAPIntValue());
  };
  if (N0.getOpcode() == ISD::AND && N0.getNode()->hasOneUse() &&
      ISD::matchBinaryPredicate(N0.getOperand(1), N1, MatchIntersect, true)) {
    if (SDValue COR = DAG.FoldConstantArithmetic(ISD::OR, SDLoc(N1), VT,
                                                 {N1, N0.getOperand(1)})) {
      SDValue IOR = DAG.getNode(ISD::OR, SDLoc(N0), VT, N0.getOperand(0), N1);
      AddToWorklist(IOR.getNode());
      return DAG.getNode(ISD::AND, SDLoc(N), VT, COR, IOR);
    }
  }

  if (SDValue Combined = visitORCommutative(DAG, N0, N1, N))
    return Combined;
  if (SDValue Combined = visitORCommutative(DAG, N1, N0, N))
    return Combined;

  // Simplify: (or (op x...), (op y...))  -> (op (or x, y))
  if (N0.getOpcode() == N1.getOpcode())
    if (SDValue V = hoistLogicOpWithSameOpcodeHands(N))
      return V;

  // See if this is some rotate idiom.
  if (SDValue Rot = MatchRotate(N0, N1, SDLoc(N)))
    return Rot;

  if (SDValue Load = MatchLoadCombine(N))
    return Load;

  // Simplify the operands using demanded-bits information.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // If OR can be rewritten into ADD, try combines based on ADD.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::ADD, VT)) &&
      DAG.haveNoCommonBitsSet(N0, N1))
    if (SDValue Combined = visitADDLike(N))
      return Combined;

  return SDValue();
}

static SDValue stripConstantMask(SelectionDAG &DAG, SDValue Op, SDValue &Mask) {
  if (Op.getOpcode() == ISD::AND &&
      DAG.isConstantIntBuildVectorOrConstantInt(Op.getOperand(1))) {
    Mask = Op.getOperand(1);
    return Op.getOperand(0);
  }
  return Op;
}

/// Match "(X shl/srl V1) & V2" where V2 may not be present.
static bool matchRotateHalf(SelectionDAG &DAG, SDValue Op, SDValue &Shift,
                            SDValue &Mask) {
  Op = stripConstantMask(DAG, Op, Mask);
  if (Op.getOpcode() == ISD::SRL || Op.getOpcode() == ISD::SHL) {
    Shift = Op;
    return true;
  }
  return false;
}

/// Helper function for visitOR to extract the needed side of a rotate idiom
/// from a shl/srl/mul/udiv.  This is meant to handle cases where
/// InstCombine merged some outside op with one of the shifts from
/// the rotate pattern.
/// \returns An empty \c SDValue if the needed shift couldn't be extracted.
/// Otherwise, returns an expansion of \p ExtractFrom based on the following
/// patterns:
///
///   (or (add v v) (shrl v bitwidth-1)):
///     expands (add v v) -> (shl v 1)
///
///   (or (mul v c0) (shrl (mul v c1) c2)):
///     expands (mul v c0) -> (shl (mul v c1) c3)
///
///   (or (udiv v c0) (shl (udiv v c1) c2)):
///     expands (udiv v c0) -> (shrl (udiv v c1) c3)
///
///   (or (shl v c0) (shrl (shl v c1) c2)):
///     expands (shl v c0) -> (shl (shl v c1) c3)
///
///   (or (shrl v c0) (shl (shrl v c1) c2)):
///     expands (shrl v c0) -> (shrl (shrl v c1) c3)
///
/// Such that in all cases, c3+c2==bitwidth(op v c1).
static SDValue extractShiftForRotate(SelectionDAG &DAG, SDValue OppShift,
                                     SDValue ExtractFrom, SDValue &Mask,
                                     const SDLoc &DL) {
  assert(OppShift && ExtractFrom && "Empty SDValue");
  assert(
      (OppShift.getOpcode() == ISD::SHL || OppShift.getOpcode() == ISD::SRL) &&
      "Existing shift must be valid as a rotate half");

  ExtractFrom = stripConstantMask(DAG, ExtractFrom, Mask);

  // Value and Type of the shift.
  SDValue OppShiftLHS = OppShift.getOperand(0);
  EVT ShiftedVT = OppShiftLHS.getValueType();

  // Amount of the existing shift.
  ConstantSDNode *OppShiftCst = isConstOrConstSplat(OppShift.getOperand(1));

  // (add v v) -> (shl v 1)
  // TODO: Should this be a general DAG canonicalization?
  if (OppShift.getOpcode() == ISD::SRL && OppShiftCst &&
      ExtractFrom.getOpcode() == ISD::ADD &&
      ExtractFrom.getOperand(0) == ExtractFrom.getOperand(1) &&
      ExtractFrom.getOperand(0) == OppShiftLHS &&
      OppShiftCst->getAPIntValue() == ShiftedVT.getScalarSizeInBits() - 1)
    return DAG.getNode(ISD::SHL, DL, ShiftedVT, OppShiftLHS,
                       DAG.getShiftAmountConstant(1, ShiftedVT, DL));

  // Preconditions:
  //    (or (op0 v c0) (shiftl/r (op0 v c1) c2))
  //
  // Find opcode of the needed shift to be extracted from (op0 v c0).
  unsigned Opcode = ISD::DELETED_NODE;
  bool IsMulOrDiv = false;
  // Set Opcode and IsMulOrDiv if the extract opcode matches the needed shift
  // opcode or its arithmetic (mul or udiv) variant.
  auto SelectOpcode = [&](unsigned NeededShift, unsigned MulOrDivVariant) {
    IsMulOrDiv = ExtractFrom.getOpcode() == MulOrDivVariant;
    if (!IsMulOrDiv && ExtractFrom.getOpcode() != NeededShift)
      return false;
    Opcode = NeededShift;
    return true;
  };
  // op0 must be either the needed shift opcode or the mul/udiv equivalent
  // that the needed shift can be extracted from.
  if ((OppShift.getOpcode() != ISD::SRL || !SelectOpcode(ISD::SHL, ISD::MUL)) &&
      (OppShift.getOpcode() != ISD::SHL || !SelectOpcode(ISD::SRL, ISD::UDIV)))
    return SDValue();

  // op0 must be the same opcode on both sides, have the same LHS argument,
  // and produce the same value type.
  if (OppShiftLHS.getOpcode() != ExtractFrom.getOpcode() ||
      OppShiftLHS.getOperand(0) != ExtractFrom.getOperand(0) ||
      ShiftedVT != ExtractFrom.getValueType())
    return SDValue();

  // Constant mul/udiv/shift amount from the RHS of the shift's LHS op.
  ConstantSDNode *OppLHSCst = isConstOrConstSplat(OppShiftLHS.getOperand(1));
  // Constant mul/udiv/shift amount from the RHS of the ExtractFrom op.
  ConstantSDNode *ExtractFromCst =
      isConstOrConstSplat(ExtractFrom.getOperand(1));
  // TODO: We should be able to handle non-uniform constant vectors for these values
  // Check that we have constant values.
  if (!OppShiftCst || !OppShiftCst->getAPIntValue() ||
      !OppLHSCst || !OppLHSCst->getAPIntValue() ||
      !ExtractFromCst || !ExtractFromCst->getAPIntValue())
    return SDValue();

  // Compute the shift amount we need to extract to complete the rotate.
  const unsigned VTWidth = ShiftedVT.getScalarSizeInBits();
  if (OppShiftCst->getAPIntValue().ugt(VTWidth))
    return SDValue();
  APInt NeededShiftAmt = VTWidth - OppShiftCst->getAPIntValue();
  // Normalize the bitwidth of the two mul/udiv/shift constant operands.
  APInt ExtractFromAmt = ExtractFromCst->getAPIntValue();
  APInt OppLHSAmt = OppLHSCst->getAPIntValue();
  zeroExtendToMatch(ExtractFromAmt, OppLHSAmt);

  // Now try extract the needed shift from the ExtractFrom op and see if the
  // result matches up with the existing shift's LHS op.
  if (IsMulOrDiv) {
    // Op to extract from is a mul or udiv by a constant.
    // Check:
    //     c2 / (1 << (bitwidth(op0 v c0) - c1)) == c0
    //     c2 % (1 << (bitwidth(op0 v c0) - c1)) == 0
    const APInt ExtractDiv = APInt::getOneBitSet(ExtractFromAmt.getBitWidth(),
                                                 NeededShiftAmt.getZExtValue());
    APInt ResultAmt;
    APInt Rem;
    APInt::udivrem(ExtractFromAmt, ExtractDiv, ResultAmt, Rem);
    if (Rem != 0 || ResultAmt != OppLHSAmt)
      return SDValue();
  } else {
    // Op to extract from is a shift by a constant.
    // Check:
    //      c2 - (bitwidth(op0 v c0) - c1) == c0
    if (OppLHSAmt != ExtractFromAmt - NeededShiftAmt.zextOrTrunc(
                                          ExtractFromAmt.getBitWidth()))
      return SDValue();
  }

  // Return the expanded shift op that should allow a rotate to be formed.
  EVT ShiftVT = OppShift.getOperand(1).getValueType();
  EVT ResVT = ExtractFrom.getValueType();
  SDValue NewShiftNode = DAG.getConstant(NeededShiftAmt, DL, ShiftVT);
  return DAG.getNode(Opcode, DL, ResVT, OppShiftLHS, NewShiftNode);
}

// Return true if we can prove that, whenever Neg and Pos are both in the
// range [0, EltSize), Neg == (Pos == 0 ? 0 : EltSize - Pos).  This means that
// for two opposing shifts shift1 and shift2 and a value X with OpBits bits:
//
//     (or (shift1 X, Neg), (shift2 X, Pos))
//
// reduces to a rotate in direction shift2 by Pos or (equivalently) a rotate
// in direction shift1 by Neg.  The range [0, EltSize) means that we only need
// to consider shift amounts with defined behavior.
//
// The IsRotate flag should be set when the LHS of both shifts is the same.
// Otherwise if matching a general funnel shift, it should be clear.
static bool matchRotateSub(SDValue Pos, SDValue Neg, unsigned EltSize,
                           SelectionDAG &DAG, bool IsRotate) {
  // If EltSize is a power of 2 then:
  //
  //  (a) (Pos == 0 ? 0 : EltSize - Pos) == (EltSize - Pos) & (EltSize - 1)
  //  (b) Neg == Neg & (EltSize - 1) whenever Neg is in [0, EltSize).
  //
  // So if EltSize is a power of 2 and Neg is (and Neg', EltSize-1), we check
  // for the stronger condition:
  //
  //     Neg & (EltSize - 1) == (EltSize - Pos) & (EltSize - 1)    [A]
  //
  // for all Neg and Pos.  Since Neg & (EltSize - 1) == Neg' & (EltSize - 1)
  // we can just replace Neg with Neg' for the rest of the function.
  //
  // In other cases we check for the even stronger condition:
  //
  //     Neg == EltSize - Pos                                    [B]
  //
  // for all Neg and Pos.  Note that the (or ...) then invokes undefined
  // behavior if Pos == 0 (and consequently Neg == EltSize).
  //
  // We could actually use [A] whenever EltSize is a power of 2, but the
  // only extra cases that it would match are those uninteresting ones
  // where Neg and Pos are never in range at the same time.  E.g. for
  // EltSize == 32, using [A] would allow a Neg of the form (sub 64, Pos)
  // as well as (sub 32, Pos), but:
  //
  //     (or (shift1 X, (sub 64, Pos)), (shift2 X, Pos))
  //
  // always invokes undefined behavior for 32-bit X.
  //
  // Below, Mask == EltSize - 1 when using [A] and is all-ones otherwise.
  //
  // NOTE: We can only do this when matching an AND and not a general
  // funnel shift.
  unsigned MaskLoBits = 0;
  if (IsRotate && Neg.getOpcode() == ISD::AND && isPowerOf2_64(EltSize)) {
    if (ConstantSDNode *NegC = isConstOrConstSplat(Neg.getOperand(1))) {
      KnownBits Known = DAG.computeKnownBits(Neg.getOperand(0));
      unsigned Bits = Log2_64(EltSize);
      if (NegC->getAPIntValue().getActiveBits() <= Bits &&
          ((NegC->getAPIntValue() | Known.Zero).countTrailingOnes() >= Bits)) {
        Neg = Neg.getOperand(0);
        MaskLoBits = Bits;
      }
    }
  }

  // Check whether Neg has the form (sub NegC, NegOp1) for some NegC and NegOp1.
  if (Neg.getOpcode() != ISD::SUB)
    return false;
  ConstantSDNode *NegC = isConstOrConstSplat(Neg.getOperand(0));
  if (!NegC)
    return false;
  SDValue NegOp1 = Neg.getOperand(1);

  // On the RHS of [A], if Pos is Pos' & (EltSize - 1), just replace Pos with
  // Pos'.  The truncation is redundant for the purpose of the equality.
  if (MaskLoBits && Pos.getOpcode() == ISD::AND) {
    if (ConstantSDNode *PosC = isConstOrConstSplat(Pos.getOperand(1))) {
      KnownBits Known = DAG.computeKnownBits(Pos.getOperand(0));
      if (PosC->getAPIntValue().getActiveBits() <= MaskLoBits &&
          ((PosC->getAPIntValue() | Known.Zero).countTrailingOnes() >=
           MaskLoBits))
        Pos = Pos.getOperand(0);
    }
  }

  // The condition we need is now:
  //
  //     (NegC - NegOp1) & Mask == (EltSize - Pos) & Mask
  //
  // If NegOp1 == Pos then we need:
  //
  //              EltSize & Mask == NegC & Mask
  //
  // (because "x & Mask" is a truncation and distributes through subtraction).
  //
  // We also need to account for a potential truncation of NegOp1 if the amount
  // has already been legalized to a shift amount type.
  APInt Width;
  if ((Pos == NegOp1) ||
      (NegOp1.getOpcode() == ISD::TRUNCATE && Pos == NegOp1.getOperand(0)))
    Width = NegC->getAPIntValue();

  // Check for cases where Pos has the form (add NegOp1, PosC) for some PosC.
  // Then the condition we want to prove becomes:
  //
  //     (NegC - NegOp1) & Mask == (EltSize - (NegOp1 + PosC)) & Mask
  //
  // which, again because "x & Mask" is a truncation, becomes:
  //
  //                NegC & Mask == (EltSize - PosC) & Mask
  //             EltSize & Mask == (NegC + PosC) & Mask
  else if (Pos.getOpcode() == ISD::ADD && Pos.getOperand(0) == NegOp1) {
    if (ConstantSDNode *PosC = isConstOrConstSplat(Pos.getOperand(1)))
      Width = PosC->getAPIntValue() + NegC->getAPIntValue();
    else
      return false;
  } else
    return false;

  // Now we just need to check that EltSize & Mask == Width & Mask.
  if (MaskLoBits)
    // EltSize & Mask is 0 since Mask is EltSize - 1.
    return Width.getLoBits(MaskLoBits) == 0;
  return Width == EltSize;
}

// A subroutine of MatchRotate used once we have found an OR of two opposite
// shifts of Shifted.  If Neg == <operand size> - Pos then the OR reduces
// to both (PosOpcode Shifted, Pos) and (NegOpcode Shifted, Neg), with the
// former being preferred if supported.  InnerPos and InnerNeg are Pos and
// Neg with outer conversions stripped away.
SDValue DAGCombiner::MatchRotatePosNeg(SDValue Shifted, SDValue Pos,
                                       SDValue Neg, SDValue InnerPos,
                                       SDValue InnerNeg, unsigned PosOpcode,
                                       unsigned NegOpcode, const SDLoc &DL) {
  // fold (or (shl x, (*ext y)),
  //          (srl x, (*ext (sub 32, y)))) ->
  //   (rotl x, y) or (rotr x, (sub 32, y))
  //
  // fold (or (shl x, (*ext (sub 32, y))),
  //          (srl x, (*ext y))) ->
  //   (rotr x, y) or (rotl x, (sub 32, y))
  EVT VT = Shifted.getValueType();
  if (matchRotateSub(InnerPos, InnerNeg, VT.getScalarSizeInBits(), DAG,
                     /*IsRotate*/ true)) {
    bool HasPos = TLI.isOperationLegalOrCustom(PosOpcode, VT);
    return DAG.getNode(HasPos ? PosOpcode : NegOpcode, DL, VT, Shifted,
                       HasPos ? Pos : Neg);
  }

  return SDValue();
}

// A subroutine of MatchRotate used once we have found an OR of two opposite
// shifts of N0 + N1.  If Neg == <operand size> - Pos then the OR reduces
// to both (PosOpcode N0, N1, Pos) and (NegOpcode N0, N1, Neg), with the
// former being preferred if supported.  InnerPos and InnerNeg are Pos and
// Neg with outer conversions stripped away.
// TODO: Merge with MatchRotatePosNeg.
SDValue DAGCombiner::MatchFunnelPosNeg(SDValue N0, SDValue N1, SDValue Pos,
                                       SDValue Neg, SDValue InnerPos,
                                       SDValue InnerNeg, unsigned PosOpcode,
                                       unsigned NegOpcode, const SDLoc &DL) {
  EVT VT = N0.getValueType();
  unsigned EltBits = VT.getScalarSizeInBits();

  // fold (or (shl x0, (*ext y)),
  //          (srl x1, (*ext (sub 32, y)))) ->
  //   (fshl x0, x1, y) or (fshr x0, x1, (sub 32, y))
  //
  // fold (or (shl x0, (*ext (sub 32, y))),
  //          (srl x1, (*ext y))) ->
  //   (fshr x0, x1, y) or (fshl x0, x1, (sub 32, y))
  if (matchRotateSub(InnerPos, InnerNeg, EltBits, DAG, /*IsRotate*/ N0 == N1)) {
    bool HasPos = TLI.isOperationLegalOrCustom(PosOpcode, VT);
    return DAG.getNode(HasPos ? PosOpcode : NegOpcode, DL, VT, N0, N1,
                       HasPos ? Pos : Neg);
  }

  // Matching the shift+xor cases, we can't easily use the xor'd shift amount
  // so for now just use the PosOpcode case if its legal.
  // TODO: When can we use the NegOpcode case?
  if (PosOpcode == ISD::FSHL && isPowerOf2_32(EltBits)) {
    auto IsBinOpImm = [](SDValue Op, unsigned BinOpc, unsigned Imm) {
      if (Op.getOpcode() != BinOpc)
        return false;
      ConstantSDNode *Cst = isConstOrConstSplat(Op.getOperand(1));
      return Cst && (Cst->getAPIntValue() == Imm);
    };

    // fold (or (shl x0, y), (srl (srl x1, 1), (xor y, 31)))
    //   -> (fshl x0, x1, y)
    if (IsBinOpImm(N1, ISD::SRL, 1) &&
        IsBinOpImm(InnerNeg, ISD::XOR, EltBits - 1) &&
        InnerPos == InnerNeg.getOperand(0) &&
        TLI.isOperationLegalOrCustom(ISD::FSHL, VT)) {
      return DAG.getNode(ISD::FSHL, DL, VT, N0, N1.getOperand(0), Pos);
    }

    // fold (or (shl (shl x0, 1), (xor y, 31)), (srl x1, y))
    //   -> (fshr x0, x1, y)
    if (IsBinOpImm(N0, ISD::SHL, 1) &&
        IsBinOpImm(InnerPos, ISD::XOR, EltBits - 1) &&
        InnerNeg == InnerPos.getOperand(0) &&
        TLI.isOperationLegalOrCustom(ISD::FSHR, VT)) {
      return DAG.getNode(ISD::FSHR, DL, VT, N0.getOperand(0), N1, Neg);
    }

    // fold (or (shl (add x0, x0), (xor y, 31)), (srl x1, y))
    //   -> (fshr x0, x1, y)
    // TODO: Should add(x,x) -> shl(x,1) be a general DAG canonicalization?
    if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N0.getOperand(1) &&
        IsBinOpImm(InnerPos, ISD::XOR, EltBits - 1) &&
        InnerNeg == InnerPos.getOperand(0) &&
        TLI.isOperationLegalOrCustom(ISD::FSHR, VT)) {
      return DAG.getNode(ISD::FSHR, DL, VT, N0.getOperand(0), N1, Neg);
    }
  }

  return SDValue();
}

// MatchRotate - Handle an 'or' of two operands.  If this is one of the many
// idioms for rotate, and if the target supports rotation instructions, generate
// a rot[lr]. This also matches funnel shift patterns, similar to rotation but
// with different shifted sources.
SDValue DAGCombiner::MatchRotate(SDValue LHS, SDValue RHS, const SDLoc &DL) {
  // Must be a legal type.  Expanded 'n promoted things won't work with rotates.
  EVT VT = LHS.getValueType();
  if (!TLI.isTypeLegal(VT))
    return SDValue();

  // The target must have at least one rotate/funnel flavor.
  bool HasROTL = hasOperation(ISD::ROTL, VT);
  bool HasROTR = hasOperation(ISD::ROTR, VT);
  bool HasFSHL = hasOperation(ISD::FSHL, VT);
  bool HasFSHR = hasOperation(ISD::FSHR, VT);
  if (!HasROTL && !HasROTR && !HasFSHL && !HasFSHR)
    return SDValue();

  // Check for truncated rotate.
  if (LHS.getOpcode() == ISD::TRUNCATE && RHS.getOpcode() == ISD::TRUNCATE &&
      LHS.getOperand(0).getValueType() == RHS.getOperand(0).getValueType()) {
    assert(LHS.getValueType() == RHS.getValueType());
    if (SDValue Rot = MatchRotate(LHS.getOperand(0), RHS.getOperand(0), DL)) {
      return DAG.getNode(ISD::TRUNCATE, SDLoc(LHS), LHS.getValueType(), Rot);
    }
  }

  // Match "(X shl/srl V1) & V2" where V2 may not be present.
  SDValue LHSShift;   // The shift.
  SDValue LHSMask;    // AND value if any.
  matchRotateHalf(DAG, LHS, LHSShift, LHSMask);

  SDValue RHSShift;   // The shift.
  SDValue RHSMask;    // AND value if any.
  matchRotateHalf(DAG, RHS, RHSShift, RHSMask);

  // If neither side matched a rotate half, bail
  if (!LHSShift && !RHSShift)
    return SDValue();

  // InstCombine may have combined a constant shl, srl, mul, or udiv with one
  // side of the rotate, so try to handle that here. In all cases we need to
  // pass the matched shift from the opposite side to compute the opcode and
  // needed shift amount to extract.  We still want to do this if both sides
  // matched a rotate half because one half may be a potential overshift that
  // can be broken down (ie if InstCombine merged two shl or srl ops into a
  // single one).

  // Have LHS side of the rotate, try to extract the needed shift from the RHS.
  if (LHSShift)
    if (SDValue NewRHSShift =
            extractShiftForRotate(DAG, LHSShift, RHS, RHSMask, DL))
      RHSShift = NewRHSShift;
  // Have RHS side of the rotate, try to extract the needed shift from the LHS.
  if (RHSShift)
    if (SDValue NewLHSShift =
            extractShiftForRotate(DAG, RHSShift, LHS, LHSMask, DL))
      LHSShift = NewLHSShift;

  // If a side is still missing, nothing else we can do.
  if (!RHSShift || !LHSShift)
    return SDValue();

  // At this point we've matched or extracted a shift op on each side.

  if (LHSShift.getOpcode() == RHSShift.getOpcode())
    return SDValue(); // Shifts must disagree.

  bool IsRotate = LHSShift.getOperand(0) == RHSShift.getOperand(0);
  if (!IsRotate && !(HasFSHL || HasFSHR))
    return SDValue(); // Requires funnel shift support.

  // Canonicalize shl to left side in a shl/srl pair.
  if (RHSShift.getOpcode() == ISD::SHL) {
    std::swap(LHS, RHS);
    std::swap(LHSShift, RHSShift);
    std::swap(LHSMask, RHSMask);
  }

  unsigned EltSizeInBits = VT.getScalarSizeInBits();
  SDValue LHSShiftArg = LHSShift.getOperand(0);
  SDValue LHSShiftAmt = LHSShift.getOperand(1);
  SDValue RHSShiftArg = RHSShift.getOperand(0);
  SDValue RHSShiftAmt = RHSShift.getOperand(1);

  // fold (or (shl x, C1), (srl x, C2)) -> (rotl x, C1)
  // fold (or (shl x, C1), (srl x, C2)) -> (rotr x, C2)
  // fold (or (shl x, C1), (srl y, C2)) -> (fshl x, y, C1)
  // fold (or (shl x, C1), (srl y, C2)) -> (fshr x, y, C2)
  // iff C1+C2 == EltSizeInBits
  auto MatchRotateSum = [EltSizeInBits](ConstantSDNode *LHS,
                                        ConstantSDNode *RHS) {
    return (LHS->getAPIntValue() + RHS->getAPIntValue()) == EltSizeInBits;
  };
  if (ISD::matchBinaryPredicate(LHSShiftAmt, RHSShiftAmt, MatchRotateSum)) {
    SDValue Res;
    if (IsRotate && (HasROTL || HasROTR))
      Res = DAG.getNode(HasROTL ? ISD::ROTL : ISD::ROTR, DL, VT, LHSShiftArg,
                        HasROTL ? LHSShiftAmt : RHSShiftAmt);
    else
      Res = DAG.getNode(HasFSHL ? ISD::FSHL : ISD::FSHR, DL, VT, LHSShiftArg,
                        RHSShiftArg, HasFSHL ? LHSShiftAmt : RHSShiftAmt);

    // If there is an AND of either shifted operand, apply it to the result.
    if (LHSMask.getNode() || RHSMask.getNode()) {
      SDValue AllOnes = DAG.getAllOnesConstant(DL, VT);
      SDValue Mask = AllOnes;

      if (LHSMask.getNode()) {
        SDValue RHSBits = DAG.getNode(ISD::SRL, DL, VT, AllOnes, RHSShiftAmt);
        Mask = DAG.getNode(ISD::AND, DL, VT, Mask,
                           DAG.getNode(ISD::OR, DL, VT, LHSMask, RHSBits));
      }
      if (RHSMask.getNode()) {
        SDValue LHSBits = DAG.getNode(ISD::SHL, DL, VT, AllOnes, LHSShiftAmt);
        Mask = DAG.getNode(ISD::AND, DL, VT, Mask,
                           DAG.getNode(ISD::OR, DL, VT, RHSMask, LHSBits));
      }

      Res = DAG.getNode(ISD::AND, DL, VT, Res, Mask);
    }

    return Res;
  }

  // If there is a mask here, and we have a variable shift, we can't be sure
  // that we're masking out the right stuff.
  if (LHSMask.getNode() || RHSMask.getNode())
    return SDValue();

  // If the shift amount is sign/zext/any-extended just peel it off.
  SDValue LExtOp0 = LHSShiftAmt;
  SDValue RExtOp0 = RHSShiftAmt;
  if ((LHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::ANY_EXTEND ||
       LHSShiftAmt.getOpcode() == ISD::TRUNCATE) &&
      (RHSShiftAmt.getOpcode() == ISD::SIGN_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::ZERO_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::ANY_EXTEND ||
       RHSShiftAmt.getOpcode() == ISD::TRUNCATE)) {
    LExtOp0 = LHSShiftAmt.getOperand(0);
    RExtOp0 = RHSShiftAmt.getOperand(0);
  }

  if (IsRotate && (HasROTL || HasROTR)) {
    SDValue TryL =
        MatchRotatePosNeg(LHSShiftArg, LHSShiftAmt, RHSShiftAmt, LExtOp0,
                          RExtOp0, ISD::ROTL, ISD::ROTR, DL);
    if (TryL)
      return TryL;

    SDValue TryR =
        MatchRotatePosNeg(RHSShiftArg, RHSShiftAmt, LHSShiftAmt, RExtOp0,
                          LExtOp0, ISD::ROTR, ISD::ROTL, DL);
    if (TryR)
      return TryR;
  }

  SDValue TryL =
      MatchFunnelPosNeg(LHSShiftArg, RHSShiftArg, LHSShiftAmt, RHSShiftAmt,
                        LExtOp0, RExtOp0, ISD::FSHL, ISD::FSHR, DL);
  if (TryL)
    return TryL;

  SDValue TryR =
      MatchFunnelPosNeg(LHSShiftArg, RHSShiftArg, RHSShiftAmt, LHSShiftAmt,
                        RExtOp0, LExtOp0, ISD::FSHR, ISD::FSHL, DL);
  if (TryR)
    return TryR;

  return SDValue();
}

namespace {

/// Represents known origin of an individual byte in load combine pattern. The
/// value of the byte is either constant zero or comes from memory.
struct ByteProvider {
  // For constant zero providers Load is set to nullptr. For memory providers
  // Load represents the node which loads the byte from memory.
  // ByteOffset is the offset of the byte in the value produced by the load.
  LoadSDNode *Load = nullptr;
  unsigned ByteOffset = 0;

  ByteProvider() = default;

  static ByteProvider getMemory(LoadSDNode *Load, unsigned ByteOffset) {
    return ByteProvider(Load, ByteOffset);
  }

  static ByteProvider getConstantZero() { return ByteProvider(nullptr, 0); }

  bool isConstantZero() const { return !Load; }
  bool isMemory() const { return Load; }

  bool operator==(const ByteProvider &Other) const {
    return Other.Load == Load && Other.ByteOffset == ByteOffset;
  }

private:
  ByteProvider(LoadSDNode *Load, unsigned ByteOffset)
      : Load(Load), ByteOffset(ByteOffset) {}
};

} // end anonymous namespace

/// Recursively traverses the expression calculating the origin of the requested
/// byte of the given value. Returns None if the provider can't be calculated.
///
/// For all the values except the root of the expression verifies that the value
/// has exactly one use and if it's not true return None. This way if the origin
/// of the byte is returned it's guaranteed that the values which contribute to
/// the byte are not used outside of this expression.
///
/// Because the parts of the expression are not allowed to have more than one
/// use this function iterates over trees, not DAGs. So it never visits the same
/// node more than once.
static const Optional<ByteProvider>
calculateByteProvider(SDValue Op, unsigned Index, unsigned Depth,
                      bool Root = false) {
  // Typical i64 by i8 pattern requires recursion up to 8 calls depth
  if (Depth == 10)
    return None;

  if (!Root && !Op.hasOneUse())
    return None;

  assert(Op.getValueType().isScalarInteger() && "can't handle other types");
  unsigned BitWidth = Op.getValueSizeInBits();
  if (BitWidth % 8 != 0)
    return None;
  unsigned ByteWidth = BitWidth / 8;
  assert(Index < ByteWidth && "invalid index requested");
  (void) ByteWidth;

  switch (Op.getOpcode()) {
  case ISD::OR: {
    auto LHS = calculateByteProvider(Op->getOperand(0), Index, Depth + 1);
    if (!LHS)
      return None;
    auto RHS = calculateByteProvider(Op->getOperand(1), Index, Depth + 1);
    if (!RHS)
      return None;

    if (LHS->isConstantZero())
      return RHS;
    if (RHS->isConstantZero())
      return LHS;
    return None;
  }
  case ISD::SHL: {
    auto ShiftOp = dyn_cast<ConstantSDNode>(Op->getOperand(1));
    if (!ShiftOp)
      return None;

    uint64_t BitShift = ShiftOp->getZExtValue();
    if (BitShift % 8 != 0)
      return None;
    uint64_t ByteShift = BitShift / 8;

    return Index < ByteShift
               ? ByteProvider::getConstantZero()
               : calculateByteProvider(Op->getOperand(0), Index - ByteShift,
                                       Depth + 1);
  }
  case ISD::ANY_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND: {
    SDValue NarrowOp = Op->getOperand(0);
    unsigned NarrowBitWidth = NarrowOp.getScalarValueSizeInBits();
    if (NarrowBitWidth % 8 != 0)
      return None;
    uint64_t NarrowByteWidth = NarrowBitWidth / 8;

    if (Index >= NarrowByteWidth)
      return Op.getOpcode() == ISD::ZERO_EXTEND
                 ? Optional<ByteProvider>(ByteProvider::getConstantZero())
                 : None;
    return calculateByteProvider(NarrowOp, Index, Depth + 1);
  }
  case ISD::BSWAP:
    return calculateByteProvider(Op->getOperand(0), ByteWidth - Index - 1,
                                 Depth + 1);
  case ISD::LOAD: {
    auto L = cast<LoadSDNode>(Op.getNode());
    if (!L->isSimple() || L->isIndexed())
      return None;

    unsigned NarrowBitWidth = L->getMemoryVT().getSizeInBits();
    if (NarrowBitWidth % 8 != 0)
      return None;
    uint64_t NarrowByteWidth = NarrowBitWidth / 8;

    if (Index >= NarrowByteWidth)
      return L->getExtensionType() == ISD::ZEXTLOAD
                 ? Optional<ByteProvider>(ByteProvider::getConstantZero())
                 : None;
    return ByteProvider::getMemory(L, Index);
  }
  }

  return None;
}

static unsigned littleEndianByteAt(unsigned BW, unsigned i) {
  return i;
}

static unsigned bigEndianByteAt(unsigned BW, unsigned i) {
  return BW - i - 1;
}

// Check if the bytes offsets we are looking at match with either big or
// little endian value loaded. Return true for big endian, false for little
// endian, and None if match failed.
static Optional<bool> isBigEndian(const ArrayRef<int64_t> ByteOffsets,
                                  int64_t FirstOffset) {
  // The endian can be decided only when it is 2 bytes at least.
  unsigned Width = ByteOffsets.size();
  if (Width < 2)
    return None;

  bool BigEndian = true, LittleEndian = true;
  for (unsigned i = 0; i < Width; i++) {
    int64_t CurrentByteOffset = ByteOffsets[i] - FirstOffset;
    LittleEndian &= CurrentByteOffset == littleEndianByteAt(Width, i);
    BigEndian &= CurrentByteOffset == bigEndianByteAt(Width, i);
    if (!BigEndian && !LittleEndian)
      return None;
  }

  assert((BigEndian != LittleEndian) && "It should be either big endian or"
                                        "little endian");
  return BigEndian;
}

static SDValue stripTruncAndExt(SDValue Value) {
  switch (Value.getOpcode()) {
  case ISD::TRUNCATE:
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND:
    return stripTruncAndExt(Value.getOperand(0));
  }
  return Value;
}

/// Match a pattern where a wide type scalar value is stored by several narrow
/// stores. Fold it into a single store or a BSWAP and a store if the targets
/// supports it.
///
/// Assuming little endian target:
///  i8 *p = ...
///  i32 val = ...
///  p[0] = (val >> 0) & 0xFF;
///  p[1] = (val >> 8) & 0xFF;
///  p[2] = (val >> 16) & 0xFF;
///  p[3] = (val >> 24) & 0xFF;
/// =>
///  *((i32)p) = val;
///
///  i8 *p = ...
///  i32 val = ...
///  p[0] = (val >> 24) & 0xFF;
///  p[1] = (val >> 16) & 0xFF;
///  p[2] = (val >> 8) & 0xFF;
///  p[3] = (val >> 0) & 0xFF;
/// =>
///  *((i32)p) = BSWAP(val);
SDValue DAGCombiner::mergeTruncStores(StoreSDNode *N) {
  // The matching looks for "store (trunc x)" patterns that appear early but are
  // likely to be replaced by truncating store nodes during combining.
  // TODO: If there is evidence that running this later would help, this
  //       limitation could be removed. Legality checks may need to be added
  //       for the created store and optional bswap/rotate.
  if (LegalOperations)
    return SDValue();

  // We only handle merging simple stores of 1-4 bytes.
  // TODO: Allow unordered atomics when wider type is legal (see D66309)
  EVT MemVT = N->getMemoryVT();
  if (!(MemVT == MVT::i8 || MemVT == MVT::i16 || MemVT == MVT::i32) ||
      !N->isSimple() || N->isIndexed())
    return SDValue();

  // Collect all of the stores in the chain.
  SDValue Chain = N->getChain();
  SmallVector<StoreSDNode *, 8> Stores = {N};
  while (auto *Store = dyn_cast<StoreSDNode>(Chain)) {
    // All stores must be the same size to ensure that we are writing all of the
    // bytes in the wide value.
    // TODO: We could allow multiple sizes by tracking each stored byte.
    if (Store->getMemoryVT() != MemVT || !Store->isSimple() ||
        Store->isIndexed())
      return SDValue();
    Stores.push_back(Store);
    Chain = Store->getChain();
  }
  // There is no reason to continue if we do not have at least a pair of stores.
  if (Stores.size() < 2)
    return SDValue();

  // Handle simple types only.
  LLVMContext &Context = *DAG.getContext();
  unsigned NumStores = Stores.size();
  unsigned NarrowNumBits = N->getMemoryVT().getScalarSizeInBits();
  unsigned WideNumBits = NumStores * NarrowNumBits;
  EVT WideVT = EVT::getIntegerVT(Context, WideNumBits);
  if (WideVT != MVT::i16 && WideVT != MVT::i32 && WideVT != MVT::i64)
    return SDValue();

  // Check if all bytes of the source value that we are looking at are stored
  // to the same base address. Collect offsets from Base address into OffsetMap.
  SDValue SourceValue;
  SmallVector<int64_t, 8> OffsetMap(NumStores, INT64_MAX);
  int64_t FirstOffset = INT64_MAX;
  StoreSDNode *FirstStore = nullptr;
  Optional<BaseIndexOffset> Base;
  for (auto Store : Stores) {
    // All the stores store different parts of the CombinedValue. A truncate is
    // required to get the partial value.
    SDValue Trunc = Store->getValue();
    if (Trunc.getOpcode() != ISD::TRUNCATE)
      return SDValue();
    // Other than the first/last part, a shift operation is required to get the
    // offset.
    int64_t Offset = 0;
    SDValue WideVal = Trunc.getOperand(0);
    if ((WideVal.getOpcode() == ISD::SRL || WideVal.getOpcode() == ISD::SRA) &&
        isa<ConstantSDNode>(WideVal.getOperand(1))) {
      // The shift amount must be a constant multiple of the narrow type.
      // It is translated to the offset address in the wide source value "y".
      //
      // x = srl y, ShiftAmtC
      // i8 z = trunc x
      // store z, ...
      uint64_t ShiftAmtC = WideVal.getConstantOperandVal(1);
      if (ShiftAmtC % NarrowNumBits != 0)
        return SDValue();

      Offset = ShiftAmtC / NarrowNumBits;
      WideVal = WideVal.getOperand(0);
    }

    // Stores must share the same source value with different offsets.
    // Truncate and extends should be stripped to get the single source value.
    if (!SourceValue)
      SourceValue = WideVal;
    else if (stripTruncAndExt(SourceValue) != stripTruncAndExt(WideVal))
      return SDValue();
    else if (SourceValue.getValueType() != WideVT) {
      if (WideVal.getValueType() == WideVT ||
          WideVal.getScalarValueSizeInBits() >
              SourceValue.getScalarValueSizeInBits())
        SourceValue = WideVal;
      // Give up if the source value type is smaller than the store size.
      if (SourceValue.getScalarValueSizeInBits() < WideVT.getScalarSizeInBits())
        return SDValue();
    }

    // Stores must share the same base address.
    BaseIndexOffset Ptr = BaseIndexOffset::match(Store, DAG);
    int64_t ByteOffsetFromBase = 0;
    if (!Base)
      Base = Ptr;
    else if (!Base->equalBaseIndex(Ptr, DAG, ByteOffsetFromBase))
      return SDValue();

    // Remember the first store.
    if (ByteOffsetFromBase < FirstOffset) {
      FirstStore = Store;
      FirstOffset = ByteOffsetFromBase;
    }
    // Map the offset in the store and the offset in the combined value, and
    // early return if it has been set before.
    if (Offset < 0 || Offset >= NumStores || OffsetMap[Offset] != INT64_MAX)
      return SDValue();
    OffsetMap[Offset] = ByteOffsetFromBase;
  }

  assert(FirstOffset != INT64_MAX && "First byte offset must be set");
  assert(FirstStore && "First store must be set");

  // Check that a store of the wide type is both allowed and fast on the target
  const DataLayout &Layout = DAG.getDataLayout();
  bool Fast = false;
  bool Allowed = TLI.allowsMemoryAccess(Context, Layout, WideVT,
                                        *FirstStore->getMemOperand(), &Fast);
  if (!Allowed || !Fast)
    return SDValue();

  // Check if the pieces of the value are going to the expected places in memory
  // to merge the stores.
  auto checkOffsets = [&](bool MatchLittleEndian) {
    if (MatchLittleEndian) {
      for (unsigned i = 0; i != NumStores; ++i)
        if (OffsetMap[i] != i * (NarrowNumBits / 8) + FirstOffset)
          return false;
    } else { // MatchBigEndian by reversing loop counter.
      for (unsigned i = 0, j = NumStores - 1; i != NumStores; ++i, --j)
        if (OffsetMap[j] != i * (NarrowNumBits / 8) + FirstOffset)
          return false;
    }
    return true;
  };

  // Check if the offsets line up for the native data layout of this target.
  bool NeedBswap = false;
  bool NeedRotate = false;
  if (!checkOffsets(Layout.isLittleEndian())) {
    // Special-case: check if byte offsets line up for the opposite endian.
    if (NarrowNumBits == 8 && checkOffsets(Layout.isBigEndian()))
      NeedBswap = true;
    else if (NumStores == 2 && checkOffsets(Layout.isBigEndian()))
      NeedRotate = true;
    else
      return SDValue();
  }

  SDLoc DL(N);
  if (WideVT != SourceValue.getValueType()) {
    assert(SourceValue.getValueType().getScalarSizeInBits() > WideNumBits &&
           "Unexpected store value to merge");
    SourceValue = DAG.getNode(ISD::TRUNCATE, DL, WideVT, SourceValue);
  }

  // Before legalize we can introduce illegal bswaps/rotates which will be later
  // converted to an explicit bswap sequence. This way we end up with a single
  // store and byte shuffling instead of several stores and byte shuffling.
  if (NeedBswap) {
    SourceValue = DAG.getNode(ISD::BSWAP, DL, WideVT, SourceValue);
  } else if (NeedRotate) {
    assert(WideNumBits % 2 == 0 && "Unexpected type for rotate");
    SDValue RotAmt = DAG.getConstant(WideNumBits / 2, DL, WideVT);
    SourceValue = DAG.getNode(ISD::ROTR, DL, WideVT, SourceValue, RotAmt);
  }

  SDValue NewStore =
      DAG.getStore(Chain, DL, SourceValue, FirstStore->getBasePtr(),
                   FirstStore->getPointerInfo(), FirstStore->getAlign());

  // Rely on other DAG combine rules to remove the other individual stores.
  DAG.ReplaceAllUsesWith(N, NewStore.getNode());
  return NewStore;
}

/// Match a pattern where a wide type scalar value is loaded by several narrow
/// loads and combined by shifts and ors. Fold it into a single load or a load
/// and a BSWAP if the targets supports it.
///
/// Assuming little endian target:
///  i8 *a = ...
///  i32 val = a[0] | (a[1] << 8) | (a[2] << 16) | (a[3] << 24)
/// =>
///  i32 val = *((i32)a)
///
///  i8 *a = ...
///  i32 val = (a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3]
/// =>
///  i32 val = BSWAP(*((i32)a))
///
/// TODO: This rule matches complex patterns with OR node roots and doesn't
/// interact well with the worklist mechanism. When a part of the pattern is
/// updated (e.g. one of the loads) its direct users are put into the worklist,
/// but the root node of the pattern which triggers the load combine is not
/// necessarily a direct user of the changed node. For example, once the address
/// of t28 load is reassociated load combine won't be triggered:
///             t25: i32 = add t4, Constant:i32<2>
///           t26: i64 = sign_extend t25
///        t27: i64 = add t2, t26
///       t28: i8,ch = load<LD1[%tmp9]> t0, t27, undef:i64
///     t29: i32 = zero_extend t28
///   t32: i32 = shl t29, Constant:i8<8>
/// t33: i32 = or t23, t32
/// As a possible fix visitLoad can check if the load can be a part of a load
/// combine pattern and add corresponding OR roots to the worklist.
SDValue DAGCombiner::MatchLoadCombine(SDNode *N) {
  assert(N->getOpcode() == ISD::OR &&
         "Can only match load combining against OR nodes");

  // Handles simple types only
  EVT VT = N->getValueType(0);
  if (VT != MVT::i16 && VT != MVT::i32 && VT != MVT::i64)
    return SDValue();
  unsigned ByteWidth = VT.getSizeInBits() / 8;

  bool IsBigEndianTarget = DAG.getDataLayout().isBigEndian();
  auto MemoryByteOffset = [&] (ByteProvider P) {
    assert(P.isMemory() && "Must be a memory byte provider");
    unsigned LoadBitWidth = P.Load->getMemoryVT().getSizeInBits();
    assert(LoadBitWidth % 8 == 0 &&
           "can only analyze providers for individual bytes not bit");
    unsigned LoadByteWidth = LoadBitWidth / 8;
    return IsBigEndianTarget
            ? bigEndianByteAt(LoadByteWidth, P.ByteOffset)
            : littleEndianByteAt(LoadByteWidth, P.ByteOffset);
  };

  Optional<BaseIndexOffset> Base;
  SDValue Chain;

  SmallPtrSet<LoadSDNode *, 8> Loads;
  Optional<ByteProvider> FirstByteProvider;
  int64_t FirstOffset = INT64_MAX;

  // Check if all the bytes of the OR we are looking at are loaded from the same
  // base address. Collect bytes offsets from Base address in ByteOffsets.
  SmallVector<int64_t, 8> ByteOffsets(ByteWidth);
  unsigned ZeroExtendedBytes = 0;
  for (int i = ByteWidth - 1; i >= 0; --i) {
    auto P = calculateByteProvider(SDValue(N, 0), i, 0, /*Root=*/true);
    if (!P)
      return SDValue();

    if (P->isConstantZero()) {
      // It's OK for the N most significant bytes to be 0, we can just
      // zero-extend the load.
      if (++ZeroExtendedBytes != (ByteWidth - static_cast<unsigned>(i)))
        return SDValue();
      continue;
    }
    assert(P->isMemory() && "provenance should either be memory or zero");

    LoadSDNode *L = P->Load;
    assert(L->hasNUsesOfValue(1, 0) && L->isSimple() &&
           !L->isIndexed() &&
           "Must be enforced by calculateByteProvider");
    assert(L->getOffset().isUndef() && "Unindexed load must have undef offset");

    // All loads must share the same chain
    SDValue LChain = L->getChain();
    if (!Chain)
      Chain = LChain;
    else if (Chain != LChain)
      return SDValue();

    // Loads must share the same base address
    BaseIndexOffset Ptr = BaseIndexOffset::match(L, DAG);
    int64_t ByteOffsetFromBase = 0;
    if (!Base)
      Base = Ptr;
    else if (!Base->equalBaseIndex(Ptr, DAG, ByteOffsetFromBase))
      return SDValue();

    // Calculate the offset of the current byte from the base address
    ByteOffsetFromBase += MemoryByteOffset(*P);
    ByteOffsets[i] = ByteOffsetFromBase;

    // Remember the first byte load
    if (ByteOffsetFromBase < FirstOffset) {
      FirstByteProvider = P;
      FirstOffset = ByteOffsetFromBase;
    }

    Loads.insert(L);
  }
  assert(!Loads.empty() && "All the bytes of the value must be loaded from "
         "memory, so there must be at least one load which produces the value");
  assert(Base && "Base address of the accessed memory location must be set");
  assert(FirstOffset != INT64_MAX && "First byte offset must be set");

  bool NeedsZext = ZeroExtendedBytes > 0;

  EVT MemVT =
      EVT::getIntegerVT(*DAG.getContext(), (ByteWidth - ZeroExtendedBytes) * 8);

  if (!MemVT.isSimple())
    return SDValue();

  // Before legalize we can introduce too wide illegal loads which will be later
  // split into legal sized loads. This enables us to combine i64 load by i8
  // patterns to a couple of i32 loads on 32 bit targets.
  if (LegalOperations &&
      !TLI.isOperationLegal(NeedsZext ? ISD::ZEXTLOAD : ISD::NON_EXTLOAD,
                            MemVT))
    return SDValue();

  // Check if the bytes of the OR we are looking at match with either big or
  // little endian value load
  Optional<bool> IsBigEndian = isBigEndian(
      makeArrayRef(ByteOffsets).drop_back(ZeroExtendedBytes), FirstOffset);
  if (!IsBigEndian.hasValue())
    return SDValue();

  assert(FirstByteProvider && "must be set");

  // Ensure that the first byte is loaded from zero offset of the first load.
  // So the combined value can be loaded from the first load address.
  if (MemoryByteOffset(*FirstByteProvider) != 0)
    return SDValue();
  LoadSDNode *FirstLoad = FirstByteProvider->Load;

  // The node we are looking at matches with the pattern, check if we can
  // replace it with a single (possibly zero-extended) load and bswap + shift if
  // needed.

  // If the load needs byte swap check if the target supports it
  bool NeedsBswap = IsBigEndianTarget != *IsBigEndian;

  // Before legalize we can introduce illegal bswaps which will be later
  // converted to an explicit bswap sequence. This way we end up with a single
  // load and byte shuffling instead of several loads and byte shuffling.
  // We do not introduce illegal bswaps when zero-extending as this tends to
  // introduce too many arithmetic instructions.
  if (NeedsBswap && (LegalOperations || NeedsZext) &&
      !TLI.isOperationLegal(ISD::BSWAP, VT))
    return SDValue();

  // If we need to bswap and zero extend, we have to insert a shift. Check that
  // it is legal.
  if (NeedsBswap && NeedsZext && LegalOperations &&
      !TLI.isOperationLegal(ISD::SHL, VT))
    return SDValue();

  // Check that a load of the wide type is both allowed and fast on the target
  bool Fast = false;
  bool Allowed =
      TLI.allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), MemVT,
                             *FirstLoad->getMemOperand(), &Fast);
  if (!Allowed || !Fast)
    return SDValue();

  SDValue NewLoad =
      DAG.getExtLoad(NeedsZext ? ISD::ZEXTLOAD : ISD::NON_EXTLOAD, SDLoc(N), VT,
                     Chain, FirstLoad->getBasePtr(),
                     FirstLoad->getPointerInfo(), MemVT, FirstLoad->getAlign());

  // Transfer chain users from old loads to the new load.
  for (LoadSDNode *L : Loads)
    DAG.ReplaceAllUsesOfValueWith(SDValue(L, 1), SDValue(NewLoad.getNode(), 1));

  if (!NeedsBswap)
    return NewLoad;

  SDValue ShiftedLoad =
      NeedsZext
          ? DAG.getNode(ISD::SHL, SDLoc(N), VT, NewLoad,
                        DAG.getShiftAmountConstant(ZeroExtendedBytes * 8, VT,
                                                   SDLoc(N), LegalOperations))
          : NewLoad;
  return DAG.getNode(ISD::BSWAP, SDLoc(N), VT, ShiftedLoad);
}

// If the target has andn, bsl, or a similar bit-select instruction,
// we want to unfold masked merge, with canonical pattern of:
//   |        A  |  |B|
//   ((x ^ y) & m) ^ y
//    |  D  |
// Into:
//   (x & m) | (y & ~m)
// If y is a constant, and the 'andn' does not work with immediates,
// we unfold into a different pattern:
//   ~(~x & m) & (m | y)
// NOTE: we don't unfold the pattern if 'xor' is actually a 'not', because at
//       the very least that breaks andnpd / andnps patterns, and because those
//       patterns are simplified in IR and shouldn't be created in the DAG
SDValue DAGCombiner::unfoldMaskedMerge(SDNode *N) {
  assert(N->getOpcode() == ISD::XOR);

  // Don't touch 'not' (i.e. where y = -1).
  if (isAllOnesOrAllOnesSplat(N->getOperand(1)))
    return SDValue();

  EVT VT = N->getValueType(0);

  // There are 3 commutable operators in the pattern,
  // so we have to deal with 8 possible variants of the basic pattern.
  SDValue X, Y, M;
  auto matchAndXor = [&X, &Y, &M](SDValue And, unsigned XorIdx, SDValue Other) {
    if (And.getOpcode() != ISD::AND || !And.hasOneUse())
      return false;
    SDValue Xor = And.getOperand(XorIdx);
    if (Xor.getOpcode() != ISD::XOR || !Xor.hasOneUse())
      return false;
    SDValue Xor0 = Xor.getOperand(0);
    SDValue Xor1 = Xor.getOperand(1);
    // Don't touch 'not' (i.e. where y = -1).
    if (isAllOnesOrAllOnesSplat(Xor1))
      return false;
    if (Other == Xor0)
      std::swap(Xor0, Xor1);
    if (Other != Xor1)
      return false;
    X = Xor0;
    Y = Xor1;
    M = And.getOperand(XorIdx ? 0 : 1);
    return true;
  };

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (!matchAndXor(N0, 0, N1) && !matchAndXor(N0, 1, N1) &&
      !matchAndXor(N1, 0, N0) && !matchAndXor(N1, 1, N0))
    return SDValue();

  // Don't do anything if the mask is constant. This should not be reachable.
  // InstCombine should have already unfolded this pattern, and DAGCombiner
  // probably shouldn't produce it, too.
  if (isa<ConstantSDNode>(M.getNode()))
    return SDValue();

  // We can transform if the target has AndNot
  if (!TLI.hasAndNot(M))
    return SDValue();

  SDLoc DL(N);

  // If Y is a constant, check that 'andn' works with immediates.
  if (!TLI.hasAndNot(Y)) {
    assert(TLI.hasAndNot(X) && "Only mask is a variable? Unreachable.");
    // If not, we need to do a bit more work to make sure andn is still used.
    SDValue NotX = DAG.getNOT(DL, X, VT);
    SDValue LHS = DAG.getNode(ISD::AND, DL, VT, NotX, M);
    SDValue NotLHS = DAG.getNOT(DL, LHS, VT);
    SDValue RHS = DAG.getNode(ISD::OR, DL, VT, M, Y);
    return DAG.getNode(ISD::AND, DL, VT, NotLHS, RHS);
  }

  SDValue LHS = DAG.getNode(ISD::AND, DL, VT, X, M);
  SDValue NotM = DAG.getNOT(DL, M, VT);
  SDValue RHS = DAG.getNode(ISD::AND, DL, VT, Y, NotM);

  return DAG.getNode(ISD::OR, DL, VT, LHS, RHS);
}

SDValue DAGCombiner::visitXOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0.getValueType();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    // fold (xor x, 0) -> x, vector edition
    if (ISD::isConstantSplatVectorAllZeros(N0.getNode()))
      return N1;
    if (ISD::isConstantSplatVectorAllZeros(N1.getNode()))
      return N0;
  }

  // fold (xor undef, undef) -> 0. This is a common idiom (misuse).
  SDLoc DL(N);
  if (N0.isUndef() && N1.isUndef())
    return DAG.getConstant(0, DL, VT);

  // fold (xor x, undef) -> undef
  if (N0.isUndef())
    return N0;
  if (N1.isUndef())
    return N1;

  // fold (xor c1, c2) -> c1^c2
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::XOR, DL, VT, {N0, N1}))
    return C;

  // canonicalize constant to RHS
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
     !DAG.isConstantIntBuildVectorOrConstantInt(N1))
    return DAG.getNode(ISD::XOR, DL, VT, N1, N0);

  // fold (xor x, 0) -> x
  if (isNullConstant(N1))
    return N0;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // reassociate xor
  if (SDValue RXOR = reassociateOps(ISD::XOR, DL, N0, N1, N->getFlags()))
    return RXOR;

  // fold !(x cc y) -> (x !cc y)
  unsigned N0Opcode = N0.getOpcode();
  SDValue LHS, RHS, CC;
  if (TLI.isConstTrueVal(N1.getNode()) &&
      isSetCCEquivalent(N0, LHS, RHS, CC, /*MatchStrict*/true)) {
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               LHS.getValueType());
    if (!LegalOperations ||
        TLI.isCondCodeLegal(NotCC, LHS.getSimpleValueType())) {
      switch (N0Opcode) {
      default:
        llvm_unreachable("Unhandled SetCC Equivalent!");
      case ISD::SETCC:
        return DAG.getSetCC(SDLoc(N0), VT, LHS, RHS, NotCC);
      case ISD::SELECT_CC:
        return DAG.getSelectCC(SDLoc(N0), LHS, RHS, N0.getOperand(2),
                               N0.getOperand(3), NotCC);
      case ISD::STRICT_FSETCC:
      case ISD::STRICT_FSETCCS: {
        if (N0.hasOneUse()) {
          // FIXME Can we handle multiple uses? Could we token factor the chain
          // results from the new/old setcc?
          SDValue SetCC =
              DAG.getSetCC(SDLoc(N0), VT, LHS, RHS, NotCC,
                           N0.getOperand(0), N0Opcode == ISD::STRICT_FSETCCS);
          CombineTo(N, SetCC);
          DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), SetCC.getValue(1));
          recursivelyDeleteUnusedNodes(N0.getNode());
          return SDValue(N, 0); // Return N so it doesn't get rechecked!
        }
        break;
      }
      }
    }
  }

  // fold (not (zext (setcc x, y))) -> (zext (not (setcc x, y)))
  if (isOneConstant(N1) && N0Opcode == ISD::ZERO_EXTEND && N0.hasOneUse() &&
      isSetCCEquivalent(N0.getOperand(0), LHS, RHS, CC)){
    SDValue V = N0.getOperand(0);
    SDLoc DL0(N0);
    V = DAG.getNode(ISD::XOR, DL0, V.getValueType(), V,
                    DAG.getConstant(1, DL0, V.getValueType()));
    AddToWorklist(V.getNode());
    return DAG.getNode(ISD::ZERO_EXTEND, DL, VT, V);
  }

  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are setcc
  if (isOneConstant(N1) && VT == MVT::i1 && N0.hasOneUse() &&
      (N0Opcode == ISD::OR || N0Opcode == ISD::AND)) {
    SDValue N00 = N0.getOperand(0), N01 = N0.getOperand(1);
    if (isOneUseSetCC(N01) || isOneUseSetCC(N00)) {
      unsigned NewOpcode = N0Opcode == ISD::AND ? ISD::OR : ISD::AND;
      N00 = DAG.getNode(ISD::XOR, SDLoc(N00), VT, N00, N1); // N00 = ~N00
      N01 = DAG.getNode(ISD::XOR, SDLoc(N01), VT, N01, N1); // N01 = ~N01
      AddToWorklist(N00.getNode()); AddToWorklist(N01.getNode());
      return DAG.getNode(NewOpcode, DL, VT, N00, N01);
    }
  }
  // fold (not (or x, y)) -> (and (not x), (not y)) iff x or y are constants
  if (isAllOnesConstant(N1) && N0.hasOneUse() &&
      (N0Opcode == ISD::OR || N0Opcode == ISD::AND)) {
    SDValue N00 = N0.getOperand(0), N01 = N0.getOperand(1);
    if (isa<ConstantSDNode>(N01) || isa<ConstantSDNode>(N00)) {
      unsigned NewOpcode = N0Opcode == ISD::AND ? ISD::OR : ISD::AND;
      N00 = DAG.getNode(ISD::XOR, SDLoc(N00), VT, N00, N1); // N00 = ~N00
      N01 = DAG.getNode(ISD::XOR, SDLoc(N01), VT, N01, N1); // N01 = ~N01
      AddToWorklist(N00.getNode()); AddToWorklist(N01.getNode());
      return DAG.getNode(NewOpcode, DL, VT, N00, N01);
    }
  }

  // fold (not (neg x)) -> (add X, -1)
  // FIXME: This can be generalized to (not (sub Y, X)) -> (add X, ~Y) if
  // Y is a constant or the subtract has a single use.
  if (isAllOnesConstant(N1) && N0.getOpcode() == ISD::SUB &&
      isNullConstant(N0.getOperand(0))) {
    return DAG.getNode(ISD::ADD, DL, VT, N0.getOperand(1),
                       DAG.getAllOnesConstant(DL, VT));
  }

  // fold (not (add X, -1)) -> (neg X)
  if (isAllOnesConstant(N1) && N0.getOpcode() == ISD::ADD &&
      isAllOnesOrAllOnesSplat(N0.getOperand(1))) {
    return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT),
                       N0.getOperand(0));
  }

  // fold (xor (and x, y), y) -> (and (not x), y)
  if (N0Opcode == ISD::AND && N0.hasOneUse() && N0->getOperand(1) == N1) {
    SDValue X = N0.getOperand(0);
    SDValue NotX = DAG.getNOT(SDLoc(X), X, VT);
    AddToWorklist(NotX.getNode());
    return DAG.getNode(ISD::AND, DL, VT, NotX, N1);
  }

  if ((N0Opcode == ISD::SRL || N0Opcode == ISD::SHL) && N0.hasOneUse()) {
    ConstantSDNode *XorC = isConstOrConstSplat(N1);
    ConstantSDNode *ShiftC = isConstOrConstSplat(N0.getOperand(1));
    unsigned BitWidth = VT.getScalarSizeInBits();
    if (XorC && ShiftC) {
      // Don't crash on an oversized shift. We can not guarantee that a bogus
      // shift has been simplified to undef.
      uint64_t ShiftAmt = ShiftC->getLimitedValue();
      if (ShiftAmt < BitWidth) {
        APInt Ones = APInt::getAllOnesValue(BitWidth);
        Ones = N0Opcode == ISD::SHL ? Ones.shl(ShiftAmt) : Ones.lshr(ShiftAmt);
        if (XorC->getAPIntValue() == Ones) {
          // If the xor constant is a shifted -1, do a 'not' before the shift:
          // xor (X << ShiftC), XorC --> (not X) << ShiftC
          // xor (X >> ShiftC), XorC --> (not X) >> ShiftC
          SDValue Not = DAG.getNOT(DL, N0.getOperand(0), VT);
          return DAG.getNode(N0Opcode, DL, VT, Not, N0.getOperand(1));
        }
      }
    }
  }

  // fold Y = sra (X, size(X)-1); xor (add (X, Y), Y) -> (abs X)
  if (TLI.isOperationLegalOrCustom(ISD::ABS, VT)) {
    SDValue A = N0Opcode == ISD::ADD ? N0 : N1;
    SDValue S = N0Opcode == ISD::SRA ? N0 : N1;
    if (A.getOpcode() == ISD::ADD && S.getOpcode() == ISD::SRA) {
      SDValue A0 = A.getOperand(0), A1 = A.getOperand(1);
      SDValue S0 = S.getOperand(0);
      if ((A0 == S && A1 == S0) || (A1 == S && A0 == S0))
        if (ConstantSDNode *C = isConstOrConstSplat(S.getOperand(1)))
          if (C->getAPIntValue() == (VT.getScalarSizeInBits() - 1))
            return DAG.getNode(ISD::ABS, DL, VT, S0);
    }
  }

  // fold (xor x, x) -> 0
  if (N0 == N1)
    return tryFoldToZero(DL, TLI, VT, DAG, LegalOperations);

  // fold (xor (shl 1, x), -1) -> (rotl ~1, x)
  // Here is a concrete example of this equivalence:
  // i16   x ==  14
  // i16 shl ==   1 << 14  == 16384 == 0b0100000000000000
  // i16 xor == ~(1 << 14) == 49151 == 0b1011111111111111
  //
  // =>
  //
  // i16     ~1      == 0b1111111111111110
  // i16 rol(~1, 14) == 0b1011111111111111
  //
  // Some additional tips to help conceptualize this transform:
  // - Try to see the operation as placing a single zero in a value of all ones.
  // - There exists no value for x which would allow the result to contain zero.
  // - Values of x larger than the bitwidth are undefined and do not require a
  //   consistent result.
  // - Pushing the zero left requires shifting one bits in from the right.
  // A rotate left of ~1 is a nice way of achieving the desired result.
  if (TLI.isOperationLegalOrCustom(ISD::ROTL, VT) && N0Opcode == ISD::SHL &&
      isAllOnesConstant(N1) && isOneConstant(N0.getOperand(0))) {
    return DAG.getNode(ISD::ROTL, DL, VT, DAG.getConstant(~1, DL, VT),
                       N0.getOperand(1));
  }

  // Simplify: xor (op x...), (op y...)  -> (op (xor x, y))
  if (N0Opcode == N1.getOpcode())
    if (SDValue V = hoistLogicOpWithSameOpcodeHands(N))
      return V;

  // Unfold  ((x ^ y) & m) ^ y  into  (x & m) | (y & ~m)  if profitable
  if (SDValue MM = unfoldMaskedMerge(N))
    return MM;

  // Simplify the expression using non-local knowledge.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  if (SDValue Combined = combineCarryDiamond(*this, DAG, TLI, N0, N1, N))
    return Combined;

  return SDValue();
}

/// If we have a shift-by-constant of a bitwise logic op that itself has a
/// shift-by-constant operand with identical opcode, we may be able to convert
/// that into 2 independent shifts followed by the logic op. This is a
/// throughput improvement.
static SDValue combineShiftOfShiftedLogic(SDNode *Shift, SelectionDAG &DAG) {
  // Match a one-use bitwise logic op.
  SDValue LogicOp = Shift->getOperand(0);
  if (!LogicOp.hasOneUse())
    return SDValue();

  unsigned LogicOpcode = LogicOp.getOpcode();
  if (LogicOpcode != ISD::AND && LogicOpcode != ISD::OR &&
      LogicOpcode != ISD::XOR)
    return SDValue();

  // Find a matching one-use shift by constant.
  unsigned ShiftOpcode = Shift->getOpcode();
  SDValue C1 = Shift->getOperand(1);
  ConstantSDNode *C1Node = isConstOrConstSplat(C1);
  assert(C1Node && "Expected a shift with constant operand");
  const APInt &C1Val = C1Node->getAPIntValue();
  auto matchFirstShift = [&](SDValue V, SDValue &ShiftOp,
                             const APInt *&ShiftAmtVal) {
    if (V.getOpcode() != ShiftOpcode || !V.hasOneUse())
      return false;

    ConstantSDNode *ShiftCNode = isConstOrConstSplat(V.getOperand(1));
    if (!ShiftCNode)
      return false;

    // Capture the shifted operand and shift amount value.
    ShiftOp = V.getOperand(0);
    ShiftAmtVal = &ShiftCNode->getAPIntValue();

    // Shift amount types do not have to match their operand type, so check that
    // the constants are the same width.
    if (ShiftAmtVal->getBitWidth() != C1Val.getBitWidth())
      return false;

    // The fold is not valid if the sum of the shift values exceeds bitwidth.
    if ((*ShiftAmtVal + C1Val).uge(V.getScalarValueSizeInBits()))
      return false;

    return true;
  };

  // Logic ops are commutative, so check each operand for a match.
  SDValue X, Y;
  const APInt *C0Val;
  if (matchFirstShift(LogicOp.getOperand(0), X, C0Val))
    Y = LogicOp.getOperand(1);
  else if (matchFirstShift(LogicOp.getOperand(1), X, C0Val))
    Y = LogicOp.getOperand(0);
  else
    return SDValue();

  // shift (logic (shift X, C0), Y), C1 -> logic (shift X, C0+C1), (shift Y, C1)
  SDLoc DL(Shift);
  EVT VT = Shift->getValueType(0);
  EVT ShiftAmtVT = Shift->getOperand(1).getValueType();
  SDValue ShiftSumC = DAG.getConstant(*C0Val + C1Val, DL, ShiftAmtVT);
  SDValue NewShift1 = DAG.getNode(ShiftOpcode, DL, VT, X, ShiftSumC);
  SDValue NewShift2 = DAG.getNode(ShiftOpcode, DL, VT, Y, C1);
  return DAG.getNode(LogicOpcode, DL, VT, NewShift1, NewShift2);
}

/// Handle transforms common to the three shifts, when the shift amount is a
/// constant.
/// We are looking for: (shift being one of shl/sra/srl)
///   shift (binop X, C0), C1
/// And want to transform into:
///   binop (shift X, C1), (shift C0, C1)
SDValue DAGCombiner::visitShiftByConstant(SDNode *N) {
  assert(isConstOrConstSplat(N->getOperand(1)) && "Expected constant operand");

  // Do not turn a 'not' into a regular xor.
  if (isBitwiseNot(N->getOperand(0)))
    return SDValue();

  // The inner binop must be one-use, since we want to replace it.
  SDValue LHS = N->getOperand(0);
  if (!LHS.hasOneUse() || !TLI.isDesirableToCommuteWithShift(N, Level))
    return SDValue();

  // TODO: This is limited to early combining because it may reveal regressions
  //       otherwise. But since we just checked a target hook to see if this is
  //       desirable, that should have filtered out cases where this interferes
  //       with some other pattern matching.
  if (!LegalTypes)
    if (SDValue R = combineShiftOfShiftedLogic(N, DAG))
      return R;

  // We want to pull some binops through shifts, so that we have (and (shift))
  // instead of (shift (and)), likewise for add, or, xor, etc.  This sort of
  // thing happens with address calculations, so it's important to canonicalize
  // it.
  switch (LHS.getOpcode()) {
  default:
    return SDValue();
  case ISD::OR:
  case ISD::XOR:
  case ISD::AND:
    break;
  case ISD::ADD:
    if (N->getOpcode() != ISD::SHL)
      return SDValue(); // only shl(add) not sr[al](add).
    break;
  }

  // We require the RHS of the binop to be a constant and not opaque as well.
  ConstantSDNode *BinOpCst = getAsNonOpaqueConstant(LHS.getOperand(1));
  if (!BinOpCst)
    return SDValue();

  // FIXME: disable this unless the input to the binop is a shift by a constant
  // or is copy/select. Enable this in other cases when figure out it's exactly
  // profitable.
  SDValue BinOpLHSVal = LHS.getOperand(0);
  bool IsShiftByConstant = (BinOpLHSVal.getOpcode() == ISD::SHL ||
                            BinOpLHSVal.getOpcode() == ISD::SRA ||
                            BinOpLHSVal.getOpcode() == ISD::SRL) &&
                           isa<ConstantSDNode>(BinOpLHSVal.getOperand(1));
  bool IsCopyOrSelect = BinOpLHSVal.getOpcode() == ISD::CopyFromReg ||
                        BinOpLHSVal.getOpcode() == ISD::SELECT;

  if (!IsShiftByConstant && !IsCopyOrSelect)
    return SDValue();

  if (IsCopyOrSelect && N->hasOneUse())
    return SDValue();

  // Fold the constants, shifting the binop RHS by the shift amount.
  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  SDValue NewRHS = DAG.getNode(N->getOpcode(), DL, VT, LHS.getOperand(1),
                               N->getOperand(1));
  assert(isa<ConstantSDNode>(NewRHS) && "Folding was not successful!");

  SDValue NewShift = DAG.getNode(N->getOpcode(), DL, VT, LHS.getOperand(0),
                                 N->getOperand(1));
  return DAG.getNode(LHS.getOpcode(), DL, VT, NewShift, NewRHS);
}

SDValue DAGCombiner::distributeTruncateThroughAnd(SDNode *N) {
  assert(N->getOpcode() == ISD::TRUNCATE);
  assert(N->getOperand(0).getOpcode() == ISD::AND);

  // (truncate:TruncVT (and N00, N01C)) -> (and (truncate:TruncVT N00), TruncC)
  EVT TruncVT = N->getValueType(0);
  if (N->hasOneUse() && N->getOperand(0).hasOneUse() &&
      TLI.isTypeDesirableForOp(ISD::AND, TruncVT)) {
    SDValue N01 = N->getOperand(0).getOperand(1);
    if (isConstantOrConstantVector(N01, /* NoOpaques */ true)) {
      SDLoc DL(N);
      SDValue N00 = N->getOperand(0).getOperand(0);
      SDValue Trunc00 = DAG.getNode(ISD::TRUNCATE, DL, TruncVT, N00);
      SDValue Trunc01 = DAG.getNode(ISD::TRUNCATE, DL, TruncVT, N01);
      AddToWorklist(Trunc00.getNode());
      AddToWorklist(Trunc01.getNode());
      return DAG.getNode(ISD::AND, DL, TruncVT, Trunc00, Trunc01);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitRotate(SDNode *N) {
  SDLoc dl(N);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  unsigned Bitsize = VT.getScalarSizeInBits();

  // fold (rot x, 0) -> x
  if (isNullOrNullSplat(N1))
    return N0;

  // fold (rot x, c) -> x iff (c % BitSize) == 0
  if (isPowerOf2_32(Bitsize) && Bitsize > 1) {
    APInt ModuloMask(N1.getScalarValueSizeInBits(), Bitsize - 1);
    if (DAG.MaskedValueIsZero(N1, ModuloMask))
      return N0;
  }

  // fold (rot x, c) -> (rot x, c % BitSize)
  bool OutOfRange = false;
  auto MatchOutOfRange = [Bitsize, &OutOfRange](ConstantSDNode *C) {
    OutOfRange |= C->getAPIntValue().uge(Bitsize);
    return true;
  };
  if (ISD::matchUnaryPredicate(N1, MatchOutOfRange) && OutOfRange) {
    EVT AmtVT = N1.getValueType();
    SDValue Bits = DAG.getConstant(Bitsize, dl, AmtVT);
    if (SDValue Amt =
            DAG.FoldConstantArithmetic(ISD::UREM, dl, AmtVT, {N1, Bits}))
      return DAG.getNode(N->getOpcode(), dl, VT, N0, Amt);
  }

  // rot i16 X, 8 --> bswap X
  auto *RotAmtC = isConstOrConstSplat(N1);
  if (RotAmtC && RotAmtC->getAPIntValue() == 8 &&
      VT.getScalarSizeInBits() == 16 && hasOperation(ISD::BSWAP, VT))
    return DAG.getNode(ISD::BSWAP, dl, VT, N0);

  // Simplify the operands using demanded-bits information.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (rot* x, (trunc (and y, c))) -> (rot* x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    if (SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode()))
      return DAG.getNode(N->getOpcode(), dl, VT, N0, NewOp1);
  }

  unsigned NextOp = N0.getOpcode();
  // fold (rot* (rot* x, c2), c1) -> (rot* x, c1 +- c2 % bitsize)
  if (NextOp == ISD::ROTL || NextOp == ISD::ROTR) {
    SDNode *C1 = DAG.isConstantIntBuildVectorOrConstantInt(N1);
    SDNode *C2 = DAG.isConstantIntBuildVectorOrConstantInt(N0.getOperand(1));
    if (C1 && C2 && C1->getValueType(0) == C2->getValueType(0)) {
      EVT ShiftVT = C1->getValueType(0);
      bool SameSide = (N->getOpcode() == NextOp);
      unsigned CombineOp = SameSide ? ISD::ADD : ISD::SUB;
      if (SDValue CombinedShift = DAG.FoldConstantArithmetic(
              CombineOp, dl, ShiftVT, {N1, N0.getOperand(1)})) {
        SDValue BitsizeC = DAG.getConstant(Bitsize, dl, ShiftVT);
        SDValue CombinedShiftNorm = DAG.FoldConstantArithmetic(
            ISD::SREM, dl, ShiftVT, {CombinedShift, BitsizeC});
        return DAG.getNode(N->getOpcode(), dl, VT, N0->getOperand(0),
                           CombinedShiftNorm);
      }
    }
  }
  return SDValue();
}

SDValue DAGCombiner::visitSHL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (SDValue V = DAG.simplifyShift(N0, N1))
    return V;

  EVT VT = N0.getValueType();
  EVT ShiftVT = N1.getValueType();
  unsigned OpSizeInBits = VT.getScalarSizeInBits();

  // fold vector ops
  if (VT.isVector()) {
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

    BuildVectorSDNode *N1CV = dyn_cast<BuildVectorSDNode>(N1);
    // If setcc produces all-one true value then:
    // (shl (and (setcc) N01CV) N1CV) -> (and (setcc) N01CV<<N1CV)
    if (N1CV && N1CV->isConstant()) {
      if (N0.getOpcode() == ISD::AND) {
        SDValue N00 = N0->getOperand(0);
        SDValue N01 = N0->getOperand(1);
        BuildVectorSDNode *N01CV = dyn_cast<BuildVectorSDNode>(N01);

        if (N01CV && N01CV->isConstant() && N00.getOpcode() == ISD::SETCC &&
            TLI.getBooleanContents(N00.getOperand(0).getValueType()) ==
                TargetLowering::ZeroOrNegativeOneBooleanContent) {
          if (SDValue C =
                  DAG.FoldConstantArithmetic(ISD::SHL, SDLoc(N), VT, {N01, N1}))
            return DAG.getNode(ISD::AND, SDLoc(N), VT, N00, C);
        }
      }
    }
  }

  ConstantSDNode *N1C = isConstOrConstSplat(N1);

  // fold (shl c1, c2) -> c1<<c2
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::SHL, SDLoc(N), VT, {N0, N1}))
    return C;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // if (shl x, c) is known to be zero, return 0
  if (DAG.MaskedValueIsZero(SDValue(N, 0),
                            APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, SDLoc(N), VT);

  // fold (shl x, (trunc (and y, c))) -> (shl x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    if (SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode()))
      return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0, NewOp1);
  }

  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (shl (shl x, c1), c2) -> 0 or (shl x, (add c1, c2))
  if (N0.getOpcode() == ISD::SHL) {
    auto MatchOutOfRange = [OpSizeInBits](ConstantSDNode *LHS,
                                          ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      return (c1 + c2).uge(OpSizeInBits);
    };
    if (ISD::matchBinaryPredicate(N1, N0.getOperand(1), MatchOutOfRange))
      return DAG.getConstant(0, SDLoc(N), VT);

    auto MatchInRange = [OpSizeInBits](ConstantSDNode *LHS,
                                       ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      return (c1 + c2).ult(OpSizeInBits);
    };
    if (ISD::matchBinaryPredicate(N1, N0.getOperand(1), MatchInRange)) {
      SDLoc DL(N);
      SDValue Sum = DAG.getNode(ISD::ADD, DL, ShiftVT, N1, N0.getOperand(1));
      return DAG.getNode(ISD::SHL, DL, VT, N0.getOperand(0), Sum);
    }
  }

  // fold (shl (ext (shl x, c1)), c2) -> (shl (ext x), (add c1, c2))
  // For this to be valid, the second form must not preserve any of the bits
  // that are shifted out by the inner shift in the first form.  This means
  // the outer shift size must be >= the number of bits added by the ext.
  // As a corollary, we don't care what kind of ext it is.
  if ((N0.getOpcode() == ISD::ZERO_EXTEND ||
       N0.getOpcode() == ISD::ANY_EXTEND ||
       N0.getOpcode() == ISD::SIGN_EXTEND) &&
      N0.getOperand(0).getOpcode() == ISD::SHL) {
    SDValue N0Op0 = N0.getOperand(0);
    SDValue InnerShiftAmt = N0Op0.getOperand(1);
    EVT InnerVT = N0Op0.getValueType();
    uint64_t InnerBitwidth = InnerVT.getScalarSizeInBits();

    auto MatchOutOfRange = [OpSizeInBits, InnerBitwidth](ConstantSDNode *LHS,
                                                         ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      return c2.uge(OpSizeInBits - InnerBitwidth) &&
             (c1 + c2).uge(OpSizeInBits);
    };
    if (ISD::matchBinaryPredicate(InnerShiftAmt, N1, MatchOutOfRange,
                                  /*AllowUndefs*/ false,
                                  /*AllowTypeMismatch*/ true))
      return DAG.getConstant(0, SDLoc(N), VT);

    auto MatchInRange = [OpSizeInBits, InnerBitwidth](ConstantSDNode *LHS,
                                                      ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      return c2.uge(OpSizeInBits - InnerBitwidth) &&
             (c1 + c2).ult(OpSizeInBits);
    };
    if (ISD::matchBinaryPredicate(InnerShiftAmt, N1, MatchInRange,
                                  /*AllowUndefs*/ false,
                                  /*AllowTypeMismatch*/ true)) {
      SDLoc DL(N);
      SDValue Ext = DAG.getNode(N0.getOpcode(), DL, VT, N0Op0.getOperand(0));
      SDValue Sum = DAG.getZExtOrTrunc(InnerShiftAmt, DL, ShiftVT);
      Sum = DAG.getNode(ISD::ADD, DL, ShiftVT, Sum, N1);
      return DAG.getNode(ISD::SHL, DL, VT, Ext, Sum);
    }
  }

  // fold (shl (zext (srl x, C)), C) -> (zext (shl (srl x, C), C))
  // Only fold this if the inner zext has no other uses to avoid increasing
  // the total number of instructions.
  if (N0.getOpcode() == ISD::ZERO_EXTEND && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::SRL) {
    SDValue N0Op0 = N0.getOperand(0);
    SDValue InnerShiftAmt = N0Op0.getOperand(1);

    auto MatchEqual = [VT](ConstantSDNode *LHS, ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2);
      return c1.ult(VT.getScalarSizeInBits()) && (c1 == c2);
    };
    if (ISD::matchBinaryPredicate(InnerShiftAmt, N1, MatchEqual,
                                  /*AllowUndefs*/ false,
                                  /*AllowTypeMismatch*/ true)) {
      SDLoc DL(N);
      EVT InnerShiftAmtVT = N0Op0.getOperand(1).getValueType();
      SDValue NewSHL = DAG.getZExtOrTrunc(N1, DL, InnerShiftAmtVT);
      NewSHL = DAG.getNode(ISD::SHL, DL, N0Op0.getValueType(), N0Op0, NewSHL);
      AddToWorklist(NewSHL.getNode());
      return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N0), VT, NewSHL);
    }
  }

  // fold (shl (sr[la] exact X,  C1), C2) -> (shl    X, (C2-C1)) if C1 <= C2
  // fold (shl (sr[la] exact X,  C1), C2) -> (sr[la] X, (C2-C1)) if C1  > C2
  // TODO - support non-uniform vector shift amounts.
  if (N1C && (N0.getOpcode() == ISD::SRL || N0.getOpcode() == ISD::SRA) &&
      N0->getFlags().hasExact()) {
    if (ConstantSDNode *N0C1 = isConstOrConstSplat(N0.getOperand(1))) {
      uint64_t C1 = N0C1->getZExtValue();
      uint64_t C2 = N1C->getZExtValue();
      SDLoc DL(N);
      if (C1 <= C2)
        return DAG.getNode(ISD::SHL, DL, VT, N0.getOperand(0),
                           DAG.getConstant(C2 - C1, DL, ShiftVT));
      return DAG.getNode(N0.getOpcode(), DL, VT, N0.getOperand(0),
                         DAG.getConstant(C1 - C2, DL, ShiftVT));
    }
  }

  // fold (shl (srl x, c1), c2) -> (and (shl x, (sub c2, c1), MASK) or
  //                               (and (srl x, (sub c1, c2), MASK)
  // Only fold this if the inner shift has no other uses -- if it does, folding
  // this will increase the total number of instructions.
  // TODO - drop hasOneUse requirement if c1 == c2?
  // TODO - support non-uniform vector shift amounts.
  if (N1C && N0.getOpcode() == ISD::SRL && N0.hasOneUse() &&
      TLI.shouldFoldConstantShiftPairToMask(N, Level)) {
    if (ConstantSDNode *N0C1 = isConstOrConstSplat(N0.getOperand(1))) {
      if (N0C1->getAPIntValue().ult(OpSizeInBits)) {
        uint64_t c1 = N0C1->getZExtValue();
        uint64_t c2 = N1C->getZExtValue();
        APInt Mask = APInt::getHighBitsSet(OpSizeInBits, OpSizeInBits - c1);
        SDValue Shift;
        if (c2 > c1) {
          Mask <<= c2 - c1;
          SDLoc DL(N);
          Shift = DAG.getNode(ISD::SHL, DL, VT, N0.getOperand(0),
                              DAG.getConstant(c2 - c1, DL, ShiftVT));
        } else {
          Mask.lshrInPlace(c1 - c2);
          SDLoc DL(N);
          Shift = DAG.getNode(ISD::SRL, DL, VT, N0.getOperand(0),
                              DAG.getConstant(c1 - c2, DL, ShiftVT));
        }
        SDLoc DL(N0);
        return DAG.getNode(ISD::AND, DL, VT, Shift,
                           DAG.getConstant(Mask, DL, VT));
      }
    }
  }

  // fold (shl (sra x, c1), c1) -> (and x, (shl -1, c1))
  if (N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1) &&
      isConstantOrConstantVector(N1, /* No Opaques */ true)) {
    SDLoc DL(N);
    SDValue AllBits = DAG.getAllOnesConstant(DL, VT);
    SDValue HiBitsMask = DAG.getNode(ISD::SHL, DL, VT, AllBits, N1);
    return DAG.getNode(ISD::AND, DL, VT, N0.getOperand(0), HiBitsMask);
  }

  // fold (shl (add x, c1), c2) -> (add (shl x, c2), c1 << c2)
  // fold (shl (or x, c1), c2) -> (or (shl x, c2), c1 << c2)
  // Variant of version done on multiply, except mul by a power of 2 is turned
  // into a shift.
  if ((N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::OR) &&
      N0.getNode()->hasOneUse() &&
      isConstantOrConstantVector(N1, /* No Opaques */ true) &&
      isConstantOrConstantVector(N0.getOperand(1), /* No Opaques */ true) &&
      TLI.isDesirableToCommuteWithShift(N, Level)) {
    SDValue Shl0 = DAG.getNode(ISD::SHL, SDLoc(N0), VT, N0.getOperand(0), N1);
    SDValue Shl1 = DAG.getNode(ISD::SHL, SDLoc(N1), VT, N0.getOperand(1), N1);
    AddToWorklist(Shl0.getNode());
    AddToWorklist(Shl1.getNode());
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, Shl0, Shl1);
  }

  // fold (shl (mul x, c1), c2) -> (mul x, c1 << c2)
  if (N0.getOpcode() == ISD::MUL && N0.getNode()->hasOneUse() &&
      isConstantOrConstantVector(N1, /* No Opaques */ true) &&
      isConstantOrConstantVector(N0.getOperand(1), /* No Opaques */ true)) {
    SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(N1), VT, N0.getOperand(1), N1);
    if (isConstantOrConstantVector(Shl))
      return DAG.getNode(ISD::MUL, SDLoc(N), VT, N0.getOperand(0), Shl);
  }

  if (N1C && !N1C->isOpaque())
    if (SDValue NewSHL = visitShiftByConstant(N))
      return NewSHL;

  // Fold (shl (vscale * C0), C1) to (vscale * (C0 << C1)).
  if (N0.getOpcode() == ISD::VSCALE)
    if (ConstantSDNode *NC1 = isConstOrConstSplat(N->getOperand(1))) {
      const APInt &C0 = N0.getConstantOperandAPInt(0);
      const APInt &C1 = NC1->getAPIntValue();
      return DAG.getVScale(SDLoc(N), VT, C0 << C1);
    }

  // Fold (shl step_vector(C0), C1) to (step_vector(C0 << C1)).
  APInt ShlVal;
  if (N0.getOpcode() == ISD::STEP_VECTOR)
    if (ISD::isConstantSplatVector(N1.getNode(), ShlVal)) {
      const APInt &C0 = N0.getConstantOperandAPInt(0);
      if (ShlVal.ult(C0.getBitWidth())) {
        APInt NewStep = C0 << ShlVal;
        return DAG.getStepVector(SDLoc(N), VT, NewStep);
      }
    }

  return SDValue();
}

// Transform a right shift of a multiply into a multiply-high.
// Examples:
// (srl (mul (zext i32:$a to i64), (zext i32:$a to i64)), 32) -> (mulhu $a, $b)
// (sra (mul (sext i32:$a to i64), (sext i32:$a to i64)), 32) -> (mulhs $a, $b)
static SDValue combineShiftToMULH(SDNode *N, SelectionDAG &DAG,
                                  const TargetLowering &TLI) {
  assert((N->getOpcode() == ISD::SRL || N->getOpcode() == ISD::SRA) &&
         "SRL or SRA node is required here!");

  // Check the shift amount. Proceed with the transformation if the shift
  // amount is constant.
  ConstantSDNode *ShiftAmtSrc = isConstOrConstSplat(N->getOperand(1));
  if (!ShiftAmtSrc)
    return SDValue();

  SDLoc DL(N);

  // The operation feeding into the shift must be a multiply.
  SDValue ShiftOperand = N->getOperand(0);
  if (ShiftOperand.getOpcode() != ISD::MUL)
    return SDValue();

  // Both operands must be equivalent extend nodes.
  SDValue LeftOp = ShiftOperand.getOperand(0);
  SDValue RightOp = ShiftOperand.getOperand(1);
  bool IsSignExt = LeftOp.getOpcode() == ISD::SIGN_EXTEND;
  bool IsZeroExt = LeftOp.getOpcode() == ISD::ZERO_EXTEND;

  if ((!(IsSignExt || IsZeroExt)) || LeftOp.getOpcode() != RightOp.getOpcode())
    return SDValue();

  EVT WideVT1 = LeftOp.getValueType();
  EVT WideVT2 = RightOp.getValueType();
  (void)WideVT2;
  // Proceed with the transformation if the wide types match.
  assert((WideVT1 == WideVT2) &&
         "Cannot have a multiply node with two different operand types.");

  EVT NarrowVT = LeftOp.getOperand(0).getValueType();
  // Check that the two extend nodes are the same type.
  if (NarrowVT !=  RightOp.getOperand(0).getValueType())
    return SDValue();

  // Proceed with the transformation if the wide type is twice as large
  // as the narrow type.
  unsigned NarrowVTSize = NarrowVT.getScalarSizeInBits();
  if (WideVT1.getScalarSizeInBits() != 2 * NarrowVTSize)
    return SDValue();

  // Check the shift amount with the narrow type size.
  // Proceed with the transformation if the shift amount is the width
  // of the narrow type.
  unsigned ShiftAmt = ShiftAmtSrc->getZExtValue();
  if (ShiftAmt != NarrowVTSize)
    return SDValue();

  // If the operation feeding into the MUL is a sign extend (sext),
  // we use mulhs. Othewise, zero extends (zext) use mulhu.
  unsigned MulhOpcode = IsSignExt ? ISD::MULHS : ISD::MULHU;

  // Combine to mulh if mulh is legal/custom for the narrow type on the target.
  if (!TLI.isOperationLegalOrCustom(MulhOpcode, NarrowVT))
    return SDValue();

  SDValue Result = DAG.getNode(MulhOpcode, DL, NarrowVT, LeftOp.getOperand(0),
                               RightOp.getOperand(0));
  return (N->getOpcode() == ISD::SRA ? DAG.getSExtOrTrunc(Result, DL, WideVT1)
                                     : DAG.getZExtOrTrunc(Result, DL, WideVT1));
}

SDValue DAGCombiner::visitSRA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (SDValue V = DAG.simplifyShift(N0, N1))
    return V;

  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarSizeInBits();

  // Arithmetic shifting an all-sign-bit value is a no-op.
  // fold (sra 0, x) -> 0
  // fold (sra -1, x) -> -1
  if (DAG.ComputeNumSignBits(N0) == OpSizeInBits)
    return N0;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  ConstantSDNode *N1C = isConstOrConstSplat(N1);

  // fold (sra c1, c2) -> (sra c1, c2)
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::SRA, SDLoc(N), VT, {N0, N1}))
    return C;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // fold (sra (shl x, c1), c1) -> sext_inreg for some c1 and target supports
  // sext_inreg.
  if (N1C && N0.getOpcode() == ISD::SHL && N1 == N0.getOperand(1)) {
    unsigned LowBits = OpSizeInBits - (unsigned)N1C->getZExtValue();
    EVT ExtVT = EVT::getIntegerVT(*DAG.getContext(), LowBits);
    if (VT.isVector())
      ExtVT = EVT::getVectorVT(*DAG.getContext(), ExtVT,
                               VT.getVectorElementCount());
    if (!LegalOperations ||
        TLI.getOperationAction(ISD::SIGN_EXTEND_INREG, ExtVT) ==
        TargetLowering::Legal)
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT,
                         N0.getOperand(0), DAG.getValueType(ExtVT));
    // Even if we can't convert to sext_inreg, we might be able to remove
    // this shift pair if the input is already sign extended.
    if (DAG.ComputeNumSignBits(N0.getOperand(0)) > N1C->getZExtValue())
      return N0.getOperand(0);
  }

  // fold (sra (sra x, c1), c2) -> (sra x, (add c1, c2))
  // clamp (add c1, c2) to max shift.
  if (N0.getOpcode() == ISD::SRA) {
    SDLoc DL(N);
    EVT ShiftVT = N1.getValueType();
    EVT ShiftSVT = ShiftVT.getScalarType();
    SmallVector<SDValue, 16> ShiftValues;

    auto SumOfShifts = [&](ConstantSDNode *LHS, ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      APInt Sum = c1 + c2;
      unsigned ShiftSum =
          Sum.uge(OpSizeInBits) ? (OpSizeInBits - 1) : Sum.getZExtValue();
      ShiftValues.push_back(DAG.getConstant(ShiftSum, DL, ShiftSVT));
      return true;
    };
    if (ISD::matchBinaryPredicate(N1, N0.getOperand(1), SumOfShifts)) {
      SDValue ShiftValue;
      if (N1.getOpcode() == ISD::BUILD_VECTOR)
        ShiftValue = DAG.getBuildVector(ShiftVT, DL, ShiftValues);
      else if (N1.getOpcode() == ISD::SPLAT_VECTOR) {
        assert(ShiftValues.size() == 1 &&
               "Expected matchBinaryPredicate to return one element for "
               "SPLAT_VECTORs");
        ShiftValue = DAG.getSplatVector(ShiftVT, DL, ShiftValues[0]);
      } else
        ShiftValue = ShiftValues[0];
      return DAG.getNode(ISD::SRA, DL, VT, N0.getOperand(0), ShiftValue);
    }
  }

  // fold (sra (shl X, m), (sub result_size, n))
  // -> (sign_extend (trunc (shl X, (sub (sub result_size, n), m)))) for
  // result_size - n != m.
  // If truncate is free for the target sext(shl) is likely to result in better
  // code.
  if (N0.getOpcode() == ISD::SHL && N1C) {
    // Get the two constanst of the shifts, CN0 = m, CN = n.
    const ConstantSDNode *N01C = isConstOrConstSplat(N0.getOperand(1));
    if (N01C) {
      LLVMContext &Ctx = *DAG.getContext();
      // Determine what the truncate's result bitsize and type would be.
      EVT TruncVT = EVT::getIntegerVT(Ctx, OpSizeInBits - N1C->getZExtValue());

      if (VT.isVector())
        TruncVT = EVT::getVectorVT(Ctx, TruncVT, VT.getVectorElementCount());

      // Determine the residual right-shift amount.
      int ShiftAmt = N1C->getZExtValue() - N01C->getZExtValue();

      // If the shift is not a no-op (in which case this should be just a sign
      // extend already), the truncated to type is legal, sign_extend is legal
      // on that type, and the truncate to that type is both legal and free,
      // perform the transform.
      if ((ShiftAmt > 0) &&
          TLI.isOperationLegalOrCustom(ISD::SIGN_EXTEND, TruncVT) &&
          TLI.isOperationLegalOrCustom(ISD::TRUNCATE, VT) &&
          TLI.isTruncateFree(VT, TruncVT)) {
        SDLoc DL(N);
        SDValue Amt = DAG.getConstant(ShiftAmt, DL,
            getShiftAmountTy(N0.getOperand(0).getValueType()));
        SDValue Shift = DAG.getNode(ISD::SRL, DL, VT,
                                    N0.getOperand(0), Amt);
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE, DL, TruncVT,
                                    Shift);
        return DAG.getNode(ISD::SIGN_EXTEND, DL,
                           N->getValueType(0), Trunc);
      }
    }
  }

  // We convert trunc/ext to opposing shifts in IR, but casts may be cheaper.
  //   sra (add (shl X, N1C), AddC), N1C -->
  //   sext (add (trunc X to (width - N1C)), AddC')
  if (N0.getOpcode() == ISD::ADD && N0.hasOneUse() && N1C &&
      N0.getOperand(0).getOpcode() == ISD::SHL &&
      N0.getOperand(0).getOperand(1) == N1 && N0.getOperand(0).hasOneUse()) {
    if (ConstantSDNode *AddC = isConstOrConstSplat(N0.getOperand(1))) {
      SDValue Shl = N0.getOperand(0);
      // Determine what the truncate's type would be and ask the target if that
      // is a free operation.
      LLVMContext &Ctx = *DAG.getContext();
      unsigned ShiftAmt = N1C->getZExtValue();
      EVT TruncVT = EVT::getIntegerVT(Ctx, OpSizeInBits - ShiftAmt);
      if (VT.isVector())
        TruncVT = EVT::getVectorVT(Ctx, TruncVT, VT.getVectorElementCount());

      // TODO: The simple type check probably belongs in the default hook
      //       implementation and/or target-specific overrides (because
      //       non-simple types likely require masking when legalized), but that
      //       restriction may conflict with other transforms.
      if (TruncVT.isSimple() && isTypeLegal(TruncVT) &&
          TLI.isTruncateFree(VT, TruncVT)) {
        SDLoc DL(N);
        SDValue Trunc = DAG.getZExtOrTrunc(Shl.getOperand(0), DL, TruncVT);
        SDValue ShiftC = DAG.getConstant(AddC->getAPIntValue().lshr(ShiftAmt).
                             trunc(TruncVT.getScalarSizeInBits()), DL, TruncVT);
        SDValue Add = DAG.getNode(ISD::ADD, DL, TruncVT, Trunc, ShiftC);
        return DAG.getSExtOrTrunc(Add, DL, VT);
      }
    }
  }

  // fold (sra x, (trunc (and y, c))) -> (sra x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    if (SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode()))
      return DAG.getNode(ISD::SRA, SDLoc(N), VT, N0, NewOp1);
  }

  // fold (sra (trunc (sra x, c1)), c2) -> (trunc (sra x, c1 + c2))
  // fold (sra (trunc (srl x, c1)), c2) -> (trunc (sra x, c1 + c2))
  //      if c1 is equal to the number of bits the trunc removes
  // TODO - support non-uniform vector shift amounts.
  if (N0.getOpcode() == ISD::TRUNCATE &&
      (N0.getOperand(0).getOpcode() == ISD::SRL ||
       N0.getOperand(0).getOpcode() == ISD::SRA) &&
      N0.getOperand(0).hasOneUse() &&
      N0.getOperand(0).getOperand(1).hasOneUse() && N1C) {
    SDValue N0Op0 = N0.getOperand(0);
    if (ConstantSDNode *LargeShift = isConstOrConstSplat(N0Op0.getOperand(1))) {
      EVT LargeVT = N0Op0.getValueType();
      unsigned TruncBits = LargeVT.getScalarSizeInBits() - OpSizeInBits;
      if (LargeShift->getAPIntValue() == TruncBits) {
        SDLoc DL(N);
        SDValue Amt = DAG.getConstant(N1C->getZExtValue() + TruncBits, DL,
                                      getShiftAmountTy(LargeVT));
        SDValue SRA =
            DAG.getNode(ISD::SRA, DL, LargeVT, N0Op0.getOperand(0), Amt);
        return DAG.getNode(ISD::TRUNCATE, DL, VT, SRA);
      }
    }
  }

  // Simplify, based on bits shifted out of the LHS.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // If the sign bit is known to be zero, switch this to a SRL.
  if (DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0, N1);

  if (N1C && !N1C->isOpaque())
    if (SDValue NewSRA = visitShiftByConstant(N))
      return NewSRA;

  // Try to transform this shift into a multiply-high if
  // it matches the appropriate pattern detected in combineShiftToMULH.
  if (SDValue MULH = combineShiftToMULH(N, DAG, TLI))
    return MULH;

  return SDValue();
}

SDValue DAGCombiner::visitSRL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  if (SDValue V = DAG.simplifyShift(N0, N1))
    return V;

  EVT VT = N0.getValueType();
  unsigned OpSizeInBits = VT.getScalarSizeInBits();

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  ConstantSDNode *N1C = isConstOrConstSplat(N1);

  // fold (srl c1, c2) -> c1 >>u c2
  if (SDValue C = DAG.FoldConstantArithmetic(ISD::SRL, SDLoc(N), VT, {N0, N1}))
    return C;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // if (srl x, c) is known to be zero, return 0
  if (N1C && DAG.MaskedValueIsZero(SDValue(N, 0),
                                   APInt::getAllOnesValue(OpSizeInBits)))
    return DAG.getConstant(0, SDLoc(N), VT);

  // fold (srl (srl x, c1), c2) -> 0 or (srl x, (add c1, c2))
  if (N0.getOpcode() == ISD::SRL) {
    auto MatchOutOfRange = [OpSizeInBits](ConstantSDNode *LHS,
                                          ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      return (c1 + c2).uge(OpSizeInBits);
    };
    if (ISD::matchBinaryPredicate(N1, N0.getOperand(1), MatchOutOfRange))
      return DAG.getConstant(0, SDLoc(N), VT);

    auto MatchInRange = [OpSizeInBits](ConstantSDNode *LHS,
                                       ConstantSDNode *RHS) {
      APInt c1 = LHS->getAPIntValue();
      APInt c2 = RHS->getAPIntValue();
      zeroExtendToMatch(c1, c2, 1 /* Overflow Bit */);
      return (c1 + c2).ult(OpSizeInBits);
    };
    if (ISD::matchBinaryPredicate(N1, N0.getOperand(1), MatchInRange)) {
      SDLoc DL(N);
      EVT ShiftVT = N1.getValueType();
      SDValue Sum = DAG.getNode(ISD::ADD, DL, ShiftVT, N1, N0.getOperand(1));
      return DAG.getNode(ISD::SRL, DL, VT, N0.getOperand(0), Sum);
    }
  }

  if (N1C && N0.getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(0).getOpcode() == ISD::SRL) {
    SDValue InnerShift = N0.getOperand(0);
    // TODO - support non-uniform vector shift amounts.
    if (auto *N001C = isConstOrConstSplat(InnerShift.getOperand(1))) {
      uint64_t c1 = N001C->getZExtValue();
      uint64_t c2 = N1C->getZExtValue();
      EVT InnerShiftVT = InnerShift.getValueType();
      EVT ShiftAmtVT = InnerShift.getOperand(1).getValueType();
      uint64_t InnerShiftSize = InnerShiftVT.getScalarSizeInBits();
      // srl (trunc (srl x, c1)), c2 --> 0 or (trunc (srl x, (add c1, c2)))
      // This is only valid if the OpSizeInBits + c1 = size of inner shift.
      if (c1 + OpSizeInBits == InnerShiftSize) {
        SDLoc DL(N);
        if (c1 + c2 >= InnerShiftSize)
          return DAG.getConstant(0, DL, VT);
        SDValue NewShiftAmt = DAG.getConstant(c1 + c2, DL, ShiftAmtVT);
        SDValue NewShift = DAG.getNode(ISD::SRL, DL, InnerShiftVT,
                                       InnerShift.getOperand(0), NewShiftAmt);
        return DAG.getNode(ISD::TRUNCATE, DL, VT, NewShift);
      }
      // In the more general case, we can clear the high bits after the shift:
      // srl (trunc (srl x, c1)), c2 --> trunc (and (srl x, (c1+c2)), Mask)
      if (N0.hasOneUse() && InnerShift.hasOneUse() &&
          c1 + c2 < InnerShiftSize) {
        SDLoc DL(N);
        SDValue NewShiftAmt = DAG.getConstant(c1 + c2, DL, ShiftAmtVT);
        SDValue NewShift = DAG.getNode(ISD::SRL, DL, InnerShiftVT,
                                       InnerShift.getOperand(0), NewShiftAmt);
        SDValue Mask = DAG.getConstant(APInt::getLowBitsSet(InnerShiftSize,
                                                            OpSizeInBits - c2),
                                       DL, InnerShiftVT);
        SDValue And = DAG.getNode(ISD::AND, DL, InnerShiftVT, NewShift, Mask);
        return DAG.getNode(ISD::TRUNCATE, DL, VT, And);
      }
    }
  }

  // fold (srl (shl x, c), c) -> (and x, cst2)
  // TODO - (srl (shl x, c1), c2).
  if (N0.getOpcode() == ISD::SHL && N0.getOperand(1) == N1 &&
      isConstantOrConstantVector(N1, /* NoOpaques */ true)) {
    SDLoc DL(N);
    SDValue Mask =
        DAG.getNode(ISD::SRL, DL, VT, DAG.getAllOnesConstant(DL, VT), N1);
    AddToWorklist(Mask.getNode());
    return DAG.getNode(ISD::AND, DL, VT, N0.getOperand(0), Mask);
  }

  // fold (srl (anyextend x), c) -> (and (anyextend (srl x, c)), mask)
  // TODO - support non-uniform vector shift amounts.
  if (N1C && N0.getOpcode() == ISD::ANY_EXTEND) {
    // Shifting in all undef bits?
    EVT SmallVT = N0.getOperand(0).getValueType();
    unsigned BitSize = SmallVT.getScalarSizeInBits();
    if (N1C->getAPIntValue().uge(BitSize))
      return DAG.getUNDEF(VT);

    if (!LegalTypes || TLI.isTypeDesirableForOp(ISD::SRL, SmallVT)) {
      uint64_t ShiftAmt = N1C->getZExtValue();
      SDLoc DL0(N0);
      SDValue SmallShift = DAG.getNode(ISD::SRL, DL0, SmallVT,
                                       N0.getOperand(0),
                          DAG.getConstant(ShiftAmt, DL0,
                                          getShiftAmountTy(SmallVT)));
      AddToWorklist(SmallShift.getNode());
      APInt Mask = APInt::getLowBitsSet(OpSizeInBits, OpSizeInBits - ShiftAmt);
      SDLoc DL(N);
      return DAG.getNode(ISD::AND, DL, VT,
                         DAG.getNode(ISD::ANY_EXTEND, DL, VT, SmallShift),
                         DAG.getConstant(Mask, DL, VT));
    }
  }

  // fold (srl (sra X, Y), 31) -> (srl X, 31).  This srl only looks at the sign
  // bit, which is unmodified by sra.
  if (N1C && N1C->getAPIntValue() == (OpSizeInBits - 1)) {
    if (N0.getOpcode() == ISD::SRA)
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0.getOperand(0), N1);
  }

  // fold (srl (ctlz x), "5") -> x  iff x has one bit set (the low bit).
  if (N1C && N0.getOpcode() == ISD::CTLZ &&
      N1C->getAPIntValue() == Log2_32(OpSizeInBits)) {
    KnownBits Known = DAG.computeKnownBits(N0.getOperand(0));

    // If any of the input bits are KnownOne, then the input couldn't be all
    // zeros, thus the result of the srl will always be zero.
    if (Known.One.getBoolValue()) return DAG.getConstant(0, SDLoc(N0), VT);

    // If all of the bits input the to ctlz node are known to be zero, then
    // the result of the ctlz is "32" and the result of the shift is one.
    APInt UnknownBits = ~Known.Zero;
    if (UnknownBits == 0) return DAG.getConstant(1, SDLoc(N0), VT);

    // Otherwise, check to see if there is exactly one bit input to the ctlz.
    if (UnknownBits.isPowerOf2()) {
      // Okay, we know that only that the single bit specified by UnknownBits
      // could be set on input to the CTLZ node. If this bit is set, the SRL
      // will return 0, if it is clear, it returns 1. Change the CTLZ/SRL pair
      // to an SRL/XOR pair, which is likely to simplify more.
      unsigned ShAmt = UnknownBits.countTrailingZeros();
      SDValue Op = N0.getOperand(0);

      if (ShAmt) {
        SDLoc DL(N0);
        Op = DAG.getNode(ISD::SRL, DL, VT, Op,
                  DAG.getConstant(ShAmt, DL,
                                  getShiftAmountTy(Op.getValueType())));
        AddToWorklist(Op.getNode());
      }

      SDLoc DL(N);
      return DAG.getNode(ISD::XOR, DL, VT,
                         Op, DAG.getConstant(1, DL, VT));
    }
  }

  // fold (srl x, (trunc (and y, c))) -> (srl x, (and (trunc y), (trunc c))).
  if (N1.getOpcode() == ISD::TRUNCATE &&
      N1.getOperand(0).getOpcode() == ISD::AND) {
    if (SDValue NewOp1 = distributeTruncateThroughAnd(N1.getNode()))
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N0, NewOp1);
  }

  // fold operands of srl based on knowledge that the low bits are not
  // demanded.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  if (N1C && !N1C->isOpaque())
    if (SDValue NewSRL = visitShiftByConstant(N))
      return NewSRL;

  // Attempt to convert a srl of a load into a narrower zero-extending load.
  if (SDValue NarrowLoad = ReduceLoadWidth(N))
    return NarrowLoad;

  // Here is a common situation. We want to optimize:
  //
  //   %a = ...
  //   %b = and i32 %a, 2
  //   %c = srl i32 %b, 1
  //   brcond i32 %c ...
  //
  // into
  //
  //   %a = ...
  //   %b = and %a, 2
  //   %c = setcc eq %b, 0
  //   brcond %c ...
  //
  // However when after the source operand of SRL is optimized into AND, the SRL
  // itself may not be optimized further. Look for it and add the BRCOND into
  // the worklist.
  if (N->hasOneUse()) {
    SDNode *Use = *N->use_begin();
    if (Use->getOpcode() == ISD::BRCOND)
      AddToWorklist(Use);
    else if (Use->getOpcode() == ISD::TRUNCATE && Use->hasOneUse()) {
      // Also look pass the truncate.
      Use = *Use->use_begin();
      if (Use->getOpcode() == ISD::BRCOND)
        AddToWorklist(Use);
    }
  }

  // Try to transform this shift into a multiply-high if
  // it matches the appropriate pattern detected in combineShiftToMULH.
  if (SDValue MULH = combineShiftToMULH(N, DAG, TLI))
    return MULH;

  return SDValue();
}

SDValue DAGCombiner::visitFunnelShift(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  bool IsFSHL = N->getOpcode() == ISD::FSHL;
  unsigned BitWidth = VT.getScalarSizeInBits();

  // fold (fshl N0, N1, 0) -> N0
  // fold (fshr N0, N1, 0) -> N1
  if (isPowerOf2_32(BitWidth))
    if (DAG.MaskedValueIsZero(
            N2, APInt(N2.getScalarValueSizeInBits(), BitWidth - 1)))
      return IsFSHL ? N0 : N1;

  auto IsUndefOrZero = [](SDValue V) {
    return V.isUndef() || isNullOrNullSplat(V, /*AllowUndefs*/ true);
  };

  // TODO - support non-uniform vector shift amounts.
  if (ConstantSDNode *Cst = isConstOrConstSplat(N2)) {
    EVT ShAmtTy = N2.getValueType();

    // fold (fsh* N0, N1, c) -> (fsh* N0, N1, c % BitWidth)
    if (Cst->getAPIntValue().uge(BitWidth)) {
      uint64_t RotAmt = Cst->getAPIntValue().urem(BitWidth);
      return DAG.getNode(N->getOpcode(), SDLoc(N), VT, N0, N1,
                         DAG.getConstant(RotAmt, SDLoc(N), ShAmtTy));
    }

    unsigned ShAmt = Cst->getZExtValue();
    if (ShAmt == 0)
      return IsFSHL ? N0 : N1;

    // fold fshl(undef_or_zero, N1, C) -> lshr(N1, BW-C)
    // fold fshr(undef_or_zero, N1, C) -> lshr(N1, C)
    // fold fshl(N0, undef_or_zero, C) -> shl(N0, C)
    // fold fshr(N0, undef_or_zero, C) -> shl(N0, BW-C)
    if (IsUndefOrZero(N0))
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N1,
                         DAG.getConstant(IsFSHL ? BitWidth - ShAmt : ShAmt,
                                         SDLoc(N), ShAmtTy));
    if (IsUndefOrZero(N1))
      return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0,
                         DAG.getConstant(IsFSHL ? ShAmt : BitWidth - ShAmt,
                                         SDLoc(N), ShAmtTy));

    // fold (fshl ld1, ld0, c) -> (ld0[ofs]) iff ld0 and ld1 are consecutive.
    // fold (fshr ld1, ld0, c) -> (ld0[ofs]) iff ld0 and ld1 are consecutive.
    // TODO - bigendian support once we have test coverage.
    // TODO - can we merge this with CombineConseutiveLoads/MatchLoadCombine?
    // TODO - permit LHS EXTLOAD if extensions are shifted out.
    if ((BitWidth % 8) == 0 && (ShAmt % 8) == 0 && !VT.isVector() &&
        !DAG.getDataLayout().isBigEndian()) {
      auto *LHS = dyn_cast<LoadSDNode>(N0);
      auto *RHS = dyn_cast<LoadSDNode>(N1);
      if (LHS && RHS && LHS->isSimple() && RHS->isSimple() &&
          LHS->getAddressSpace() == RHS->getAddressSpace() &&
          (LHS->hasOneUse() || RHS->hasOneUse()) && ISD::isNON_EXTLoad(RHS) &&
          ISD::isNON_EXTLoad(LHS)) {
        if (DAG.areNonVolatileConsecutiveLoads(LHS, RHS, BitWidth / 8, 1)) {
          SDLoc DL(RHS);
          uint64_t PtrOff =
              IsFSHL ? (((BitWidth - ShAmt) % BitWidth) / 8) : (ShAmt / 8);
          Align NewAlign = commonAlignment(RHS->getAlign(), PtrOff);
          bool Fast = false;
          if (TLI.allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), VT,
                                     RHS->getAddressSpace(), NewAlign,
                                     RHS->getMemOperand()->getFlags(), &Fast) &&
              Fast) {
            SDValue NewPtr = DAG.getMemBasePlusOffset(
                RHS->getBasePtr(), TypeSize::Fixed(PtrOff), DL);
            AddToWorklist(NewPtr.getNode());
            SDValue Load = DAG.getLoad(
                VT, DL, RHS->getChain(), NewPtr,
                RHS->getPointerInfo().getWithOffset(PtrOff), NewAlign,
                RHS->getMemOperand()->getFlags(), RHS->getAAInfo());
            // Replace the old load's chain with the new load's chain.
            WorklistRemover DeadNodes(*this);
            DAG.ReplaceAllUsesOfValueWith(N1.getValue(1), Load.getValue(1));
            return Load;
          }
        }
      }
    }
  }

  // fold fshr(undef_or_zero, N1, N2) -> lshr(N1, N2)
  // fold fshl(N0, undef_or_zero, N2) -> shl(N0, N2)
  // iff We know the shift amount is in range.
  // TODO: when is it worth doing SUB(BW, N2) as well?
  if (isPowerOf2_32(BitWidth)) {
    APInt ModuloBits(N2.getScalarValueSizeInBits(), BitWidth - 1);
    if (IsUndefOrZero(N0) && !IsFSHL && DAG.MaskedValueIsZero(N2, ~ModuloBits))
      return DAG.getNode(ISD::SRL, SDLoc(N), VT, N1, N2);
    if (IsUndefOrZero(N1) && IsFSHL && DAG.MaskedValueIsZero(N2, ~ModuloBits))
      return DAG.getNode(ISD::SHL, SDLoc(N), VT, N0, N2);
  }

  // fold (fshl N0, N0, N2) -> (rotl N0, N2)
  // fold (fshr N0, N0, N2) -> (rotr N0, N2)
  // TODO: Investigate flipping this rotate if only one is legal, if funnel shift
  // is legal as well we might be better off avoiding non-constant (BW - N2).
  unsigned RotOpc = IsFSHL ? ISD::ROTL : ISD::ROTR;
  if (N0 == N1 && hasOperation(RotOpc, VT))
    return DAG.getNode(RotOpc, SDLoc(N), VT, N0, N2);

  // Simplify, based on bits shifted out of N0/N1.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

// Given a ABS node, detect the following pattern:
// (ABS (SUB (EXTEND a), (EXTEND b))).
// Generates UABD/SABD instruction.
static SDValue combineABSToABD(SDNode *N, SelectionDAG &DAG,
                               const TargetLowering &TLI) {
  SDValue AbsOp1 = N->getOperand(0);
  SDValue Op0, Op1;

  if (AbsOp1.getOpcode() != ISD::SUB)
    return SDValue();

  Op0 = AbsOp1.getOperand(0);
  Op1 = AbsOp1.getOperand(1);

  unsigned Opc0 = Op0.getOpcode();
  // Check if the operands of the sub are (zero|sign)-extended.
  if (Opc0 != Op1.getOpcode() ||
      (Opc0 != ISD::ZERO_EXTEND && Opc0 != ISD::SIGN_EXTEND))
    return SDValue();

  EVT VT1 = Op0.getOperand(0).getValueType();
  EVT VT2 = Op1.getOperand(0).getValueType();
  // Check if the operands are of same type and valid size.
  unsigned ABDOpcode = (Opc0 == ISD::SIGN_EXTEND) ? ISD::ABDS : ISD::ABDU;
  if (VT1 != VT2 || !TLI.isOperationLegalOrCustom(ABDOpcode, VT1))
    return SDValue();

  Op0 = Op0.getOperand(0);
  Op1 = Op1.getOperand(0);
  SDValue ABD =
      DAG.getNode(ABDOpcode, SDLoc(N), Op0->getValueType(0), Op0, Op1);
  return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), N->getValueType(0), ABD);
}

SDValue DAGCombiner::visitABS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (abs c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::ABS, SDLoc(N), VT, N0);
  // fold (abs (abs x)) -> (abs x)
  if (N0.getOpcode() == ISD::ABS)
    return N0;
  // fold (abs x) -> x iff not-negative
  if (DAG.SignBitIsZero(N0))
    return N0;

  if (SDValue ABD = combineABSToABD(N, DAG, TLI))
    return ABD;

  return SDValue();
}

SDValue DAGCombiner::visitBSWAP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (bswap c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::BSWAP, SDLoc(N), VT, N0);
  // fold (bswap (bswap x)) -> x
  if (N0.getOpcode() == ISD::BSWAP)
    return N0->getOperand(0);
  return SDValue();
}

SDValue DAGCombiner::visitBITREVERSE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (bitreverse c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::BITREVERSE, SDLoc(N), VT, N0);
  // fold (bitreverse (bitreverse x)) -> x
  if (N0.getOpcode() == ISD::BITREVERSE)
    return N0.getOperand(0);
  return SDValue();
}

SDValue DAGCombiner::visitCTLZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctlz c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTLZ, SDLoc(N), VT, N0);

  // If the value is known never to be zero, switch to the undef version.
  if (!LegalOperations || TLI.isOperationLegal(ISD::CTLZ_ZERO_UNDEF, VT)) {
    if (DAG.isKnownNeverZero(N0))
      return DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitCTLZ_ZERO_UNDEF(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctlz_zero_undef c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTTZ(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTTZ, SDLoc(N), VT, N0);

  // If the value is known never to be zero, switch to the undef version.
  if (!LegalOperations || TLI.isOperationLegal(ISD::CTTZ_ZERO_UNDEF, VT)) {
    if (DAG.isKnownNeverZero(N0))
      return DAG.getNode(ISD::CTTZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  }

  return SDValue();
}

SDValue DAGCombiner::visitCTTZ_ZERO_UNDEF(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (cttz_zero_undef c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTTZ_ZERO_UNDEF, SDLoc(N), VT, N0);
  return SDValue();
}

SDValue DAGCombiner::visitCTPOP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ctpop c1) -> c2
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::CTPOP, SDLoc(N), VT, N0);
  return SDValue();
}

// FIXME: This should be checking for no signed zeros on individual operands, as
// well as no nans.
static bool isLegalToCombineMinNumMaxNum(SelectionDAG &DAG, SDValue LHS,
                                         SDValue RHS,
                                         const TargetLowering &TLI) {
  const TargetOptions &Options = DAG.getTarget().Options;
  EVT VT = LHS.getValueType();

  return Options.NoSignedZerosFPMath && VT.isFloatingPoint() &&
         TLI.isProfitableToCombineMinNumMaxNum(VT) &&
         DAG.isKnownNeverNaN(LHS) && DAG.isKnownNeverNaN(RHS);
}

/// Generate Min/Max node
static SDValue combineMinNumMaxNum(const SDLoc &DL, EVT VT, SDValue LHS,
                                   SDValue RHS, SDValue True, SDValue False,
                                   ISD::CondCode CC, const TargetLowering &TLI,
                                   SelectionDAG &DAG) {
  if (!(LHS == True && RHS == False) && !(LHS == False && RHS == True))
    return SDValue();

  EVT TransformVT = TLI.getTypeToTransformTo(*DAG.getContext(), VT);
  switch (CC) {
  case ISD::SETOLT:
  case ISD::SETOLE:
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETULT:
  case ISD::SETULE: {
    // Since it's known never nan to get here already, either fminnum or
    // fminnum_ieee are OK. Try the ieee version first, since it's fminnum is
    // expanded in terms of it.
    unsigned IEEEOpcode = (LHS == True) ? ISD::FMINNUM_IEEE : ISD::FMAXNUM_IEEE;
    if (TLI.isOperationLegalOrCustom(IEEEOpcode, VT))
      return DAG.getNode(IEEEOpcode, DL, VT, LHS, RHS);

    unsigned Opcode = (LHS == True) ? ISD::FMINNUM : ISD::FMAXNUM;
    if (TLI.isOperationLegalOrCustom(Opcode, TransformVT))
      return DAG.getNode(Opcode, DL, VT, LHS, RHS);
    return SDValue();
  }
  case ISD::SETOGT:
  case ISD::SETOGE:
  case ISD::SETGT:
  case ISD::SETGE:
  case ISD::SETUGT:
  case ISD::SETUGE: {
    unsigned IEEEOpcode = (LHS == True) ? ISD::FMAXNUM_IEEE : ISD::FMINNUM_IEEE;
    if (TLI.isOperationLegalOrCustom(IEEEOpcode, VT))
      return DAG.getNode(IEEEOpcode, DL, VT, LHS, RHS);

    unsigned Opcode = (LHS == True) ? ISD::FMAXNUM : ISD::FMINNUM;
    if (TLI.isOperationLegalOrCustom(Opcode, TransformVT))
      return DAG.getNode(Opcode, DL, VT, LHS, RHS);
    return SDValue();
  }
  default:
    return SDValue();
  }
}

/// If a (v)select has a condition value that is a sign-bit test, try to smear
/// the condition operand sign-bit across the value width and use it as a mask.
static SDValue foldSelectOfConstantsUsingSra(SDNode *N, SelectionDAG &DAG) {
  SDValue Cond = N->getOperand(0);
  SDValue C1 = N->getOperand(1);
  SDValue C2 = N->getOperand(2);
  if (!isConstantOrConstantVector(C1) || !isConstantOrConstantVector(C2))
    return SDValue();

  EVT VT = N->getValueType(0);
  if (Cond.getOpcode() != ISD::SETCC || !Cond.hasOneUse() ||
      VT != Cond.getOperand(0).getValueType())
    return SDValue();

  // The inverted-condition + commuted-select variants of these patterns are
  // canonicalized to these forms in IR.
  SDValue X = Cond.getOperand(0);
  SDValue CondC = Cond.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
  if (CC == ISD::SETGT && isAllOnesOrAllOnesSplat(CondC) &&
      isAllOnesOrAllOnesSplat(C2)) {
    // i32 X > -1 ? C1 : -1 --> (X >>s 31) | C1
    SDLoc DL(N);
    SDValue ShAmtC = DAG.getConstant(X.getScalarValueSizeInBits() - 1, DL, VT);
    SDValue Sra = DAG.getNode(ISD::SRA, DL, VT, X, ShAmtC);
    return DAG.getNode(ISD::OR, DL, VT, Sra, C1);
  }
  if (CC == ISD::SETLT && isNullOrNullSplat(CondC) && isNullOrNullSplat(C2)) {
    // i8 X < 0 ? C1 : 0 --> (X >>s 7) & C1
    SDLoc DL(N);
    SDValue ShAmtC = DAG.getConstant(X.getScalarValueSizeInBits() - 1, DL, VT);
    SDValue Sra = DAG.getNode(ISD::SRA, DL, VT, X, ShAmtC);
    return DAG.getNode(ISD::AND, DL, VT, Sra, C1);
  }
  return SDValue();
}

SDValue DAGCombiner::foldSelectOfConstants(SDNode *N) {
  SDValue Cond = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  EVT CondVT = Cond.getValueType();
  SDLoc DL(N);

  if (!VT.isInteger())
    return SDValue();

  auto *C1 = dyn_cast<ConstantSDNode>(N1);
  auto *C2 = dyn_cast<ConstantSDNode>(N2);
  if (!C1 || !C2)
    return SDValue();

  // Only do this before legalization to avoid conflicting with target-specific
  // transforms in the other direction (create a select from a zext/sext). There
  // is also a target-independent combine here in DAGCombiner in the other
  // direction for (select Cond, -1, 0) when the condition is not i1.
  if (CondVT == MVT::i1 && !LegalOperations) {
    if (C1->isNullValue() && C2->isOne()) {
      // select Cond, 0, 1 --> zext (!Cond)
      SDValue NotCond = DAG.getNOT(DL, Cond, MVT::i1);
      if (VT != MVT::i1)
        NotCond = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, NotCond);
      return NotCond;
    }
    if (C1->isNullValue() && C2->isAllOnesValue()) {
      // select Cond, 0, -1 --> sext (!Cond)
      SDValue NotCond = DAG.getNOT(DL, Cond, MVT::i1);
      if (VT != MVT::i1)
        NotCond = DAG.getNode(ISD::SIGN_EXTEND, DL, VT, NotCond);
      return NotCond;
    }
    if (C1->isOne() && C2->isNullValue()) {
      // select Cond, 1, 0 --> zext (Cond)
      if (VT != MVT::i1)
        Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Cond);
      return Cond;
    }
    if (C1->isAllOnesValue() && C2->isNullValue()) {
      // select Cond, -1, 0 --> sext (Cond)
      if (VT != MVT::i1)
        Cond = DAG.getNode(ISD::SIGN_EXTEND, DL, VT, Cond);
      return Cond;
    }

    // Use a target hook because some targets may prefer to transform in the
    // other direction.
    if (TLI.convertSelectOfConstantsToMath(VT)) {
      // For any constants that differ by 1, we can transform the select into an
      // extend and add.
      const APInt &C1Val = C1->getAPIntValue();
      const APInt &C2Val = C2->getAPIntValue();
      if (C1Val - 1 == C2Val) {
        // select Cond, C1, C1-1 --> add (zext Cond), C1-1
        if (VT != MVT::i1)
          Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Cond);
        return DAG.getNode(ISD::ADD, DL, VT, Cond, N2);
      }
      if (C1Val + 1 == C2Val) {
        // select Cond, C1, C1+1 --> add (sext Cond), C1+1
        if (VT != MVT::i1)
          Cond = DAG.getNode(ISD::SIGN_EXTEND, DL, VT, Cond);
        return DAG.getNode(ISD::ADD, DL, VT, Cond, N2);
      }

      // select Cond, Pow2, 0 --> (zext Cond) << log2(Pow2)
      if (C1Val.isPowerOf2() && C2Val.isNullValue()) {
        if (VT != MVT::i1)
          Cond = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Cond);
        SDValue ShAmtC = DAG.getConstant(C1Val.exactLogBase2(), DL, VT);
        return DAG.getNode(ISD::SHL, DL, VT, Cond, ShAmtC);
      }

      if (SDValue V = foldSelectOfConstantsUsingSra(N, DAG))
        return V;
    }

    return SDValue();
  }

  // fold (select Cond, 0, 1) -> (xor Cond, 1)
  // We can't do this reliably if integer based booleans have different contents
  // to floating point based booleans. This is because we can't tell whether we
  // have an integer-based boolean or a floating-point-based boolean unless we
  // can find the SETCC that produced it and inspect its operands. This is
  // fairly easy if C is the SETCC node, but it can potentially be
  // undiscoverable (or not reasonably discoverable). For example, it could be
  // in another basic block or it could require searching a complicated
  // expression.
  if (CondVT.isInteger() &&
      TLI.getBooleanContents(/*isVec*/false, /*isFloat*/true) ==
          TargetLowering::ZeroOrOneBooleanContent &&
      TLI.getBooleanContents(/*isVec*/false, /*isFloat*/false) ==
          TargetLowering::ZeroOrOneBooleanContent &&
      C1->isNullValue() && C2->isOne()) {
    SDValue NotCond =
        DAG.getNode(ISD::XOR, DL, CondVT, Cond, DAG.getConstant(1, DL, CondVT));
    if (VT.bitsEq(CondVT))
      return NotCond;
    return DAG.getZExtOrTrunc(NotCond, DL, VT);
  }

  return SDValue();
}

static SDValue foldBoolSelectToLogic(SDNode *N, SelectionDAG &DAG) {
  assert((N->getOpcode() == ISD::SELECT || N->getOpcode() == ISD::VSELECT) &&
         "Expected a (v)select");
  SDValue Cond = N->getOperand(0);
  SDValue T = N->getOperand(1), F = N->getOperand(2);
  EVT VT = N->getValueType(0);
  if (VT != Cond.getValueType() || VT.getScalarSizeInBits() != 1)
    return SDValue();

  // select Cond, Cond, F --> or Cond, F
  // select Cond, 1, F    --> or Cond, F
  if (Cond == T || isOneOrOneSplat(T, /* AllowUndefs */ true))
    return DAG.getNode(ISD::OR, SDLoc(N), VT, Cond, F);

  // select Cond, T, Cond --> and Cond, T
  // select Cond, T, 0    --> and Cond, T
  if (Cond == F || isNullOrNullSplat(F, /* AllowUndefs */ true))
    return DAG.getNode(ISD::AND, SDLoc(N), VT, Cond, T);

  // select Cond, T, 1 --> or (not Cond), T
  if (isOneOrOneSplat(F, /* AllowUndefs */ true)) {
    SDValue NotCond = DAG.getNOT(SDLoc(N), Cond, VT);
    return DAG.getNode(ISD::OR, SDLoc(N), VT, NotCond, T);
  }

  // select Cond, 0, F --> and (not Cond), F
  if (isNullOrNullSplat(T, /* AllowUndefs */ true)) {
    SDValue NotCond = DAG.getNOT(SDLoc(N), Cond, VT);
    return DAG.getNode(ISD::AND, SDLoc(N), VT, NotCond, F);
  }

  return SDValue();
}

SDValue DAGCombiner::visitSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  EVT VT0 = N0.getValueType();
  SDLoc DL(N);
  SDNodeFlags Flags = N->getFlags();

  if (SDValue V = DAG.simplifySelect(N0, N1, N2))
    return V;

  if (SDValue V = foldSelectOfConstants(N))
    return V;

  if (SDValue V = foldBoolSelectToLogic(N, DAG))
    return V;

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N1, N2))
    return SDValue(N, 0); // Don't revisit N.

  if (VT0 == MVT::i1) {
    // The code in this block deals with the following 2 equivalences:
    //    select(C0|C1, x, y) <=> select(C0, x, select(C1, x, y))
    //    select(C0&C1, x, y) <=> select(C0, select(C1, x, y), y)
    // The target can specify its preferred form with the
    // shouldNormalizeToSelectSequence() callback. However we always transform
    // to the right anyway if we find the inner select exists in the DAG anyway
    // and we always transform to the left side if we know that we can further
    // optimize the combination of the conditions.
    bool normalizeToSequence =
        TLI.shouldNormalizeToSelectSequence(*DAG.getContext(), VT);
    // select (and Cond0, Cond1), X, Y
    //   -> select Cond0, (select Cond1, X, Y), Y
    if (N0->getOpcode() == ISD::AND && N0->hasOneUse()) {
      SDValue Cond0 = N0->getOperand(0);
      SDValue Cond1 = N0->getOperand(1);
      SDValue InnerSelect =
          DAG.getNode(ISD::SELECT, DL, N1.getValueType(), Cond1, N1, N2, Flags);
      if (normalizeToSequence || !InnerSelect.use_empty())
        return DAG.getNode(ISD::SELECT, DL, N1.getValueType(), Cond0,
                           InnerSelect, N2, Flags);
      // Cleanup on failure.
      if (InnerSelect.use_empty())
        recursivelyDeleteUnusedNodes(InnerSelect.getNode());
    }
    // select (or Cond0, Cond1), X, Y -> select Cond0, X, (select Cond1, X, Y)
    if (N0->getOpcode() == ISD::OR && N0->hasOneUse()) {
      SDValue Cond0 = N0->getOperand(0);
      SDValue Cond1 = N0->getOperand(1);
      SDValue InnerSelect = DAG.getNode(ISD::SELECT, DL, N1.getValueType(),
                                        Cond1, N1, N2, Flags);
      if (normalizeToSequence || !InnerSelect.use_empty())
        return DAG.getNode(ISD::SELECT, DL, N1.getValueType(), Cond0, N1,
                           InnerSelect, Flags);
      // Cleanup on failure.
      if (InnerSelect.use_empty())
        recursivelyDeleteUnusedNodes(InnerSelect.getNode());
    }

    // select Cond0, (select Cond1, X, Y), Y -> select (and Cond0, Cond1), X, Y
    if (N1->getOpcode() == ISD::SELECT && N1->hasOneUse()) {
      SDValue N1_0 = N1->getOperand(0);
      SDValue N1_1 = N1->getOperand(1);
      SDValue N1_2 = N1->getOperand(2);
      if (N1_2 == N2 && N0.getValueType() == N1_0.getValueType()) {
        // Create the actual and node if we can generate good code for it.
        if (!normalizeToSequence) {
          SDValue And = DAG.getNode(ISD::AND, DL, N0.getValueType(), N0, N1_0);
          return DAG.getNode(ISD::SELECT, DL, N1.getValueType(), And, N1_1,
                             N2, Flags);
        }
        // Otherwise see if we can optimize the "and" to a better pattern.
        if (SDValue Combined = visitANDLike(N0, N1_0, N)) {
          return DAG.getNode(ISD::SELECT, DL, N1.getValueType(), Combined, N1_1,
                             N2, Flags);
        }
      }
    }
    // select Cond0, X, (select Cond1, X, Y) -> select (or Cond0, Cond1), X, Y
    if (N2->getOpcode() == ISD::SELECT && N2->hasOneUse()) {
      SDValue N2_0 = N2->getOperand(0);
      SDValue N2_1 = N2->getOperand(1);
      SDValue N2_2 = N2->getOperand(2);
      if (N2_1 == N1 && N0.getValueType() == N2_0.getValueType()) {
        // Create the actual or node if we can generate good code for it.
        if (!normalizeToSequence) {
          SDValue Or = DAG.getNode(ISD::OR, DL, N0.getValueType(), N0, N2_0);
          return DAG.getNode(ISD::SELECT, DL, N1.getValueType(), Or, N1,
                             N2_2, Flags);
        }
        // Otherwise see if we can optimize to a better pattern.
        if (SDValue Combined = visitORLike(N0, N2_0, N))
          return DAG.getNode(ISD::SELECT, DL, N1.getValueType(), Combined, N1,
                             N2_2, Flags);
      }
    }
  }

  // select (not Cond), N1, N2 -> select Cond, N2, N1
  if (SDValue F = extractBooleanFlip(N0, DAG, TLI, false)) {
    SDValue SelectOp = DAG.getSelect(DL, VT, F, N2, N1);
    SelectOp->setFlags(Flags);
    return SelectOp;
  }

  // Fold selects based on a setcc into other things, such as min/max/abs.
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue Cond0 = N0.getOperand(0), Cond1 = N0.getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();

    // select (fcmp lt x, y), x, y -> fminnum x, y
    // select (fcmp gt x, y), x, y -> fmaxnum x, y
    //
    // This is OK if we don't care what happens if either operand is a NaN.
    if (N0.hasOneUse() && isLegalToCombineMinNumMaxNum(DAG, N1, N2, TLI))
      if (SDValue FMinMax = combineMinNumMaxNum(DL, VT, Cond0, Cond1, N1, N2,
                                                CC, TLI, DAG))
        return FMinMax;

    // Use 'unsigned add with overflow' to optimize an unsigned saturating add.
    // This is conservatively limited to pre-legal-operations to give targets
    // a chance to reverse the transform if they want to do that. Also, it is
    // unlikely that the pattern would be formed late, so it's probably not
    // worth going through the other checks.
    if (!LegalOperations && TLI.isOperationLegalOrCustom(ISD::UADDO, VT) &&
        CC == ISD::SETUGT && N0.hasOneUse() && isAllOnesConstant(N1) &&
        N2.getOpcode() == ISD::ADD && Cond0 == N2.getOperand(0)) {
      auto *C = dyn_cast<ConstantSDNode>(N2.getOperand(1));
      auto *NotC = dyn_cast<ConstantSDNode>(Cond1);
      if (C && NotC && C->getAPIntValue() == ~NotC->getAPIntValue()) {
        // select (setcc Cond0, ~C, ugt), -1, (add Cond0, C) -->
        // uaddo Cond0, C; select uaddo.1, -1, uaddo.0
        //
        // The IR equivalent of this transform would have this form:
        //   %a = add %x, C
        //   %c = icmp ugt %x, ~C
        //   %r = select %c, -1, %a
        //   =>
        //   %u = call {iN,i1} llvm.uadd.with.overflow(%x, C)
        //   %u0 = extractvalue %u, 0
        //   %u1 = extractvalue %u, 1
        //   %r = select %u1, -1, %u0
        SDVTList VTs = DAG.getVTList(VT, VT0);
        SDValue UAO = DAG.getNode(ISD::UADDO, DL, VTs, Cond0, N2.getOperand(1));
        return DAG.getSelect(DL, VT, UAO.getValue(1), N1, UAO.getValue(0));
      }
    }

    if (TLI.isOperationLegal(ISD::SELECT_CC, VT) ||
        (!LegalOperations &&
         TLI.isOperationLegalOrCustom(ISD::SELECT_CC, VT))) {
      // Any flags available in a select/setcc fold will be on the setcc as they
      // migrated from fcmp
      Flags = N0.getNode()->getFlags();
      SDValue SelectNode = DAG.getNode(ISD::SELECT_CC, DL, VT, Cond0, Cond1, N1,
                                       N2, N0.getOperand(2));
      SelectNode->setFlags(Flags);
      return SelectNode;
    }

    if (SDValue NewSel = SimplifySelect(DL, N0, N1, N2))
      return NewSel;
  }

  if (!VT.isVector())
    if (SDValue BinOp = foldSelectOfBinops(N))
      return BinOp;

  return SDValue();
}

// This function assumes all the vselect's arguments are CONCAT_VECTOR
// nodes and that the condition is a BV of ConstantSDNodes (or undefs).
static SDValue ConvertSelectToConcatVector(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  SDValue Cond = N->getOperand(0);
  SDValue LHS = N->getOperand(1);
  SDValue RHS = N->getOperand(2);
  EVT VT = N->getValueType(0);
  int NumElems = VT.getVectorNumElements();
  assert(LHS.getOpcode() == ISD::CONCAT_VECTORS &&
         RHS.getOpcode() == ISD::CONCAT_VECTORS &&
         Cond.getOpcode() == ISD::BUILD_VECTOR);

  // CONCAT_VECTOR can take an arbitrary number of arguments. We only care about
  // binary ones here.
  if (LHS->getNumOperands() != 2 || RHS->getNumOperands() != 2)
    return SDValue();

  // We're sure we have an even number of elements due to the
  // concat_vectors we have as arguments to vselect.
  // Skip BV elements until we find one that's not an UNDEF
  // After we find an UNDEF element, keep looping until we get to half the
  // length of the BV and see if all the non-undef nodes are the same.
  ConstantSDNode *BottomHalf = nullptr;
  for (int i = 0; i < NumElems / 2; ++i) {
    if (Cond->getOperand(i)->isUndef())
      continue;

    if (BottomHalf == nullptr)
      BottomHalf = cast<ConstantSDNode>(Cond.getOperand(i));
    else if (Cond->getOperand(i).getNode() != BottomHalf)
      return SDValue();
  }

  // Do the same for the second half of the BuildVector
  ConstantSDNode *TopHalf = nullptr;
  for (int i = NumElems / 2; i < NumElems; ++i) {
    if (Cond->getOperand(i)->isUndef())
      continue;

    if (TopHalf == nullptr)
      TopHalf = cast<ConstantSDNode>(Cond.getOperand(i));
    else if (Cond->getOperand(i).getNode() != TopHalf)
      return SDValue();
  }

  assert(TopHalf && BottomHalf &&
         "One half of the selector was all UNDEFs and the other was all the "
         "same value. This should have been addressed before this function.");
  return DAG.getNode(
      ISD::CONCAT_VECTORS, DL, VT,
      BottomHalf->isNullValue() ? RHS->getOperand(0) : LHS->getOperand(0),
      TopHalf->isNullValue() ? RHS->getOperand(1) : LHS->getOperand(1));
}

bool refineUniformBase(SDValue &BasePtr, SDValue &Index, SelectionDAG &DAG) {
  if (!isNullConstant(BasePtr) || Index.getOpcode() != ISD::ADD)
    return false;

  // For now we check only the LHS of the add.
  SDValue LHS = Index.getOperand(0);
  SDValue SplatVal = DAG.getSplatValue(LHS);
  if (!SplatVal)
    return false;

  BasePtr = SplatVal;
  Index = Index.getOperand(1);
  return true;
}

// Fold sext/zext of index into index type.
bool refineIndexType(MaskedGatherScatterSDNode *MGS, SDValue &Index,
                     bool Scaled, SelectionDAG &DAG) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  if (Index.getOpcode() == ISD::ZERO_EXTEND) {
    SDValue Op = Index.getOperand(0);
    MGS->setIndexType(Scaled ? ISD::UNSIGNED_SCALED : ISD::UNSIGNED_UNSCALED);
    if (TLI.shouldRemoveExtendFromGSIndex(Op.getValueType())) {
      Index = Op;
      return true;
    }
  }

  if (Index.getOpcode() == ISD::SIGN_EXTEND) {
    SDValue Op = Index.getOperand(0);
    MGS->setIndexType(Scaled ? ISD::SIGNED_SCALED : ISD::SIGNED_UNSCALED);
    if (TLI.shouldRemoveExtendFromGSIndex(Op.getValueType())) {
      Index = Op;
      return true;
    }
  }

  return false;
}

SDValue DAGCombiner::visitMSCATTER(SDNode *N) {
  MaskedScatterSDNode *MSC = cast<MaskedScatterSDNode>(N);
  SDValue Mask = MSC->getMask();
  SDValue Chain = MSC->getChain();
  SDValue Index = MSC->getIndex();
  SDValue Scale = MSC->getScale();
  SDValue StoreVal = MSC->getValue();
  SDValue BasePtr = MSC->getBasePtr();
  SDLoc DL(N);

  // Zap scatters with a zero mask.
  if (ISD::isConstantSplatVectorAllZeros(Mask.getNode()))
    return Chain;

  if (refineUniformBase(BasePtr, Index, DAG)) {
    SDValue Ops[] = {Chain, StoreVal, Mask, BasePtr, Index, Scale};
    return DAG.getMaskedScatter(
        DAG.getVTList(MVT::Other), MSC->getMemoryVT(), DL, Ops,
        MSC->getMemOperand(), MSC->getIndexType(), MSC->isTruncatingStore());
  }

  if (refineIndexType(MSC, Index, MSC->isIndexScaled(), DAG)) {
    SDValue Ops[] = {Chain, StoreVal, Mask, BasePtr, Index, Scale};
    return DAG.getMaskedScatter(
        DAG.getVTList(MVT::Other), MSC->getMemoryVT(), DL, Ops,
        MSC->getMemOperand(), MSC->getIndexType(), MSC->isTruncatingStore());
  }

  return SDValue();
}

SDValue DAGCombiner::visitMSTORE(SDNode *N) {
  MaskedStoreSDNode *MST = cast<MaskedStoreSDNode>(N);
  SDValue Mask = MST->getMask();
  SDValue Chain = MST->getChain();
  SDLoc DL(N);

  // Zap masked stores with a zero mask.
  if (ISD::isConstantSplatVectorAllZeros(Mask.getNode()))
    return Chain;

  // If this is a masked load with an all ones mask, we can use a unmasked load.
  // FIXME: Can we do this for indexed, compressing, or truncating stores?
  if (ISD::isConstantSplatVectorAllOnes(Mask.getNode()) &&
      MST->isUnindexed() && !MST->isCompressingStore() &&
      !MST->isTruncatingStore())
    return DAG.getStore(MST->getChain(), SDLoc(N), MST->getValue(),
                        MST->getBasePtr(), MST->getMemOperand());

  // Try transforming N to an indexed store.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  return SDValue();
}

SDValue DAGCombiner::visitMGATHER(SDNode *N) {
  MaskedGatherSDNode *MGT = cast<MaskedGatherSDNode>(N);
  SDValue Mask = MGT->getMask();
  SDValue Chain = MGT->getChain();
  SDValue Index = MGT->getIndex();
  SDValue Scale = MGT->getScale();
  SDValue PassThru = MGT->getPassThru();
  SDValue BasePtr = MGT->getBasePtr();
  SDLoc DL(N);

  // Zap gathers with a zero mask.
  if (ISD::isConstantSplatVectorAllZeros(Mask.getNode()))
    return CombineTo(N, PassThru, MGT->getChain());

  if (refineUniformBase(BasePtr, Index, DAG)) {
    SDValue Ops[] = {Chain, PassThru, Mask, BasePtr, Index, Scale};
    return DAG.getMaskedGather(DAG.getVTList(N->getValueType(0), MVT::Other),
                               MGT->getMemoryVT(), DL, Ops,
                               MGT->getMemOperand(), MGT->getIndexType(),
                               MGT->getExtensionType());
  }

  if (refineIndexType(MGT, Index, MGT->isIndexScaled(), DAG)) {
    SDValue Ops[] = {Chain, PassThru, Mask, BasePtr, Index, Scale};
    return DAG.getMaskedGather(DAG.getVTList(N->getValueType(0), MVT::Other),
                               MGT->getMemoryVT(), DL, Ops,
                               MGT->getMemOperand(), MGT->getIndexType(),
                               MGT->getExtensionType());
  }

  return SDValue();
}

SDValue DAGCombiner::visitMLOAD(SDNode *N) {
  MaskedLoadSDNode *MLD = cast<MaskedLoadSDNode>(N);
  SDValue Mask = MLD->getMask();
  SDLoc DL(N);

  // Zap masked loads with a zero mask.
  if (ISD::isConstantSplatVectorAllZeros(Mask.getNode()))
    return CombineTo(N, MLD->getPassThru(), MLD->getChain());

  // If this is a masked load with an all ones mask, we can use a unmasked load.
  // FIXME: Can we do this for indexed, expanding, or extending loads?
  if (ISD::isConstantSplatVectorAllOnes(Mask.getNode()) &&
      MLD->isUnindexed() && !MLD->isExpandingLoad() &&
      MLD->getExtensionType() == ISD::NON_EXTLOAD) {
    SDValue NewLd = DAG.getLoad(N->getValueType(0), SDLoc(N), MLD->getChain(),
                                MLD->getBasePtr(), MLD->getMemOperand());
    return CombineTo(N, NewLd, NewLd.getValue(1));
  }

  // Try transforming N to an indexed load.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  return SDValue();
}

/// A vector select of 2 constant vectors can be simplified to math/logic to
/// avoid a variable select instruction and possibly avoid constant loads.
SDValue DAGCombiner::foldVSelectOfConstants(SDNode *N) {
  SDValue Cond = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  if (!Cond.hasOneUse() || Cond.getScalarValueSizeInBits() != 1 ||
      !TLI.convertSelectOfConstantsToMath(VT) ||
      !ISD::isBuildVectorOfConstantSDNodes(N1.getNode()) ||
      !ISD::isBuildVectorOfConstantSDNodes(N2.getNode()))
    return SDValue();

  // Check if we can use the condition value to increment/decrement a single
  // constant value. This simplifies a select to an add and removes a constant
  // load/materialization from the general case.
  bool AllAddOne = true;
  bool AllSubOne = true;
  unsigned Elts = VT.getVectorNumElements();
  for (unsigned i = 0; i != Elts; ++i) {
    SDValue N1Elt = N1.getOperand(i);
    SDValue N2Elt = N2.getOperand(i);
    if (N1Elt.isUndef() || N2Elt.isUndef())
      continue;
    if (N1Elt.getValueType() != N2Elt.getValueType())
      continue;

    const APInt &C1 = cast<ConstantSDNode>(N1Elt)->getAPIntValue();
    const APInt &C2 = cast<ConstantSDNode>(N2Elt)->getAPIntValue();
    if (C1 != C2 + 1)
      AllAddOne = false;
    if (C1 != C2 - 1)
      AllSubOne = false;
  }

  // Further simplifications for the extra-special cases where the constants are
  // all 0 or all -1 should be implemented as folds of these patterns.
  SDLoc DL(N);
  if (AllAddOne || AllSubOne) {
    // vselect <N x i1> Cond, C+1, C --> add (zext Cond), C
    // vselect <N x i1> Cond, C-1, C --> add (sext Cond), C
    auto ExtendOpcode = AllAddOne ? ISD::ZERO_EXTEND : ISD::SIGN_EXTEND;
    SDValue ExtendedCond = DAG.getNode(ExtendOpcode, DL, VT, Cond);
    return DAG.getNode(ISD::ADD, DL, VT, ExtendedCond, N2);
  }

  // select Cond, Pow2C, 0 --> (zext Cond) << log2(Pow2C)
  APInt Pow2C;
  if (ISD::isConstantSplatVector(N1.getNode(), Pow2C) && Pow2C.isPowerOf2() &&
      isNullOrNullSplat(N2)) {
    SDValue ZextCond = DAG.getZExtOrTrunc(Cond, DL, VT);
    SDValue ShAmtC = DAG.getConstant(Pow2C.exactLogBase2(), DL, VT);
    return DAG.getNode(ISD::SHL, DL, VT, ZextCond, ShAmtC);
  }

  if (SDValue V = foldSelectOfConstantsUsingSra(N, DAG))
    return V;

  // The general case for select-of-constants:
  // vselect <N x i1> Cond, C1, C2 --> xor (and (sext Cond), (C1^C2)), C2
  // ...but that only makes sense if a vselect is slower than 2 logic ops, so
  // leave that to a machine-specific pass.
  return SDValue();
}

SDValue DAGCombiner::visitVSELECT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  if (SDValue V = DAG.simplifySelect(N0, N1, N2))
    return V;

  if (SDValue V = foldBoolSelectToLogic(N, DAG))
    return V;

  // vselect (not Cond), N1, N2 -> vselect Cond, N2, N1
  if (SDValue F = extractBooleanFlip(N0, DAG, TLI, false))
    return DAG.getSelect(DL, VT, F, N2, N1);

  // Canonicalize integer abs.
  // vselect (setg[te] X,  0),  X, -X ->
  // vselect (setgt    X, -1),  X, -X ->
  // vselect (setl[te] X,  0), -X,  X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N0.getOpcode() == ISD::SETCC) {
    SDValue LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
    bool isAbs = false;
    bool RHSIsAllZeros = ISD::isBuildVectorAllZeros(RHS.getNode());

    if (((RHSIsAllZeros && (CC == ISD::SETGT || CC == ISD::SETGE)) ||
         (ISD::isBuildVectorAllOnes(RHS.getNode()) && CC == ISD::SETGT)) &&
        N1 == LHS && N2.getOpcode() == ISD::SUB && N1 == N2.getOperand(1))
      isAbs = ISD::isBuildVectorAllZeros(N2.getOperand(0).getNode());
    else if ((RHSIsAllZeros && (CC == ISD::SETLT || CC == ISD::SETLE)) &&
             N2 == LHS && N1.getOpcode() == ISD::SUB && N2 == N1.getOperand(1))
      isAbs = ISD::isBuildVectorAllZeros(N1.getOperand(0).getNode());

    if (isAbs) {
      if (TLI.isOperationLegalOrCustom(ISD::ABS, VT))
        return DAG.getNode(ISD::ABS, DL, VT, LHS);

      SDValue Shift = DAG.getNode(ISD::SRA, DL, VT, LHS,
                                  DAG.getConstant(VT.getScalarSizeInBits() - 1,
                                                  DL, getShiftAmountTy(VT)));
      SDValue Add = DAG.getNode(ISD::ADD, DL, VT, LHS, Shift);
      AddToWorklist(Shift.getNode());
      AddToWorklist(Add.getNode());
      return DAG.getNode(ISD::XOR, DL, VT, Add, Shift);
    }

    // vselect x, y (fcmp lt x, y) -> fminnum x, y
    // vselect x, y (fcmp gt x, y) -> fmaxnum x, y
    //
    // This is OK if we don't care about what happens if either operand is a
    // NaN.
    //
    if (N0.hasOneUse() && isLegalToCombineMinNumMaxNum(DAG, LHS, RHS, TLI)) {
      if (SDValue FMinMax =
              combineMinNumMaxNum(DL, VT, LHS, RHS, N1, N2, CC, TLI, DAG))
        return FMinMax;
    }

    // If this select has a condition (setcc) with narrower operands than the
    // select, try to widen the compare to match the select width.
    // TODO: This should be extended to handle any constant.
    // TODO: This could be extended to handle non-loading patterns, but that
    //       requires thorough testing to avoid regressions.
    if (isNullOrNullSplat(RHS)) {
      EVT NarrowVT = LHS.getValueType();
      EVT WideVT = N1.getValueType().changeVectorElementTypeToInteger();
      EVT SetCCVT = getSetCCResultType(LHS.getValueType());
      unsigned SetCCWidth = SetCCVT.getScalarSizeInBits();
      unsigned WideWidth = WideVT.getScalarSizeInBits();
      bool IsSigned = isSignedIntSetCC(CC);
      auto LoadExtOpcode = IsSigned ? ISD::SEXTLOAD : ISD::ZEXTLOAD;
      if (LHS.getOpcode() == ISD::LOAD && LHS.hasOneUse() &&
          SetCCWidth != 1 && SetCCWidth < WideWidth &&
          TLI.isLoadExtLegalOrCustom(LoadExtOpcode, WideVT, NarrowVT) &&
          TLI.isOperationLegalOrCustom(ISD::SETCC, WideVT)) {
        // Both compare operands can be widened for free. The LHS can use an
        // extended load, and the RHS is a constant:
        //   vselect (ext (setcc load(X), C)), N1, N2 -->
        //   vselect (setcc extload(X), C'), N1, N2
        auto ExtOpcode = IsSigned ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
        SDValue WideLHS = DAG.getNode(ExtOpcode, DL, WideVT, LHS);
        SDValue WideRHS = DAG.getNode(ExtOpcode, DL, WideVT, RHS);
        EVT WideSetCCVT = getSetCCResultType(WideVT);
        SDValue WideSetCC = DAG.getSetCC(DL, WideSetCCVT, WideLHS, WideRHS, CC);
        return DAG.getSelect(DL, N1.getValueType(), WideSetCC, N1, N2);
      }
    }

    // Match VSELECTs into add with unsigned saturation.
    if (hasOperation(ISD::UADDSAT, VT)) {
      // Check if one of the arms of the VSELECT is vector with all bits set.
      // If it's on the left side invert the predicate to simplify logic below.
      SDValue Other;
      ISD::CondCode SatCC = CC;
      if (ISD::isConstantSplatVectorAllOnes(N1.getNode())) {
        Other = N2;
        SatCC = ISD::getSetCCInverse(SatCC, VT.getScalarType());
      } else if (ISD::isConstantSplatVectorAllOnes(N2.getNode())) {
        Other = N1;
      }

      if (Other && Other.getOpcode() == ISD::ADD) {
        SDValue CondLHS = LHS, CondRHS = RHS;
        SDValue OpLHS = Other.getOperand(0), OpRHS = Other.getOperand(1);

        // Canonicalize condition operands.
        if (SatCC == ISD::SETUGE) {
          std::swap(CondLHS, CondRHS);
          SatCC = ISD::SETULE;
        }

        // We can test against either of the addition operands.
        // x <= x+y ? x+y : ~0 --> uaddsat x, y
        // x+y >= x ? x+y : ~0 --> uaddsat x, y
        if (SatCC == ISD::SETULE && Other == CondRHS &&
            (OpLHS == CondLHS || OpRHS == CondLHS))
          return DAG.getNode(ISD::UADDSAT, DL, VT, OpLHS, OpRHS);

        if (OpRHS.getOpcode() == CondRHS.getOpcode() &&
            (OpRHS.getOpcode() == ISD::BUILD_VECTOR ||
             OpRHS.getOpcode() == ISD::SPLAT_VECTOR) &&
            CondLHS == OpLHS) {
          // If the RHS is a constant we have to reverse the const
          // canonicalization.
          // x >= ~C ? x+C : ~0 --> uaddsat x, C
          auto MatchUADDSAT = [](ConstantSDNode *Op, ConstantSDNode *Cond) {
            return Cond->getAPIntValue() == ~Op->getAPIntValue();
          };
          if (SatCC == ISD::SETULE &&
              ISD::matchBinaryPredicate(OpRHS, CondRHS, MatchUADDSAT))
            return DAG.getNode(ISD::UADDSAT, DL, VT, OpLHS, OpRHS);
        }
      }
    }

    // Match VSELECTs into sub with unsigned saturation.
    if (hasOperation(ISD::USUBSAT, VT)) {
      // Check if one of the arms of the VSELECT is a zero vector. If it's on
      // the left side invert the predicate to simplify logic below.
      SDValue Other;
      ISD::CondCode SatCC = CC;
      if (ISD::isConstantSplatVectorAllZeros(N1.getNode())) {
        Other = N2;
        SatCC = ISD::getSetCCInverse(SatCC, VT.getScalarType());
      } else if (ISD::isConstantSplatVectorAllZeros(N2.getNode())) {
        Other = N1;
      }

      if (Other && Other.getNumOperands() == 2) {
        SDValue CondRHS = RHS;
        SDValue OpLHS = Other.getOperand(0), OpRHS = Other.getOperand(1);

        if (Other.getOpcode() == ISD::SUB &&
            LHS.getOpcode() == ISD::ZERO_EXTEND && LHS.getOperand(0) == OpLHS &&
            OpRHS.getOpcode() == ISD::TRUNCATE && OpRHS.getOperand(0) == RHS) {
          // Look for a general sub with unsigned saturation first.
          // zext(x) >= y ? x - trunc(y) : 0
          // --> usubsat(x,trunc(umin(y,SatLimit)))
          // zext(x) >  y ? x - trunc(y) : 0
          // --> usubsat(x,trunc(umin(y,SatLimit)))
          if (SatCC == ISD::SETUGE || SatCC == ISD::SETUGT)
            return getTruncatedUSUBSAT(VT, LHS.getValueType(), LHS, RHS, DAG,
                                       DL);
        }

        if (OpLHS == LHS) {
          // Look for a general sub with unsigned saturation first.
          // x >= y ? x-y : 0 --> usubsat x, y
          // x >  y ? x-y : 0 --> usubsat x, y
          if ((SatCC == ISD::SETUGE || SatCC == ISD::SETUGT) &&
              Other.getOpcode() == ISD::SUB && OpRHS == CondRHS)
            return DAG.getNode(ISD::USUBSAT, DL, VT, OpLHS, OpRHS);

          if (OpRHS.getOpcode() == ISD::BUILD_VECTOR ||
              OpRHS.getOpcode() == ISD::SPLAT_VECTOR) {
            if (CondRHS.getOpcode() == ISD::BUILD_VECTOR ||
                CondRHS.getOpcode() == ISD::SPLAT_VECTOR) {
              // If the RHS is a constant we have to reverse the const
              // canonicalization.
              // x > C-1 ? x+-C : 0 --> usubsat x, C
              auto MatchUSUBSAT = [](ConstantSDNode *Op, ConstantSDNode *Cond) {
                return (!Op && !Cond) ||
                       (Op && Cond &&
                        Cond->getAPIntValue() == (-Op->getAPIntValue() - 1));
              };
              if (SatCC == ISD::SETUGT && Other.getOpcode() == ISD::ADD &&
                  ISD::matchBinaryPredicate(OpRHS, CondRHS, MatchUSUBSAT,
                                            /*AllowUndefs*/ true)) {
                OpRHS = DAG.getNode(ISD::SUB, DL, VT,
                                    DAG.getConstant(0, DL, VT), OpRHS);
                return DAG.getNode(ISD::USUBSAT, DL, VT, OpLHS, OpRHS);
              }

              // Another special case: If C was a sign bit, the sub has been
              // canonicalized into a xor.
              // FIXME: Would it be better to use computeKnownBits to determine
              //        whether it's safe to decanonicalize the xor?
              // x s< 0 ? x^C : 0 --> usubsat x, C
              APInt SplatValue;
              if (SatCC == ISD::SETLT && Other.getOpcode() == ISD::XOR &&
                  ISD::isConstantSplatVector(OpRHS.getNode(), SplatValue) &&
                  ISD::isConstantSplatVectorAllZeros(CondRHS.getNode()) &&
                  SplatValue.isSignMask()) {
                // Note that we have to rebuild the RHS constant here to
                // ensure we don't rely on particular values of undef lanes.
                OpRHS = DAG.getConstant(SplatValue, DL, VT);
                return DAG.getNode(ISD::USUBSAT, DL, VT, OpLHS, OpRHS);
              }
            }
          }
        }
      }
    }
  }

  if (SimplifySelectOps(N, N1, N2))
    return SDValue(N, 0);  // Don't revisit N.

  // Fold (vselect all_ones, N1, N2) -> N1
  if (ISD::isConstantSplatVectorAllOnes(N0.getNode()))
    return N1;
  // Fold (vselect all_zeros, N1, N2) -> N2
  if (ISD::isConstantSplatVectorAllZeros(N0.getNode()))
    return N2;

  // The ConvertSelectToConcatVector function is assuming both the above
  // checks for (vselect (build_vector all{ones,zeros) ...) have been made
  // and addressed.
  if (N1.getOpcode() == ISD::CONCAT_VECTORS &&
      N2.getOpcode() == ISD::CONCAT_VECTORS &&
      ISD::isBuildVectorOfConstantSDNodes(N0.getNode())) {
    if (SDValue CV = ConvertSelectToConcatVector(N, DAG))
      return CV;
  }

  if (SDValue V = foldVSelectOfConstants(N))
    return V;

  return SDValue();
}

SDValue DAGCombiner::visitSELECT_CC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDValue N3 = N->getOperand(3);
  SDValue N4 = N->getOperand(4);
  ISD::CondCode CC = cast<CondCodeSDNode>(N4)->get();

  // fold select_cc lhs, rhs, x, x, cc -> x
  if (N2 == N3)
    return N2;

  // Determine if the condition we're dealing with is constant
  if (SDValue SCC = SimplifySetCC(getSetCCResultType(N0.getValueType()), N0, N1,
                                  CC, SDLoc(N), false)) {
    AddToWorklist(SCC.getNode());

    if (ConstantSDNode *SCCC = dyn_cast<ConstantSDNode>(SCC.getNode())) {
      if (!SCCC->isNullValue())
        return N2;    // cond always true -> true val
      else
        return N3;    // cond always false -> false val
    } else if (SCC->isUndef()) {
      // When the condition is UNDEF, just return the first operand. This is
      // coherent the DAG creation, no setcc node is created in this case
      return N2;
    } else if (SCC.getOpcode() == ISD::SETCC) {
      // Fold to a simpler select_cc
      SDValue SelectOp = DAG.getNode(
          ISD::SELECT_CC, SDLoc(N), N2.getValueType(), SCC.getOperand(0),
          SCC.getOperand(1), N2, N3, SCC.getOperand(2));
      SelectOp->setFlags(SCC->getFlags());
      return SelectOp;
    }
  }

  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDValue(N, 0);  // Don't revisit N.

  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(SDLoc(N), N0, N1, N2, N3, CC);
}

SDValue DAGCombiner::visitSETCC(SDNode *N) {
  // setcc is very commonly used as an argument to brcond. This pattern
  // also lend itself to numerous combines and, as a result, it is desired
  // we keep the argument to a brcond as a setcc as much as possible.
  bool PreferSetCC =
      N->hasOneUse() && N->use_begin()->getOpcode() == ISD::BRCOND;

  ISD::CondCode Cond = cast<CondCodeSDNode>(N->getOperand(2))->get();
  EVT VT = N->getValueType(0);

  //   SETCC(FREEZE(X), CONST, Cond)
  // =>
  //   FREEZE(SETCC(X, CONST, Cond))
  // This is correct if FREEZE(X) has one use and SETCC(FREEZE(X), CONST, Cond)
  // isn't equivalent to true or false.
  // For example, SETCC(FREEZE(X), -128, SETULT) cannot be folded to
  // FREEZE(SETCC(X, -128, SETULT)) because X can be poison.
  //
  // This transformation is beneficial because visitBRCOND can fold
  // BRCOND(FREEZE(X)) to BRCOND(X).

  // Conservatively optimize integer comparisons only.
  if (PreferSetCC) {
    // Do this only when SETCC is going to be used by BRCOND.

    SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
    ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
    bool Updated = false;

    // Is 'X Cond C' always true or false?
    auto IsAlwaysTrueOrFalse = [](ISD::CondCode Cond, ConstantSDNode *C) {
      bool False = (Cond == ISD::SETULT && C->isNullValue()) ||
                   (Cond == ISD::SETLT  && C->isMinSignedValue()) ||
                   (Cond == ISD::SETUGT && C->isAllOnesValue()) ||
                   (Cond == ISD::SETGT  && C->isMaxSignedValue());
      bool True =  (Cond == ISD::SETULE && C->isAllOnesValue()) ||
                   (Cond == ISD::SETLE  && C->isMaxSignedValue()) ||
                   (Cond == ISD::SETUGE && C->isNullValue()) ||
                   (Cond == ISD::SETGE  && C->isMinSignedValue());
      return True || False;
    };

    if (N0->getOpcode() == ISD::FREEZE && N0.hasOneUse() && N1C) {
      if (!IsAlwaysTrueOrFalse(Cond, N1C)) {
        N0 = N0->getOperand(0);
        Updated = true;
      }
    }
    if (N1->getOpcode() == ISD::FREEZE && N1.hasOneUse() && N0C) {
      if (!IsAlwaysTrueOrFalse(ISD::getSetCCSwappedOperands(Cond),
                               N0C)) {
        N1 = N1->getOperand(0);
        Updated = true;
      }
    }

    if (Updated)
      return DAG.getFreeze(DAG.getSetCC(SDLoc(N), VT, N0, N1, Cond));
  }

  SDValue Combined = SimplifySetCC(VT, N->getOperand(0), N->getOperand(1), Cond,
                                   SDLoc(N), !PreferSetCC);

  if (!Combined)
    return SDValue();

  // If we prefer to have a setcc, and we don't, we'll try our best to
  // recreate one using rebuildSetCC.
  if (PreferSetCC && Combined.getOpcode() != ISD::SETCC) {
    SDValue NewSetCC = rebuildSetCC(Combined);

    // We don't have anything interesting to combine to.
    if (NewSetCC.getNode() == N)
      return SDValue();

    if (NewSetCC)
      return NewSetCC;
  }

  return Combined;
}

SDValue DAGCombiner::visitSETCCCARRY(SDNode *N) {
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue Carry = N->getOperand(2);
  SDValue Cond = N->getOperand(3);

  // If Carry is false, fold to a regular SETCC.
  if (isNullConstant(Carry))
    return DAG.getNode(ISD::SETCC, SDLoc(N), N->getVTList(), LHS, RHS, Cond);

  return SDValue();
}

/// Check if N satisfies:
///   N is used once.
///   N is a Load.
///   The load is compatible with ExtOpcode. It means
///     If load has explicit zero/sign extension, ExpOpcode must have the same
///     extension.
///     Otherwise returns true.
static bool isCompatibleLoad(SDValue N, unsigned ExtOpcode) {
  if (!N.hasOneUse())
    return false;

  if (!isa<LoadSDNode>(N))
    return false;

  LoadSDNode *Load = cast<LoadSDNode>(N);
  ISD::LoadExtType LoadExt = Load->getExtensionType();
  if (LoadExt == ISD::NON_EXTLOAD || LoadExt == ISD::EXTLOAD)
    return true;

  // Now LoadExt is either SEXTLOAD or ZEXTLOAD, ExtOpcode must have the same
  // extension.
  if ((LoadExt == ISD::SEXTLOAD && ExtOpcode != ISD::SIGN_EXTEND) ||
      (LoadExt == ISD::ZEXTLOAD && ExtOpcode != ISD::ZERO_EXTEND))
    return false;

  return true;
}

/// Fold
///   (sext (select c, load x, load y)) -> (select c, sextload x, sextload y)
///   (zext (select c, load x, load y)) -> (select c, zextload x, zextload y)
///   (aext (select c, load x, load y)) -> (select c, extload x, extload y)
/// This function is called by the DAGCombiner when visiting sext/zext/aext
/// dag nodes (see for example method DAGCombiner::visitSIGN_EXTEND).
static SDValue tryToFoldExtendSelectLoad(SDNode *N, const TargetLowering &TLI,
                                         SelectionDAG &DAG) {
  unsigned Opcode = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  assert((Opcode == ISD::SIGN_EXTEND || Opcode == ISD::ZERO_EXTEND ||
          Opcode == ISD::ANY_EXTEND) &&
         "Expected EXTEND dag node in input!");

  if (!(N0->getOpcode() == ISD::SELECT || N0->getOpcode() == ISD::VSELECT) ||
      !N0.hasOneUse())
    return SDValue();

  SDValue Op1 = N0->getOperand(1);
  SDValue Op2 = N0->getOperand(2);
  if (!isCompatibleLoad(Op1, Opcode) || !isCompatibleLoad(Op2, Opcode))
    return SDValue();

  auto ExtLoadOpcode = ISD::EXTLOAD;
  if (Opcode == ISD::SIGN_EXTEND)
    ExtLoadOpcode = ISD::SEXTLOAD;
  else if (Opcode == ISD::ZERO_EXTEND)
    ExtLoadOpcode = ISD::ZEXTLOAD;

  LoadSDNode *Load1 = cast<LoadSDNode>(Op1);
  LoadSDNode *Load2 = cast<LoadSDNode>(Op2);
  if (!TLI.isLoadExtLegal(ExtLoadOpcode, VT, Load1->getMemoryVT()) ||
      !TLI.isLoadExtLegal(ExtLoadOpcode, VT, Load2->getMemoryVT()))
    return SDValue();

  SDValue Ext1 = DAG.getNode(Opcode, DL, VT, Op1);
  SDValue Ext2 = DAG.getNode(Opcode, DL, VT, Op2);
  return DAG.getSelect(DL, VT, N0->getOperand(0), Ext1, Ext2);
}

/// Try to fold a sext/zext/aext dag node into a ConstantSDNode or
/// a build_vector of constants.
/// This function is called by the DAGCombiner when visiting sext/zext/aext
/// dag nodes (see for example method DAGCombiner::visitSIGN_EXTEND).
/// Vector extends are not folded if operations are legal; this is to
/// avoid introducing illegal build_vector dag nodes.
static SDValue tryToFoldExtendOfConstant(SDNode *N, const TargetLowering &TLI,
                                         SelectionDAG &DAG, bool LegalTypes) {
  unsigned Opcode = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  assert((Opcode == ISD::SIGN_EXTEND || Opcode == ISD::ZERO_EXTEND ||
         Opcode == ISD::ANY_EXTEND || Opcode == ISD::SIGN_EXTEND_VECTOR_INREG ||
         Opcode == ISD::ZERO_EXTEND_VECTOR_INREG)
         && "Expected EXTEND dag node in input!");

  // fold (sext c1) -> c1
  // fold (zext c1) -> c1
  // fold (aext c1) -> c1
  if (isa<ConstantSDNode>(N0))
    return DAG.getNode(Opcode, DL, VT, N0);

  // fold (sext (select cond, c1, c2)) -> (select cond, sext c1, sext c2)
  // fold (zext (select cond, c1, c2)) -> (select cond, zext c1, zext c2)
  // fold (aext (select cond, c1, c2)) -> (select cond, sext c1, sext c2)
  if (N0->getOpcode() == ISD::SELECT) {
    SDValue Op1 = N0->getOperand(1);
    SDValue Op2 = N0->getOperand(2);
    if (isa<ConstantSDNode>(Op1) && isa<ConstantSDNode>(Op2) &&
        (Opcode != ISD::ZERO_EXTEND || !TLI.isZExtFree(N0.getValueType(), VT))) {
      // For any_extend, choose sign extension of the constants to allow a
      // possible further transform to sign_extend_inreg.i.e.
      //
      // t1: i8 = select t0, Constant:i8<-1>, Constant:i8<0>
      // t2: i64 = any_extend t1
      // -->
      // t3: i64 = select t0, Constant:i64<-1>, Constant:i64<0>
      // -->
      // t4: i64 = sign_extend_inreg t3
      unsigned FoldOpc = Opcode;
      if (FoldOpc == ISD::ANY_EXTEND)
        FoldOpc = ISD::SIGN_EXTEND;
      return DAG.getSelect(DL, VT, N0->getOperand(0),
                           DAG.getNode(FoldOpc, DL, VT, Op1),
                           DAG.getNode(FoldOpc, DL, VT, Op2));
    }
  }

  // fold (sext (build_vector AllConstants) -> (build_vector AllConstants)
  // fold (zext (build_vector AllConstants) -> (build_vector AllConstants)
  // fold (aext (build_vector AllConstants) -> (build_vector AllConstants)
  EVT SVT = VT.getScalarType();
  if (!(VT.isVector() && (!LegalTypes || TLI.isTypeLegal(SVT)) &&
      ISD::isBuildVectorOfConstantSDNodes(N0.getNode())))
    return SDValue();

  // We can fold this node into a build_vector.
  unsigned VTBits = SVT.getSizeInBits();
  unsigned EVTBits = N0->getValueType(0).getScalarSizeInBits();
  SmallVector<SDValue, 8> Elts;
  unsigned NumElts = VT.getVectorNumElements();

  // For zero-extensions, UNDEF elements still guarantee to have the upper
  // bits set to zero.
  bool IsZext =
      Opcode == ISD::ZERO_EXTEND || Opcode == ISD::ZERO_EXTEND_VECTOR_INREG;

  for (unsigned i = 0; i != NumElts; ++i) {
    SDValue Op = N0.getOperand(i);
    if (Op.isUndef()) {
      Elts.push_back(IsZext ? DAG.getConstant(0, DL, SVT) : DAG.getUNDEF(SVT));
      continue;
    }

    SDLoc DL(Op);
    // Get the constant value and if needed trunc it to the size of the type.
    // Nodes like build_vector might have constants wider than the scalar type.
    APInt C = cast<ConstantSDNode>(Op)->getAPIntValue().zextOrTrunc(EVTBits);
    if (Opcode == ISD::SIGN_EXTEND || Opcode == ISD::SIGN_EXTEND_VECTOR_INREG)
      Elts.push_back(DAG.getConstant(C.sext(VTBits), DL, SVT));
    else
      Elts.push_back(DAG.getConstant(C.zext(VTBits), DL, SVT));
  }

  return DAG.getBuildVector(VT, DL, Elts);
}

// ExtendUsesToFormExtLoad - Trying to extend uses of a load to enable this:
// "fold ({s|z|a}ext (load x)) -> ({s|z|a}ext (truncate ({s|z|a}extload x)))"
// transformation. Returns true if extension are possible and the above
// mentioned transformation is profitable.
static bool ExtendUsesToFormExtLoad(EVT VT, SDNode *N, SDValue N0,
                                    unsigned ExtOpc,
                                    SmallVectorImpl<SDNode *> &ExtendNodes,
                                    const TargetLowering &TLI) {
  bool HasCopyToRegUses = false;
  bool isTruncFree = TLI.isTruncateFree(VT, N0.getValueType());
  for (SDNode::use_iterator UI = N0.getNode()->use_begin(),
                            UE = N0.getNode()->use_end();
       UI != UE; ++UI) {
    SDNode *User = *UI;
    if (User == N)
      continue;
    if (UI.getUse().getResNo() != N0.getResNo())
      continue;
    // FIXME: Only extend SETCC N, N and SETCC N, c for now.
    if (ExtOpc != ISD::ANY_EXTEND && User->getOpcode() == ISD::SETCC) {
      ISD::CondCode CC = cast<CondCodeSDNode>(User->getOperand(2))->get();
      if (ExtOpc == ISD::ZERO_EXTEND && ISD::isSignedIntSetCC(CC))
        // Sign bits will be lost after a zext.
        return false;
      bool Add = false;
      for (unsigned i = 0; i != 2; ++i) {
        SDValue UseOp = User->getOperand(i);
        if (UseOp == N0)
          continue;
        if (!isa<ConstantSDNode>(UseOp))
          return false;
        Add = true;
      }
      if (Add)
        ExtendNodes.push_back(User);
      continue;
    }
    // If truncates aren't free and there are users we can't
    // extend, it isn't worthwhile.
    if (!isTruncFree)
      return false;
    // Remember if this value is live-out.
    if (User->getOpcode() == ISD::CopyToReg)
      HasCopyToRegUses = true;
  }

  if (HasCopyToRegUses) {
    bool BothLiveOut = false;
    for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
         UI != UE; ++UI) {
      SDUse &Use = UI.getUse();
      if (Use.getResNo() == 0 && Use.getUser()->getOpcode() == ISD::CopyToReg) {
        BothLiveOut = true;
        break;
      }
    }
    if (BothLiveOut)
      // Both unextended and extended values are live out. There had better be
      // a good reason for the transformation.
      return ExtendNodes.size();
  }
  return true;
}

void DAGCombiner::ExtendSetCCUses(const SmallVectorImpl<SDNode *> &SetCCs,
                                  SDValue OrigLoad, SDValue ExtLoad,
                                  ISD::NodeType ExtType) {
  // Extend SetCC uses if necessary.
  SDLoc DL(ExtLoad);
  for (SDNode *SetCC : SetCCs) {
    SmallVector<SDValue, 4> Ops;

    for (unsigned j = 0; j != 2; ++j) {
      SDValue SOp = SetCC->getOperand(j);
      if (SOp == OrigLoad)
        Ops.push_back(ExtLoad);
      else
        Ops.push_back(DAG.getNode(ExtType, DL, ExtLoad->getValueType(0), SOp));
    }

    Ops.push_back(SetCC->getOperand(2));
    CombineTo(SetCC, DAG.getNode(ISD::SETCC, DL, SetCC->getValueType(0), Ops));
  }
}

// FIXME: Bring more similar combines here, common to sext/zext (maybe aext?).
SDValue DAGCombiner::CombineExtLoad(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT DstVT = N->getValueType(0);
  EVT SrcVT = N0.getValueType();

  assert((N->getOpcode() == ISD::SIGN_EXTEND ||
          N->getOpcode() == ISD::ZERO_EXTEND) &&
         "Unexpected node type (not an extend)!");

  // fold (sext (load x)) to multiple smaller sextloads; same for zext.
  // For example, on a target with legal v4i32, but illegal v8i32, turn:
  //   (v8i32 (sext (v8i16 (load x))))
  // into:
  //   (v8i32 (concat_vectors (v4i32 (sextload x)),
  //                          (v4i32 (sextload (x + 16)))))
  // Where uses of the original load, i.e.:
  //   (v8i16 (load x))
  // are replaced with:
  //   (v8i16 (truncate
  //     (v8i32 (concat_vectors (v4i32 (sextload x)),
  //                            (v4i32 (sextload (x + 16)))))))
  //
  // This combine is only applicable to illegal, but splittable, vectors.
  // All legal types, and illegal non-vector types, are handled elsewhere.
  // This combine is controlled by TargetLowering::isVectorLoadExtDesirable.
  //
  if (N0->getOpcode() != ISD::LOAD)
    return SDValue();

  LoadSDNode *LN0 = cast<LoadSDNode>(N0);

  if (!ISD::isNON_EXTLoad(LN0) || !ISD::isUNINDEXEDLoad(LN0) ||
      !N0.hasOneUse() || !LN0->isSimple() ||
      !DstVT.isVector() || !DstVT.isPow2VectorType() ||
      !TLI.isVectorLoadExtDesirable(SDValue(N, 0)))
    return SDValue();

  SmallVector<SDNode *, 4> SetCCs;
  if (!ExtendUsesToFormExtLoad(DstVT, N, N0, N->getOpcode(), SetCCs, TLI))
    return SDValue();

  ISD::LoadExtType ExtType =
      N->getOpcode() == ISD::SIGN_EXTEND ? ISD::SEXTLOAD : ISD::ZEXTLOAD;

  // Try to split the vector types to get down to legal types.
  EVT SplitSrcVT = SrcVT;
  EVT SplitDstVT = DstVT;
  while (!TLI.isLoadExtLegalOrCustom(ExtType, SplitDstVT, SplitSrcVT) &&
         SplitSrcVT.getVectorNumElements() > 1) {
    SplitDstVT = DAG.GetSplitDestVTs(SplitDstVT).first;
    SplitSrcVT = DAG.GetSplitDestVTs(SplitSrcVT).first;
  }

  if (!TLI.isLoadExtLegalOrCustom(ExtType, SplitDstVT, SplitSrcVT))
    return SDValue();

  assert(!DstVT.isScalableVector() && "Unexpected scalable vector type");

  SDLoc DL(N);
  const unsigned NumSplits =
      DstVT.getVectorNumElements() / SplitDstVT.getVectorNumElements();
  const unsigned Stride = SplitSrcVT.getStoreSize();
  SmallVector<SDValue, 4> Loads;
  SmallVector<SDValue, 4> Chains;

  SDValue BasePtr = LN0->getBasePtr();
  for (unsigned Idx = 0; Idx < NumSplits; Idx++) {
    const unsigned Offset = Idx * Stride;
    const Align Align = commonAlignment(LN0->getAlign(), Offset);

    SDValue SplitLoad = DAG.getExtLoad(
        ExtType, SDLoc(LN0), SplitDstVT, LN0->getChain(), BasePtr,
        LN0->getPointerInfo().getWithOffset(Offset), SplitSrcVT, Align,
        LN0->getMemOperand()->getFlags(), LN0->getAAInfo());

    BasePtr = DAG.getMemBasePlusOffset(BasePtr, TypeSize::Fixed(Stride), DL);

    Loads.push_back(SplitLoad.getValue(0));
    Chains.push_back(SplitLoad.getValue(1));
  }

  SDValue NewChain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
  SDValue NewValue = DAG.getNode(ISD::CONCAT_VECTORS, DL, DstVT, Loads);

  // Simplify TF.
  AddToWorklist(NewChain.getNode());

  CombineTo(N, NewValue);

  // Replace uses of the original load (before extension)
  // with a truncate of the concatenated sextloaded vectors.
  SDValue Trunc =
      DAG.getNode(ISD::TRUNCATE, SDLoc(N0), N0.getValueType(), NewValue);
  ExtendSetCCUses(SetCCs, N0, NewValue, (ISD::NodeType)N->getOpcode());
  CombineTo(N0.getNode(), Trunc, NewChain);
  return SDValue(N, 0); // Return N so it doesn't get rechecked!
}

// fold (zext (and/or/xor (shl/shr (load x), cst), cst)) ->
//      (and/or/xor (shl/shr (zextload x), (zext cst)), (zext cst))
SDValue DAGCombiner::CombineZExtLogicopShiftLoad(SDNode *N) {
  assert(N->getOpcode() == ISD::ZERO_EXTEND);
  EVT VT = N->getValueType(0);
  EVT OrigVT = N->getOperand(0).getValueType();
  if (TLI.isZExtFree(OrigVT, VT))
    return SDValue();

  // and/or/xor
  SDValue N0 = N->getOperand(0);
  if (!(N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
        N0.getOpcode() == ISD::XOR) ||
      N0.getOperand(1).getOpcode() != ISD::Constant ||
      (LegalOperations && !TLI.isOperationLegal(N0.getOpcode(), VT)))
    return SDValue();

  // shl/shr
  SDValue N1 = N0->getOperand(0);
  if (!(N1.getOpcode() == ISD::SHL || N1.getOpcode() == ISD::SRL) ||
      N1.getOperand(1).getOpcode() != ISD::Constant ||
      (LegalOperations && !TLI.isOperationLegal(N1.getOpcode(), VT)))
    return SDValue();

  // load
  if (!isa<LoadSDNode>(N1.getOperand(0)))
    return SDValue();
  LoadSDNode *Load = cast<LoadSDNode>(N1.getOperand(0));
  EVT MemVT = Load->getMemoryVT();
  if (!TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, MemVT) ||
      Load->getExtensionType() == ISD::SEXTLOAD || Load->isIndexed())
    return SDValue();


  // If the shift op is SHL, the logic op must be AND, otherwise the result
  // will be wrong.
  if (N1.getOpcode() == ISD::SHL && N0.getOpcode() != ISD::AND)
    return SDValue();

  if (!N0.hasOneUse() || !N1.hasOneUse())
    return SDValue();

  SmallVector<SDNode*, 4> SetCCs;
  if (!ExtendUsesToFormExtLoad(VT, N1.getNode(), N1.getOperand(0),
                               ISD::ZERO_EXTEND, SetCCs, TLI))
    return SDValue();

  // Actually do the transformation.
  SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(Load), VT,
                                   Load->getChain(), Load->getBasePtr(),
                                   Load->getMemoryVT(), Load->getMemOperand());

  SDLoc DL1(N1);
  SDValue Shift = DAG.getNode(N1.getOpcode(), DL1, VT, ExtLoad,
                              N1.getOperand(1));

  APInt Mask = N0.getConstantOperandAPInt(1).zext(VT.getSizeInBits());
  SDLoc DL0(N0);
  SDValue And = DAG.getNode(N0.getOpcode(), DL0, VT, Shift,
                            DAG.getConstant(Mask, DL0, VT));

  ExtendSetCCUses(SetCCs, N1.getOperand(0), ExtLoad, ISD::ZERO_EXTEND);
  CombineTo(N, And);
  if (SDValue(Load, 0).hasOneUse()) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(Load, 1), ExtLoad.getValue(1));
  } else {
    SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(Load),
                                Load->getValueType(0), ExtLoad);
    CombineTo(Load, Trunc, ExtLoad.getValue(1));
  }

  // N0 is dead at this point.
  recursivelyDeleteUnusedNodes(N0.getNode());

  return SDValue(N,0); // Return N so it doesn't get rechecked!
}

/// If we're narrowing or widening the result of a vector select and the final
/// size is the same size as a setcc (compare) feeding the select, then try to
/// apply the cast operation to the select's operands because matching vector
/// sizes for a select condition and other operands should be more efficient.
SDValue DAGCombiner::matchVSelectOpSizesWithSetCC(SDNode *Cast) {
  unsigned CastOpcode = Cast->getOpcode();
  assert((CastOpcode == ISD::SIGN_EXTEND || CastOpcode == ISD::ZERO_EXTEND ||
          CastOpcode == ISD::TRUNCATE || CastOpcode == ISD::FP_EXTEND ||
          CastOpcode == ISD::FP_ROUND) &&
         "Unexpected opcode for vector select narrowing/widening");

  // We only do this transform before legal ops because the pattern may be
  // obfuscated by target-specific operations after legalization. Do not create
  // an illegal select op, however, because that may be difficult to lower.
  EVT VT = Cast->getValueType(0);
  if (LegalOperations || !TLI.isOperationLegalOrCustom(ISD::VSELECT, VT))
    return SDValue();

  SDValue VSel = Cast->getOperand(0);
  if (VSel.getOpcode() != ISD::VSELECT || !VSel.hasOneUse() ||
      VSel.getOperand(0).getOpcode() != ISD::SETCC)
    return SDValue();

  // Does the setcc have the same vector size as the casted select?
  SDValue SetCC = VSel.getOperand(0);
  EVT SetCCVT = getSetCCResultType(SetCC.getOperand(0).getValueType());
  if (SetCCVT.getSizeInBits() != VT.getSizeInBits())
    return SDValue();

  // cast (vsel (setcc X), A, B) --> vsel (setcc X), (cast A), (cast B)
  SDValue A = VSel.getOperand(1);
  SDValue B = VSel.getOperand(2);
  SDValue CastA, CastB;
  SDLoc DL(Cast);
  if (CastOpcode == ISD::FP_ROUND) {
    // FP_ROUND (fptrunc) has an extra flag operand to pass along.
    CastA = DAG.getNode(CastOpcode, DL, VT, A, Cast->getOperand(1));
    CastB = DAG.getNode(CastOpcode, DL, VT, B, Cast->getOperand(1));
  } else {
    CastA = DAG.getNode(CastOpcode, DL, VT, A);
    CastB = DAG.getNode(CastOpcode, DL, VT, B);
  }
  return DAG.getNode(ISD::VSELECT, DL, VT, SetCC, CastA, CastB);
}

// fold ([s|z]ext ([s|z]extload x)) -> ([s|z]ext (truncate ([s|z]extload x)))
// fold ([s|z]ext (     extload x)) -> ([s|z]ext (truncate ([s|z]extload x)))
static SDValue tryToFoldExtOfExtload(SelectionDAG &DAG, DAGCombiner &Combiner,
                                     const TargetLowering &TLI, EVT VT,
                                     bool LegalOperations, SDNode *N,
                                     SDValue N0, ISD::LoadExtType ExtLoadType) {
  SDNode *N0Node = N0.getNode();
  bool isAExtLoad = (ExtLoadType == ISD::SEXTLOAD) ? ISD::isSEXTLoad(N0Node)
                                                   : ISD::isZEXTLoad(N0Node);
  if ((!isAExtLoad && !ISD::isEXTLoad(N0Node)) ||
      !ISD::isUNINDEXEDLoad(N0Node) || !N0.hasOneUse())
    return SDValue();

  LoadSDNode *LN0 = cast<LoadSDNode>(N0);
  EVT MemVT = LN0->getMemoryVT();
  if ((LegalOperations || !LN0->isSimple() ||
       VT.isVector()) &&
      !TLI.isLoadExtLegal(ExtLoadType, VT, MemVT))
    return SDValue();

  SDValue ExtLoad =
      DAG.getExtLoad(ExtLoadType, SDLoc(LN0), VT, LN0->getChain(),
                     LN0->getBasePtr(), MemVT, LN0->getMemOperand());
  Combiner.CombineTo(N, ExtLoad);
  DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 1), ExtLoad.getValue(1));
  if (LN0->use_empty())
    Combiner.recursivelyDeleteUnusedNodes(LN0);
  return SDValue(N, 0); // Return N so it doesn't get rechecked!
}

// fold ([s|z]ext (load x)) -> ([s|z]ext (truncate ([s|z]extload x)))
// Only generate vector extloads when 1) they're legal, and 2) they are
// deemed desirable by the target.
static SDValue tryToFoldExtOfLoad(SelectionDAG &DAG, DAGCombiner &Combiner,
                                  const TargetLowering &TLI, EVT VT,
                                  bool LegalOperations, SDNode *N, SDValue N0,
                                  ISD::LoadExtType ExtLoadType,
                                  ISD::NodeType ExtOpc) {
  if (!ISD::isNON_EXTLoad(N0.getNode()) ||
      !ISD::isUNINDEXEDLoad(N0.getNode()) ||
      ((LegalOperations || VT.isVector() ||
        !cast<LoadSDNode>(N0)->isSimple()) &&
       !TLI.isLoadExtLegal(ExtLoadType, VT, N0.getValueType())))
    return {};

  bool DoXform = true;
  SmallVector<SDNode *, 4> SetCCs;
  if (!N0.hasOneUse())
    DoXform = ExtendUsesToFormExtLoad(VT, N, N0, ExtOpc, SetCCs, TLI);
  if (VT.isVector())
    DoXform &= TLI.isVectorLoadExtDesirable(SDValue(N, 0));
  if (!DoXform)
    return {};

  LoadSDNode *LN0 = cast<LoadSDNode>(N0);
  SDValue ExtLoad = DAG.getExtLoad(ExtLoadType, SDLoc(LN0), VT, LN0->getChain(),
                                   LN0->getBasePtr(), N0.getValueType(),
                                   LN0->getMemOperand());
  Combiner.ExtendSetCCUses(SetCCs, N0, ExtLoad, ExtOpc);
  // If the load value is used only by N, replace it via CombineTo N.
  bool NoReplaceTrunc = SDValue(LN0, 0).hasOneUse();
  Combiner.CombineTo(N, ExtLoad);
  if (NoReplaceTrunc) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 1), ExtLoad.getValue(1));
    Combiner.recursivelyDeleteUnusedNodes(LN0);
  } else {
    SDValue Trunc =
        DAG.getNode(ISD::TRUNCATE, SDLoc(N0), N0.getValueType(), ExtLoad);
    Combiner.CombineTo(LN0, Trunc, ExtLoad.getValue(1));
  }
  return SDValue(N, 0); // Return N so it doesn't get rechecked!
}

static SDValue tryToFoldExtOfMaskedLoad(SelectionDAG &DAG,
                                        const TargetLowering &TLI, EVT VT,
                                        SDNode *N, SDValue N0,
                                        ISD::LoadExtType ExtLoadType,
                                        ISD::NodeType ExtOpc) {
  if (!N0.hasOneUse())
    return SDValue();

  MaskedLoadSDNode *Ld = dyn_cast<MaskedLoadSDNode>(N0);
  if (!Ld || Ld->getExtensionType() != ISD::NON_EXTLOAD)
    return SDValue();

  if (!TLI.isLoadExtLegal(ExtLoadType, VT, Ld->getValueType(0)))
    return SDValue();

  if (!TLI.isVectorLoadExtDesirable(SDValue(N, 0)))
    return SDValue();

  SDLoc dl(Ld);
  SDValue PassThru = DAG.getNode(ExtOpc, dl, VT, Ld->getPassThru());
  SDValue NewLoad = DAG.getMaskedLoad(
      VT, dl, Ld->getChain(), Ld->getBasePtr(), Ld->getOffset(), Ld->getMask(),
      PassThru, Ld->getMemoryVT(), Ld->getMemOperand(), Ld->getAddressingMode(),
      ExtLoadType, Ld->isExpandingLoad());
  DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1), SDValue(NewLoad.getNode(), 1));
  return NewLoad;
}

static SDValue foldExtendedSignBitTest(SDNode *N, SelectionDAG &DAG,
                                       bool LegalOperations) {
  assert((N->getOpcode() == ISD::SIGN_EXTEND ||
          N->getOpcode() == ISD::ZERO_EXTEND) && "Expected sext or zext");

  SDValue SetCC = N->getOperand(0);
  if (LegalOperations || SetCC.getOpcode() != ISD::SETCC ||
      !SetCC.hasOneUse() || SetCC.getValueType() != MVT::i1)
    return SDValue();

  SDValue X = SetCC.getOperand(0);
  SDValue Ones = SetCC.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(SetCC.getOperand(2))->get();
  EVT VT = N->getValueType(0);
  EVT XVT = X.getValueType();
  // setge X, C is canonicalized to setgt, so we do not need to match that
  // pattern. The setlt sibling is folded in SimplifySelectCC() because it does
  // not require the 'not' op.
  if (CC == ISD::SETGT && isAllOnesConstant(Ones) && VT == XVT) {
    // Invert and smear/shift the sign bit:
    // sext i1 (setgt iN X, -1) --> sra (not X), (N - 1)
    // zext i1 (setgt iN X, -1) --> srl (not X), (N - 1)
    SDLoc DL(N);
    unsigned ShCt = VT.getSizeInBits() - 1;
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    if (!TLI.shouldAvoidTransformToShift(VT, ShCt)) {
      SDValue NotX = DAG.getNOT(DL, X, VT);
      SDValue ShiftAmount = DAG.getConstant(ShCt, DL, VT);
      auto ShiftOpcode =
        N->getOpcode() == ISD::SIGN_EXTEND ? ISD::SRA : ISD::SRL;
      return DAG.getNode(ShiftOpcode, DL, VT, NotX, ShiftAmount);
    }
  }
  return SDValue();
}

SDValue DAGCombiner::foldSextSetcc(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  if (N0.getOpcode() != ISD::SETCC)
    return SDValue();

  SDValue N00 = N0.getOperand(0);
  SDValue N01 = N0.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
  EVT VT = N->getValueType(0);
  EVT N00VT = N00.getValueType();
  SDLoc DL(N);

  // On some architectures (such as SSE/NEON/etc) the SETCC result type is
  // the same size as the compared operands. Try to optimize sext(setcc())
  // if this is the case.
  if (VT.isVector() && !LegalOperations &&
      TLI.getBooleanContents(N00VT) ==
          TargetLowering::ZeroOrNegativeOneBooleanContent) {
    EVT SVT = getSetCCResultType(N00VT);

    // If we already have the desired type, don't change it.
    if (SVT != N0.getValueType()) {
      // We know that the # elements of the results is the same as the
      // # elements of the compare (and the # elements of the compare result
      // for that matter).  Check to see that they are the same size.  If so,
      // we know that the element size of the sext'd result matches the
      // element size of the compare operands.
      if (VT.getSizeInBits() == SVT.getSizeInBits())
        return DAG.getSetCC(DL, VT, N00, N01, CC);

      // If the desired elements are smaller or larger than the source
      // elements, we can use a matching integer vector type and then
      // truncate/sign extend.
      EVT MatchingVecType = N00VT.changeVectorElementTypeToInteger();
      if (SVT == MatchingVecType) {
        SDValue VsetCC = DAG.getSetCC(DL, MatchingVecType, N00, N01, CC);
        return DAG.getSExtOrTrunc(VsetCC, DL, VT);
      }
    }

    // Try to eliminate the sext of a setcc by zexting the compare operands.
    if (N0.hasOneUse() && TLI.isOperationLegalOrCustom(ISD::SETCC, VT) &&
        !TLI.isOperationLegalOrCustom(ISD::SETCC, SVT)) {
      bool IsSignedCmp = ISD::isSignedIntSetCC(CC);
      unsigned LoadOpcode = IsSignedCmp ? ISD::SEXTLOAD : ISD::ZEXTLOAD;
      unsigned ExtOpcode = IsSignedCmp ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;

      // We have an unsupported narrow vector compare op that would be legal
      // if extended to the destination type. See if the compare operands
      // can be freely extended to the destination type.
      auto IsFreeToExtend = [&](SDValue V) {
        if (isConstantOrConstantVector(V, /*NoOpaques*/ true))
          return true;
        // Match a simple, non-extended load that can be converted to a
        // legal {z/s}ext-load.
        // TODO: Allow widening of an existing {z/s}ext-load?
        if (!(ISD::isNON_EXTLoad(V.getNode()) &&
              ISD::isUNINDEXEDLoad(V.getNode()) &&
              cast<LoadSDNode>(V)->isSimple() &&
              TLI.isLoadExtLegal(LoadOpcode, VT, V.getValueType())))
          return false;

        // Non-chain users of this value must either be the setcc in this
        // sequence or extends that can be folded into the new {z/s}ext-load.
        for (SDNode::use_iterator UI = V->use_begin(), UE = V->use_end();
             UI != UE; ++UI) {
          // Skip uses of the chain and the setcc.
          SDNode *User = *UI;
          if (UI.getUse().getResNo() != 0 || User == N0.getNode())
            continue;
          // Extra users must have exactly the same cast we are about to create.
          // TODO: This restriction could be eased if ExtendUsesToFormExtLoad()
          //       is enhanced similarly.
          if (User->getOpcode() != ExtOpcode || User->getValueType(0) != VT)
            return false;
        }
        return true;
      };

      if (IsFreeToExtend(N00) && IsFreeToExtend(N01)) {
        SDValue Ext0 = DAG.getNode(ExtOpcode, DL, VT, N00);
        SDValue Ext1 = DAG.getNode(ExtOpcode, DL, VT, N01);
        return DAG.getSetCC(DL, VT, Ext0, Ext1, CC);
      }
    }
  }

  // sext(setcc x, y, cc) -> (select (setcc x, y, cc), T, 0)
  // Here, T can be 1 or -1, depending on the type of the setcc and
  // getBooleanContents().
  unsigned SetCCWidth = N0.getScalarValueSizeInBits();

  // To determine the "true" side of the select, we need to know the high bit
  // of the value returned by the setcc if it evaluates to true.
  // If the type of the setcc is i1, then the true case of the select is just
  // sext(i1 1), that is, -1.
  // If the type of the setcc is larger (say, i8) then the value of the high
  // bit depends on getBooleanContents(), so ask TLI for a real "true" value
  // of the appropriate width.
  SDValue ExtTrueVal = (SetCCWidth == 1)
                           ? DAG.getAllOnesConstant(DL, VT)
                           : DAG.getBoolConstant(true, DL, VT, N00VT);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  if (SDValue SCC = SimplifySelectCC(DL, N00, N01, ExtTrueVal, Zero, CC, true))
    return SCC;

  if (!VT.isVector() && !TLI.convertSelectOfConstantsToMath(VT)) {
    EVT SetCCVT = getSetCCResultType(N00VT);
    // Don't do this transform for i1 because there's a select transform
    // that would reverse it.
    // TODO: We should not do this transform at all without a target hook
    // because a sext is likely cheaper than a select?
    if (SetCCVT.getScalarSizeInBits() != 1 &&
        (!LegalOperations || TLI.isOperationLegal(ISD::SETCC, N00VT))) {
      SDValue SetCC = DAG.getSetCC(DL, SetCCVT, N00, N01, CC);
      return DAG.getSelect(DL, VT, SetCC, ExtTrueVal, Zero);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  if (SDValue Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes))
    return Res;

  // fold (sext (sext x)) -> (sext x)
  // fold (sext (aext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, DL, VT, N0.getOperand(0));

  if (N0.getOpcode() == ISD::TRUNCATE) {
    // fold (sext (truncate (load x))) -> (sext (smaller load x))
    // fold (sext (truncate (srl (load x), c))) -> (sext (smaller load (x+c/n)))
    if (SDValue NarrowLoad = ReduceLoadWidth(N0.getNode())) {
      SDNode *oye = N0.getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }

    // See if the value being truncated is already sign extended.  If so, just
    // eliminate the trunc/sext pair.
    SDValue Op = N0.getOperand(0);
    unsigned OpBits   = Op.getScalarValueSizeInBits();
    unsigned MidBits  = N0.getScalarValueSizeInBits();
    unsigned DestBits = VT.getScalarSizeInBits();
    unsigned NumSignBits = DAG.ComputeNumSignBits(Op);

    if (OpBits == DestBits) {
      // Op is i32, Mid is i8, and Dest is i32.  If Op has more than 24 sign
      // bits, it is already ready.
      if (NumSignBits > DestBits-MidBits)
        return Op;
    } else if (OpBits < DestBits) {
      // Op is i32, Mid is i8, and Dest is i64.  If Op has more than 24 sign
      // bits, just sext from i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::SIGN_EXTEND, DL, VT, Op);
    } else {
      // Op is i64, Mid is i8, and Dest is i32.  If Op has more than 56 sign
      // bits, just truncate to i32.
      if (NumSignBits > OpBits-MidBits)
        return DAG.getNode(ISD::TRUNCATE, DL, VT, Op);
    }

    // fold (sext (truncate x)) -> (sextinreg x).
    if (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND_INREG,
                                                 N0.getValueType())) {
      if (OpBits < DestBits)
        Op = DAG.getNode(ISD::ANY_EXTEND, SDLoc(N0), VT, Op);
      else if (OpBits > DestBits)
        Op = DAG.getNode(ISD::TRUNCATE, SDLoc(N0), VT, Op);
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, Op,
                         DAG.getValueType(N0.getValueType()));
    }
  }

  // Try to simplify (sext (load x)).
  if (SDValue foldedExt =
          tryToFoldExtOfLoad(DAG, *this, TLI, VT, LegalOperations, N, N0,
                             ISD::SEXTLOAD, ISD::SIGN_EXTEND))
    return foldedExt;

  if (SDValue foldedExt =
      tryToFoldExtOfMaskedLoad(DAG, TLI, VT, N, N0, ISD::SEXTLOAD,
                               ISD::SIGN_EXTEND))
    return foldedExt;

  // fold (sext (load x)) to multiple smaller sextloads.
  // Only on illegal but splittable vectors.
  if (SDValue ExtLoad = CombineExtLoad(N))
    return ExtLoad;

  // Try to simplify (sext (sextload x)).
  if (SDValue foldedExt = tryToFoldExtOfExtload(
          DAG, *this, TLI, VT, LegalOperations, N, N0, ISD::SEXTLOAD))
    return foldedExt;

  // fold (sext (and/or/xor (load x), cst)) ->
  //      (and/or/xor (sextload x), (sext cst))
  if ((N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
       N0.getOpcode() == ISD::XOR) &&
      isa<LoadSDNode>(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      (!LegalOperations && TLI.isOperationLegal(N0.getOpcode(), VT))) {
    LoadSDNode *LN00 = cast<LoadSDNode>(N0.getOperand(0));
    EVT MemVT = LN00->getMemoryVT();
    if (TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, MemVT) &&
      LN00->getExtensionType() != ISD::ZEXTLOAD && LN00->isUnindexed()) {
      SmallVector<SDNode*, 4> SetCCs;
      bool DoXform = ExtendUsesToFormExtLoad(VT, N0.getNode(), N0.getOperand(0),
                                             ISD::SIGN_EXTEND, SetCCs, TLI);
      if (DoXform) {
        SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(LN00), VT,
                                         LN00->getChain(), LN00->getBasePtr(),
                                         LN00->getMemoryVT(),
                                         LN00->getMemOperand());
        APInt Mask = N0.getConstantOperandAPInt(1).sext(VT.getSizeInBits());
        SDValue And = DAG.getNode(N0.getOpcode(), DL, VT,
                                  ExtLoad, DAG.getConstant(Mask, DL, VT));
        ExtendSetCCUses(SetCCs, N0.getOperand(0), ExtLoad, ISD::SIGN_EXTEND);
        bool NoReplaceTruncAnd = !N0.hasOneUse();
        bool NoReplaceTrunc = SDValue(LN00, 0).hasOneUse();
        CombineTo(N, And);
        // If N0 has multiple uses, change other uses as well.
        if (NoReplaceTruncAnd) {
          SDValue TruncAnd =
              DAG.getNode(ISD::TRUNCATE, DL, N0.getValueType(), And);
          CombineTo(N0.getNode(), TruncAnd);
        }
        if (NoReplaceTrunc) {
          DAG.ReplaceAllUsesOfValueWith(SDValue(LN00, 1), ExtLoad.getValue(1));
        } else {
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(LN00),
                                      LN00->getValueType(0), ExtLoad);
          CombineTo(LN00, Trunc, ExtLoad.getValue(1));
        }
        return SDValue(N,0); // Return N so it doesn't get rechecked!
      }
    }
  }

  if (SDValue V = foldExtendedSignBitTest(N, DAG, LegalOperations))
    return V;

  if (SDValue V = foldSextSetcc(N))
    return V;

  // fold (sext x) -> (zext x) if the sign bit is known zero.
  if ((!LegalOperations || TLI.isOperationLegal(ISD::ZERO_EXTEND, VT)) &&
      DAG.SignBitIsZero(N0))
    return DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0);

  if (SDValue NewVSel = matchVSelectOpSizesWithSetCC(N))
    return NewVSel;

  // Eliminate this sign extend by doing a negation in the destination type:
  // sext i32 (0 - (zext i8 X to i32)) to i64 --> 0 - (zext i8 X to i64)
  if (N0.getOpcode() == ISD::SUB && N0.hasOneUse() &&
      isNullOrNullSplat(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::ZERO_EXTEND &&
      TLI.isOperationLegalOrCustom(ISD::SUB, VT)) {
    SDValue Zext = DAG.getZExtOrTrunc(N0.getOperand(1).getOperand(0), DL, VT);
    return DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT), Zext);
  }
  // Eliminate this sign extend by doing a decrement in the destination type:
  // sext i32 ((zext i8 X to i32) + (-1)) to i64 --> (zext i8 X to i64) + (-1)
  if (N0.getOpcode() == ISD::ADD && N0.hasOneUse() &&
      isAllOnesOrAllOnesSplat(N0.getOperand(1)) &&
      N0.getOperand(0).getOpcode() == ISD::ZERO_EXTEND &&
      TLI.isOperationLegalOrCustom(ISD::ADD, VT)) {
    SDValue Zext = DAG.getZExtOrTrunc(N0.getOperand(0).getOperand(0), DL, VT);
    return DAG.getNode(ISD::ADD, DL, VT, Zext, DAG.getAllOnesConstant(DL, VT));
  }

  // fold sext (not i1 X) -> add (zext i1 X), -1
  // TODO: This could be extended to handle bool vectors.
  if (N0.getValueType() == MVT::i1 && isBitwiseNot(N0) && N0.hasOneUse() &&
      (!LegalOperations || (TLI.isOperationLegal(ISD::ZERO_EXTEND, VT) &&
                            TLI.isOperationLegal(ISD::ADD, VT)))) {
    // If we can eliminate the 'not', the sext form should be better
    if (SDValue NewXor = visitXOR(N0.getNode())) {
      // Returning N0 is a form of in-visit replacement that may have
      // invalidated N0.
      if (NewXor.getNode() == N0.getNode()) {
        // Return SDValue here as the xor should have already been replaced in
        // this sext.
        return SDValue();
      } else {
        // Return a new sext with the new xor.
        return DAG.getNode(ISD::SIGN_EXTEND, DL, VT, NewXor);
      }
    }

    SDValue Zext = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0));
    return DAG.getNode(ISD::ADD, DL, VT, Zext, DAG.getAllOnesConstant(DL, VT));
  }

  if (SDValue Res = tryToFoldExtendSelectLoad(N, TLI, DAG))
    return Res;

  return SDValue();
}

// isTruncateOf - If N is a truncate of some other value, return true, record
// the value being truncated in Op and which of Op's bits are zero/one in Known.
// This function computes KnownBits to avoid a duplicated call to
// computeKnownBits in the caller.
static bool isTruncateOf(SelectionDAG &DAG, SDValue N, SDValue &Op,
                         KnownBits &Known) {
  if (N->getOpcode() == ISD::TRUNCATE) {
    Op = N->getOperand(0);
    Known = DAG.computeKnownBits(Op);
    return true;
  }

  if (N.getOpcode() != ISD::SETCC ||
      N.getValueType().getScalarType() != MVT::i1 ||
      cast<CondCodeSDNode>(N.getOperand(2))->get() != ISD::SETNE)
    return false;

  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  assert(Op0.getValueType() == Op1.getValueType());

  if (isNullOrNullSplat(Op0))
    Op = Op1;
  else if (isNullOrNullSplat(Op1))
    Op = Op0;
  else
    return false;

  Known = DAG.computeKnownBits(Op);

  return (Known.Zero | 1).isAllOnesValue();
}

/// Given an extending node with a pop-count operand, if the target does not
/// support a pop-count in the narrow source type but does support it in the
/// destination type, widen the pop-count to the destination type.
static SDValue widenCtPop(SDNode *Extend, SelectionDAG &DAG) {
  assert((Extend->getOpcode() == ISD::ZERO_EXTEND ||
          Extend->getOpcode() == ISD::ANY_EXTEND) && "Expected extend op");

  SDValue CtPop = Extend->getOperand(0);
  if (CtPop.getOpcode() != ISD::CTPOP || !CtPop.hasOneUse())
    return SDValue();

  EVT VT = Extend->getValueType(0);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.isOperationLegalOrCustom(ISD::CTPOP, CtPop.getValueType()) ||
      !TLI.isOperationLegalOrCustom(ISD::CTPOP, VT))
    return SDValue();

  // zext (ctpop X) --> ctpop (zext X)
  SDLoc DL(Extend);
  SDValue NewZext = DAG.getZExtOrTrunc(CtPop.getOperand(0), DL, VT);
  return DAG.getNode(ISD::CTPOP, DL, VT, NewZext);
}

SDValue DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (SDValue Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes))
    return Res;

  // fold (zext (zext x)) -> (zext x)
  // fold (zext (aext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N), VT,
                       N0.getOperand(0));

  // fold (zext (truncate x)) -> (zext x) or
  //      (zext (truncate x)) -> (truncate x)
  // This is valid when the truncated bits of x are already zero.
  SDValue Op;
  KnownBits Known;
  if (isTruncateOf(DAG, N0, Op, Known)) {
    APInt TruncatedBits =
      (Op.getScalarValueSizeInBits() == N0.getScalarValueSizeInBits()) ?
      APInt(Op.getScalarValueSizeInBits(), 0) :
      APInt::getBitsSet(Op.getScalarValueSizeInBits(),
                        N0.getScalarValueSizeInBits(),
                        std::min(Op.getScalarValueSizeInBits(),
                                 VT.getScalarSizeInBits()));
    if (TruncatedBits.isSubsetOf(Known.Zero))
      return DAG.getZExtOrTrunc(Op, SDLoc(N), VT);
  }

  // fold (zext (truncate x)) -> (and x, mask)
  if (N0.getOpcode() == ISD::TRUNCATE) {
    // fold (zext (truncate (load x))) -> (zext (smaller load x))
    // fold (zext (truncate (srl (load x), c))) -> (zext (smaller load (x+c/n)))
    if (SDValue NarrowLoad = ReduceLoadWidth(N0.getNode())) {
      SDNode *oye = N0.getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }

    EVT SrcVT = N0.getOperand(0).getValueType();
    EVT MinVT = N0.getValueType();

    // Try to mask before the extension to avoid having to generate a larger mask,
    // possibly over several sub-vectors.
    if (SrcVT.bitsLT(VT) && VT.isVector()) {
      if (!LegalOperations || (TLI.isOperationLegal(ISD::AND, SrcVT) &&
                               TLI.isOperationLegal(ISD::ZERO_EXTEND, VT))) {
        SDValue Op = N0.getOperand(0);
        Op = DAG.getZeroExtendInReg(Op, SDLoc(N), MinVT);
        AddToWorklist(Op.getNode());
        SDValue ZExtOrTrunc = DAG.getZExtOrTrunc(Op, SDLoc(N), VT);
        // Transfer the debug info; the new node is equivalent to N0.
        DAG.transferDbgValues(N0, ZExtOrTrunc);
        return ZExtOrTrunc;
      }
    }

    if (!LegalOperations || TLI.isOperationLegal(ISD::AND, VT)) {
      SDValue Op = DAG.getAnyExtOrTrunc(N0.getOperand(0), SDLoc(N), VT);
      AddToWorklist(Op.getNode());
      SDValue And = DAG.getZeroExtendInReg(Op, SDLoc(N), MinVT);
      // We may safely transfer the debug info describing the truncate node over
      // to the equivalent and operation.
      DAG.transferDbgValues(N0, And);
      return And;
    }
  }

  // Fold (zext (and (trunc x), cst)) -> (and x, cst),
  // if either of the casts is not free.
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      (!TLI.isTruncateFree(N0.getOperand(0).getOperand(0).getValueType(),
                           N0.getValueType()) ||
       !TLI.isZExtFree(N0.getValueType(), VT))) {
    SDValue X = N0.getOperand(0).getOperand(0);
    X = DAG.getAnyExtOrTrunc(X, SDLoc(X), VT);
    APInt Mask = N0.getConstantOperandAPInt(1).zext(VT.getSizeInBits());
    SDLoc DL(N);
    return DAG.getNode(ISD::AND, DL, VT,
                       X, DAG.getConstant(Mask, DL, VT));
  }

  // Try to simplify (zext (load x)).
  if (SDValue foldedExt =
          tryToFoldExtOfLoad(DAG, *this, TLI, VT, LegalOperations, N, N0,
                             ISD::ZEXTLOAD, ISD::ZERO_EXTEND))
    return foldedExt;

  if (SDValue foldedExt =
      tryToFoldExtOfMaskedLoad(DAG, TLI, VT, N, N0, ISD::ZEXTLOAD,
                               ISD::ZERO_EXTEND))
    return foldedExt;

  // fold (zext (load x)) to multiple smaller zextloads.
  // Only on illegal but splittable vectors.
  if (SDValue ExtLoad = CombineExtLoad(N))
    return ExtLoad;

  // fold (zext (and/or/xor (load x), cst)) ->
  //      (and/or/xor (zextload x), (zext cst))
  // Unless (and (load x) cst) will match as a zextload already and has
  // additional users.
  if ((N0.getOpcode() == ISD::AND || N0.getOpcode() == ISD::OR ||
       N0.getOpcode() == ISD::XOR) &&
      isa<LoadSDNode>(N0.getOperand(0)) &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      (!LegalOperations && TLI.isOperationLegal(N0.getOpcode(), VT))) {
    LoadSDNode *LN00 = cast<LoadSDNode>(N0.getOperand(0));
    EVT MemVT = LN00->getMemoryVT();
    if (TLI.isLoadExtLegal(ISD::ZEXTLOAD, VT, MemVT) &&
        LN00->getExtensionType() != ISD::SEXTLOAD && LN00->isUnindexed()) {
      bool DoXform = true;
      SmallVector<SDNode*, 4> SetCCs;
      if (!N0.hasOneUse()) {
        if (N0.getOpcode() == ISD::AND) {
          auto *AndC = cast<ConstantSDNode>(N0.getOperand(1));
          EVT LoadResultTy = AndC->getValueType(0);
          EVT ExtVT;
          if (isAndLoadExtLoad(AndC, LN00, LoadResultTy, ExtVT))
            DoXform = false;
        }
      }
      if (DoXform)
        DoXform = ExtendUsesToFormExtLoad(VT, N0.getNode(), N0.getOperand(0),
                                          ISD::ZERO_EXTEND, SetCCs, TLI);
      if (DoXform) {
        SDValue ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, SDLoc(LN00), VT,
                                         LN00->getChain(), LN00->getBasePtr(),
                                         LN00->getMemoryVT(),
                                         LN00->getMemOperand());
        APInt Mask = N0.getConstantOperandAPInt(1).zext(VT.getSizeInBits());
        SDLoc DL(N);
        SDValue And = DAG.getNode(N0.getOpcode(), DL, VT,
                                  ExtLoad, DAG.getConstant(Mask, DL, VT));
        ExtendSetCCUses(SetCCs, N0.getOperand(0), ExtLoad, ISD::ZERO_EXTEND);
        bool NoReplaceTruncAnd = !N0.hasOneUse();
        bool NoReplaceTrunc = SDValue(LN00, 0).hasOneUse();
        CombineTo(N, And);
        // If N0 has multiple uses, change other uses as well.
        if (NoReplaceTruncAnd) {
          SDValue TruncAnd =
              DAG.getNode(ISD::TRUNCATE, DL, N0.getValueType(), And);
          CombineTo(N0.getNode(), TruncAnd);
        }
        if (NoReplaceTrunc) {
          DAG.ReplaceAllUsesOfValueWith(SDValue(LN00, 1), ExtLoad.getValue(1));
        } else {
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SDLoc(LN00),
                                      LN00->getValueType(0), ExtLoad);
          CombineTo(LN00, Trunc, ExtLoad.getValue(1));
        }
        return SDValue(N,0); // Return N so it doesn't get rechecked!
      }
    }
  }

  // fold (zext (and/or/xor (shl/shr (load x), cst), cst)) ->
  //      (and/or/xor (shl/shr (zextload x), (zext cst)), (zext cst))
  if (SDValue ZExtLoad = CombineZExtLogicopShiftLoad(N))
    return ZExtLoad;

  // Try to simplify (zext (zextload x)).
  if (SDValue foldedExt = tryToFoldExtOfExtload(
          DAG, *this, TLI, VT, LegalOperations, N, N0, ISD::ZEXTLOAD))
    return foldedExt;

  if (SDValue V = foldExtendedSignBitTest(N, DAG, LegalOperations))
    return V;

  if (N0.getOpcode() == ISD::SETCC) {
    // Only do this before legalize for now.
    if (!LegalOperations && VT.isVector() &&
        N0.getValueType().getVectorElementType() == MVT::i1) {
      EVT N00VT = N0.getOperand(0).getValueType();
      if (getSetCCResultType(N00VT) == N0.getValueType())
        return SDValue();

      // We know that the # elements of the results is the same as the #
      // elements of the compare (and the # elements of the compare result for
      // that matter). Check to see that they are the same size. If so, we know
      // that the element size of the sext'd result matches the element size of
      // the compare operands.
      SDLoc DL(N);
      if (VT.getSizeInBits() == N00VT.getSizeInBits()) {
        // zext(setcc) -> zext_in_reg(vsetcc) for vectors.
        SDValue VSetCC = DAG.getNode(ISD::SETCC, DL, VT, N0.getOperand(0),
                                     N0.getOperand(1), N0.getOperand(2));
        return DAG.getZeroExtendInReg(VSetCC, DL, N0.getValueType());
      }

      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/any extend followed by zext_in_reg.
      EVT MatchingVectorType = N00VT.changeVectorElementTypeToInteger();
      SDValue VsetCC =
          DAG.getNode(ISD::SETCC, DL, MatchingVectorType, N0.getOperand(0),
                      N0.getOperand(1), N0.getOperand(2));
      return DAG.getZeroExtendInReg(DAG.getAnyExtOrTrunc(VsetCC, DL, VT), DL,
                                    N0.getValueType());
    }

    // zext(setcc x,y,cc) -> zext(select x, y, true, false, cc)
    SDLoc DL(N);
    EVT N0VT = N0.getValueType();
    EVT N00VT = N0.getOperand(0).getValueType();
    if (SDValue SCC = SimplifySelectCC(
            DL, N0.getOperand(0), N0.getOperand(1),
            DAG.getBoolConstant(true, DL, N0VT, N00VT),
            DAG.getBoolConstant(false, DL, N0VT, N00VT),
            cast<CondCodeSDNode>(N0.getOperand(2))->get(), true))
      return DAG.getNode(ISD::ZERO_EXTEND, DL, VT, SCC);
  }

  // (zext (shl (zext x), cst)) -> (shl (zext x), cst)
  if ((N0.getOpcode() == ISD::SHL || N0.getOpcode() == ISD::SRL) &&
      isa<ConstantSDNode>(N0.getOperand(1)) &&
      N0.getOperand(0).getOpcode() == ISD::ZERO_EXTEND &&
      N0.hasOneUse()) {
    SDValue ShAmt = N0.getOperand(1);
    if (N0.getOpcode() == ISD::SHL) {
      SDValue InnerZExt = N0.getOperand(0);
      // If the original shl may be shifting out bits, do not perform this
      // transformation.
      unsigned KnownZeroBits = InnerZExt.getValueSizeInBits() -
        InnerZExt.getOperand(0).getValueSizeInBits();
      if (cast<ConstantSDNode>(ShAmt)->getAPIntValue().ugt(KnownZeroBits))
        return SDValue();
    }

    SDLoc DL(N);

    // Ensure that the shift amount is wide enough for the shifted value.
    if (Log2_32_Ceil(VT.getSizeInBits()) > ShAmt.getValueSizeInBits())
      ShAmt = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, ShAmt);

    return DAG.getNode(N0.getOpcode(), DL, VT,
                       DAG.getNode(ISD::ZERO_EXTEND, DL, VT, N0.getOperand(0)),
                       ShAmt);
  }

  if (SDValue NewVSel = matchVSelectOpSizesWithSetCC(N))
    return NewVSel;

  if (SDValue NewCtPop = widenCtPop(N, DAG))
    return NewCtPop;

  if (SDValue Res = tryToFoldExtendSelectLoad(N, TLI, DAG))
    return Res;

  return SDValue();
}

SDValue DAGCombiner::visitANY_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (SDValue Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes))
    return Res;

  // fold (aext (aext x)) -> (aext x)
  // fold (aext (zext x)) -> (zext x)
  // fold (aext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::ANY_EXTEND  ||
      N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, N0.getOperand(0));

  // fold (aext (truncate (load x))) -> (aext (smaller load x))
  // fold (aext (truncate (srl (load x), c))) -> (aext (small load (x+c/n)))
  if (N0.getOpcode() == ISD::TRUNCATE) {
    if (SDValue NarrowLoad = ReduceLoadWidth(N0.getNode())) {
      SDNode *oye = N0.getOperand(0).getNode();
      if (NarrowLoad.getNode() != N0.getNode()) {
        CombineTo(N0.getNode(), NarrowLoad);
        // CombineTo deleted the truncate, if needed, but not what's under it.
        AddToWorklist(oye);
      }
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  // fold (aext (truncate x))
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getAnyExtOrTrunc(N0.getOperand(0), SDLoc(N), VT);

  // Fold (aext (and (trunc x), cst)) -> (and x, cst)
  // if the trunc is not free.
  if (N0.getOpcode() == ISD::AND &&
      N0.getOperand(0).getOpcode() == ISD::TRUNCATE &&
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      !TLI.isTruncateFree(N0.getOperand(0).getOperand(0).getValueType(),
                          N0.getValueType())) {
    SDLoc DL(N);
    SDValue X = N0.getOperand(0).getOperand(0);
    X = DAG.getAnyExtOrTrunc(X, DL, VT);
    APInt Mask = N0.getConstantOperandAPInt(1).zext(VT.getSizeInBits());
    return DAG.getNode(ISD::AND, DL, VT,
                       X, DAG.getConstant(Mask, DL, VT));
  }

  // fold (aext (load x)) -> (aext (truncate (extload x)))
  // None of the supported targets knows how to perform load and any_ext
  // on vectors in one instruction, so attempt to fold to zext instead.
  if (VT.isVector()) {
    // Try to simplify (zext (load x)).
    if (SDValue foldedExt =
            tryToFoldExtOfLoad(DAG, *this, TLI, VT, LegalOperations, N, N0,
                               ISD::ZEXTLOAD, ISD::ZERO_EXTEND))
      return foldedExt;
  } else if (ISD::isNON_EXTLoad(N0.getNode()) &&
             ISD::isUNINDEXEDLoad(N0.getNode()) &&
             TLI.isLoadExtLegal(ISD::EXTLOAD, VT, N0.getValueType())) {
    bool DoXform = true;
    SmallVector<SDNode *, 4> SetCCs;
    if (!N0.hasOneUse())
      DoXform =
          ExtendUsesToFormExtLoad(VT, N, N0, ISD::ANY_EXTEND, SetCCs, TLI);
    if (DoXform) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, SDLoc(N), VT,
                                       LN0->getChain(), LN0->getBasePtr(),
                                       N0.getValueType(), LN0->getMemOperand());
      ExtendSetCCUses(SetCCs, N0, ExtLoad, ISD::ANY_EXTEND);
      // If the load value is used only by N, replace it via CombineTo N.
      bool NoReplaceTrunc = N0.hasOneUse();
      CombineTo(N, ExtLoad);
      if (NoReplaceTrunc) {
        DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 1), ExtLoad.getValue(1));
        recursivelyDeleteUnusedNodes(LN0);
      } else {
        SDValue Trunc =
            DAG.getNode(ISD::TRUNCATE, SDLoc(N0), N0.getValueType(), ExtLoad);
        CombineTo(LN0, Trunc, ExtLoad.getValue(1));
      }
      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }
  }

  // fold (aext (zextload x)) -> (aext (truncate (zextload x)))
  // fold (aext (sextload x)) -> (aext (truncate (sextload x)))
  // fold (aext ( extload x)) -> (aext (truncate (extload  x)))
  if (N0.getOpcode() == ISD::LOAD && !ISD::isNON_EXTLoad(N0.getNode()) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) && N0.hasOneUse()) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    ISD::LoadExtType ExtType = LN0->getExtensionType();
    EVT MemVT = LN0->getMemoryVT();
    if (!LegalOperations || TLI.isLoadExtLegal(ExtType, VT, MemVT)) {
      SDValue ExtLoad = DAG.getExtLoad(ExtType, SDLoc(N),
                                       VT, LN0->getChain(), LN0->getBasePtr(),
                                       MemVT, LN0->getMemOperand());
      CombineTo(N, ExtLoad);
      DAG.ReplaceAllUsesOfValueWith(SDValue(LN0, 1), ExtLoad.getValue(1));
      recursivelyDeleteUnusedNodes(LN0);
      return SDValue(N, 0);   // Return N so it doesn't get rechecked!
    }
  }

  if (N0.getOpcode() == ISD::SETCC) {
    // For vectors:
    // aext(setcc) -> vsetcc
    // aext(setcc) -> truncate(vsetcc)
    // aext(setcc) -> aext(vsetcc)
    // Only do this before legalize for now.
    if (VT.isVector() && !LegalOperations) {
      EVT N00VT = N0.getOperand(0).getValueType();
      if (getSetCCResultType(N00VT) == N0.getValueType())
        return SDValue();

      // We know that the # elements of the results is the same as the
      // # elements of the compare (and the # elements of the compare result
      // for that matter).  Check to see that they are the same size.  If so,
      // we know that the element size of the sext'd result matches the
      // element size of the compare operands.
      if (VT.getSizeInBits() == N00VT.getSizeInBits())
        return DAG.getSetCC(SDLoc(N), VT, N0.getOperand(0),
                             N0.getOperand(1),
                             cast<CondCodeSDNode>(N0.getOperand(2))->get());

      // If the desired elements are smaller or larger than the source
      // elements we can use a matching integer vector type and then
      // truncate/any extend
      EVT MatchingVectorType = N00VT.changeVectorElementTypeToInteger();
      SDValue VsetCC =
        DAG.getSetCC(SDLoc(N), MatchingVectorType, N0.getOperand(0),
                      N0.getOperand(1),
                      cast<CondCodeSDNode>(N0.getOperand(2))->get());
      return DAG.getAnyExtOrTrunc(VsetCC, SDLoc(N), VT);
    }

    // aext(setcc x,y,cc) -> select_cc x, y, 1, 0, cc
    SDLoc DL(N);
    if (SDValue SCC = SimplifySelectCC(
            DL, N0.getOperand(0), N0.getOperand(1), DAG.getConstant(1, DL, VT),
            DAG.getConstant(0, DL, VT),
            cast<CondCodeSDNode>(N0.getOperand(2))->get(), true))
      return SCC;
  }

  if (SDValue NewCtPop = widenCtPop(N, DAG))
    return NewCtPop;

  if (SDValue Res = tryToFoldExtendSelectLoad(N, TLI, DAG))
    return Res;

  return SDValue();
}

SDValue DAGCombiner::visitAssertExt(SDNode *N) {
  unsigned Opcode = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT AssertVT = cast<VTSDNode>(N1)->getVT();

  // fold (assert?ext (assert?ext x, vt), vt) -> (assert?ext x, vt)
  if (N0.getOpcode() == Opcode &&
      AssertVT == cast<VTSDNode>(N0.getOperand(1))->getVT())
    return N0;

  if (N0.getOpcode() == ISD::TRUNCATE && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == Opcode) {
    // We have an assert, truncate, assert sandwich. Make one stronger assert
    // by asserting on the smallest asserted type to the larger source type.
    // This eliminates the later assert:
    // assert (trunc (assert X, i8) to iN), i1 --> trunc (assert X, i1) to iN
    // assert (trunc (assert X, i1) to iN), i8 --> trunc (assert X, i1) to iN
    SDValue BigA = N0.getOperand(0);
    EVT BigA_AssertVT = cast<VTSDNode>(BigA.getOperand(1))->getVT();
    assert(BigA_AssertVT.bitsLE(N0.getValueType()) &&
           "Asserting zero/sign-extended bits to a type larger than the "
           "truncated destination does not provide information");

    SDLoc DL(N);
    EVT MinAssertVT = AssertVT.bitsLT(BigA_AssertVT) ? AssertVT : BigA_AssertVT;
    SDValue MinAssertVTVal = DAG.getValueType(MinAssertVT);
    SDValue NewAssert = DAG.getNode(Opcode, DL, BigA.getValueType(),
                                    BigA.getOperand(0), MinAssertVTVal);
    return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewAssert);
  }

  // If we have (AssertZext (truncate (AssertSext X, iX)), iY) and Y is smaller
  // than X. Just move the AssertZext in front of the truncate and drop the
  // AssertSExt.
  if (N0.getOpcode() == ISD::TRUNCATE && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::AssertSext &&
      Opcode == ISD::AssertZext) {
    SDValue BigA = N0.getOperand(0);
    EVT BigA_AssertVT = cast<VTSDNode>(BigA.getOperand(1))->getVT();
    assert(BigA_AssertVT.bitsLE(N0.getValueType()) &&
           "Asserting zero/sign-extended bits to a type larger than the "
           "truncated destination does not provide information");

    if (AssertVT.bitsLT(BigA_AssertVT)) {
      SDLoc DL(N);
      SDValue NewAssert = DAG.getNode(Opcode, DL, BigA.getValueType(),
                                      BigA.getOperand(0), N1);
      return DAG.getNode(ISD::TRUNCATE, DL, N->getValueType(0), NewAssert);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitAssertAlign(SDNode *N) {
  SDLoc DL(N);

  Align AL = cast<AssertAlignSDNode>(N)->getAlign();
  SDValue N0 = N->getOperand(0);

  // Fold (assertalign (assertalign x, AL0), AL1) ->
  // (assertalign x, max(AL0, AL1))
  if (auto *AAN = dyn_cast<AssertAlignSDNode>(N0))
    return DAG.getAssertAlign(DL, N0.getOperand(0),
                              std::max(AL, AAN->getAlign()));

  // In rare cases, there are trivial arithmetic ops in source operands. Sink
  // this assert down to source operands so that those arithmetic ops could be
  // exposed to the DAG combining.
  switch (N0.getOpcode()) {
  default:
    break;
  case ISD::ADD:
  case ISD::SUB: {
    unsigned AlignShift = Log2(AL);
    SDValue LHS = N0.getOperand(0);
    SDValue RHS = N0.getOperand(1);
    unsigned LHSAlignShift = DAG.computeKnownBits(LHS).countMinTrailingZeros();
    unsigned RHSAlignShift = DAG.computeKnownBits(RHS).countMinTrailingZeros();
    if (LHSAlignShift >= AlignShift || RHSAlignShift >= AlignShift) {
      if (LHSAlignShift < AlignShift)
        LHS = DAG.getAssertAlign(DL, LHS, AL);
      if (RHSAlignShift < AlignShift)
        RHS = DAG.getAssertAlign(DL, RHS, AL);
      return DAG.getNode(N0.getOpcode(), DL, N0.getValueType(), LHS, RHS);
    }
    break;
  }
  }

  return SDValue();
}

/// If the result of a wider load is shifted to right of N  bits and then
/// truncated to a narrower type and where N is a multiple of number of bits of
/// the narrower type, transform it to a narrower load from address + N / num of
/// bits of new type. Also narrow the load if the result is masked with an AND
/// to effectively produce a smaller type. If the result is to be extended, also
/// fold the extension to form a extending load.
SDValue DAGCombiner::ReduceLoadWidth(SDNode *N) {
  unsigned Opc = N->getOpcode();

  ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT ExtVT = VT;

  // This transformation isn't valid for vector loads.
  if (VT.isVector())
    return SDValue();

  unsigned ShAmt = 0;
  bool HasShiftedOffset = false;
  // Special case: SIGN_EXTEND_INREG is basically truncating to ExtVT then
  // extended to VT.
  if (Opc == ISD::SIGN_EXTEND_INREG) {
    ExtType = ISD::SEXTLOAD;
    ExtVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  } else if (Opc == ISD::SRL) {
    // Another special-case: SRL is basically zero-extending a narrower value,
    // or it maybe shifting a higher subword, half or byte into the lowest
    // bits.
    ExtType = ISD::ZEXTLOAD;
    N0 = SDValue(N, 0);

    auto *LN0 = dyn_cast<LoadSDNode>(N0.getOperand(0));
    auto *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (!N01 || !LN0)
      return SDValue();

    uint64_t ShiftAmt = N01->getZExtValue();
    uint64_t MemoryWidth = LN0->getMemoryVT().getScalarSizeInBits();
    if (LN0->getExtensionType() != ISD::SEXTLOAD && MemoryWidth > ShiftAmt)
      ExtVT = EVT::getIntegerVT(*DAG.getContext(), MemoryWidth - ShiftAmt);
    else
      ExtVT = EVT::getIntegerVT(*DAG.getContext(),
                                VT.getScalarSizeInBits() - ShiftAmt);
  } else if (Opc == ISD::AND) {
    // An AND with a constant mask is the same as a truncate + zero-extend.
    auto AndC = dyn_cast<ConstantSDNode>(N->getOperand(1));
    if (!AndC)
      return SDValue();

    const APInt &Mask = AndC->getAPIntValue();
    unsigned ActiveBits = 0;
    if (Mask.isMask()) {
      ActiveBits = Mask.countTrailingOnes();
    } else if (Mask.isShiftedMask()) {
      ShAmt = Mask.countTrailingZeros();
      APInt ShiftedMask = Mask.lshr(ShAmt);
      ActiveBits = ShiftedMask.countTrailingOnes();
      HasShiftedOffset = true;
    } else
      return SDValue();

    ExtType = ISD::ZEXTLOAD;
    ExtVT = EVT::getIntegerVT(*DAG.getContext(), ActiveBits);
  }

  if (N0.getOpcode() == ISD::SRL && N0.hasOneUse()) {
    SDValue SRL = N0;
    if (auto *ConstShift = dyn_cast<ConstantSDNode>(SRL.getOperand(1))) {
      ShAmt = ConstShift->getZExtValue();
      unsigned EVTBits = ExtVT.getScalarSizeInBits();
      // Is the shift amount a multiple of size of VT?
      if ((ShAmt & (EVTBits-1)) == 0) {
        N0 = N0.getOperand(0);
        // Is the load width a multiple of size of VT?
        if ((N0.getScalarValueSizeInBits() & (EVTBits - 1)) != 0)
          return SDValue();
      }

      // At this point, we must have a load or else we can't do the transform.
      auto *LN0 = dyn_cast<LoadSDNode>(N0);
      if (!LN0) return SDValue();

      // Because a SRL must be assumed to *need* to zero-extend the high bits
      // (as opposed to anyext the high bits), we can't combine the zextload
      // lowering of SRL and an sextload.
      if (LN0->getExtensionType() == ISD::SEXTLOAD)
        return SDValue();

      // If the shift amount is larger than the input type then we're not
      // accessing any of the loaded bytes.  If the load was a zextload/extload
      // then the result of the shift+trunc is zero/undef (handled elsewhere).
      if (ShAmt >= LN0->getMemoryVT().getSizeInBits())
        return SDValue();

      // If the SRL is only used by a masking AND, we may be able to adjust
      // the ExtVT to make the AND redundant.
      SDNode *Mask = *(SRL->use_begin());
      if (Mask->getOpcode() == ISD::AND &&
          isa<ConstantSDNode>(Mask->getOperand(1))) {
        const APInt& ShiftMask = Mask->getConstantOperandAPInt(1);
        if (ShiftMask.isMask()) {
          EVT MaskedVT = EVT::getIntegerVT(*DAG.getContext(),
                                           ShiftMask.countTrailingOnes());
          // If the mask is smaller, recompute the type.
          if ((ExtVT.getScalarSizeInBits() > MaskedVT.getScalarSizeInBits()) &&
              TLI.isLoadExtLegal(ExtType, N0.getValueType(), MaskedVT))
            ExtVT = MaskedVT;
        }
      }
    }
  }

  // If the load is shifted left (and the result isn't shifted back right),
  // we can fold the truncate through the shift.
  unsigned ShLeftAmt = 0;
  if (ShAmt == 0 && N0.getOpcode() == ISD::SHL && N0.hasOneUse() &&
      ExtVT == VT && TLI.isNarrowingProfitable(N0.getValueType(), VT)) {
    if (ConstantSDNode *N01 = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      ShLeftAmt = N01->getZExtValue();
      N0 = N0.getOperand(0);
    }
  }

  // If we haven't found a load, we can't narrow it.
  if (!isa<LoadSDNode>(N0))
    return SDValue();

  LoadSDNode *LN0 = cast<LoadSDNode>(N0);
  // Reducing the width of a volatile load is illegal.  For atomics, we may be
  // able to reduce the width provided we never widen again. (see D66309)
  if (!LN0->isSimple() ||
      !isLegalNarrowLdSt(LN0, ExtType, ExtVT, ShAmt))
    return SDValue();

  auto AdjustBigEndianShift = [&](unsigned ShAmt) {
    unsigned LVTStoreBits =
        LN0->getMemoryVT().getStoreSizeInBits().getFixedSize();
    unsigned EVTStoreBits = ExtVT.getStoreSizeInBits().getFixedSize();
    return LVTStoreBits - EVTStoreBits - ShAmt;
  };

  // For big endian targets, we need to adjust the offset to the pointer to
  // load the correct bytes.
  if (DAG.getDataLayout().isBigEndian())
    ShAmt = AdjustBigEndianShift(ShAmt);

  uint64_t PtrOff = ShAmt / 8;
  Align NewAlign = commonAlignment(LN0->getAlign(), PtrOff);
  SDLoc DL(LN0);
  // The original load itself didn't wrap, so an offset within it doesn't.
  SDNodeFlags Flags;
  Flags.setNoUnsignedWrap(true);
  SDValue NewPtr = DAG.getMemBasePlusOffset(LN0->getBasePtr(),
                                            TypeSize::Fixed(PtrOff), DL, Flags);
  AddToWorklist(NewPtr.getNode());

  SDValue Load;
  if (ExtType == ISD::NON_EXTLOAD)
    Load = DAG.getLoad(VT, DL, LN0->getChain(), NewPtr,
                       LN0->getPointerInfo().getWithOffset(PtrOff), NewAlign,
                       LN0->getMemOperand()->getFlags(), LN0->getAAInfo());
  else
    Load = DAG.getExtLoad(ExtType, DL, VT, LN0->getChain(), NewPtr,
                          LN0->getPointerInfo().getWithOffset(PtrOff), ExtVT,
                          NewAlign, LN0->getMemOperand()->getFlags(),
                          LN0->getAAInfo());

  // Replace the old load's chain with the new load's chain.
  WorklistRemover DeadNodes(*this);
  DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1));

  // Shift the result left, if we've swallowed a left shift.
  SDValue Result = Load;
  if (ShLeftAmt != 0) {
    EVT ShImmTy = getShiftAmountTy(Result.getValueType());
    if (!isUIntN(ShImmTy.getScalarSizeInBits(), ShLeftAmt))
      ShImmTy = VT;
    // If the shift amount is as large as the result size (but, presumably,
    // no larger than the source) then the useful bits of the result are
    // zero; we can't simply return the shortened shift, because the result
    // of that operation is undefined.
    if (ShLeftAmt >= VT.getScalarSizeInBits())
      Result = DAG.getConstant(0, DL, VT);
    else
      Result = DAG.getNode(ISD::SHL, DL, VT,
                          Result, DAG.getConstant(ShLeftAmt, DL, ShImmTy));
  }

  if (HasShiftedOffset) {
    // Recalculate the shift amount after it has been altered to calculate
    // the offset.
    if (DAG.getDataLayout().isBigEndian())
      ShAmt = AdjustBigEndianShift(ShAmt);

    // We're using a shifted mask, so the load now has an offset. This means
    // that data has been loaded into the lower bytes than it would have been
    // before, so we need to shl the loaded data into the correct position in the
    // register.
    SDValue ShiftC = DAG.getConstant(ShAmt, DL, VT);
    Result = DAG.getNode(ISD::SHL, DL, VT, Result, ShiftC);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result);
  }

  // Return the new loaded value.
  return Result;
}

SDValue DAGCombiner::visitSIGN_EXTEND_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT ExtVT = cast<VTSDNode>(N1)->getVT();
  unsigned VTBits = VT.getScalarSizeInBits();
  unsigned ExtVTBits = ExtVT.getScalarSizeInBits();

  // sext_vector_inreg(undef) = 0 because the top bit will all be the same.
  if (N0.isUndef())
    return DAG.getConstant(0, SDLoc(N), VT);

  // fold (sext_in_reg c1) -> c1
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0))
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, N0, N1);

  // If the input is already sign extended, just drop the extension.
  if (DAG.ComputeNumSignBits(N0) >= (VTBits - ExtVTBits + 1))
    return N0;

  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      ExtVT.bitsLT(cast<VTSDNode>(N0.getOperand(1))->getVT()))
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, N0.getOperand(0),
                       N1);

  // fold (sext_in_reg (sext x)) -> (sext x)
  // fold (sext_in_reg (aext x)) -> (sext x)
  // if x is small enough or if we know that x has more than 1 sign bit and the
  // sign_extend_inreg is extending from one of them.
  if (N0.getOpcode() == ISD::SIGN_EXTEND || N0.getOpcode() == ISD::ANY_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    unsigned N00Bits = N00.getScalarValueSizeInBits();
    if ((N00Bits <= ExtVTBits ||
         (N00Bits - DAG.ComputeNumSignBits(N00)) < ExtVTBits) &&
        (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND, VT)))
      return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, N00);
  }

  // fold (sext_in_reg (*_extend_vector_inreg x)) -> (sext_vector_inreg x)
  // if x is small enough or if we know that x has more than 1 sign bit and the
  // sign_extend_inreg is extending from one of them.
  if (N0.getOpcode() == ISD::ANY_EXTEND_VECTOR_INREG ||
      N0.getOpcode() == ISD::SIGN_EXTEND_VECTOR_INREG ||
      N0.getOpcode() == ISD::ZERO_EXTEND_VECTOR_INREG) {
    SDValue N00 = N0.getOperand(0);
    unsigned N00Bits = N00.getScalarValueSizeInBits();
    unsigned DstElts = N0.getValueType().getVectorMinNumElements();
    unsigned SrcElts = N00.getValueType().getVectorMinNumElements();
    bool IsZext = N0.getOpcode() == ISD::ZERO_EXTEND_VECTOR_INREG;
    APInt DemandedSrcElts = APInt::getLowBitsSet(SrcElts, DstElts);
    if ((N00Bits == ExtVTBits ||
         (!IsZext && (N00Bits < ExtVTBits ||
                      (N00Bits - DAG.ComputeNumSignBits(N00, DemandedSrcElts)) <
                          ExtVTBits))) &&
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::SIGN_EXTEND_VECTOR_INREG, VT)))
      return DAG.getNode(ISD::SIGN_EXTEND_VECTOR_INREG, SDLoc(N), VT, N00);
  }

  // fold (sext_in_reg (zext x)) -> (sext x)
  // iff we are extending the source sign bit.
  if (N0.getOpcode() == ISD::ZERO_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getScalarValueSizeInBits() == ExtVTBits &&
        (!LegalOperations || TLI.isOperationLegal(ISD::SIGN_EXTEND, VT)))
      return DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, N00, N1);
  }

  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is known zero.
  if (DAG.MaskedValueIsZero(N0, APInt::getOneBitSet(VTBits, ExtVTBits - 1)))
    return DAG.getZeroExtendInReg(N0, SDLoc(N), ExtVT);

  // fold operands of sext_in_reg based on knowledge that the top bits are not
  // demanded.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // fold (sext_in_reg (load x)) -> (smaller sextload x)
  // fold (sext_in_reg (srl (load x), c)) -> (smaller sextload (x+c/evtbits))
  if (SDValue NarrowLoad = ReduceLoadWidth(N))
    return NarrowLoad;

  // fold (sext_in_reg (srl X, 24), i8) -> (sra X, 24)
  // fold (sext_in_reg (srl X, 23), i8) -> (sra X, 23) iff possible.
  // We already fold "(sext_in_reg (srl X, 25), i8) -> srl X, 25" above.
  if (N0.getOpcode() == ISD::SRL) {
    if (auto *ShAmt = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if (ShAmt->getAPIntValue().ule(VTBits - ExtVTBits)) {
        // We can turn this into an SRA iff the input to the SRL is already sign
        // extended enough.
        unsigned InSignBits = DAG.ComputeNumSignBits(N0.getOperand(0));
        if (((VTBits - ExtVTBits) - ShAmt->getZExtValue()) < InSignBits)
          return DAG.getNode(ISD::SRA, SDLoc(N), VT, N0.getOperand(0),
                             N0.getOperand(1));
      }
  }

  // fold (sext_inreg (extload x)) -> (sextload x)
  // If sextload is not supported by target, we can only do the combine when
  // load has one use. Doing otherwise can block folding the extload with other
  // extends that the target does support.
  if (ISD::isEXTLoad(N0.getNode()) &&
      ISD::isUNINDEXEDLoad(N0.getNode()) &&
      ExtVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && cast<LoadSDNode>(N0)->isSimple() &&
        N0.hasOneUse()) ||
       TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, ExtVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), ExtVT,
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    AddToWorklist(ExtLoad.getNode());
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  // fold (sext_inreg (zextload x)) -> (sextload x) iff load has one use
  if (ISD::isZEXTLoad(N0.getNode()) && ISD::isUNINDEXEDLoad(N0.getNode()) &&
      N0.hasOneUse() &&
      ExtVT == cast<LoadSDNode>(N0)->getMemoryVT() &&
      ((!LegalOperations && cast<LoadSDNode>(N0)->isSimple()) &&
       TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, ExtVT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), ExtVT,
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  // fold (sext_inreg (masked_load x)) -> (sext_masked_load x)
  // ignore it if the masked load is already sign extended
  if (MaskedLoadSDNode *Ld = dyn_cast<MaskedLoadSDNode>(N0)) {
    if (ExtVT == Ld->getMemoryVT() && N0.hasOneUse() &&
        Ld->getExtensionType() != ISD::LoadExtType::NON_EXTLOAD &&
        TLI.isLoadExtLegal(ISD::SEXTLOAD, VT, ExtVT)) {
      SDValue ExtMaskedLoad = DAG.getMaskedLoad(
          VT, SDLoc(N), Ld->getChain(), Ld->getBasePtr(), Ld->getOffset(),
          Ld->getMask(), Ld->getPassThru(), ExtVT, Ld->getMemOperand(),
          Ld->getAddressingMode(), ISD::SEXTLOAD, Ld->isExpandingLoad());
      CombineTo(N, ExtMaskedLoad);
      CombineTo(N0.getNode(), ExtMaskedLoad, ExtMaskedLoad.getValue(1));
      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }
  }

  // fold (sext_inreg (masked_gather x)) -> (sext_masked_gather x)
  if (auto *GN0 = dyn_cast<MaskedGatherSDNode>(N0)) {
    if (SDValue(GN0, 0).hasOneUse() &&
        ExtVT == GN0->getMemoryVT() &&
        TLI.isVectorLoadExtDesirable(SDValue(SDValue(GN0, 0)))) {
      SDValue Ops[] = {GN0->getChain(),   GN0->getPassThru(), GN0->getMask(),
                       GN0->getBasePtr(), GN0->getIndex(),    GN0->getScale()};

      SDValue ExtLoad = DAG.getMaskedGather(
          DAG.getVTList(VT, MVT::Other), ExtVT, SDLoc(N), Ops,
          GN0->getMemOperand(), GN0->getIndexType(), ISD::SEXTLOAD);

      CombineTo(N, ExtLoad);
      CombineTo(N0.getNode(), ExtLoad, ExtLoad.getValue(1));
      AddToWorklist(ExtLoad.getNode());
      return SDValue(N, 0); // Return N so it doesn't get rechecked!
    }
  }

  // Form (sext_inreg (bswap >> 16)) or (sext_inreg (rotl (bswap) 16))
  if (ExtVTBits <= 16 && N0.getOpcode() == ISD::OR) {
    if (SDValue BSwap = MatchBSwapHWordLow(N0.getNode(), N0.getOperand(0),
                                           N0.getOperand(1), false))
      return DAG.getNode(ISD::SIGN_EXTEND_INREG, SDLoc(N), VT, BSwap, N1);
  }

  return SDValue();
}

SDValue DAGCombiner::visitEXTEND_VECTOR_INREG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // {s/z}ext_vector_inreg(undef) = 0 because the top bits must be the same.
  if (N0.isUndef())
    return DAG.getConstant(0, SDLoc(N), VT);

  if (SDValue Res = tryToFoldExtendOfConstant(N, TLI, DAG, LegalTypes))
    return Res;

  if (SimplifyDemandedVectorElts(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

SDValue DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT SrcVT = N0.getValueType();
  bool isLE = DAG.getDataLayout().isLittleEndian();

  // noop truncate
  if (SrcVT == VT)
    return N0;

  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0.getOperand(0));

  // fold (truncate c1) -> c1
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0)) {
    SDValue C = DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0);
    if (C.getNode() != N)
      return C;
  }

  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND ||
      N0.getOpcode() == ISD::SIGN_EXTEND ||
      N0.getOpcode() == ISD::ANY_EXTEND) {
    // if the source is smaller than the dest, we still need an extend.
    if (N0.getOperand(0).getValueType().bitsLT(VT))
      return DAG.getNode(N0.getOpcode(), SDLoc(N), VT, N0.getOperand(0));
    // if the source is larger than the dest, than we just need the truncate.
    if (N0.getOperand(0).getValueType().bitsGT(VT))
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, N0.getOperand(0));
    // if the source and dest are the same type, we can drop both the extend
    // and the truncate.
    return N0.getOperand(0);
  }

  // If this is anyext(trunc), don't fold it, allow ourselves to be folded.
  if (N->hasOneUse() && (N->use_begin()->getOpcode() == ISD::ANY_EXTEND))
    return SDValue();

  // Fold extract-and-trunc into a narrow extract. For example:
  //   i64 x = EXTRACT_VECTOR_ELT(v2i64 val, i32 1)
  //   i32 y = TRUNCATE(i64 x)
  //        -- becomes --
  //   v16i8 b = BITCAST (v2i64 val)
  //   i8 x = EXTRACT_VECTOR_ELT(v16i8 b, i32 8)
  //
  // Note: We only run this optimization after type legalization (which often
  // creates this pattern) and before operation legalization after which
  // we need to be more careful about the vector instructions that we generate.
  if (N0.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      LegalTypes && !LegalOperations && N0->hasOneUse() && VT != MVT::i1) {
    EVT VecTy = N0.getOperand(0).getValueType();
    EVT ExTy = N0.getValueType();
    EVT TrTy = N->getValueType(0);

    auto EltCnt = VecTy.getVectorElementCount();
    unsigned SizeRatio = ExTy.getSizeInBits()/TrTy.getSizeInBits();
    auto NewEltCnt = EltCnt * SizeRatio;

    EVT NVT = EVT::getVectorVT(*DAG.getContext(), TrTy, NewEltCnt);
    assert(NVT.getSizeInBits() == VecTy.getSizeInBits() && "Invalid Size");

    SDValue EltNo = N0->getOperand(1);
    if (isa<ConstantSDNode>(EltNo) && isTypeLegal(NVT)) {
      int Elt = cast<ConstantSDNode>(EltNo)->getZExtValue();
      int Index = isLE ? (Elt*SizeRatio) : (Elt*SizeRatio + (SizeRatio-1));

      SDLoc DL(N);
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, TrTy,
                         DAG.getBitcast(NVT, N0.getOperand(0)),
                         DAG.getVectorIdxConstant(Index, DL));
    }
  }

  // trunc (select c, a, b) -> select c, (trunc a), (trunc b)
  if (N0.getOpcode() == ISD::SELECT && N0.hasOneUse()) {
    if ((!LegalOperations || TLI.isOperationLegal(ISD::SELECT, SrcVT)) &&
        TLI.isTruncateFree(SrcVT, VT)) {
      SDLoc SL(N0);
      SDValue Cond = N0.getOperand(0);
      SDValue TruncOp0 = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(1));
      SDValue TruncOp1 = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(2));
      return DAG.getNode(ISD::SELECT, SDLoc(N), VT, Cond, TruncOp0, TruncOp1);
    }
  }

  // trunc (shl x, K) -> shl (trunc x), K => K < VT.getScalarSizeInBits()
  if (N0.getOpcode() == ISD::SHL && N0.hasOneUse() &&
      (!LegalOperations || TLI.isOperationLegal(ISD::SHL, VT)) &&
      TLI.isTypeDesirableForOp(ISD::SHL, VT)) {
    SDValue Amt = N0.getOperand(1);
    KnownBits Known = DAG.computeKnownBits(Amt);
    unsigned Size = VT.getScalarSizeInBits();
    if (Known.getBitWidth() - Known.countMinLeadingZeros() <= Log2_32(Size)) {
      SDLoc SL(N);
      EVT AmtVT = TLI.getShiftAmountTy(VT, DAG.getDataLayout());

      SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(0));
      if (AmtVT != Amt.getValueType()) {
        Amt = DAG.getZExtOrTrunc(Amt, SL, AmtVT);
        AddToWorklist(Amt.getNode());
      }
      return DAG.getNode(ISD::SHL, SL, VT, Trunc, Amt);
    }
  }

  if (SDValue V = foldSubToUSubSat(VT, N0.getNode()))
    return V;

  // Attempt to pre-truncate BUILD_VECTOR sources.
  if (N0.getOpcode() == ISD::BUILD_VECTOR && !LegalOperations &&
      TLI.isTruncateFree(SrcVT.getScalarType(), VT.getScalarType()) &&
      // Avoid creating illegal types if running after type legalizer.
      (!LegalTypes || TLI.isTypeLegal(VT.getScalarType()))) {
    SDLoc DL(N);
    EVT SVT = VT.getScalarType();
    SmallVector<SDValue, 8> TruncOps;
    for (const SDValue &Op : N0->op_values()) {
      SDValue TruncOp = DAG.getNode(ISD::TRUNCATE, DL, SVT, Op);
      TruncOps.push_back(TruncOp);
    }
    return DAG.getBuildVector(VT, DL, TruncOps);
  }

  // Fold a series of buildvector, bitcast, and truncate if possible.
  // For example fold
  //   (2xi32 trunc (bitcast ((4xi32)buildvector x, x, y, y) 2xi64)) to
  //   (2xi32 (buildvector x, y)).
  if (Level == AfterLegalizeVectorOps && VT.isVector() &&
      N0.getOpcode() == ISD::BITCAST && N0.hasOneUse() &&
      N0.getOperand(0).getOpcode() == ISD::BUILD_VECTOR &&
      N0.getOperand(0).hasOneUse()) {
    SDValue BuildVect = N0.getOperand(0);
    EVT BuildVectEltTy = BuildVect.getValueType().getVectorElementType();
    EVT TruncVecEltTy = VT.getVectorElementType();

    // Check that the element types match.
    if (BuildVectEltTy == TruncVecEltTy) {
      // Now we only need to compute the offset of the truncated elements.
      unsigned BuildVecNumElts =  BuildVect.getNumOperands();
      unsigned TruncVecNumElts = VT.getVectorNumElements();
      unsigned TruncEltOffset = BuildVecNumElts / TruncVecNumElts;

      assert((BuildVecNumElts % TruncVecNumElts) == 0 &&
             "Invalid number of elements");

      SmallVector<SDValue, 8> Opnds;
      for (unsigned i = 0, e = BuildVecNumElts; i != e; i += TruncEltOffset)
        Opnds.push_back(BuildVect.getOperand(i));

      return DAG.getBuildVector(VT, SDLoc(N), Opnds);
    }
  }

  // See if we can simplify the input to this truncate through knowledge that
  // only the low bits are being used.
  // For example "trunc (or (shl x, 8), y)" // -> trunc y
  // Currently we only perform this optimization on scalars because vectors
  // may have different active low bits.
  if (!VT.isVector()) {
    APInt Mask =
        APInt::getLowBitsSet(N0.getValueSizeInBits(), VT.getSizeInBits());
    if (SDValue Shorter = DAG.GetDemandedBits(N0, Mask))
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Shorter);
  }

  // fold (truncate (load x)) -> (smaller load x)
  // fold (truncate (srl (load x), c)) -> (smaller load (x+c/evtbits))
  if (!LegalTypes || TLI.isTypeDesirableForOp(N0.getOpcode(), VT)) {
    if (SDValue Reduced = ReduceLoadWidth(N))
      return Reduced;

    // Handle the case where the load remains an extending load even
    // after truncation.
    if (N0.hasOneUse() && ISD::isUNINDEXEDLoad(N0.getNode())) {
      LoadSDNode *LN0 = cast<LoadSDNode>(N0);
      if (LN0->isSimple() && LN0->getMemoryVT().bitsLT(VT)) {
        SDValue NewLoad = DAG.getExtLoad(LN0->getExtensionType(), SDLoc(LN0),
                                         VT, LN0->getChain(), LN0->getBasePtr(),
                                         LN0->getMemoryVT(),
                                         LN0->getMemOperand());
        DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), NewLoad.getValue(1));
        return NewLoad;
      }
    }
  }

  // fold (trunc (concat ... x ...)) -> (concat ..., (trunc x), ...)),
  // where ... are all 'undef'.
  if (N0.getOpcode() == ISD::CONCAT_VECTORS && !LegalTypes) {
    SmallVector<EVT, 8> VTs;
    SDValue V;
    unsigned Idx = 0;
    unsigned NumDefs = 0;

    for (unsigned i = 0, e = N0.getNumOperands(); i != e; ++i) {
      SDValue X = N0.getOperand(i);
      if (!X.isUndef()) {
        V = X;
        Idx = i;
        NumDefs++;
      }
      // Stop if more than one members are non-undef.
      if (NumDefs > 1)
        break;

      VTs.push_back(EVT::getVectorVT(*DAG.getContext(),
                                     VT.getVectorElementType(),
                                     X.getValueType().getVectorElementCount()));
    }

    if (NumDefs == 0)
      return DAG.getUNDEF(VT);

    if (NumDefs == 1) {
      assert(V.getNode() && "The single defined operand is empty!");
      SmallVector<SDValue, 8> Opnds;
      for (unsigned i = 0, e = VTs.size(); i != e; ++i) {
        if (i != Idx) {
          Opnds.push_back(DAG.getUNDEF(VTs[i]));
          continue;
        }
        SDValue NV = DAG.getNode(ISD::TRUNCATE, SDLoc(V), VTs[i], V);
        AddToWorklist(NV.getNode());
        Opnds.push_back(NV);
      }
      return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Opnds);
    }
  }

  // Fold truncate of a bitcast of a vector to an extract of the low vector
  // element.
  //
  // e.g. trunc (i64 (bitcast v2i32:x)) -> extract_vector_elt v2i32:x, idx
  if (N0.getOpcode() == ISD::BITCAST && !VT.isVector()) {
    SDValue VecSrc = N0.getOperand(0);
    EVT VecSrcVT = VecSrc.getValueType();
    if (VecSrcVT.isVector() && VecSrcVT.getScalarType() == VT &&
        (!LegalOperations ||
         TLI.isOperationLegal(ISD::EXTRACT_VECTOR_ELT, VecSrcVT))) {
      SDLoc SL(N);

      unsigned Idx = isLE ? 0 : VecSrcVT.getVectorNumElements() - 1;
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, VT, VecSrc,
                         DAG.getVectorIdxConstant(Idx, SL));
    }
  }

  // Simplify the operands using demanded-bits information.
  if (SimplifyDemandedBits(SDValue(N, 0)))
    return SDValue(N, 0);

  // (trunc adde(X, Y, Carry)) -> (adde trunc(X), trunc(Y), Carry)
  // (trunc addcarry(X, Y, Carry)) -> (addcarry trunc(X), trunc(Y), Carry)
  // When the adde's carry is not used.
  if ((N0.getOpcode() == ISD::ADDE || N0.getOpcode() == ISD::ADDCARRY) &&
      N0.hasOneUse() && !N0.getNode()->hasAnyUseOfValue(1) &&
      // We only do for addcarry before legalize operation
      ((!LegalOperations && N0.getOpcode() == ISD::ADDCARRY) ||
       TLI.isOperationLegal(N0.getOpcode(), VT))) {
    SDLoc SL(N);
    auto X = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(0));
    auto Y = DAG.getNode(ISD::TRUNCATE, SL, VT, N0.getOperand(1));
    auto VTs = DAG.getVTList(VT, N0->getValueType(1));
    return DAG.getNode(N0.getOpcode(), SL, VTs, X, Y, N0.getOperand(2));
  }

  // fold (truncate (extract_subvector(ext x))) ->
  //      (extract_subvector x)
  // TODO: This can be generalized to cover cases where the truncate and extract
  // do not fully cancel each other out.
  if (!LegalTypes && N0.getOpcode() == ISD::EXTRACT_SUBVECTOR) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getOpcode() == ISD::SIGN_EXTEND ||
        N00.getOpcode() == ISD::ZERO_EXTEND ||
        N00.getOpcode() == ISD::ANY_EXTEND) {
      if (N00.getOperand(0)->getValueType(0).getVectorElementType() ==
          VT.getVectorElementType())
        return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SDLoc(N0->getOperand(0)), VT,
                           N00.getOperand(0), N0.getOperand(1));
    }
  }

  if (SDValue NewVSel = matchVSelectOpSizesWithSetCC(N))
    return NewVSel;

  // Narrow a suitable binary operation with a non-opaque constant operand by
  // moving it ahead of the truncate. This is limited to pre-legalization
  // because targets may prefer a wider type during later combines and invert
  // this transform.
  switch (N0.getOpcode()) {
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    if (!LegalOperations && N0.hasOneUse() &&
        (isConstantOrConstantVector(N0.getOperand(0), true) ||
         isConstantOrConstantVector(N0.getOperand(1), true))) {
      // TODO: We already restricted this to pre-legalization, but for vectors
      // we are extra cautious to not create an unsupported operation.
      // Target-specific changes are likely needed to avoid regressions here.
      if (VT.isScalarInteger() || TLI.isOperationLegal(N0.getOpcode(), VT)) {
        SDLoc DL(N);
        SDValue NarrowL = DAG.getNode(ISD::TRUNCATE, DL, VT, N0.getOperand(0));
        SDValue NarrowR = DAG.getNode(ISD::TRUNCATE, DL, VT, N0.getOperand(1));
        return DAG.getNode(N0.getOpcode(), DL, VT, NarrowL, NarrowR);
      }
    }
    break;
  case ISD::USUBSAT:
    // Truncate the USUBSAT only if LHS is a known zero-extension, its not
    // enough to know that the upper bits are zero we must ensure that we don't
    // introduce an extra truncate.
    if (!LegalOperations && N0.hasOneUse() &&
        N0.getOperand(0).getOpcode() == ISD::ZERO_EXTEND &&
        N0.getOperand(0).getOperand(0).getScalarValueSizeInBits() <=
            VT.getScalarSizeInBits() &&
        hasOperation(N0.getOpcode(), VT)) {
      return getTruncatedUSUBSAT(VT, SrcVT, N0.getOperand(0), N0.getOperand(1),
                                 DAG, SDLoc(N));
    }
    break;
  }

  return SDValue();
}

static SDNode *getBuildPairElt(SDNode *N, unsigned i) {
  SDValue Elt = N->getOperand(i);
  if (Elt.getOpcode() != ISD::MERGE_VALUES)
    return Elt.getNode();
  return Elt.getOperand(Elt.getResNo()).getNode();
}

/// build_pair (load, load) -> load
/// if load locations are consecutive.
SDValue DAGCombiner::CombineConsecutiveLoads(SDNode *N, EVT VT) {
  assert(N->getOpcode() == ISD::BUILD_PAIR);

  auto *LD1 = dyn_cast<LoadSDNode>(getBuildPairElt(N, 0));
  auto *LD2 = dyn_cast<LoadSDNode>(getBuildPairElt(N, 1));

  // A BUILD_PAIR is always having the least significant part in elt 0 and the
  // most significant part in elt 1. So when combining into one large load, we
  // need to consider the endianness.
  if (DAG.getDataLayout().isBigEndian())
    std::swap(LD1, LD2);

  if (!LD1 || !LD2 || !ISD::isNON_EXTLoad(LD1) || !ISD::isNON_EXTLoad(LD2) ||
      !LD1->hasOneUse() || !LD2->hasOneUse() ||
      LD1->getAddressSpace() != LD2->getAddressSpace())
    return SDValue();

  EVT LD1VT = LD1->getValueType(0);
  unsigned LD1Bytes = LD1VT.getStoreSize();
  if (DAG.areNonVolatileConsecutiveLoads(LD2, LD1, LD1Bytes, 1)) {
    Align Alignment = LD1->getAlign();
    Align NewAlign = DAG.getDataLayout().getABITypeAlign(
        VT.getTypeForEVT(*DAG.getContext()));

    if (NewAlign <= Alignment &&
        (!LegalOperations || TLI.isOperationLegal(ISD::LOAD, VT)))
      return DAG.getLoad(VT, SDLoc(N), LD1->getChain(), LD1->getBasePtr(),
                         LD1->getPointerInfo(), Alignment);
  }

  return SDValue();
}

static unsigned getPPCf128HiElementSelector(const SelectionDAG &DAG) {
  // On little-endian machines, bitcasting from ppcf128 to i128 does swap the Hi
  // and Lo parts; on big-endian machines it doesn't.
  return DAG.getDataLayout().isBigEndian() ? 1 : 0;
}

static SDValue foldBitcastedFPLogic(SDNode *N, SelectionDAG &DAG,
                                    const TargetLowering &TLI) {
  // If this is not a bitcast to an FP type or if the target doesn't have
  // IEEE754-compliant FP logic, we're done.
  EVT VT = N->getValueType(0);
  if (!VT.isFloatingPoint() || !TLI.hasBitPreservingFPLogic(VT))
    return SDValue();

  // TODO: Handle cases where the integer constant is a different scalar
  // bitwidth to the FP.
  SDValue N0 = N->getOperand(0);
  EVT SourceVT = N0.getValueType();
  if (VT.getScalarSizeInBits() != SourceVT.getScalarSizeInBits())
    return SDValue();

  unsigned FPOpcode;
  APInt SignMask;
  switch (N0.getOpcode()) {
  case ISD::AND:
    FPOpcode = ISD::FABS;
    SignMask = ~APInt::getSignMask(SourceVT.getScalarSizeInBits());
    break;
  case ISD::XOR:
    FPOpcode = ISD::FNEG;
    SignMask = APInt::getSignMask(SourceVT.getScalarSizeInBits());
    break;
  case ISD::OR:
    FPOpcode = ISD::FABS;
    SignMask = APInt::getSignMask(SourceVT.getScalarSizeInBits());
    break;
  default:
    return SDValue();
  }

  // Fold (bitcast int (and (bitcast fp X to int), 0x7fff...) to fp) -> fabs X
  // Fold (bitcast int (xor (bitcast fp X to int), 0x8000...) to fp) -> fneg X
  // Fold (bitcast int (or (bitcast fp X to int), 0x8000...) to fp) ->
  //   fneg (fabs X)
  SDValue LogicOp0 = N0.getOperand(0);
  ConstantSDNode *LogicOp1 = isConstOrConstSplat(N0.getOperand(1), true);
  if (LogicOp1 && LogicOp1->getAPIntValue() == SignMask &&
      LogicOp0.getOpcode() == ISD::BITCAST &&
      LogicOp0.getOperand(0).getValueType() == VT) {
    SDValue FPOp = DAG.getNode(FPOpcode, SDLoc(N), VT, LogicOp0.getOperand(0));
    NumFPLogicOpsConv++;
    if (N0.getOpcode() == ISD::OR)
      return DAG.getNode(ISD::FNEG, SDLoc(N), VT, FPOp);
    return FPOp;
  }

  return SDValue();
}

SDValue DAGCombiner::visitBITCAST(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (N0.isUndef())
    return DAG.getUNDEF(VT);

  // If the input is a BUILD_VECTOR with all constant elements, fold this now.
  // Only do this before legalize types, unless both types are integer and the
  // scalar type is legal. Only do this before legalize ops, since the target
  // maybe depending on the bitcast.
  // First check to see if this is all constant.
  // TODO: Support FP bitcasts after legalize types.
  if (VT.isVector() &&
      (!LegalTypes ||
       (!LegalOperations && VT.isInteger() && N0.getValueType().isInteger() &&
        TLI.isTypeLegal(VT.getVectorElementType()))) &&
      N0.getOpcode() == ISD::BUILD_VECTOR && N0.getNode()->hasOneUse() &&
      cast<BuildVectorSDNode>(N0)->isConstant())
    return ConstantFoldBITCASTofBUILD_VECTOR(N0.getNode(),
                                             VT.getVectorElementType());

  // If the input is a constant, let getNode fold it.
  if (isIntOrFPConstant(N0)) {
    // If we can't allow illegal operations, we need to check that this is just
    // a fp -> int or int -> conversion and that the resulting operation will
    // be legal.
    if (!LegalOperations ||
        (isa<ConstantSDNode>(N0) && VT.isFloatingPoint() && !VT.isVector() &&
         TLI.isOperationLegal(ISD::ConstantFP, VT)) ||
        (isa<ConstantFPSDNode>(N0) && VT.isInteger() && !VT.isVector() &&
         TLI.isOperationLegal(ISD::Constant, VT))) {
      SDValue C = DAG.getBitcast(VT, N0);
      if (C.getNode() != N)
        return C;
    }
  }

  // (conv (conv x, t1), t2) -> (conv x, t2)
  if (N0.getOpcode() == ISD::BITCAST)
    return DAG.getBitcast(VT, N0.getOperand(0));

  // fold (conv (load x)) -> (load (conv*)x)
  // If the resultant load doesn't need a higher alignment than the original!
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      // Do not remove the cast if the types differ in endian layout.
      TLI.hasBigEndianPartOrdering(N0.getValueType(), DAG.getDataLayout()) ==
          TLI.hasBigEndianPartOrdering(VT, DAG.getDataLayout()) &&
      // If the load is volatile, we only want to change the load type if the
      // resulting load is legal. Otherwise we might increase the number of
      // memory accesses. We don't care if the original type was legal or not
      // as we assume software couldn't rely on the number of accesses of an
      // illegal type.
      ((!LegalOperations && cast<LoadSDNode>(N0)->isSimple()) ||
       TLI.isOperationLegal(ISD::LOAD, VT))) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);

    if (TLI.isLoadBitCastBeneficial(N0.getValueType(), VT, DAG,
                                    *LN0->getMemOperand())) {
      SDValue Load =
          DAG.getLoad(VT, SDLoc(N), LN0->getChain(), LN0->getBasePtr(),
                      LN0->getPointerInfo(), LN0->getAlign(),
                      LN0->getMemOperand()->getFlags(), LN0->getAAInfo());
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), Load.getValue(1));
      return Load;
    }
  }

  if (SDValue V = foldBitcastedFPLogic(N, DAG, TLI))
    return V;

  // fold (bitconvert (fneg x)) -> (xor (bitconvert x), signbit)
  // fold (bitconvert (fabs x)) -> (and (bitconvert x), (not signbit))
  //
  // For ppc_fp128:
  // fold (bitcast (fneg x)) ->
  //     flipbit = signbit
  //     (xor (bitcast x) (build_pair flipbit, flipbit))
  //
  // fold (bitcast (fabs x)) ->
  //     flipbit = (and (extract_element (bitcast x), 0), signbit)
  //     (xor (bitcast x) (build_pair flipbit, flipbit))
  // This often reduces constant pool loads.
  if (((N0.getOpcode() == ISD::FNEG && !TLI.isFNegFree(N0.getValueType())) ||
       (N0.getOpcode() == ISD::FABS && !TLI.isFAbsFree(N0.getValueType()))) &&
      N0.getNode()->hasOneUse() && VT.isInteger() &&
      !VT.isVector() && !N0.getValueType().isVector()) {
    SDValue NewConv = DAG.getBitcast(VT, N0.getOperand(0));
    AddToWorklist(NewConv.getNode());

    SDLoc DL(N);
    if (N0.getValueType() == MVT::ppcf128 && !LegalTypes) {
      assert(VT.getSizeInBits() == 128);
      SDValue SignBit = DAG.getConstant(
          APInt::getSignMask(VT.getSizeInBits() / 2), SDLoc(N0), MVT::i64);
      SDValue FlipBit;
      if (N0.getOpcode() == ISD::FNEG) {
        FlipBit = SignBit;
        AddToWorklist(FlipBit.getNode());
      } else {
        assert(N0.getOpcode() == ISD::FABS);
        SDValue Hi =
            DAG.getNode(ISD::EXTRACT_ELEMENT, SDLoc(NewConv), MVT::i64, NewConv,
                        DAG.getIntPtrConstant(getPPCf128HiElementSelector(DAG),
                                              SDLoc(NewConv)));
        AddToWorklist(Hi.getNode());
        FlipBit = DAG.getNode(ISD::AND, SDLoc(N0), MVT::i64, Hi, SignBit);
        AddToWorklist(FlipBit.getNode());
      }
      SDValue FlipBits =
          DAG.getNode(ISD::BUILD_PAIR, SDLoc(N0), VT, FlipBit, FlipBit);
      AddToWorklist(FlipBits.getNode());
      return DAG.getNode(ISD::XOR, DL, VT, NewConv, FlipBits);
    }
    APInt SignBit = APInt::getSignMask(VT.getSizeInBits());
    if (N0.getOpcode() == ISD::FNEG)
      return DAG.getNode(ISD::XOR, DL, VT,
                         NewConv, DAG.getConstant(SignBit, DL, VT));
    assert(N0.getOpcode() == ISD::FABS);
    return DAG.getNode(ISD::AND, DL, VT,
                       NewConv, DAG.getConstant(~SignBit, DL, VT));
  }

  // fold (bitconvert (fcopysign cst, x)) ->
  //         (or (and (bitconvert x), sign), (and cst, (not sign)))
  // Note that we don't handle (copysign x, cst) because this can always be
  // folded to an fneg or fabs.
  //
  // For ppc_fp128:
  // fold (bitcast (fcopysign cst, x)) ->
  //     flipbit = (and (extract_element
  //                     (xor (bitcast cst), (bitcast x)), 0),
  //                    signbit)
  //     (xor (bitcast cst) (build_pair flipbit, flipbit))
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse() &&
      isa<ConstantFPSDNode>(N0.getOperand(0)) &&
      VT.isInteger() && !VT.isVector()) {
    unsigned OrigXWidth = N0.getOperand(1).getValueSizeInBits();
    EVT IntXVT = EVT::getIntegerVT(*DAG.getContext(), OrigXWidth);
    if (isTypeLegal(IntXVT)) {
      SDValue X = DAG.getBitcast(IntXVT, N0.getOperand(1));
      AddToWorklist(X.getNode());

      // If X has a different width than the result/lhs, sext it or truncate it.
      unsigned VTWidth = VT.getSizeInBits();
      if (OrigXWidth < VTWidth) {
        X = DAG.getNode(ISD::SIGN_EXTEND, SDLoc(N), VT, X);
        AddToWorklist(X.getNode());
      } else if (OrigXWidth > VTWidth) {
        // To get the sign bit in the right place, we have to shift it right
        // before truncating.
        SDLoc DL(X);
        X = DAG.getNode(ISD::SRL, DL,
                        X.getValueType(), X,
                        DAG.getConstant(OrigXWidth-VTWidth, DL,
                                        X.getValueType()));
        AddToWorklist(X.getNode());
        X = DAG.getNode(ISD::TRUNCATE, SDLoc(X), VT, X);
        AddToWorklist(X.getNode());
      }

      if (N0.getValueType() == MVT::ppcf128 && !LegalTypes) {
        APInt SignBit = APInt::getSignMask(VT.getSizeInBits() / 2);
        SDValue Cst = DAG.getBitcast(VT, N0.getOperand(0));
        AddToWorklist(Cst.getNode());
        SDValue X = DAG.getBitcast(VT, N0.getOperand(1));
        AddToWorklist(X.getNode());
        SDValue XorResult = DAG.getNode(ISD::XOR, SDLoc(N0), VT, Cst, X);
        AddToWorklist(XorResult.getNode());
        SDValue XorResult64 = DAG.getNode(
            ISD::EXTRACT_ELEMENT, SDLoc(XorResult), MVT::i64, XorResult,
            DAG.getIntPtrConstant(getPPCf128HiElementSelector(DAG),
                                  SDLoc(XorResult)));
        AddToWorklist(XorResult64.getNode());
        SDValue FlipBit =
            DAG.getNode(ISD::AND, SDLoc(XorResult64), MVT::i64, XorResult64,
                        DAG.getConstant(SignBit, SDLoc(XorResult64), MVT::i64));
        AddToWorklist(FlipBit.getNode());
        SDValue FlipBits =
            DAG.getNode(ISD::BUILD_PAIR, SDLoc(N0), VT, FlipBit, FlipBit);
        AddToWorklist(FlipBits.getNode());
        return DAG.getNode(ISD::XOR, SDLoc(N), VT, Cst, FlipBits);
      }
      APInt SignBit = APInt::getSignMask(VT.getSizeInBits());
      X = DAG.getNode(ISD::AND, SDLoc(X), VT,
                      X, DAG.getConstant(SignBit, SDLoc(X), VT));
      AddToWorklist(X.getNode());

      SDValue Cst = DAG.getBitcast(VT, N0.getOperand(0));
      Cst = DAG.getNode(ISD::AND, SDLoc(Cst), VT,
                        Cst, DAG.getConstant(~SignBit, SDLoc(Cst), VT));
      AddToWorklist(Cst.getNode());

      return DAG.getNode(ISD::OR, SDLoc(N), VT, X, Cst);
    }
  }

  // bitconvert(build_pair(ld, ld)) -> ld iff load locations are consecutive.
  if (N0.getOpcode() == ISD::BUILD_PAIR)
    if (SDValue CombineLD = CombineConsecutiveLoads(N0.getNode(), VT))
      return CombineLD;

  // Remove double bitcasts from shuffles - this is often a legacy of
  // XformToShuffleWithZero being used to combine bitmaskings (of
  // float vectors bitcast to integer vectors) into shuffles.
  // bitcast(shuffle(bitcast(s0),bitcast(s1))) -> shuffle(s0,s1)
  if (Level < AfterLegalizeDAG && TLI.isTypeLegal(VT) && VT.isVector() &&
      N0->getOpcode() == ISD::VECTOR_SHUFFLE && N0.hasOneUse() &&
      VT.getVectorNumElements() >= N0.getValueType().getVectorNumElements() &&
      !(VT.getVectorNumElements() % N0.getValueType().getVectorNumElements())) {
    ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N0);

    // If operands are a bitcast, peek through if it casts the original VT.
    // If operands are a constant, just bitcast back to original VT.
    auto PeekThroughBitcast = [&](SDValue Op) {
      if (Op.getOpcode() == ISD::BITCAST &&
          Op.getOperand(0).getValueType() == VT)
        return SDValue(Op.getOperand(0));
      if (Op.isUndef() || ISD::isBuildVectorOfConstantSDNodes(Op.getNode()) ||
          ISD::isBuildVectorOfConstantFPSDNodes(Op.getNode()))
        return DAG.getBitcast(VT, Op);
      return SDValue();
    };

    // FIXME: If either input vector is bitcast, try to convert the shuffle to
    // the result type of this bitcast. This would eliminate at least one
    // bitcast. See the transform in InstCombine.
    SDValue SV0 = PeekThroughBitcast(N0->getOperand(0));
    SDValue SV1 = PeekThroughBitcast(N0->getOperand(1));
    if (!(SV0 && SV1))
      return SDValue();

    int MaskScale =
        VT.getVectorNumElements() / N0.getValueType().getVectorNumElements();
    SmallVector<int, 8> NewMask;
    for (int M : SVN->getMask())
      for (int i = 0; i != MaskScale; ++i)
        NewMask.push_back(M < 0 ? -1 : M * MaskScale + i);

    SDValue LegalShuffle =
        TLI.buildLegalVectorShuffle(VT, SDLoc(N), SV0, SV1, NewMask, DAG);
    if (LegalShuffle)
      return LegalShuffle;
  }

  return SDValue();
}

SDValue DAGCombiner::visitBUILD_PAIR(SDNode *N) {
  EVT VT = N->getValueType(0);
  return CombineConsecutiveLoads(N, VT);
}

SDValue DAGCombiner::visitFREEZE(SDNode *N) {
  SDValue N0 = N->getOperand(0);

  if (DAG.isGuaranteedNotToBeUndefOrPoison(N0, /*PoisonOnly*/ false))
    return N0;

  return SDValue();
}

/// We know that BV is a build_vector node with Constant, ConstantFP or Undef
/// operands. DstEltVT indicates the destination element value type.
SDValue DAGCombiner::
ConstantFoldBITCASTofBUILD_VECTOR(SDNode *BV, EVT DstEltVT) {
  EVT SrcEltVT = BV->getValueType(0).getVectorElementType();

  // If this is already the right type, we're done.
  if (SrcEltVT == DstEltVT) return SDValue(BV, 0);

  unsigned SrcBitSize = SrcEltVT.getSizeInBits();
  unsigned DstBitSize = DstEltVT.getSizeInBits();

  // If this is a conversion of N elements of one type to N elements of another
  // type, convert each element.  This handles FP<->INT cases.
  if (SrcBitSize == DstBitSize) {
    SmallVector<SDValue, 8> Ops;
    for (SDValue Op : BV->op_values()) {
      // If the vector element type is not legal, the BUILD_VECTOR operands
      // are promoted and implicitly truncated.  Make that explicit here.
      if (Op.getValueType() != SrcEltVT)
        Op = DAG.getNode(ISD::TRUNCATE, SDLoc(BV), SrcEltVT, Op);
      Ops.push_back(DAG.getBitcast(DstEltVT, Op));
      AddToWorklist(Ops.back().getNode());
    }
    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                              BV->getValueType(0).getVectorNumElements());
    return DAG.getBuildVector(VT, SDLoc(BV), Ops);
  }

  // Otherwise, we're growing or shrinking the elements.  To avoid having to
  // handle annoying details of growing/shrinking FP values, we convert them to
  // int first.
  if (SrcEltVT.isFloatingPoint()) {
    // Convert the input float vector to a int vector where the elements are the
    // same sizes.
    EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), SrcEltVT.getSizeInBits());
    BV = ConstantFoldBITCASTofBUILD_VECTOR(BV, IntVT).getNode();
    SrcEltVT = IntVT;
  }

  // Now we know the input is an integer vector.  If the output is a FP type,
  // convert to integer first, then to FP of the right size.
  if (DstEltVT.isFloatingPoint()) {
    EVT TmpVT = EVT::getIntegerVT(*DAG.getContext(), DstEltVT.getSizeInBits());
    SDNode *Tmp = ConstantFoldBITCASTofBUILD_VECTOR(BV, TmpVT).getNode();

    // Next, convert to FP elements of the same size.
    return ConstantFoldBITCASTofBUILD_VECTOR(Tmp, DstEltVT);
  }

  SDLoc DL(BV);

  // Okay, we know the src/dst types are both integers of differing types.
  // Handling growing first.
  assert(SrcEltVT.isInteger() && DstEltVT.isInteger());
  if (SrcBitSize < DstBitSize) {
    unsigned NumInputsPerOutput = DstBitSize/SrcBitSize;

    SmallVector<SDValue, 8> Ops;
    for (unsigned i = 0, e = BV->getNumOperands(); i != e;
         i += NumInputsPerOutput) {
      bool isLE = DAG.getDataLayout().isLittleEndian();
      APInt NewBits = APInt(DstBitSize, 0);
      bool EltIsUndef = true;
      for (unsigned j = 0; j != NumInputsPerOutput; ++j) {
        // Shift the previously computed bits over.
        NewBits <<= SrcBitSize;
        SDValue Op = BV->getOperand(i+ (isLE ? (NumInputsPerOutput-j-1) : j));
        if (Op.isUndef()) continue;
        EltIsUndef = false;

        NewBits |= cast<ConstantSDNode>(Op)->getAPIntValue().
                   zextOrTrunc(SrcBitSize).zext(DstBitSize);
      }

      if (EltIsUndef)
        Ops.push_back(DAG.getUNDEF(DstEltVT));
      else
        Ops.push_back(DAG.getConstant(NewBits, DL, DstEltVT));
    }

    EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT, Ops.size());
    return DAG.getBuildVector(VT, DL, Ops);
  }

  // Finally, this must be the case where we are shrinking elements: each input
  // turns into multiple outputs.
  unsigned NumOutputsPerInput = SrcBitSize/DstBitSize;
  EVT VT = EVT::getVectorVT(*DAG.getContext(), DstEltVT,
                            NumOutputsPerInput*BV->getNumOperands());
  SmallVector<SDValue, 8> Ops;

  for (const SDValue &Op : BV->op_values()) {
    if (Op.isUndef()) {
      Ops.append(NumOutputsPerInput, DAG.getUNDEF(DstEltVT));
      continue;
    }

    APInt OpVal = cast<ConstantSDNode>(Op)->
                  getAPIntValue().zextOrTrunc(SrcBitSize);

    for (unsigned j = 0; j != NumOutputsPerInput; ++j) {
      APInt ThisVal = OpVal.trunc(DstBitSize);
      Ops.push_back(DAG.getConstant(ThisVal, DL, DstEltVT));
      OpVal.lshrInPlace(DstBitSize);
    }

    // For big endian targets, swap the order of the pieces of each element.
    if (DAG.getDataLayout().isBigEndian())
      std::reverse(Ops.end()-NumOutputsPerInput, Ops.end());
  }

  return DAG.getBuildVector(VT, DL, Ops);
}

/// Try to perform FMA combining on a given FADD node.
SDValue DAGCombiner::visitFADDForFMACombine(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  const TargetOptions &Options = DAG.getTarget().Options;

  // Floating-point multiply-add with intermediate rounding.
  bool HasFMAD = (LegalOperations && TLI.isFMADLegal(DAG, N));

  // Floating-point multiply-add without intermediate rounding.
  bool HasFMA =
      TLI.isFMAFasterThanFMulAndFAdd(DAG.getMachineFunction(), VT) &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FMA, VT));

  // No valid opcode, do not combine.
  if (!HasFMAD && !HasFMA)
    return SDValue();

  bool CanReassociate =
      Options.UnsafeFPMath || N->getFlags().hasAllowReassociation();
  bool AllowFusionGlobally = (Options.AllowFPOpFusion == FPOpFusion::Fast ||
                              Options.UnsafeFPMath || HasFMAD);
  // If the addition is not contractable, do not combine.
  if (!AllowFusionGlobally && !N->getFlags().hasAllowContract())
    return SDValue();

  if (TLI.generateFMAsInMachineCombiner(VT, OptLevel))
    return SDValue();

  // Always prefer FMAD to FMA for precision.
  unsigned PreferredFusedOpcode = HasFMAD ? ISD::FMAD : ISD::FMA;
  bool Aggressive = TLI.enableAggressiveFMAFusion(VT);

  // Is the node an FMUL and contractable either due to global flags or
  // SDNodeFlags.
  auto isContractableFMUL = [AllowFusionGlobally](SDValue N) {
    if (N.getOpcode() != ISD::FMUL)
      return false;
    return AllowFusionGlobally || N->getFlags().hasAllowContract();
  };
  // If we have two choices trying to fold (fadd (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  if (Aggressive && isContractableFMUL(N0) && isContractableFMUL(N1)) {
    if (N0.getNode()->use_size() > N1.getNode()->use_size())
      std::swap(N0, N1);
  }

  // fold (fadd (fmul x, y), z) -> (fma x, y, z)
  if (isContractableFMUL(N0) && (Aggressive || N0->hasOneUse())) {
    return DAG.getNode(PreferredFusedOpcode, SL, VT, N0.getOperand(0),
                       N0.getOperand(1), N1);
  }

  // fold (fadd x, (fmul y, z)) -> (fma y, z, x)
  // Note: Commutes FADD operands.
  if (isContractableFMUL(N1) && (Aggressive || N1->hasOneUse())) {
    return DAG.getNode(PreferredFusedOpcode, SL, VT, N1.getOperand(0),
                       N1.getOperand(1), N0);
  }

  // fadd (fma A, B, (fmul C, D)), E --> fma A, B, (fma C, D, E)
  // fadd E, (fma A, B, (fmul C, D)) --> fma A, B, (fma C, D, E)
  // This requires reassociation because it changes the order of operations.
  SDValue FMA, E;
  if (CanReassociate && N0.getOpcode() == PreferredFusedOpcode &&
      N0.getOperand(2).getOpcode() == ISD::FMUL && N0.hasOneUse() &&
      N0.getOperand(2).hasOneUse()) {
    FMA = N0;
    E = N1;
  } else if (CanReassociate && N1.getOpcode() == PreferredFusedOpcode &&
             N1.getOperand(2).getOpcode() == ISD::FMUL && N1.hasOneUse() &&
             N1.getOperand(2).hasOneUse()) {
    FMA = N1;
    E = N0;
  }
  if (FMA && E) {
    SDValue A = FMA.getOperand(0);
    SDValue B = FMA.getOperand(1);
    SDValue C = FMA.getOperand(2).getOperand(0);
    SDValue D = FMA.getOperand(2).getOperand(1);
    SDValue CDE = DAG.getNode(PreferredFusedOpcode, SL, VT, C, D, E);
    return DAG.getNode(PreferredFusedOpcode, SL, VT, A, B, CDE);
  }

  // Look through FP_EXTEND nodes to do more combining.

  // fold (fadd (fpext (fmul x, y)), z) -> (fma (fpext x), (fpext y), z)
  if (N0.getOpcode() == ISD::FP_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (isContractableFMUL(N00) &&
        TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                            N00.getValueType())) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         DAG.getNode(ISD::FP_EXTEND, SL, VT, N00.getOperand(0)),
                         DAG.getNode(ISD::FP_EXTEND, SL, VT, N00.getOperand(1)),
                         N1);
    }
  }

  // fold (fadd x, (fpext (fmul y, z))) -> (fma (fpext y), (fpext z), x)
  // Note: Commutes FADD operands.
  if (N1.getOpcode() == ISD::FP_EXTEND) {
    SDValue N10 = N1.getOperand(0);
    if (isContractableFMUL(N10) &&
        TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                            N10.getValueType())) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         DAG.getNode(ISD::FP_EXTEND, SL, VT, N10.getOperand(0)),
                         DAG.getNode(ISD::FP_EXTEND, SL, VT, N10.getOperand(1)),
                         N0);
    }
  }

  // More folding opportunities when target permits.
  if (Aggressive) {
    // fold (fadd (fma x, y, (fpext (fmul u, v))), z)
    //   -> (fma x, y, (fma (fpext u), (fpext v), z))
    auto FoldFAddFMAFPExtFMul = [&](SDValue X, SDValue Y, SDValue U, SDValue V,
                                    SDValue Z) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT, X, Y,
                         DAG.getNode(PreferredFusedOpcode, SL, VT,
                                     DAG.getNode(ISD::FP_EXTEND, SL, VT, U),
                                     DAG.getNode(ISD::FP_EXTEND, SL, VT, V),
                                     Z));
    };
    if (N0.getOpcode() == PreferredFusedOpcode) {
      SDValue N02 = N0.getOperand(2);
      if (N02.getOpcode() == ISD::FP_EXTEND) {
        SDValue N020 = N02.getOperand(0);
        if (isContractableFMUL(N020) &&
            TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                                N020.getValueType())) {
          return FoldFAddFMAFPExtFMul(N0.getOperand(0), N0.getOperand(1),
                                      N020.getOperand(0), N020.getOperand(1),
                                      N1);
        }
      }
    }

    // fold (fadd (fpext (fma x, y, (fmul u, v))), z)
    //   -> (fma (fpext x), (fpext y), (fma (fpext u), (fpext v), z))
    // FIXME: This turns two single-precision and one double-precision
    // operation into two double-precision operations, which might not be
    // interesting for all targets, especially GPUs.
    auto FoldFAddFPExtFMAFMul = [&](SDValue X, SDValue Y, SDValue U, SDValue V,
                                    SDValue Z) {
      return DAG.getNode(
          PreferredFusedOpcode, SL, VT, DAG.getNode(ISD::FP_EXTEND, SL, VT, X),
          DAG.getNode(ISD::FP_EXTEND, SL, VT, Y),
          DAG.getNode(PreferredFusedOpcode, SL, VT,
                      DAG.getNode(ISD::FP_EXTEND, SL, VT, U),
                      DAG.getNode(ISD::FP_EXTEND, SL, VT, V), Z));
    };
    if (N0.getOpcode() == ISD::FP_EXTEND) {
      SDValue N00 = N0.getOperand(0);
      if (N00.getOpcode() == PreferredFusedOpcode) {
        SDValue N002 = N00.getOperand(2);
        if (isContractableFMUL(N002) &&
            TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                                N00.getValueType())) {
          return FoldFAddFPExtFMAFMul(N00.getOperand(0), N00.getOperand(1),
                                      N002.getOperand(0), N002.getOperand(1),
                                      N1);
        }
      }
    }

    // fold (fadd x, (fma y, z, (fpext (fmul u, v)))
    //   -> (fma y, z, (fma (fpext u), (fpext v), x))
    if (N1.getOpcode() == PreferredFusedOpcode) {
      SDValue N12 = N1.getOperand(2);
      if (N12.getOpcode() == ISD::FP_EXTEND) {
        SDValue N120 = N12.getOperand(0);
        if (isContractableFMUL(N120) &&
            TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                                N120.getValueType())) {
          return FoldFAddFMAFPExtFMul(N1.getOperand(0), N1.getOperand(1),
                                      N120.getOperand(0), N120.getOperand(1),
                                      N0);
        }
      }
    }

    // fold (fadd x, (fpext (fma y, z, (fmul u, v)))
    //   -> (fma (fpext y), (fpext z), (fma (fpext u), (fpext v), x))
    // FIXME: This turns two single-precision and one double-precision
    // operation into two double-precision operations, which might not be
    // interesting for all targets, especially GPUs.
    if (N1.getOpcode() == ISD::FP_EXTEND) {
      SDValue N10 = N1.getOperand(0);
      if (N10.getOpcode() == PreferredFusedOpcode) {
        SDValue N102 = N10.getOperand(2);
        if (isContractableFMUL(N102) &&
            TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                                N10.getValueType())) {
          return FoldFAddFPExtFMAFMul(N10.getOperand(0), N10.getOperand(1),
                                      N102.getOperand(0), N102.getOperand(1),
                                      N0);
        }
      }
    }
  }

  return SDValue();
}

/// Try to perform FMA combining on a given FSUB node.
SDValue DAGCombiner::visitFSUBForFMACombine(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  const TargetOptions &Options = DAG.getTarget().Options;
  // Floating-point multiply-add with intermediate rounding.
  bool HasFMAD = (LegalOperations && TLI.isFMADLegal(DAG, N));

  // Floating-point multiply-add without intermediate rounding.
  bool HasFMA =
      TLI.isFMAFasterThanFMulAndFAdd(DAG.getMachineFunction(), VT) &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FMA, VT));

  // No valid opcode, do not combine.
  if (!HasFMAD && !HasFMA)
    return SDValue();

  const SDNodeFlags Flags = N->getFlags();
  bool AllowFusionGlobally = (Options.AllowFPOpFusion == FPOpFusion::Fast ||
                              Options.UnsafeFPMath || HasFMAD);

  // If the subtraction is not contractable, do not combine.
  if (!AllowFusionGlobally && !N->getFlags().hasAllowContract())
    return SDValue();

  if (TLI.generateFMAsInMachineCombiner(VT, OptLevel))
    return SDValue();

  // Always prefer FMAD to FMA for precision.
  unsigned PreferredFusedOpcode = HasFMAD ? ISD::FMAD : ISD::FMA;
  bool Aggressive = TLI.enableAggressiveFMAFusion(VT);
  bool NoSignedZero = Options.NoSignedZerosFPMath || Flags.hasNoSignedZeros();

  // Is the node an FMUL and contractable either due to global flags or
  // SDNodeFlags.
  auto isContractableFMUL = [AllowFusionGlobally](SDValue N) {
    if (N.getOpcode() != ISD::FMUL)
      return false;
    return AllowFusionGlobally || N->getFlags().hasAllowContract();
  };

  // fold (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
  auto tryToFoldXYSubZ = [&](SDValue XY, SDValue Z) {
    if (isContractableFMUL(XY) && (Aggressive || XY->hasOneUse())) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT, XY.getOperand(0),
                         XY.getOperand(1), DAG.getNode(ISD::FNEG, SL, VT, Z));
    }
    return SDValue();
  };

  // fold (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
  // Note: Commutes FSUB operands.
  auto tryToFoldXSubYZ = [&](SDValue X, SDValue YZ) {
    if (isContractableFMUL(YZ) && (Aggressive || YZ->hasOneUse())) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         DAG.getNode(ISD::FNEG, SL, VT, YZ.getOperand(0)),
                         YZ.getOperand(1), X);
    }
    return SDValue();
  };

  // If we have two choices trying to fold (fsub (fmul u, v), (fmul x, y)),
  // prefer to fold the multiply with fewer uses.
  if (isContractableFMUL(N0) && isContractableFMUL(N1) &&
      (N0.getNode()->use_size() > N1.getNode()->use_size())) {
    // fold (fsub (fmul a, b), (fmul c, d)) -> (fma (fneg c), d, (fmul a, b))
    if (SDValue V = tryToFoldXSubYZ(N0, N1))
      return V;
    // fold (fsub (fmul a, b), (fmul c, d)) -> (fma a, b, (fneg (fmul c, d)))
    if (SDValue V = tryToFoldXYSubZ(N0, N1))
      return V;
  } else {
    // fold (fsub (fmul x, y), z) -> (fma x, y, (fneg z))
    if (SDValue V = tryToFoldXYSubZ(N0, N1))
      return V;
    // fold (fsub x, (fmul y, z)) -> (fma (fneg y), z, x)
    if (SDValue V = tryToFoldXSubYZ(N0, N1))
      return V;
  }

  // fold (fsub (fneg (fmul, x, y)), z) -> (fma (fneg x), y, (fneg z))
  if (N0.getOpcode() == ISD::FNEG && isContractableFMUL(N0.getOperand(0)) &&
      (Aggressive || (N0->hasOneUse() && N0.getOperand(0).hasOneUse()))) {
    SDValue N00 = N0.getOperand(0).getOperand(0);
    SDValue N01 = N0.getOperand(0).getOperand(1);
    return DAG.getNode(PreferredFusedOpcode, SL, VT,
                       DAG.getNode(ISD::FNEG, SL, VT, N00), N01,
                       DAG.getNode(ISD::FNEG, SL, VT, N1));
  }

  // Look through FP_EXTEND nodes to do more combining.

  // fold (fsub (fpext (fmul x, y)), z)
  //   -> (fma (fpext x), (fpext y), (fneg z))
  if (N0.getOpcode() == ISD::FP_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (isContractableFMUL(N00) &&
        TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                            N00.getValueType())) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT,
                         DAG.getNode(ISD::FP_EXTEND, SL, VT, N00.getOperand(0)),
                         DAG.getNode(ISD::FP_EXTEND, SL, VT, N00.getOperand(1)),
                         DAG.getNode(ISD::FNEG, SL, VT, N1));
    }
  }

  // fold (fsub x, (fpext (fmul y, z)))
  //   -> (fma (fneg (fpext y)), (fpext z), x)
  // Note: Commutes FSUB operands.
  if (N1.getOpcode() == ISD::FP_EXTEND) {
    SDValue N10 = N1.getOperand(0);
    if (isContractableFMUL(N10) &&
        TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                            N10.getValueType())) {
      return DAG.getNode(
          PreferredFusedOpcode, SL, VT,
          DAG.getNode(ISD::FNEG, SL, VT,
                      DAG.getNode(ISD::FP_EXTEND, SL, VT, N10.getOperand(0))),
          DAG.getNode(ISD::FP_EXTEND, SL, VT, N10.getOperand(1)), N0);
    }
  }

  // fold (fsub (fpext (fneg (fmul, x, y))), z)
  //   -> (fneg (fma (fpext x), (fpext y), z))
  // Note: This could be removed with appropriate canonicalization of the
  // input expression into (fneg (fadd (fpext (fmul, x, y)), z). However, the
  // orthogonal flags -fp-contract=fast and -enable-unsafe-fp-math prevent
  // from implementing the canonicalization in visitFSUB.
  if (N0.getOpcode() == ISD::FP_EXTEND) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getOpcode() == ISD::FNEG) {
      SDValue N000 = N00.getOperand(0);
      if (isContractableFMUL(N000) &&
          TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                              N00.getValueType())) {
        return DAG.getNode(
            ISD::FNEG, SL, VT,
            DAG.getNode(PreferredFusedOpcode, SL, VT,
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N000.getOperand(0)),
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N000.getOperand(1)),
                        N1));
      }
    }
  }

  // fold (fsub (fneg (fpext (fmul, x, y))), z)
  //   -> (fneg (fma (fpext x)), (fpext y), z)
  // Note: This could be removed with appropriate canonicalization of the
  // input expression into (fneg (fadd (fpext (fmul, x, y)), z). However, the
  // orthogonal flags -fp-contract=fast and -enable-unsafe-fp-math prevent
  // from implementing the canonicalization in visitFSUB.
  if (N0.getOpcode() == ISD::FNEG) {
    SDValue N00 = N0.getOperand(0);
    if (N00.getOpcode() == ISD::FP_EXTEND) {
      SDValue N000 = N00.getOperand(0);
      if (isContractableFMUL(N000) &&
          TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                              N000.getValueType())) {
        return DAG.getNode(
            ISD::FNEG, SL, VT,
            DAG.getNode(PreferredFusedOpcode, SL, VT,
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N000.getOperand(0)),
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N000.getOperand(1)),
                        N1));
      }
    }
  }

  auto isReassociable = [Options](SDNode *N) {
    return Options.UnsafeFPMath || N->getFlags().hasAllowReassociation();
  };

  auto isContractableAndReassociableFMUL = [isContractableFMUL,
                                            isReassociable](SDValue N) {
    return isContractableFMUL(N) && isReassociable(N.getNode());
  };

  // More folding opportunities when target permits.
  if (Aggressive && isReassociable(N)) {
    bool CanFuse = Options.UnsafeFPMath || N->getFlags().hasAllowContract();
    // fold (fsub (fma x, y, (fmul u, v)), z)
    //   -> (fma x, y (fma u, v, (fneg z)))
    if (CanFuse && N0.getOpcode() == PreferredFusedOpcode &&
        isContractableAndReassociableFMUL(N0.getOperand(2)) &&
        N0->hasOneUse() && N0.getOperand(2)->hasOneUse()) {
      return DAG.getNode(PreferredFusedOpcode, SL, VT, N0.getOperand(0),
                         N0.getOperand(1),
                         DAG.getNode(PreferredFusedOpcode, SL, VT,
                                     N0.getOperand(2).getOperand(0),
                                     N0.getOperand(2).getOperand(1),
                                     DAG.getNode(ISD::FNEG, SL, VT, N1)));
    }

    // fold (fsub x, (fma y, z, (fmul u, v)))
    //   -> (fma (fneg y), z, (fma (fneg u), v, x))
    if (CanFuse && N1.getOpcode() == PreferredFusedOpcode &&
        isContractableAndReassociableFMUL(N1.getOperand(2)) &&
        N1->hasOneUse() && NoSignedZero) {
      SDValue N20 = N1.getOperand(2).getOperand(0);
      SDValue N21 = N1.getOperand(2).getOperand(1);
      return DAG.getNode(
          PreferredFusedOpcode, SL, VT,
          DAG.getNode(ISD::FNEG, SL, VT, N1.getOperand(0)), N1.getOperand(1),
          DAG.getNode(PreferredFusedOpcode, SL, VT,
                      DAG.getNode(ISD::FNEG, SL, VT, N20), N21, N0));
    }

    // fold (fsub (fma x, y, (fpext (fmul u, v))), z)
    //   -> (fma x, y (fma (fpext u), (fpext v), (fneg z)))
    if (N0.getOpcode() == PreferredFusedOpcode &&
        N0->hasOneUse()) {
      SDValue N02 = N0.getOperand(2);
      if (N02.getOpcode() == ISD::FP_EXTEND) {
        SDValue N020 = N02.getOperand(0);
        if (isContractableAndReassociableFMUL(N020) &&
            TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                                N020.getValueType())) {
          return DAG.getNode(
              PreferredFusedOpcode, SL, VT, N0.getOperand(0), N0.getOperand(1),
              DAG.getNode(
                  PreferredFusedOpcode, SL, VT,
                  DAG.getNode(ISD::FP_EXTEND, SL, VT, N020.getOperand(0)),
                  DAG.getNode(ISD::FP_EXTEND, SL, VT, N020.getOperand(1)),
                  DAG.getNode(ISD::FNEG, SL, VT, N1)));
        }
      }
    }

    // fold (fsub (fpext (fma x, y, (fmul u, v))), z)
    //   -> (fma (fpext x), (fpext y),
    //           (fma (fpext u), (fpext v), (fneg z)))
    // FIXME: This turns two single-precision and one double-precision
    // operation into two double-precision operations, which might not be
    // interesting for all targets, especially GPUs.
    if (N0.getOpcode() == ISD::FP_EXTEND) {
      SDValue N00 = N0.getOperand(0);
      if (N00.getOpcode() == PreferredFusedOpcode) {
        SDValue N002 = N00.getOperand(2);
        if (isContractableAndReassociableFMUL(N002) &&
            TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                                N00.getValueType())) {
          return DAG.getNode(
              PreferredFusedOpcode, SL, VT,
              DAG.getNode(ISD::FP_EXTEND, SL, VT, N00.getOperand(0)),
              DAG.getNode(ISD::FP_EXTEND, SL, VT, N00.getOperand(1)),
              DAG.getNode(
                  PreferredFusedOpcode, SL, VT,
                  DAG.getNode(ISD::FP_EXTEND, SL, VT, N002.getOperand(0)),
                  DAG.getNode(ISD::FP_EXTEND, SL, VT, N002.getOperand(1)),
                  DAG.getNode(ISD::FNEG, SL, VT, N1)));
        }
      }
    }

    // fold (fsub x, (fma y, z, (fpext (fmul u, v))))
    //   -> (fma (fneg y), z, (fma (fneg (fpext u)), (fpext v), x))
    if (N1.getOpcode() == PreferredFusedOpcode &&
        N1.getOperand(2).getOpcode() == ISD::FP_EXTEND &&
        N1->hasOneUse()) {
      SDValue N120 = N1.getOperand(2).getOperand(0);
      if (isContractableAndReassociableFMUL(N120) &&
          TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                              N120.getValueType())) {
        SDValue N1200 = N120.getOperand(0);
        SDValue N1201 = N120.getOperand(1);
        return DAG.getNode(
            PreferredFusedOpcode, SL, VT,
            DAG.getNode(ISD::FNEG, SL, VT, N1.getOperand(0)), N1.getOperand(1),
            DAG.getNode(PreferredFusedOpcode, SL, VT,
                        DAG.getNode(ISD::FNEG, SL, VT,
                                    DAG.getNode(ISD::FP_EXTEND, SL, VT, N1200)),
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N1201), N0));
      }
    }

    // fold (fsub x, (fpext (fma y, z, (fmul u, v))))
    //   -> (fma (fneg (fpext y)), (fpext z),
    //           (fma (fneg (fpext u)), (fpext v), x))
    // FIXME: This turns two single-precision and one double-precision
    // operation into two double-precision operations, which might not be
    // interesting for all targets, especially GPUs.
    if (N1.getOpcode() == ISD::FP_EXTEND &&
        N1.getOperand(0).getOpcode() == PreferredFusedOpcode) {
      SDValue CvtSrc = N1.getOperand(0);
      SDValue N100 = CvtSrc.getOperand(0);
      SDValue N101 = CvtSrc.getOperand(1);
      SDValue N102 = CvtSrc.getOperand(2);
      if (isContractableAndReassociableFMUL(N102) &&
          TLI.isFPExtFoldable(DAG, PreferredFusedOpcode, VT,
                              CvtSrc.getValueType())) {
        SDValue N1020 = N102.getOperand(0);
        SDValue N1021 = N102.getOperand(1);
        return DAG.getNode(
            PreferredFusedOpcode, SL, VT,
            DAG.getNode(ISD::FNEG, SL, VT,
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N100)),
            DAG.getNode(ISD::FP_EXTEND, SL, VT, N101),
            DAG.getNode(PreferredFusedOpcode, SL, VT,
                        DAG.getNode(ISD::FNEG, SL, VT,
                                    DAG.getNode(ISD::FP_EXTEND, SL, VT, N1020)),
                        DAG.getNode(ISD::FP_EXTEND, SL, VT, N1021), N0));
      }
    }
  }

  return SDValue();
}

/// Try to perform FMA combining on a given FMUL node based on the distributive
/// law x * (y + 1) = x * y + x and variants thereof (commuted versions,
/// subtraction instead of addition).
SDValue DAGCombiner::visitFMULForFMADistributiveCombine(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  assert(N->getOpcode() == ISD::FMUL && "Expected FMUL Operation");

  const TargetOptions &Options = DAG.getTarget().Options;

  // The transforms below are incorrect when x == 0 and y == inf, because the
  // intermediate multiplication produces a nan.
  if (!Options.NoInfsFPMath)
    return SDValue();

  // Floating-point multiply-add without intermediate rounding.
  bool HasFMA =
      (Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath) &&
      TLI.isFMAFasterThanFMulAndFAdd(DAG.getMachineFunction(), VT) &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FMA, VT));

  // Floating-point multiply-add with intermediate rounding. This can result
  // in a less precise result due to the changed rounding order.
  bool HasFMAD = Options.UnsafeFPMath &&
                 (LegalOperations && TLI.isFMADLegal(DAG, N));

  // No valid opcode, do not combine.
  if (!HasFMAD && !HasFMA)
    return SDValue();

  // Always prefer FMAD to FMA for precision.
  unsigned PreferredFusedOpcode = HasFMAD ? ISD::FMAD : ISD::FMA;
  bool Aggressive = TLI.enableAggressiveFMAFusion(VT);

  // fold (fmul (fadd x0, +1.0), y) -> (fma x0, y, y)
  // fold (fmul (fadd x0, -1.0), y) -> (fma x0, y, (fneg y))
  auto FuseFADD = [&](SDValue X, SDValue Y) {
    if (X.getOpcode() == ISD::FADD && (Aggressive || X->hasOneUse())) {
      if (auto *C = isConstOrConstSplatFP(X.getOperand(1), true)) {
        if (C->isExactlyValue(+1.0))
          return DAG.getNode(PreferredFusedOpcode, SL, VT, X.getOperand(0), Y,
                             Y);
        if (C->isExactlyValue(-1.0))
          return DAG.getNode(PreferredFusedOpcode, SL, VT, X.getOperand(0), Y,
                             DAG.getNode(ISD::FNEG, SL, VT, Y));
      }
    }
    return SDValue();
  };

  if (SDValue FMA = FuseFADD(N0, N1))
    return FMA;
  if (SDValue FMA = FuseFADD(N1, N0))
    return FMA;

  // fold (fmul (fsub +1.0, x1), y) -> (fma (fneg x1), y, y)
  // fold (fmul (fsub -1.0, x1), y) -> (fma (fneg x1), y, (fneg y))
  // fold (fmul (fsub x0, +1.0), y) -> (fma x0, y, (fneg y))
  // fold (fmul (fsub x0, -1.0), y) -> (fma x0, y, y)
  auto FuseFSUB = [&](SDValue X, SDValue Y) {
    if (X.getOpcode() == ISD::FSUB && (Aggressive || X->hasOneUse())) {
      if (auto *C0 = isConstOrConstSplatFP(X.getOperand(0), true)) {
        if (C0->isExactlyValue(+1.0))
          return DAG.getNode(PreferredFusedOpcode, SL, VT,
                             DAG.getNode(ISD::FNEG, SL, VT, X.getOperand(1)), Y,
                             Y);
        if (C0->isExactlyValue(-1.0))
          return DAG.getNode(PreferredFusedOpcode, SL, VT,
                             DAG.getNode(ISD::FNEG, SL, VT, X.getOperand(1)), Y,
                             DAG.getNode(ISD::FNEG, SL, VT, Y));
      }
      if (auto *C1 = isConstOrConstSplatFP(X.getOperand(1), true)) {
        if (C1->isExactlyValue(+1.0))
          return DAG.getNode(PreferredFusedOpcode, SL, VT, X.getOperand(0), Y,
                             DAG.getNode(ISD::FNEG, SL, VT, Y));
        if (C1->isExactlyValue(-1.0))
          return DAG.getNode(PreferredFusedOpcode, SL, VT, X.getOperand(0), Y,
                             Y);
      }
    }
    return SDValue();
  };

  if (SDValue FMA = FuseFSUB(N0, N1))
    return FMA;
  if (SDValue FMA = FuseFSUB(N1, N0))
    return FMA;

  return SDValue();
}

SDValue DAGCombiner::visitFADD(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  bool N0CFP = DAG.isConstantFPBuildVectorOrConstantFP(N0);
  bool N1CFP = DAG.isConstantFPBuildVectorOrConstantFP(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;
  SDNodeFlags Flags = N->getFlags();
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  if (SDValue R = DAG.simplifyFPBinop(N->getOpcode(), N0, N1, Flags))
    return R;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (fadd c1, c2) -> c1 + c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FADD, DL, VT, N0, N1);

  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, DL, VT, N1, N0);

  // N0 + -0.0 --> N0 (also allowed with +0.0 and fast-math)
  ConstantFPSDNode *N1C = isConstOrConstSplatFP(N1, true);
  if (N1C && N1C->isZero())
    if (N1C->isNegative() || Options.NoSignedZerosFPMath || Flags.hasNoSignedZeros())
      return N0;

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // fold (fadd A, (fneg B)) -> (fsub A, B)
  if (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FSUB, VT))
    if (SDValue NegN1 = TLI.getCheaperNegatedExpression(
            N1, DAG, LegalOperations, ForCodeSize))
      return DAG.getNode(ISD::FSUB, DL, VT, N0, NegN1);

  // fold (fadd (fneg A), B) -> (fsub B, A)
  if (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::FSUB, VT))
    if (SDValue NegN0 = TLI.getCheaperNegatedExpression(
            N0, DAG, LegalOperations, ForCodeSize))
      return DAG.getNode(ISD::FSUB, DL, VT, N1, NegN0);

  auto isFMulNegTwo = [](SDValue FMul) {
    if (!FMul.hasOneUse() || FMul.getOpcode() != ISD::FMUL)
      return false;
    auto *C = isConstOrConstSplatFP(FMul.getOperand(1), true);
    return C && C->isExactlyValue(-2.0);
  };

  // fadd (fmul B, -2.0), A --> fsub A, (fadd B, B)
  if (isFMulNegTwo(N0)) {
    SDValue B = N0.getOperand(0);
    SDValue Add = DAG.getNode(ISD::FADD, DL, VT, B, B);
    return DAG.getNode(ISD::FSUB, DL, VT, N1, Add);
  }
  // fadd A, (fmul B, -2.0) --> fsub A, (fadd B, B)
  if (isFMulNegTwo(N1)) {
    SDValue B = N1.getOperand(0);
    SDValue Add = DAG.getNode(ISD::FADD, DL, VT, B, B);
    return DAG.getNode(ISD::FSUB, DL, VT, N0, Add);
  }

  // No FP constant should be created after legalization as Instruction
  // Selection pass has a hard time dealing with FP constants.
  bool AllowNewConst = (Level < AfterLegalizeDAG);

  // If nnan is enabled, fold lots of things.
  if ((Options.NoNaNsFPMath || Flags.hasNoNaNs()) && AllowNewConst) {
    // If allowed, fold (fadd (fneg x), x) -> 0.0
    if (N0.getOpcode() == ISD::FNEG && N0.getOperand(0) == N1)
      return DAG.getConstantFP(0.0, DL, VT);

    // If allowed, fold (fadd x, (fneg x)) -> 0.0
    if (N1.getOpcode() == ISD::FNEG && N1.getOperand(0) == N0)
      return DAG.getConstantFP(0.0, DL, VT);
  }

  // If 'unsafe math' or reassoc and nsz, fold lots of things.
  // TODO: break out portions of the transformations below for which Unsafe is
  //       considered and which do not require both nsz and reassoc
  if (((Options.UnsafeFPMath && Options.NoSignedZerosFPMath) ||
       (Flags.hasAllowReassociation() && Flags.hasNoSignedZeros())) &&
      AllowNewConst) {
    // fadd (fadd x, c1), c2 -> fadd x, c1 + c2
    if (N1CFP && N0.getOpcode() == ISD::FADD &&
        DAG.isConstantFPBuildVectorOrConstantFP(N0.getOperand(1))) {
      SDValue NewC = DAG.getNode(ISD::FADD, DL, VT, N0.getOperand(1), N1);
      return DAG.getNode(ISD::FADD, DL, VT, N0.getOperand(0), NewC);
    }

    // We can fold chains of FADD's of the same value into multiplications.
    // This transform is not safe in general because we are reducing the number
    // of rounding steps.
    if (TLI.isOperationLegalOrCustom(ISD::FMUL, VT) && !N0CFP && !N1CFP) {
      if (N0.getOpcode() == ISD::FMUL) {
        bool CFP00 = DAG.isConstantFPBuildVectorOrConstantFP(N0.getOperand(0));
        bool CFP01 = DAG.isConstantFPBuildVectorOrConstantFP(N0.getOperand(1));

        // (fadd (fmul x, c), x) -> (fmul x, c+1)
        if (CFP01 && !CFP00 && N0.getOperand(0) == N1) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, N0.getOperand(1),
                                       DAG.getConstantFP(1.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N1, NewCFP);
        }

        // (fadd (fmul x, c), (fadd x, x)) -> (fmul x, c+2)
        if (CFP01 && !CFP00 && N1.getOpcode() == ISD::FADD &&
            N1.getOperand(0) == N1.getOperand(1) &&
            N0.getOperand(0) == N1.getOperand(0)) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, N0.getOperand(1),
                                       DAG.getConstantFP(2.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N0.getOperand(0), NewCFP);
        }
      }

      if (N1.getOpcode() == ISD::FMUL) {
        bool CFP10 = DAG.isConstantFPBuildVectorOrConstantFP(N1.getOperand(0));
        bool CFP11 = DAG.isConstantFPBuildVectorOrConstantFP(N1.getOperand(1));

        // (fadd x, (fmul x, c)) -> (fmul x, c+1)
        if (CFP11 && !CFP10 && N1.getOperand(0) == N0) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, N1.getOperand(1),
                                       DAG.getConstantFP(1.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N0, NewCFP);
        }

        // (fadd (fadd x, x), (fmul x, c)) -> (fmul x, c+2)
        if (CFP11 && !CFP10 && N0.getOpcode() == ISD::FADD &&
            N0.getOperand(0) == N0.getOperand(1) &&
            N1.getOperand(0) == N0.getOperand(0)) {
          SDValue NewCFP = DAG.getNode(ISD::FADD, DL, VT, N1.getOperand(1),
                                       DAG.getConstantFP(2.0, DL, VT));
          return DAG.getNode(ISD::FMUL, DL, VT, N1.getOperand(0), NewCFP);
        }
      }

      if (N0.getOpcode() == ISD::FADD) {
        bool CFP00 = DAG.isConstantFPBuildVectorOrConstantFP(N0.getOperand(0));
        // (fadd (fadd x, x), x) -> (fmul x, 3.0)
        if (!CFP00 && N0.getOperand(0) == N0.getOperand(1) &&
            (N0.getOperand(0) == N1)) {
          return DAG.getNode(ISD::FMUL, DL, VT, N1,
                             DAG.getConstantFP(3.0, DL, VT));
        }
      }

      if (N1.getOpcode() == ISD::FADD) {
        bool CFP10 = DAG.isConstantFPBuildVectorOrConstantFP(N1.getOperand(0));
        // (fadd x, (fadd x, x)) -> (fmul x, 3.0)
        if (!CFP10 && N1.getOperand(0) == N1.getOperand(1) &&
            N1.getOperand(0) == N0) {
          return DAG.getNode(ISD::FMUL, DL, VT, N0,
                             DAG.getConstantFP(3.0, DL, VT));
        }
      }

      // (fadd (fadd x, x), (fadd x, x)) -> (fmul x, 4.0)
      if (N0.getOpcode() == ISD::FADD && N1.getOpcode() == ISD::FADD &&
          N0.getOperand(0) == N0.getOperand(1) &&
          N1.getOperand(0) == N1.getOperand(1) &&
          N0.getOperand(0) == N1.getOperand(0)) {
        return DAG.getNode(ISD::FMUL, DL, VT, N0.getOperand(0),
                           DAG.getConstantFP(4.0, DL, VT));
      }
    }
  } // enable-unsafe-fp-math

  // FADD -> FMA combines:
  if (SDValue Fused = visitFADDForFMACombine(N)) {
    AddToWorklist(Fused.getNode());
    return Fused;
  }
  return SDValue();
}

SDValue DAGCombiner::visitSTRICT_FADD(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  SDValue N0 = N->getOperand(1);
  SDValue N1 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  EVT ChainVT = N->getValueType(1);
  SDLoc DL(N);
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  // fold (strict_fadd A, (fneg B)) -> (strict_fsub A, B)
  if (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::STRICT_FSUB, VT))
    if (SDValue NegN1 = TLI.getCheaperNegatedExpression(
            N1, DAG, LegalOperations, ForCodeSize)) {
      return DAG.getNode(ISD::STRICT_FSUB, DL, DAG.getVTList(VT, ChainVT),
                         {Chain, N0, NegN1});
    }

  // fold (strict_fadd (fneg A), B) -> (strict_fsub B, A)
  if (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::STRICT_FSUB, VT))
    if (SDValue NegN0 = TLI.getCheaperNegatedExpression(
            N0, DAG, LegalOperations, ForCodeSize)) {
      return DAG.getNode(ISD::STRICT_FSUB, DL, DAG.getVTList(VT, ChainVT),
                         {Chain, N1, NegN0});
    }
  return SDValue();
}

SDValue DAGCombiner::visitFSUB(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = isConstOrConstSplatFP(N0, true);
  ConstantFPSDNode *N1CFP = isConstOrConstSplatFP(N1, true);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;
  const SDNodeFlags Flags = N->getFlags();
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  if (SDValue R = DAG.simplifyFPBinop(N->getOpcode(), N0, N1, Flags))
    return R;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FSUB, DL, VT, N0, N1);

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  // (fsub A, 0) -> A
  if (N1CFP && N1CFP->isZero()) {
    if (!N1CFP->isNegative() || Options.NoSignedZerosFPMath ||
        Flags.hasNoSignedZeros()) {
      return N0;
    }
  }

  if (N0 == N1) {
    // (fsub x, x) -> 0.0
    if (Options.NoNaNsFPMath || Flags.hasNoNaNs())
      return DAG.getConstantFP(0.0f, DL, VT);
  }

  // (fsub -0.0, N1) -> -N1
  if (N0CFP && N0CFP->isZero()) {
    if (N0CFP->isNegative() ||
        (Options.NoSignedZerosFPMath || Flags.hasNoSignedZeros())) {
      // We cannot replace an FSUB(+-0.0,X) with FNEG(X) when denormals are
      // flushed to zero, unless all users treat denorms as zero (DAZ).
      // FIXME: This transform will change the sign of a NaN and the behavior
      // of a signaling NaN. It is only valid when a NoNaN flag is present.
      DenormalMode DenormMode = DAG.getDenormalMode(VT);
      if (DenormMode == DenormalMode::getIEEE()) {
        if (SDValue NegN1 =
                TLI.getNegatedExpression(N1, DAG, LegalOperations, ForCodeSize))
          return NegN1;
        if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
          return DAG.getNode(ISD::FNEG, DL, VT, N1);
      }
    }
  }

  if (((Options.UnsafeFPMath && Options.NoSignedZerosFPMath) ||
       (Flags.hasAllowReassociation() && Flags.hasNoSignedZeros())) &&
      N1.getOpcode() == ISD::FADD) {
    // X - (X + Y) -> -Y
    if (N0 == N1->getOperand(0))
      return DAG.getNode(ISD::FNEG, DL, VT, N1->getOperand(1));
    // X - (Y + X) -> -Y
    if (N0 == N1->getOperand(1))
      return DAG.getNode(ISD::FNEG, DL, VT, N1->getOperand(0));
  }

  // fold (fsub A, (fneg B)) -> (fadd A, B)
  if (SDValue NegN1 =
          TLI.getNegatedExpression(N1, DAG, LegalOperations, ForCodeSize))
    return DAG.getNode(ISD::FADD, DL, VT, N0, NegN1);

  // FSUB -> FMA combines:
  if (SDValue Fused = visitFSUBForFMACombine(N)) {
    AddToWorklist(Fused.getNode());
    return Fused;
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMUL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = isConstOrConstSplatFP(N0, true);
  ConstantFPSDNode *N1CFP = isConstOrConstSplatFP(N1, true);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;
  const SDNodeFlags Flags = N->getFlags();
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  if (SDValue R = DAG.simplifyFPBinop(N->getOpcode(), N0, N1, Flags))
    return R;

  // fold vector ops
  if (VT.isVector()) {
    // This just handles C1 * C2 for vectors. Other vector folds are below.
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;
  }

  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FMUL, DL, VT, N0, N1);

  // canonicalize constant to RHS
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0) &&
     !DAG.isConstantFPBuildVectorOrConstantFP(N1))
    return DAG.getNode(ISD::FMUL, DL, VT, N1, N0);

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  if (Options.UnsafeFPMath || Flags.hasAllowReassociation()) {
    // fmul (fmul X, C1), C2 -> fmul X, C1 * C2
    if (DAG.isConstantFPBuildVectorOrConstantFP(N1) &&
        N0.getOpcode() == ISD::FMUL) {
      SDValue N00 = N0.getOperand(0);
      SDValue N01 = N0.getOperand(1);
      // Avoid an infinite loop by making sure that N00 is not a constant
      // (the inner multiply has not been constant folded yet).
      if (DAG.isConstantFPBuildVectorOrConstantFP(N01) &&
          !DAG.isConstantFPBuildVectorOrConstantFP(N00)) {
        SDValue MulConsts = DAG.getNode(ISD::FMUL, DL, VT, N01, N1);
        return DAG.getNode(ISD::FMUL, DL, VT, N00, MulConsts);
      }
    }

    // Match a special-case: we convert X * 2.0 into fadd.
    // fmul (fadd X, X), C -> fmul X, 2.0 * C
    if (N0.getOpcode() == ISD::FADD && N0.hasOneUse() &&
        N0.getOperand(0) == N0.getOperand(1)) {
      const SDValue Two = DAG.getConstantFP(2.0, DL, VT);
      SDValue MulConsts = DAG.getNode(ISD::FMUL, DL, VT, Two, N1);
      return DAG.getNode(ISD::FMUL, DL, VT, N0.getOperand(0), MulConsts);
    }
  }

  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, DL, VT, N0, N0);

  // fold (fmul X, -1.0) -> (fneg X)
  if (N1CFP && N1CFP->isExactlyValue(-1.0))
    if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
      return DAG.getNode(ISD::FNEG, DL, VT, N0);

  // -N0 * -N1 --> N0 * N1
  TargetLowering::NegatibleCost CostN0 =
      TargetLowering::NegatibleCost::Expensive;
  TargetLowering::NegatibleCost CostN1 =
      TargetLowering::NegatibleCost::Expensive;
  SDValue NegN0 =
      TLI.getNegatedExpression(N0, DAG, LegalOperations, ForCodeSize, CostN0);
  SDValue NegN1 =
      TLI.getNegatedExpression(N1, DAG, LegalOperations, ForCodeSize, CostN1);
  if (NegN0 && NegN1 &&
      (CostN0 == TargetLowering::NegatibleCost::Cheaper ||
       CostN1 == TargetLowering::NegatibleCost::Cheaper))
    return DAG.getNode(ISD::FMUL, DL, VT, NegN0, NegN1);

  // fold (fmul X, (select (fcmp X > 0.0), -1.0, 1.0)) -> (fneg (fabs X))
  // fold (fmul X, (select (fcmp X > 0.0), 1.0, -1.0)) -> (fabs X)
  if (Flags.hasNoNaNs() && Flags.hasNoSignedZeros() &&
      (N0.getOpcode() == ISD::SELECT || N1.getOpcode() == ISD::SELECT) &&
      TLI.isOperationLegal(ISD::FABS, VT)) {
    SDValue Select = N0, X = N1;
    if (Select.getOpcode() != ISD::SELECT)
      std::swap(Select, X);

    SDValue Cond = Select.getOperand(0);
    auto TrueOpnd  = dyn_cast<ConstantFPSDNode>(Select.getOperand(1));
    auto FalseOpnd = dyn_cast<ConstantFPSDNode>(Select.getOperand(2));

    if (TrueOpnd && FalseOpnd &&
        Cond.getOpcode() == ISD::SETCC && Cond.getOperand(0) == X &&
        isa<ConstantFPSDNode>(Cond.getOperand(1)) &&
        cast<ConstantFPSDNode>(Cond.getOperand(1))->isExactlyValue(0.0)) {
      ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
      switch (CC) {
      default: break;
      case ISD::SETOLT:
      case ISD::SETULT:
      case ISD::SETOLE:
      case ISD::SETULE:
      case ISD::SETLT:
      case ISD::SETLE:
        std::swap(TrueOpnd, FalseOpnd);
        LLVM_FALLTHROUGH;
      case ISD::SETOGT:
      case ISD::SETUGT:
      case ISD::SETOGE:
      case ISD::SETUGE:
      case ISD::SETGT:
      case ISD::SETGE:
        if (TrueOpnd->isExactlyValue(-1.0) && FalseOpnd->isExactlyValue(1.0) &&
            TLI.isOperationLegal(ISD::FNEG, VT))
          return DAG.getNode(ISD::FNEG, DL, VT,
                   DAG.getNode(ISD::FABS, DL, VT, X));
        if (TrueOpnd->isExactlyValue(1.0) && FalseOpnd->isExactlyValue(-1.0))
          return DAG.getNode(ISD::FABS, DL, VT, X);

        break;
      }
    }
  }

  // FMUL -> FMA combines:
  if (SDValue Fused = visitFMULForFMADistributiveCombine(N)) {
    AddToWorklist(Fused.getNode());
    return Fused;
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMA(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;
  // FMA nodes have flags that propagate to the created nodes.
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  bool UnsafeFPMath =
      Options.UnsafeFPMath || N->getFlags().hasAllowReassociation();

  // Constant fold FMA.
  if (isa<ConstantFPSDNode>(N0) &&
      isa<ConstantFPSDNode>(N1) &&
      isa<ConstantFPSDNode>(N2)) {
    return DAG.getNode(ISD::FMA, DL, VT, N0, N1, N2);
  }

  // (-N0 * -N1) + N2 --> (N0 * N1) + N2
  TargetLowering::NegatibleCost CostN0 =
      TargetLowering::NegatibleCost::Expensive;
  TargetLowering::NegatibleCost CostN1 =
      TargetLowering::NegatibleCost::Expensive;
  SDValue NegN0 =
      TLI.getNegatedExpression(N0, DAG, LegalOperations, ForCodeSize, CostN0);
  SDValue NegN1 =
      TLI.getNegatedExpression(N1, DAG, LegalOperations, ForCodeSize, CostN1);
  if (NegN0 && NegN1 &&
      (CostN0 == TargetLowering::NegatibleCost::Cheaper ||
       CostN1 == TargetLowering::NegatibleCost::Cheaper))
    return DAG.getNode(ISD::FMA, DL, VT, NegN0, NegN1, N2);

  if (UnsafeFPMath) {
    if (N0CFP && N0CFP->isZero())
      return N2;
    if (N1CFP && N1CFP->isZero())
      return N2;
  }

  if (N0CFP && N0CFP->isExactlyValue(1.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N1, N2);
  if (N1CFP && N1CFP->isExactlyValue(1.0))
    return DAG.getNode(ISD::FADD, SDLoc(N), VT, N0, N2);

  // Canonicalize (fma c, x, y) -> (fma x, c, y)
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0) &&
     !DAG.isConstantFPBuildVectorOrConstantFP(N1))
    return DAG.getNode(ISD::FMA, SDLoc(N), VT, N1, N0, N2);

  if (UnsafeFPMath) {
    // (fma x, c1, (fmul x, c2)) -> (fmul x, c1+c2)
    if (N2.getOpcode() == ISD::FMUL && N0 == N2.getOperand(0) &&
        DAG.isConstantFPBuildVectorOrConstantFP(N1) &&
        DAG.isConstantFPBuildVectorOrConstantFP(N2.getOperand(1))) {
      return DAG.getNode(ISD::FMUL, DL, VT, N0,
                         DAG.getNode(ISD::FADD, DL, VT, N1, N2.getOperand(1)));
    }

    // (fma (fmul x, c1), c2, y) -> (fma x, c1*c2, y)
    if (N0.getOpcode() == ISD::FMUL &&
        DAG.isConstantFPBuildVectorOrConstantFP(N1) &&
        DAG.isConstantFPBuildVectorOrConstantFP(N0.getOperand(1))) {
      return DAG.getNode(ISD::FMA, DL, VT, N0.getOperand(0),
                         DAG.getNode(ISD::FMUL, DL, VT, N1, N0.getOperand(1)),
                         N2);
    }
  }

  // (fma x, -1, y) -> (fadd (fneg x), y)
  if (N1CFP) {
    if (N1CFP->isExactlyValue(1.0))
      return DAG.getNode(ISD::FADD, DL, VT, N0, N2);

    if (N1CFP->isExactlyValue(-1.0) &&
        (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))) {
      SDValue RHSNeg = DAG.getNode(ISD::FNEG, DL, VT, N0);
      AddToWorklist(RHSNeg.getNode());
      return DAG.getNode(ISD::FADD, DL, VT, N2, RHSNeg);
    }

    // fma (fneg x), K, y -> fma x -K, y
    if (N0.getOpcode() == ISD::FNEG &&
        (TLI.isOperationLegal(ISD::ConstantFP, VT) ||
         (N1.hasOneUse() && !TLI.isFPImmLegal(N1CFP->getValueAPF(), VT,
                                              ForCodeSize)))) {
      return DAG.getNode(ISD::FMA, DL, VT, N0.getOperand(0),
                         DAG.getNode(ISD::FNEG, DL, VT, N1), N2);
    }
  }

  if (UnsafeFPMath) {
    // (fma x, c, x) -> (fmul x, (c+1))
    if (N1CFP && N0 == N2) {
      return DAG.getNode(
          ISD::FMUL, DL, VT, N0,
          DAG.getNode(ISD::FADD, DL, VT, N1, DAG.getConstantFP(1.0, DL, VT)));
    }

    // (fma x, c, (fneg x)) -> (fmul x, (c-1))
    if (N1CFP && N2.getOpcode() == ISD::FNEG && N2.getOperand(0) == N0) {
      return DAG.getNode(
          ISD::FMUL, DL, VT, N0,
          DAG.getNode(ISD::FADD, DL, VT, N1, DAG.getConstantFP(-1.0, DL, VT)));
    }
  }

  // fold ((fma (fneg X), Y, (fneg Z)) -> fneg (fma X, Y, Z))
  // fold ((fma X, (fneg Y), (fneg Z)) -> fneg (fma X, Y, Z))
  if (!TLI.isFNegFree(VT))
    if (SDValue Neg = TLI.getCheaperNegatedExpression(
            SDValue(N, 0), DAG, LegalOperations, ForCodeSize))
      return DAG.getNode(ISD::FNEG, DL, VT, Neg);
  return SDValue();
}

// Combine multiple FDIVs with the same divisor into multiple FMULs by the
// reciprocal.
// E.g., (a / D; b / D;) -> (recip = 1.0 / D; a * recip; b * recip)
// Notice that this is not always beneficial. One reason is different targets
// may have different costs for FDIV and FMUL, so sometimes the cost of two
// FDIVs may be lower than the cost of one FDIV and two FMULs. Another reason
// is the critical path is increased from "one FDIV" to "one FDIV + one FMUL".
SDValue DAGCombiner::combineRepeatedFPDivisors(SDNode *N) {
  // TODO: Limit this transform based on optsize/minsize - it always creates at
  //       least 1 extra instruction. But the perf win may be substantial enough
  //       that only minsize should restrict this.
  bool UnsafeMath = DAG.getTarget().Options.UnsafeFPMath;
  const SDNodeFlags Flags = N->getFlags();
  if (LegalDAG || (!UnsafeMath && !Flags.hasAllowReciprocal()))
    return SDValue();

  // Skip if current node is a reciprocal/fneg-reciprocal.
  SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = isConstOrConstSplatFP(N0, /* AllowUndefs */ true);
  if (N0CFP && (N0CFP->isExactlyValue(1.0) || N0CFP->isExactlyValue(-1.0)))
    return SDValue();

  // Exit early if the target does not want this transform or if there can't
  // possibly be enough uses of the divisor to make the transform worthwhile.
  unsigned MinUses = TLI.combineRepeatedFPDivisors();

  // For splat vectors, scale the number of uses by the splat factor. If we can
  // convert the division into a scalar op, that will likely be much faster.
  unsigned NumElts = 1;
  EVT VT = N->getValueType(0);
  if (VT.isVector() && DAG.isSplatValue(N1))
    NumElts = VT.getVectorNumElements();

  if (!MinUses || (N1->use_size() * NumElts) < MinUses)
    return SDValue();

  // Find all FDIV users of the same divisor.
  // Use a set because duplicates may be present in the user list.
  SetVector<SDNode *> Users;
  for (auto *U : N1->uses()) {
    if (U->getOpcode() == ISD::FDIV && U->getOperand(1) == N1) {
      // Skip X/sqrt(X) that has not been simplified to sqrt(X) yet.
      if (U->getOperand(1).getOpcode() == ISD::FSQRT &&
          U->getOperand(0) == U->getOperand(1).getOperand(0) &&
          U->getFlags().hasAllowReassociation() &&
          U->getFlags().hasNoSignedZeros())
        continue;

      // This division is eligible for optimization only if global unsafe math
      // is enabled or if this division allows reciprocal formation.
      if (UnsafeMath || U->getFlags().hasAllowReciprocal())
        Users.insert(U);
    }
  }

  // Now that we have the actual number of divisor uses, make sure it meets
  // the minimum threshold specified by the target.
  if ((Users.size() * NumElts) < MinUses)
    return SDValue();

  SDLoc DL(N);
  SDValue FPOne = DAG.getConstantFP(1.0, DL, VT);
  SDValue Reciprocal = DAG.getNode(ISD::FDIV, DL, VT, FPOne, N1, Flags);

  // Dividend / Divisor -> Dividend * Reciprocal
  for (auto *U : Users) {
    SDValue Dividend = U->getOperand(0);
    if (Dividend != FPOne) {
      SDValue NewNode = DAG.getNode(ISD::FMUL, SDLoc(U), VT, Dividend,
                                    Reciprocal, Flags);
      CombineTo(U, NewNode);
    } else if (U != Reciprocal.getNode()) {
      // In the absence of fast-math-flags, this user node is always the
      // same node as Reciprocal, but with FMF they may be different nodes.
      CombineTo(U, Reciprocal);
    }
  }
  return SDValue(N, 0);  // N was replaced.
}

SDValue DAGCombiner::visitFDIV(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);
  const TargetOptions &Options = DAG.getTarget().Options;
  SDNodeFlags Flags = N->getFlags();
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  if (SDValue R = DAG.simplifyFPBinop(N->getOpcode(), N0, N1, Flags))
    return R;

  // fold vector ops
  if (VT.isVector())
    if (SDValue FoldedVOp = SimplifyVBinOp(N))
      return FoldedVOp;

  // fold (fdiv c1, c2) -> c1/c2
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FDIV, SDLoc(N), VT, N0, N1);

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  if (SDValue V = combineRepeatedFPDivisors(N))
    return V;

  if (Options.UnsafeFPMath || Flags.hasAllowReciprocal()) {
    // fold (fdiv X, c2) -> fmul X, 1/c2 if losing precision is acceptable.
    if (N1CFP) {
      // Compute the reciprocal 1.0 / c2.
      const APFloat &N1APF = N1CFP->getValueAPF();
      APFloat Recip(N1APF.getSemantics(), 1); // 1.0
      APFloat::opStatus st = Recip.divide(N1APF, APFloat::rmNearestTiesToEven);
      // Only do the transform if the reciprocal is a legal fp immediate that
      // isn't too nasty (eg NaN, denormal, ...).
      if ((st == APFloat::opOK || st == APFloat::opInexact) && // Not too nasty
          (!LegalOperations ||
           // FIXME: custom lowering of ConstantFP might fail (see e.g. ARM
           // backend)... we should handle this gracefully after Legalize.
           // TLI.isOperationLegalOrCustom(ISD::ConstantFP, VT) ||
           TLI.isOperationLegal(ISD::ConstantFP, VT) ||
           TLI.isFPImmLegal(Recip, VT, ForCodeSize)))
        return DAG.getNode(ISD::FMUL, DL, VT, N0,
                           DAG.getConstantFP(Recip, DL, VT));
    }

    // If this FDIV is part of a reciprocal square root, it may be folded
    // into a target-specific square root estimate instruction.
    if (N1.getOpcode() == ISD::FSQRT) {
      if (SDValue RV = buildRsqrtEstimate(N1.getOperand(0), Flags))
        return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
    } else if (N1.getOpcode() == ISD::FP_EXTEND &&
               N1.getOperand(0).getOpcode() == ISD::FSQRT) {
      if (SDValue RV =
              buildRsqrtEstimate(N1.getOperand(0).getOperand(0), Flags)) {
        RV = DAG.getNode(ISD::FP_EXTEND, SDLoc(N1), VT, RV);
        AddToWorklist(RV.getNode());
        return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
      }
    } else if (N1.getOpcode() == ISD::FP_ROUND &&
               N1.getOperand(0).getOpcode() == ISD::FSQRT) {
      if (SDValue RV =
              buildRsqrtEstimate(N1.getOperand(0).getOperand(0), Flags)) {
        RV = DAG.getNode(ISD::FP_ROUND, SDLoc(N1), VT, RV, N1.getOperand(1));
        AddToWorklist(RV.getNode());
        return DAG.getNode(ISD::FMUL, DL, VT, N0, RV);
      }
    } else if (N1.getOpcode() == ISD::FMUL) {
      // Look through an FMUL. Even though this won't remove the FDIV directly,
      // it's still worthwhile to get rid of the FSQRT if possible.
      SDValue Sqrt, Y;
      if (N1.getOperand(0).getOpcode() == ISD::FSQRT) {
        Sqrt = N1.getOperand(0);
        Y = N1.getOperand(1);
      } else if (N1.getOperand(1).getOpcode() == ISD::FSQRT) {
        Sqrt = N1.getOperand(1);
        Y = N1.getOperand(0);
      }
      if (Sqrt.getNode()) {
        // If the other multiply operand is known positive, pull it into the
        // sqrt. That will eliminate the division if we convert to an estimate.
        if (Flags.hasAllowReassociation() && N1.hasOneUse() &&
            N1->getFlags().hasAllowReassociation() && Sqrt.hasOneUse()) {
          SDValue A;
          if (Y.getOpcode() == ISD::FABS && Y.hasOneUse())
            A = Y.getOperand(0);
          else if (Y == Sqrt.getOperand(0))
            A = Y;
          if (A) {
            // X / (fabs(A) * sqrt(Z)) --> X / sqrt(A*A*Z) --> X * rsqrt(A*A*Z)
            // X / (A * sqrt(A))       --> X / sqrt(A*A*A) --> X * rsqrt(A*A*A)
            SDValue AA = DAG.getNode(ISD::FMUL, DL, VT, A, A);
            SDValue AAZ =
                DAG.getNode(ISD::FMUL, DL, VT, AA, Sqrt.getOperand(0));
            if (SDValue Rsqrt = buildRsqrtEstimate(AAZ, Flags))
              return DAG.getNode(ISD::FMUL, DL, VT, N0, Rsqrt);

            // Estimate creation failed. Clean up speculatively created nodes.
            recursivelyDeleteUnusedNodes(AAZ.getNode());
          }
        }

        // We found a FSQRT, so try to make this fold:
        // X / (Y * sqrt(Z)) -> X * (rsqrt(Z) / Y)
        if (SDValue Rsqrt = buildRsqrtEstimate(Sqrt.getOperand(0), Flags)) {
          SDValue Div = DAG.getNode(ISD::FDIV, SDLoc(N1), VT, Rsqrt, Y);
          AddToWorklist(Div.getNode());
          return DAG.getNode(ISD::FMUL, DL, VT, N0, Div);
        }
      }
    }

    // Fold into a reciprocal estimate and multiply instead of a real divide.
    if (Options.NoInfsFPMath || Flags.hasNoInfs())
      if (SDValue RV = BuildDivEstimate(N0, N1, Flags))
        return RV;
  }

  // Fold X/Sqrt(X) -> Sqrt(X)
  if ((Options.NoSignedZerosFPMath || Flags.hasNoSignedZeros()) &&
      (Options.UnsafeFPMath || Flags.hasAllowReassociation()))
    if (N1.getOpcode() == ISD::FSQRT && N0 == N1.getOperand(0))
      return N1;

  // (fdiv (fneg X), (fneg Y)) -> (fdiv X, Y)
  TargetLowering::NegatibleCost CostN0 =
      TargetLowering::NegatibleCost::Expensive;
  TargetLowering::NegatibleCost CostN1 =
      TargetLowering::NegatibleCost::Expensive;
  SDValue NegN0 =
      TLI.getNegatedExpression(N0, DAG, LegalOperations, ForCodeSize, CostN0);
  SDValue NegN1 =
      TLI.getNegatedExpression(N1, DAG, LegalOperations, ForCodeSize, CostN1);
  if (NegN0 && NegN1 &&
      (CostN0 == TargetLowering::NegatibleCost::Cheaper ||
       CostN1 == TargetLowering::NegatibleCost::Cheaper))
    return DAG.getNode(ISD::FDIV, SDLoc(N), VT, NegN0, NegN1);

  return SDValue();
}

SDValue DAGCombiner::visitFREM(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  EVT VT = N->getValueType(0);
  SDNodeFlags Flags = N->getFlags();
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  if (SDValue R = DAG.simplifyFPBinop(N->getOpcode(), N0, N1, Flags))
    return R;

  // fold (frem c1, c2) -> fmod(c1,c2)
  if (N0CFP && N1CFP)
    return DAG.getNode(ISD::FREM, SDLoc(N), VT, N0, N1);

  if (SDValue NewSel = foldBinOpIntoSelect(N))
    return NewSel;

  return SDValue();
}

SDValue DAGCombiner::visitFSQRT(SDNode *N) {
  SDNodeFlags Flags = N->getFlags();
  const TargetOptions &Options = DAG.getTarget().Options;

  // Require 'ninf' flag since sqrt(+Inf) = +Inf, but the estimation goes as:
  // sqrt(+Inf) == rsqrt(+Inf) * +Inf = 0 * +Inf = NaN
  if (!Flags.hasApproximateFuncs() ||
      (!Options.NoInfsFPMath && !Flags.hasNoInfs()))
    return SDValue();

  SDValue N0 = N->getOperand(0);
  if (TLI.isFsqrtCheap(N0, DAG))
    return SDValue();

  // FSQRT nodes have flags that propagate to the created nodes.
  // TODO: If this is N0/sqrt(N0), and we reach this node before trying to
  //       transform the fdiv, we may produce a sub-optimal estimate sequence
  //       because the reciprocal calculation may not have to filter out a
  //       0.0 input.
  return buildSqrtEstimate(N0, Flags);
}

/// copysign(x, fp_extend(y)) -> copysign(x, y)
/// copysign(x, fp_round(y)) -> copysign(x, y)
static inline bool CanCombineFCOPYSIGN_EXTEND_ROUND(SDNode *N) {
  SDValue N1 = N->getOperand(1);
  if ((N1.getOpcode() == ISD::FP_EXTEND ||
       N1.getOpcode() == ISD::FP_ROUND)) {
    EVT N1VT = N1->getValueType(0);
    EVT N1Op0VT = N1->getOperand(0).getValueType();

    // Always fold no-op FP casts.
    if (N1VT == N1Op0VT)
      return true;

    // Do not optimize out type conversion of f128 type yet.
    // For some targets like x86_64, configuration is changed to keep one f128
    // value in one SSE register, but instruction selection cannot handle
    // FCOPYSIGN on SSE registers yet.
    if (N1Op0VT == MVT::f128)
      return false;

    // Avoid mismatched vector operand types, for better instruction selection.
    if (N1Op0VT.isVector())
      return false;

    return true;
  }
  return false;
}

SDValue DAGCombiner::visitFCOPYSIGN(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  bool N0CFP = DAG.isConstantFPBuildVectorOrConstantFP(N0);
  bool N1CFP = DAG.isConstantFPBuildVectorOrConstantFP(N1);
  EVT VT = N->getValueType(0);

  if (N0CFP && N1CFP) // Constant fold
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT, N0, N1);

  if (ConstantFPSDNode *N1C = isConstOrConstSplatFP(N->getOperand(1))) {
    const APFloat &V = N1C->getValueAPF();
    // copysign(x, c1) -> fabs(x)       iff ispos(c1)
    // copysign(x, c1) -> fneg(fabs(x)) iff isneg(c1)
    if (!V.isNegative()) {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FABS, VT))
        return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);
    } else {
      if (!LegalOperations || TLI.isOperationLegal(ISD::FNEG, VT))
        return DAG.getNode(ISD::FNEG, SDLoc(N), VT,
                           DAG.getNode(ISD::FABS, SDLoc(N0), VT, N0));
    }
  }

  // copysign(fabs(x), y) -> copysign(x, y)
  // copysign(fneg(x), y) -> copysign(x, y)
  // copysign(copysign(x,z), y) -> copysign(x, y)
  if (N0.getOpcode() == ISD::FABS || N0.getOpcode() == ISD::FNEG ||
      N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT, N0.getOperand(0), N1);

  // copysign(x, abs(y)) -> abs(x)
  if (N1.getOpcode() == ISD::FABS)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);

  // copysign(x, copysign(y,z)) -> copysign(x, z)
  if (N1.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT, N0, N1.getOperand(1));

  // copysign(x, fp_extend(y)) -> copysign(x, y)
  // copysign(x, fp_round(y)) -> copysign(x, y)
  if (CanCombineFCOPYSIGN_EXTEND_ROUND(N))
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT, N0, N1.getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitFPOW(SDNode *N) {
  ConstantFPSDNode *ExponentC = isConstOrConstSplatFP(N->getOperand(1));
  if (!ExponentC)
    return SDValue();
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  // Try to convert x ** (1/3) into cube root.
  // TODO: Handle the various flavors of long double.
  // TODO: Since we're approximating, we don't need an exact 1/3 exponent.
  //       Some range near 1/3 should be fine.
  EVT VT = N->getValueType(0);
  if ((VT == MVT::f32 && ExponentC->getValueAPF().isExactlyValue(1.0f/3.0f)) ||
      (VT == MVT::f64 && ExponentC->getValueAPF().isExactlyValue(1.0/3.0))) {
    // pow(-0.0, 1/3) = +0.0; cbrt(-0.0) = -0.0.
    // pow(-inf, 1/3) = +inf; cbrt(-inf) = -inf.
    // pow(-val, 1/3) =  nan; cbrt(-val) = -num.
    // For regular numbers, rounding may cause the results to differ.
    // Therefore, we require { nsz ninf nnan afn } for this transform.
    // TODO: We could select out the special cases if we don't have nsz/ninf.
    SDNodeFlags Flags = N->getFlags();
    if (!Flags.hasNoSignedZeros() || !Flags.hasNoInfs() || !Flags.hasNoNaNs() ||
        !Flags.hasApproximateFuncs())
      return SDValue();

    // Do not create a cbrt() libcall if the target does not have it, and do not
    // turn a pow that has lowering support into a cbrt() libcall.
    if (!DAG.getLibInfo().has(LibFunc_cbrt) ||
        (!DAG.getTargetLoweringInfo().isOperationExpand(ISD::FPOW, VT) &&
         DAG.getTargetLoweringInfo().isOperationExpand(ISD::FCBRT, VT)))
      return SDValue();

    return DAG.getNode(ISD::FCBRT, SDLoc(N), VT, N->getOperand(0));
  }

  // Try to convert x ** (1/4) and x ** (3/4) into square roots.
  // x ** (1/2) is canonicalized to sqrt, so we do not bother with that case.
  // TODO: This could be extended (using a target hook) to handle smaller
  // power-of-2 fractional exponents.
  bool ExponentIs025 = ExponentC->getValueAPF().isExactlyValue(0.25);
  bool ExponentIs075 = ExponentC->getValueAPF().isExactlyValue(0.75);
  if (ExponentIs025 || ExponentIs075) {
    // pow(-0.0, 0.25) = +0.0; sqrt(sqrt(-0.0)) = -0.0.
    // pow(-inf, 0.25) = +inf; sqrt(sqrt(-inf)) =  NaN.
    // pow(-0.0, 0.75) = +0.0; sqrt(-0.0) * sqrt(sqrt(-0.0)) = +0.0.
    // pow(-inf, 0.75) = +inf; sqrt(-inf) * sqrt(sqrt(-inf)) =  NaN.
    // For regular numbers, rounding may cause the results to differ.
    // Therefore, we require { nsz ninf afn } for this transform.
    // TODO: We could select out the special cases if we don't have nsz/ninf.
    SDNodeFlags Flags = N->getFlags();

    // We only need no signed zeros for the 0.25 case.
    if ((!Flags.hasNoSignedZeros() && ExponentIs025) || !Flags.hasNoInfs() ||
        !Flags.hasApproximateFuncs())
      return SDValue();

    // Don't double the number of libcalls. We are trying to inline fast code.
    if (!DAG.getTargetLoweringInfo().isOperationLegalOrCustom(ISD::FSQRT, VT))
      return SDValue();

    // Assume that libcalls are the smallest code.
    // TODO: This restriction should probably be lifted for vectors.
    if (ForCodeSize)
      return SDValue();

    // pow(X, 0.25) --> sqrt(sqrt(X))
    SDLoc DL(N);
    SDValue Sqrt = DAG.getNode(ISD::FSQRT, DL, VT, N->getOperand(0));
    SDValue SqrtSqrt = DAG.getNode(ISD::FSQRT, DL, VT, Sqrt);
    if (ExponentIs025)
      return SqrtSqrt;
    // pow(X, 0.75) --> sqrt(X) * sqrt(sqrt(X))
    return DAG.getNode(ISD::FMUL, DL, VT, Sqrt, SqrtSqrt);
  }

  return SDValue();
}

static SDValue foldFPToIntToFP(SDNode *N, SelectionDAG &DAG,
                               const TargetLowering &TLI) {
  // This optimization is guarded by a function attribute because it may produce
  // unexpected results. Ie, programs may be relying on the platform-specific
  // undefined behavior when the float-to-int conversion overflows.
  const Function &F = DAG.getMachineFunction().getFunction();
  Attribute StrictOverflow = F.getFnAttribute("strict-float-cast-overflow");
  if (StrictOverflow.getValueAsString().equals("false"))
    return SDValue();

  // We only do this if the target has legal ftrunc. Otherwise, we'd likely be
  // replacing casts with a libcall. We also must be allowed to ignore -0.0
  // because FTRUNC will return -0.0 for (-1.0, -0.0), but using integer
  // conversions would return +0.0.
  // FIXME: We should be able to use node-level FMF here.
  // TODO: If strict math, should we use FABS (+ range check for signed cast)?
  EVT VT = N->getValueType(0);
  if (!TLI.isOperationLegal(ISD::FTRUNC, VT) ||
      !DAG.getTarget().Options.NoSignedZerosFPMath)
    return SDValue();

  // fptosi/fptoui round towards zero, so converting from FP to integer and
  // back is the same as an 'ftrunc': [us]itofp (fpto[us]i X) --> ftrunc X
  SDValue N0 = N->getOperand(0);
  if (N->getOpcode() == ISD::SINT_TO_FP && N0.getOpcode() == ISD::FP_TO_SINT &&
      N0.getOperand(0).getValueType() == VT)
    return DAG.getNode(ISD::FTRUNC, SDLoc(N), VT, N0.getOperand(0));

  if (N->getOpcode() == ISD::UINT_TO_FP && N0.getOpcode() == ISD::FP_TO_UINT &&
      N0.getOperand(0).getValueType() == VT)
    return DAG.getNode(ISD::FTRUNC, SDLoc(N), VT, N0.getOperand(0));

  return SDValue();
}

SDValue DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // [us]itofp(undef) = 0, because the result value is bounded.
  if (N0.isUndef())
    return DAG.getConstantFP(0.0, SDLoc(N), VT);

  // fold (sint_to_fp c1) -> c1fp
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      // ...but only if the target supports immediate floating-point values
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(ISD::ConstantFP, VT)))
    return DAG.getNode(ISD::SINT_TO_FP, SDLoc(N), VT, N0);

  // If the input is a legal type, and SINT_TO_FP is not legal on this target,
  // but UINT_TO_FP is legal on this target, try to convert.
  if (!hasOperation(ISD::SINT_TO_FP, OpVT) &&
      hasOperation(ISD::UINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to UINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), VT, N0);
  }

  // The next optimizations are desirable only if SELECT_CC can be lowered.
  // fold (sint_to_fp (setcc x, y, cc)) -> (select (setcc x, y, cc), -1.0, 0.0)
  if (N0.getOpcode() == ISD::SETCC && N0.getValueType() == MVT::i1 &&
      !VT.isVector() &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::ConstantFP, VT))) {
    SDLoc DL(N);
    return DAG.getSelect(DL, VT, N0, DAG.getConstantFP(-1.0, DL, VT),
                         DAG.getConstantFP(0.0, DL, VT));
  }

  // fold (sint_to_fp (zext (setcc x, y, cc))) ->
  //      (select (setcc x, y, cc), 1.0, 0.0)
  if (N0.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getOperand(0).getOpcode() == ISD::SETCC && !VT.isVector() &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::ConstantFP, VT))) {
    SDLoc DL(N);
    return DAG.getSelect(DL, VT, N0.getOperand(0),
                         DAG.getConstantFP(1.0, DL, VT),
                         DAG.getConstantFP(0.0, DL, VT));
  }

  if (SDValue FTrunc = foldFPToIntToFP(N, DAG, TLI))
    return FTrunc;

  return SDValue();
}

SDValue DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  EVT OpVT = N0.getValueType();

  // [us]itofp(undef) = 0, because the result value is bounded.
  if (N0.isUndef())
    return DAG.getConstantFP(0.0, SDLoc(N), VT);

  // fold (uint_to_fp c1) -> c1fp
  if (DAG.isConstantIntBuildVectorOrConstantInt(N0) &&
      // ...but only if the target supports immediate floating-point values
      (!LegalOperations ||
       TLI.isOperationLegalOrCustom(ISD::ConstantFP, VT)))
    return DAG.getNode(ISD::UINT_TO_FP, SDLoc(N), VT, N0);

  // If the input is a legal type, and UINT_TO_FP is not legal on this target,
  // but SINT_TO_FP is legal on this target, try to convert.
  if (!hasOperation(ISD::UINT_TO_FP, OpVT) &&
      hasOperation(ISD::SINT_TO_FP, OpVT)) {
    // If the sign bit is known to be zero, we can change this to SINT_TO_FP.
    if (DAG.SignBitIsZero(N0))
      return DAG.getNode(ISD::SINT_TO_FP, SDLoc(N), VT, N0);
  }

  // fold (uint_to_fp (setcc x, y, cc)) -> (select (setcc x, y, cc), 1.0, 0.0)
  if (N0.getOpcode() == ISD::SETCC && !VT.isVector() &&
      (!LegalOperations || TLI.isOperationLegalOrCustom(ISD::ConstantFP, VT))) {
    SDLoc DL(N);
    return DAG.getSelect(DL, VT, N0, DAG.getConstantFP(1.0, DL, VT),
                         DAG.getConstantFP(0.0, DL, VT));
  }

  if (SDValue FTrunc = foldFPToIntToFP(N, DAG, TLI))
    return FTrunc;

  return SDValue();
}

// Fold (fp_to_{s/u}int ({s/u}int_to_fpx)) -> zext x, sext x, trunc x, or x
static SDValue FoldIntToFPToInt(SDNode *N, SelectionDAG &DAG) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  if (N0.getOpcode() != ISD::UINT_TO_FP && N0.getOpcode() != ISD::SINT_TO_FP)
    return SDValue();

  SDValue Src = N0.getOperand(0);
  EVT SrcVT = Src.getValueType();
  bool IsInputSigned = N0.getOpcode() == ISD::SINT_TO_FP;
  bool IsOutputSigned = N->getOpcode() == ISD::FP_TO_SINT;

  // We can safely assume the conversion won't overflow the output range,
  // because (for example) (uint8_t)18293.f is undefined behavior.

  // Since we can assume the conversion won't overflow, our decision as to
  // whether the input will fit in the float should depend on the minimum
  // of the input range and output range.

  // This means this is also safe for a signed input and unsigned output, since
  // a negative input would lead to undefined behavior.
  unsigned InputSize = (int)SrcVT.getScalarSizeInBits() - IsInputSigned;
  unsigned OutputSize = (int)VT.getScalarSizeInBits() - IsOutputSigned;
  unsigned ActualSize = std::min(InputSize, OutputSize);
  const fltSemantics &sem = DAG.EVTToAPFloatSemantics(N0.getValueType());

  // We can only fold away the float conversion if the input range can be
  // represented exactly in the float range.
  if (APFloat::semanticsPrecision(sem) >= ActualSize) {
    if (VT.getScalarSizeInBits() > SrcVT.getScalarSizeInBits()) {
      unsigned ExtOp = IsInputSigned && IsOutputSigned ? ISD::SIGN_EXTEND
                                                       : ISD::ZERO_EXTEND;
      return DAG.getNode(ExtOp, SDLoc(N), VT, Src);
    }
    if (VT.getScalarSizeInBits() < SrcVT.getScalarSizeInBits())
      return DAG.getNode(ISD::TRUNCATE, SDLoc(N), VT, Src);
    return DAG.getBitcast(VT, Src);
  }
  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_sint undef) -> undef
  if (N0.isUndef())
    return DAG.getUNDEF(VT);

  // fold (fp_to_sint c1fp) -> c1
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FP_TO_SINT, SDLoc(N), VT, N0);

  return FoldIntToFPToInt(N, DAG);
}

SDValue DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fp_to_uint undef) -> undef
  if (N0.isUndef())
    return DAG.getUNDEF(VT);

  // fold (fp_to_uint c1fp) -> c1
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FP_TO_UINT, SDLoc(N), VT, N0);

  return FoldIntToFPToInt(N, DAG);
}

SDValue DAGCombiner::visitFP_ROUND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  EVT VT = N->getValueType(0);

  // fold (fp_round c1fp) -> c1fp
  if (N0CFP)
    return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT, N0, N1);

  // fold (fp_round (fp_extend x)) -> x
  if (N0.getOpcode() == ISD::FP_EXTEND && VT == N0.getOperand(0).getValueType())
    return N0.getOperand(0);

  // fold (fp_round (fp_round x)) -> (fp_round x)
  if (N0.getOpcode() == ISD::FP_ROUND) {
    const bool NIsTrunc = N->getConstantOperandVal(1) == 1;
    const bool N0IsTrunc = N0.getConstantOperandVal(1) == 1;

    // Skip this folding if it results in an fp_round from f80 to f16.
    //
    // f80 to f16 always generates an expensive (and as yet, unimplemented)
    // libcall to __truncxfhf2 instead of selecting native f16 conversion
    // instructions from f32 or f64.  Moreover, the first (value-preserving)
    // fp_round from f80 to either f32 or f64 may become a NOP in platforms like
    // x86.
    if (N0.getOperand(0).getValueType() == MVT::f80 && VT == MVT::f16)
      return SDValue();

    // If the first fp_round isn't a value preserving truncation, it might
    // introduce a tie in the second fp_round, that wouldn't occur in the
    // single-step fp_round we want to fold to.
    // In other words, double rounding isn't the same as rounding.
    // Also, this is a value preserving truncation iff both fp_round's are.
    if (DAG.getTarget().Options.UnsafeFPMath || N0IsTrunc) {
      SDLoc DL(N);
      return DAG.getNode(ISD::FP_ROUND, DL, VT, N0.getOperand(0),
                         DAG.getIntPtrConstant(NIsTrunc && N0IsTrunc, DL));
    }
  }

  // fold (fp_round (copysign X, Y)) -> (copysign (fp_round X), Y)
  if (N0.getOpcode() == ISD::FCOPYSIGN && N0.getNode()->hasOneUse()) {
    SDValue Tmp = DAG.getNode(ISD::FP_ROUND, SDLoc(N0), VT,
                              N0.getOperand(0), N1);
    AddToWorklist(Tmp.getNode());
    return DAG.getNode(ISD::FCOPYSIGN, SDLoc(N), VT,
                       Tmp, N0.getOperand(1));
  }

  if (SDValue NewVSel = matchVSelectOpSizesWithSetCC(N))
    return NewVSel;

  return SDValue();
}

SDValue DAGCombiner::visitFP_EXTEND(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // If this is fp_round(fpextend), don't fold it, allow ourselves to be folded.
  if (N->hasOneUse() &&
      N->use_begin()->getOpcode() == ISD::FP_ROUND)
    return SDValue();

  // fold (fp_extend c1fp) -> c1fp
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, N0);

  // fold (fp_extend (fp16_to_fp op)) -> (fp16_to_fp op)
  if (N0.getOpcode() == ISD::FP16_TO_FP &&
      TLI.getOperationAction(ISD::FP16_TO_FP, VT) == TargetLowering::Legal)
    return DAG.getNode(ISD::FP16_TO_FP, SDLoc(N), VT, N0.getOperand(0));

  // Turn fp_extend(fp_round(X, 1)) -> x since the fp_round doesn't affect the
  // value of X.
  if (N0.getOpcode() == ISD::FP_ROUND
      && N0.getConstantOperandVal(1) == 1) {
    SDValue In = N0.getOperand(0);
    if (In.getValueType() == VT) return In;
    if (VT.bitsLT(In.getValueType()))
      return DAG.getNode(ISD::FP_ROUND, SDLoc(N), VT,
                         In, N0.getOperand(1));
    return DAG.getNode(ISD::FP_EXTEND, SDLoc(N), VT, In);
  }

  // fold (fpext (load x)) -> (fpext (fptrunc (extload x)))
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
       TLI.isLoadExtLegal(ISD::EXTLOAD, VT, N0.getValueType())) {
    LoadSDNode *LN0 = cast<LoadSDNode>(N0);
    SDValue ExtLoad = DAG.getExtLoad(ISD::EXTLOAD, SDLoc(N), VT,
                                     LN0->getChain(),
                                     LN0->getBasePtr(), N0.getValueType(),
                                     LN0->getMemOperand());
    CombineTo(N, ExtLoad);
    CombineTo(N0.getNode(),
              DAG.getNode(ISD::FP_ROUND, SDLoc(N0),
                          N0.getValueType(), ExtLoad,
                          DAG.getIntPtrConstant(1, SDLoc(N0))),
              ExtLoad.getValue(1));
    return SDValue(N, 0);   // Return N so it doesn't get rechecked!
  }

  if (SDValue NewVSel = matchVSelectOpSizesWithSetCC(N))
    return NewVSel;

  return SDValue();
}

SDValue DAGCombiner::visitFCEIL(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fceil c1) -> fceil(c1)
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FCEIL, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFTRUNC(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ftrunc c1) -> ftrunc(c1)
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FTRUNC, SDLoc(N), VT, N0);

  // fold ftrunc (known rounded int x) -> x
  // ftrunc is a part of fptosi/fptoui expansion on some targets, so this is
  // likely to be generated to extract integer from a rounded floating value.
  switch (N0.getOpcode()) {
  default: break;
  case ISD::FRINT:
  case ISD::FTRUNC:
  case ISD::FNEARBYINT:
  case ISD::FFLOOR:
  case ISD::FCEIL:
    return N0;
  }

  return SDValue();
}

SDValue DAGCombiner::visitFFLOOR(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (ffloor c1) -> ffloor(c1)
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FFLOOR, SDLoc(N), VT, N0);

  return SDValue();
}

SDValue DAGCombiner::visitFNEG(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  // Constant fold FNEG.
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FNEG, SDLoc(N), VT, N0);

  if (SDValue NegN0 =
          TLI.getNegatedExpression(N0, DAG, LegalOperations, ForCodeSize))
    return NegN0;

  // -(X-Y) -> (Y-X) is unsafe because when X==Y, -0.0 != +0.0
  // FIXME: This is duplicated in getNegatibleCost, but getNegatibleCost doesn't
  // know it was called from a context with a nsz flag if the input fsub does
  // not.
  if (N0.getOpcode() == ISD::FSUB &&
      (DAG.getTarget().Options.NoSignedZerosFPMath ||
       N->getFlags().hasNoSignedZeros()) && N0.hasOneUse()) {
    return DAG.getNode(ISD::FSUB, SDLoc(N), VT, N0.getOperand(1),
                       N0.getOperand(0));
  }

  if (SDValue Cast = foldSignChangeInBitcast(N))
    return Cast;

  return SDValue();
}

static SDValue visitFMinMax(SelectionDAG &DAG, SDNode *N,
                            APFloat (*Op)(const APFloat &, const APFloat &)) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  const ConstantFPSDNode *N0CFP = isConstOrConstSplatFP(N0);
  const ConstantFPSDNode *N1CFP = isConstOrConstSplatFP(N1);
  const SDNodeFlags Flags = N->getFlags();
  unsigned Opc = N->getOpcode();
  bool PropagatesNaN = Opc == ISD::FMINIMUM || Opc == ISD::FMAXIMUM;
  bool IsMin = Opc == ISD::FMINNUM || Opc == ISD::FMINIMUM;
  SelectionDAG::FlagInserter FlagsInserter(DAG, N);

  if (N0CFP && N1CFP) {
    const APFloat &C0 = N0CFP->getValueAPF();
    const APFloat &C1 = N1CFP->getValueAPF();
    return DAG.getConstantFP(Op(C0, C1), SDLoc(N), VT);
  }

  // Canonicalize to constant on RHS.
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0) &&
      !DAG.isConstantFPBuildVectorOrConstantFP(N1))
    return DAG.getNode(N->getOpcode(), SDLoc(N), VT, N1, N0);

  if (N1CFP) {
    const APFloat &AF = N1CFP->getValueAPF();

    // minnum(X, nan) -> X
    // maxnum(X, nan) -> X
    // minimum(X, nan) -> nan
    // maximum(X, nan) -> nan
    if (AF.isNaN())
      return PropagatesNaN ? N->getOperand(1) : N->getOperand(0);

    // In the following folds, inf can be replaced with the largest finite
    // float, if the ninf flag is set.
    if (AF.isInfinity() || (Flags.hasNoInfs() && AF.isLargest())) {
      // minnum(X, -inf) -> -inf
      // maxnum(X, +inf) -> +inf
      // minimum(X, -inf) -> -inf if nnan
      // maximum(X, +inf) -> +inf if nnan
      if (IsMin == AF.isNegative() && (!PropagatesNaN || Flags.hasNoNaNs()))
        return N->getOperand(1);

      // minnum(X, +inf) -> X if nnan
      // maxnum(X, -inf) -> X if nnan
      // minimum(X, +inf) -> X
      // maximum(X, -inf) -> X
      if (IsMin != AF.isNegative() && (PropagatesNaN || Flags.hasNoNaNs()))
        return N->getOperand(0);
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitFMINNUM(SDNode *N) {
  return visitFMinMax(DAG, N, minnum);
}

SDValue DAGCombiner::visitFMAXNUM(SDNode *N) {
  return visitFMinMax(DAG, N, maxnum);
}

SDValue DAGCombiner::visitFMINIMUM(SDNode *N) {
  return visitFMinMax(DAG, N, minimum);
}

SDValue DAGCombiner::visitFMAXIMUM(SDNode *N) {
  return visitFMinMax(DAG, N, maximum);
}

SDValue DAGCombiner::visitFABS(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fold (fabs c1) -> fabs(c1)
  if (DAG.isConstantFPBuildVectorOrConstantFP(N0))
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0);

  // fold (fabs (fabs x)) -> (fabs x)
  if (N0.getOpcode() == ISD::FABS)
    return N->getOperand(0);

  // fold (fabs (fneg x)) -> (fabs x)
  // fold (fabs (fcopysign x, y)) -> (fabs x)
  if (N0.getOpcode() == ISD::FNEG || N0.getOpcode() == ISD::FCOPYSIGN)
    return DAG.getNode(ISD::FABS, SDLoc(N), VT, N0.getOperand(0));

  if (SDValue Cast = foldSignChangeInBitcast(N))
    return Cast;

  return SDValue();
}

SDValue DAGCombiner::visitBRCOND(SDNode *N) {
  SDValue Chain = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);

  // BRCOND(FREEZE(cond)) is equivalent to BRCOND(cond) (both are
  // nondeterministic jumps).
  if (N1->getOpcode() == ISD::FREEZE && N1.hasOneUse()) {
    return DAG.getNode(ISD::BRCOND, SDLoc(N), MVT::Other, Chain,
                       N1->getOperand(0), N2);
  }

  // If N is a constant we could fold this into a fallthrough or unconditional
  // branch. However that doesn't happen very often in normal code, because
  // Instcombine/SimplifyCFG should have handled the available opportunities.
  // If we did this folding here, it would be necessary to update the
  // MachineBasicBlock CFG, which is awkward.

  // fold a brcond with a setcc condition into a BR_CC node if BR_CC is legal
  // on the target.
  if (N1.getOpcode() == ISD::SETCC &&
      TLI.isOperationLegalOrCustom(ISD::BR_CC,
                                   N1.getOperand(0).getValueType())) {
    return DAG.getNode(ISD::BR_CC, SDLoc(N), MVT::Other,
                       Chain, N1.getOperand(2),
                       N1.getOperand(0), N1.getOperand(1), N2);
  }

  if (N1.hasOneUse()) {
    // rebuildSetCC calls visitXor which may change the Chain when there is a
    // STRICT_FSETCC/STRICT_FSETCCS involved. Use a handle to track changes.
    HandleSDNode ChainHandle(Chain);
    if (SDValue NewN1 = rebuildSetCC(N1))
      return DAG.getNode(ISD::BRCOND, SDLoc(N), MVT::Other,
                         ChainHandle.getValue(), NewN1, N2);
  }

  return SDValue();
}

SDValue DAGCombiner::rebuildSetCC(SDValue N) {
  if (N.getOpcode() == ISD::SRL ||
      (N.getOpcode() == ISD::TRUNCATE &&
       (N.getOperand(0).hasOneUse() &&
        N.getOperand(0).getOpcode() == ISD::SRL))) {
    // Look pass the truncate.
    if (N.getOpcode() == ISD::TRUNCATE)
      N = N.getOperand(0);

    // Match this pattern so that we can generate simpler code:
    //
    //   %a = ...
    //   %b = and i32 %a, 2
    //   %c = srl i32 %b, 1
    //   brcond i32 %c ...
    //
    // into
    //
    //   %a = ...
    //   %b = and i32 %a, 2
    //   %c = setcc eq %b, 0
    //   brcond %c ...
    //
    // This applies only when the AND constant value has one bit set and the
    // SRL constant is equal to the log2 of the AND constant. The back-end is
    // smart enough to convert the result into a TEST/JMP sequence.
    SDValue Op0 = N.getOperand(0);
    SDValue Op1 = N.getOperand(1);

    if (Op0.getOpcode() == ISD::AND && Op1.getOpcode() == ISD::Constant) {
      SDValue AndOp1 = Op0.getOperand(1);

      if (AndOp1.getOpcode() == ISD::Constant) {
        const APInt &AndConst = cast<ConstantSDNode>(AndOp1)->getAPIntValue();

        if (AndConst.isPowerOf2() &&
            cast<ConstantSDNode>(Op1)->getAPIntValue() == AndConst.logBase2()) {
          SDLoc DL(N);
          return DAG.getSetCC(DL, getSetCCResultType(Op0.getValueType()),
                              Op0, DAG.getConstant(0, DL, Op0.getValueType()),
                              ISD::SETNE);
        }
      }
    }
  }

  // Transform (brcond (xor x, y)) -> (brcond (setcc, x, y, ne))
  // Transform (brcond (xor (xor x, y), -1)) -> (brcond (setcc, x, y, eq))
  if (N.getOpcode() == ISD::XOR) {
    // Because we may call this on a speculatively constructed
    // SimplifiedSetCC Node, we need to simplify this node first.
    // Ideally this should be folded into SimplifySetCC and not
    // here. For now, grab a handle to N so we don't lose it from
    // replacements interal to the visit.
    HandleSDNode XORHandle(N);
    while (N.getOpcode() == ISD::XOR) {
      SDValue Tmp = visitXOR(N.getNode());
      // No simplification done.
      if (!Tmp.getNode())
        break;
      // Returning N is form in-visit replacement that may invalidated
      // N. Grab value from Handle.
      if (Tmp.getNode() == N.getNode())
        N = XORHandle.getValue();
      else // Node simplified. Try simplifying again.
        N = Tmp;
    }

    if (N.getOpcode() != ISD::XOR)
      return N;

    SDValue Op0 = N->getOperand(0);
    SDValue Op1 = N->getOperand(1);

    if (Op0.getOpcode() != ISD::SETCC && Op1.getOpcode() != ISD::SETCC) {
      bool Equal = false;
      // (brcond (xor (xor x, y), -1)) -> (brcond (setcc x, y, eq))
      if (isBitwiseNot(N) && Op0.hasOneUse() && Op0.getOpcode() == ISD::XOR &&
          Op0.getValueType() == MVT::i1) {
        N = Op0;
        Op0 = N->getOperand(0);
        Op1 = N->getOperand(1);
        Equal = true;
      }

      EVT SetCCVT = N.getValueType();
      if (LegalTypes)
        SetCCVT = getSetCCResultType(SetCCVT);
      // Replace the uses of XOR with SETCC
      return DAG.getSetCC(SDLoc(N), SetCCVT, Op0, Op1,
                          Equal ? ISD::SETEQ : ISD::SETNE);
    }
  }

  return SDValue();
}

// Operand List for BR_CC: Chain, CondCC, CondLHS, CondRHS, DestBB.
//
SDValue DAGCombiner::visitBR_CC(SDNode *N) {
  CondCodeSDNode *CC = cast<CondCodeSDNode>(N->getOperand(1));
  SDValue CondLHS = N->getOperand(2), CondRHS = N->getOperand(3);

  // If N is a constant we could fold this into a fallthrough or unconditional
  // branch. However that doesn't happen very often in normal code, because
  // Instcombine/SimplifyCFG should have handled the available opportunities.
  // If we did this folding here, it would be necessary to update the
  // MachineBasicBlock CFG, which is awkward.

  // Use SimplifySetCC to simplify SETCC's.
  SDValue Simp = SimplifySetCC(getSetCCResultType(CondLHS.getValueType()),
                               CondLHS, CondRHS, CC->get(), SDLoc(N),
                               false);
  if (Simp.getNode()) AddToWorklist(Simp.getNode());

  // fold to a simpler setcc
  if (Simp.getNode() && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, SDLoc(N), MVT::Other,
                       N->getOperand(0), Simp.getOperand(2),
                       Simp.getOperand(0), Simp.getOperand(1),
                       N->getOperand(4));

  return SDValue();
}

static bool getCombineLoadStoreParts(SDNode *N, unsigned Inc, unsigned Dec,
                                     bool &IsLoad, bool &IsMasked, SDValue &Ptr,
                                     const TargetLowering &TLI) {
  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    if (LD->isIndexed())
      return false;
    EVT VT = LD->getMemoryVT();
    if (!TLI.isIndexedLoadLegal(Inc, VT) && !TLI.isIndexedLoadLegal(Dec, VT))
      return false;
    Ptr = LD->getBasePtr();
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    if (ST->isIndexed())
      return false;
    EVT VT = ST->getMemoryVT();
    if (!TLI.isIndexedStoreLegal(Inc, VT) && !TLI.isIndexedStoreLegal(Dec, VT))
      return false;
    Ptr = ST->getBasePtr();
    IsLoad = false;
  } else if (MaskedLoadSDNode *LD = dyn_cast<MaskedLoadSDNode>(N)) {
    if (LD->isIndexed())
      return false;
    EVT VT = LD->getMemoryVT();
    if (!TLI.isIndexedMaskedLoadLegal(Inc, VT) &&
        !TLI.isIndexedMaskedLoadLegal(Dec, VT))
      return false;
    Ptr = LD->getBasePtr();
    IsMasked = true;
  } else if (MaskedStoreSDNode *ST = dyn_cast<MaskedStoreSDNode>(N)) {
    if (ST->isIndexed())
      return false;
    EVT VT = ST->getMemoryVT();
    if (!TLI.isIndexedMaskedStoreLegal(Inc, VT) &&
        !TLI.isIndexedMaskedStoreLegal(Dec, VT))
      return false;
    Ptr = ST->getBasePtr();
    IsLoad = false;
    IsMasked = true;
  } else {
    return false;
  }
  return true;
}

/// Try turning a load/store into a pre-indexed load/store when the base
/// pointer is an add or subtract and it has other uses besides the load/store.
/// After the transformation, the new indexed load/store has effectively folded
/// the add/subtract in and all of its other uses are redirected to the
/// new load/store.
bool DAGCombiner::CombineToPreIndexedLoadStore(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  bool IsLoad = true;
  bool IsMasked = false;
  SDValue Ptr;
  if (!getCombineLoadStoreParts(N, ISD::PRE_INC, ISD::PRE_DEC, IsLoad, IsMasked,
                                Ptr, TLI))
    return false;

  // If the pointer is not an add/sub, or if it doesn't have multiple uses, bail
  // out.  There is no reason to make this a preinc/predec.
  if ((Ptr.getOpcode() != ISD::ADD && Ptr.getOpcode() != ISD::SUB) ||
      Ptr.getNode()->hasOneUse())
    return false;

  // Ask the target to do addressing mode selection.
  SDValue BasePtr;
  SDValue Offset;
  ISD::MemIndexedMode AM = ISD::UNINDEXED;
  if (!TLI.getPreIndexedAddressParts(N, BasePtr, Offset, AM, DAG))
    return false;

  // Backends without true r+i pre-indexed forms may need to pass a
  // constant base with a variable offset so that constant coercion
  // will work with the patterns in canonical form.
  bool Swapped = false;
  if (isa<ConstantSDNode>(BasePtr)) {
    std::swap(BasePtr, Offset);
    Swapped = true;
  }

  // Don't create a indexed load / store with zero offset.
  if (isNullConstant(Offset))
    return false;

  // Try turning it into a pre-indexed load / store except when:
  // 1) The new base ptr is a frame index.
  // 2) If N is a store and the new base ptr is either the same as or is a
  //    predecessor of the value being stored.
  // 3) Another use of old base ptr is a predecessor of N. If ptr is folded
  //    that would create a cycle.
  // 4) All uses are load / store ops that use it as old base ptr.

  // Check #1.  Preinc'ing a frame index would require copying the stack pointer
  // (plus the implicit offset) to a register to preinc anyway.
  if (isa<FrameIndexSDNode>(BasePtr) || isa<RegisterSDNode>(BasePtr))
    return false;

  // Check #2.
  if (!IsLoad) {
    SDValue Val = IsMasked ? cast<MaskedStoreSDNode>(N)->getValue()
                           : cast<StoreSDNode>(N)->getValue();

    // Would require a copy.
    if (Val == BasePtr)
      return false;

    // Would create a cycle.
    if (Val == Ptr || Ptr->isPredecessorOf(Val.getNode()))
      return false;
  }

  // Caches for hasPredecessorHelper.
  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 16> Worklist;
  Worklist.push_back(N);

  // If the offset is a constant, there may be other adds of constants that
  // can be folded with this one. We should do this to avoid having to keep
  // a copy of the original base pointer.
  SmallVector<SDNode *, 16> OtherUses;
  if (isa<ConstantSDNode>(Offset))
    for (SDNode::use_iterator UI = BasePtr.getNode()->use_begin(),
                              UE = BasePtr.getNode()->use_end();
         UI != UE; ++UI) {
      SDUse &Use = UI.getUse();
      // Skip the use that is Ptr and uses of other results from BasePtr's
      // node (important for nodes that return multiple results).
      if (Use.getUser() == Ptr.getNode() || Use != BasePtr)
        continue;

      if (SDNode::hasPredecessorHelper(Use.getUser(), Visited, Worklist))
        continue;

      if (Use.getUser()->getOpcode() != ISD::ADD &&
          Use.getUser()->getOpcode() != ISD::SUB) {
        OtherUses.clear();
        break;
      }

      SDValue Op1 = Use.getUser()->getOperand((UI.getOperandNo() + 1) & 1);
      if (!isa<ConstantSDNode>(Op1)) {
        OtherUses.clear();
        break;
      }

      // FIXME: In some cases, we can be smarter about this.
      if (Op1.getValueType() != Offset.getValueType()) {
        OtherUses.clear();
        break;
      }

      OtherUses.push_back(Use.getUser());
    }

  if (Swapped)
    std::swap(BasePtr, Offset);

  // Now check for #3 and #4.
  bool RealUse = false;

  for (SDNode *Use : Ptr.getNode()->uses()) {
    if (Use == N)
      continue;
    if (SDNode::hasPredecessorHelper(Use, Visited, Worklist))
      return false;

    // If Ptr may be folded in addressing mode of other use, then it's
    // not profitable to do this transformation.
    if (!canFoldInAddressingMode(Ptr.getNode(), Use, DAG, TLI))
      RealUse = true;
  }

  if (!RealUse)
    return false;

  SDValue Result;
  if (!IsMasked) {
    if (IsLoad)
      Result = DAG.getIndexedLoad(SDValue(N, 0), SDLoc(N), BasePtr, Offset, AM);
    else
      Result =
          DAG.getIndexedStore(SDValue(N, 0), SDLoc(N), BasePtr, Offset, AM);
  } else {
    if (IsLoad)
      Result = DAG.getIndexedMaskedLoad(SDValue(N, 0), SDLoc(N), BasePtr,
                                        Offset, AM);
    else
      Result = DAG.getIndexedMaskedStore(SDValue(N, 0), SDLoc(N), BasePtr,
                                         Offset, AM);
  }
  ++PreIndexedNodes;
  ++NodesCombined;
  LLVM_DEBUG(dbgs() << "\nReplacing.4 "; N->dump(&DAG); dbgs() << "\nWith: ";
             Result.getNode()->dump(&DAG); dbgs() << '\n');
  WorklistRemover DeadNodes(*this);
  if (IsLoad) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0));
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2));
  } else {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1));
  }

  // Finally, since the node is now dead, remove it from the graph.
  deleteAndRecombine(N);

  if (Swapped)
    std::swap(BasePtr, Offset);

  // Replace other uses of BasePtr that can be updated to use Ptr
  for (unsigned i = 0, e = OtherUses.size(); i != e; ++i) {
    unsigned OffsetIdx = 1;
    if (OtherUses[i]->getOperand(OffsetIdx).getNode() == BasePtr.getNode())
      OffsetIdx = 0;
    assert(OtherUses[i]->getOperand(!OffsetIdx).getNode() ==
           BasePtr.getNode() && "Expected BasePtr operand");

    // We need to replace ptr0 in the following expression:
    //   x0 * offset0 + y0 * ptr0 = t0
    // knowing that
    //   x1 * offset1 + y1 * ptr0 = t1 (the indexed load/store)
    //
    // where x0, x1, y0 and y1 in {-1, 1} are given by the types of the
    // indexed load/store and the expression that needs to be re-written.
    //
    // Therefore, we have:
    //   t0 = (x0 * offset0 - x1 * y0 * y1 *offset1) + (y0 * y1) * t1

    auto *CN = cast<ConstantSDNode>(OtherUses[i]->getOperand(OffsetIdx));
    const APInt &Offset0 = CN->getAPIntValue();
    const APInt &Offset1 = cast<ConstantSDNode>(Offset)->getAPIntValue();
    int X0 = (OtherUses[i]->getOpcode() == ISD::SUB && OffsetIdx == 1) ? -1 : 1;
    int Y0 = (OtherUses[i]->getOpcode() == ISD::SUB && OffsetIdx == 0) ? -1 : 1;
    int X1 = (AM == ISD::PRE_DEC && !Swapped) ? -1 : 1;
    int Y1 = (AM == ISD::PRE_DEC && Swapped) ? -1 : 1;

    unsigned Opcode = (Y0 * Y1 < 0) ? ISD::SUB : ISD::ADD;

    APInt CNV = Offset0;
    if (X0 < 0) CNV = -CNV;
    if (X1 * Y0 * Y1 < 0) CNV = CNV + Offset1;
    else CNV = CNV - Offset1;

    SDLoc DL(OtherUses[i]);

    // We can now generate the new expression.
    SDValue NewOp1 = DAG.getConstant(CNV, DL, CN->getValueType(0));
    SDValue NewOp2 = Result.getValue(IsLoad ? 1 : 0);

    SDValue NewUse = DAG.getNode(Opcode,
                                 DL,
                                 OtherUses[i]->getValueType(0), NewOp1, NewOp2);
    DAG.ReplaceAllUsesOfValueWith(SDValue(OtherUses[i], 0), NewUse);
    deleteAndRecombine(OtherUses[i]);
  }

  // Replace the uses of Ptr with uses of the updated base value.
  DAG.ReplaceAllUsesOfValueWith(Ptr, Result.getValue(IsLoad ? 1 : 0));
  deleteAndRecombine(Ptr.getNode());
  AddToWorklist(Result.getNode());

  return true;
}

static bool shouldCombineToPostInc(SDNode *N, SDValue Ptr, SDNode *PtrUse,
                                   SDValue &BasePtr, SDValue &Offset,
                                   ISD::MemIndexedMode &AM,
                                   SelectionDAG &DAG,
                                   const TargetLowering &TLI) {
  if (PtrUse == N ||
      (PtrUse->getOpcode() != ISD::ADD && PtrUse->getOpcode() != ISD::SUB))
    return false;

  if (!TLI.getPostIndexedAddressParts(N, PtrUse, BasePtr, Offset, AM, DAG))
    return false;

  // Don't create a indexed load / store with zero offset.
  if (isNullConstant(Offset))
    return false;

  if (isa<FrameIndexSDNode>(BasePtr) || isa<RegisterSDNode>(BasePtr))
    return false;

  SmallPtrSet<const SDNode *, 32> Visited;
  for (SDNode *Use : BasePtr.getNode()->uses()) {
    if (Use == Ptr.getNode())
      continue;

    // No if there's a later user which could perform the index instead.
    if (isa<MemSDNode>(Use)) {
      bool IsLoad = true;
      bool IsMasked = false;
      SDValue OtherPtr;
      if (getCombineLoadStoreParts(Use, ISD::POST_INC, ISD::POST_DEC, IsLoad,
                                   IsMasked, OtherPtr, TLI)) {
        SmallVector<const SDNode *, 2> Worklist;
        Worklist.push_back(Use);
        if (SDNode::hasPredecessorHelper(N, Visited, Worklist))
          return false;
      }
    }

    // If all the uses are load / store addresses, then don't do the
    // transformation.
    if (Use->getOpcode() == ISD::ADD || Use->getOpcode() == ISD::SUB) {
      for (SDNode *UseUse : Use->uses())
        if (canFoldInAddressingMode(Use, UseUse, DAG, TLI))
          return false;
    }
  }
  return true;
}

static SDNode *getPostIndexedLoadStoreOp(SDNode *N, bool &IsLoad,
                                         bool &IsMasked, SDValue &Ptr,
                                         SDValue &BasePtr, SDValue &Offset,
                                         ISD::MemIndexedMode &AM,
                                         SelectionDAG &DAG,
                                         const TargetLowering &TLI) {
  if (!getCombineLoadStoreParts(N, ISD::POST_INC, ISD::POST_DEC, IsLoad,
                                IsMasked, Ptr, TLI) ||
      Ptr.getNode()->hasOneUse())
    return nullptr;

  // Try turning it into a post-indexed load / store except when
  // 1) All uses are load / store ops that use it as base ptr (and
  //    it may be folded as addressing mmode).
  // 2) Op must be independent of N, i.e. Op is neither a predecessor
  //    nor a successor of N. Otherwise, if Op is folded that would
  //    create a cycle.
  for (SDNode *Op : Ptr->uses()) {
    // Check for #1.
    if (!shouldCombineToPostInc(N, Ptr, Op, BasePtr, Offset, AM, DAG, TLI))
      continue;

    // Check for #2.
    SmallPtrSet<const SDNode *, 32> Visited;
    SmallVector<const SDNode *, 8> Worklist;
    // Ptr is predecessor to both N and Op.
    Visited.insert(Ptr.getNode());
    Worklist.push_back(N);
    Worklist.push_back(Op);
    if (!SDNode::hasPredecessorHelper(N, Visited, Worklist) &&
        !SDNode::hasPredecessorHelper(Op, Visited, Worklist))
      return Op;
  }
  return nullptr;
}

/// Try to combine a load/store with a add/sub of the base pointer node into a
/// post-indexed load/store. The transformation folded the add/subtract into the
/// new indexed load/store effectively and all of its uses are redirected to the
/// new load/store.
bool DAGCombiner::CombineToPostIndexedLoadStore(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  bool IsLoad = true;
  bool IsMasked = false;
  SDValue Ptr;
  SDValue BasePtr;
  SDValue Offset;
  ISD::MemIndexedMode AM = ISD::UNINDEXED;
  SDNode *Op = getPostIndexedLoadStoreOp(N, IsLoad, IsMasked, Ptr, BasePtr,
                                         Offset, AM, DAG, TLI);
  if (!Op)
    return false;

  SDValue Result;
  if (!IsMasked)
    Result = IsLoad ? DAG.getIndexedLoad(SDValue(N, 0), SDLoc(N), BasePtr,
                                         Offset, AM)
                    : DAG.getIndexedStore(SDValue(N, 0), SDLoc(N),
                                          BasePtr, Offset, AM);
  else
    Result = IsLoad ? DAG.getIndexedMaskedLoad(SDValue(N, 0), SDLoc(N),
                                               BasePtr, Offset, AM)
                    : DAG.getIndexedMaskedStore(SDValue(N, 0), SDLoc(N),
                                                BasePtr, Offset, AM);
  ++PostIndexedNodes;
  ++NodesCombined;
  LLVM_DEBUG(dbgs() << "\nReplacing.5 "; N->dump(&DAG);
             dbgs() << "\nWith: "; Result.getNode()->dump(&DAG);
             dbgs() << '\n');
  WorklistRemover DeadNodes(*this);
  if (IsLoad) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(0));
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Result.getValue(2));
  } else {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Result.getValue(1));
  }

  // Finally, since the node is now dead, remove it from the graph.
  deleteAndRecombine(N);

  // Replace the uses of Use with uses of the updated base value.
  DAG.ReplaceAllUsesOfValueWith(SDValue(Op, 0),
                                Result.getValue(IsLoad ? 1 : 0));
  deleteAndRecombine(Op);
  return true;
}

/// Return the base-pointer arithmetic from an indexed \p LD.
SDValue DAGCombiner::SplitIndexingFromLoad(LoadSDNode *LD) {
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  assert(AM != ISD::UNINDEXED);
  SDValue BP = LD->getOperand(1);
  SDValue Inc = LD->getOperand(2);

  // Some backends use TargetConstants for load offsets, but don't expect
  // TargetConstants in general ADD nodes. We can convert these constants into
  // regular Constants (if the constant is not opaque).
  assert((Inc.getOpcode() != ISD::TargetConstant ||
          !cast<ConstantSDNode>(Inc)->isOpaque()) &&
         "Cannot split out indexing using opaque target constants");
  if (Inc.getOpcode() == ISD::TargetConstant) {
    ConstantSDNode *ConstInc = cast<ConstantSDNode>(Inc);
    Inc = DAG.getConstant(*ConstInc->getConstantIntValue(), SDLoc(Inc),
                          ConstInc->getValueType(0));
  }

  unsigned Opc =
      (AM == ISD::PRE_INC || AM == ISD::POST_INC ? ISD::ADD : ISD::SUB);
  return DAG.getNode(Opc, SDLoc(LD), BP.getSimpleValueType(), BP, Inc);
}

static inline ElementCount numVectorEltsOrZero(EVT T) {
  return T.isVector() ? T.getVectorElementCount() : ElementCount::getFixed(0);
}

bool DAGCombiner::getTruncatedStoreValue(StoreSDNode *ST, SDValue &Val) {
  Val = ST->getValue();
  EVT STType = Val.getValueType();
  EVT STMemType = ST->getMemoryVT();
  if (STType == STMemType)
    return true;
  if (isTypeLegal(STMemType))
    return false; // fail.
  if (STType.isFloatingPoint() && STMemType.isFloatingPoint() &&
      TLI.isOperationLegal(ISD::FTRUNC, STMemType)) {
    Val = DAG.getNode(ISD::FTRUNC, SDLoc(ST), STMemType, Val);
    return true;
  }
  if (numVectorEltsOrZero(STType) == numVectorEltsOrZero(STMemType) &&
      STType.isInteger() && STMemType.isInteger()) {
    Val = DAG.getNode(ISD::TRUNCATE, SDLoc(ST), STMemType, Val);
    return true;
  }
  if (STType.getSizeInBits() == STMemType.getSizeInBits()) {
    Val = DAG.getBitcast(STMemType, Val);
    return true;
  }
  return false; // fail.
}

bool DAGCombiner::extendLoadedValueToExtension(LoadSDNode *LD, SDValue &Val) {
  EVT LDMemType = LD->getMemoryVT();
  EVT LDType = LD->getValueType(0);
  assert(Val.getValueType() == LDMemType &&
         "Attempting to extend value of non-matching type");
  if (LDType == LDMemType)
    return true;
  if (LDMemType.isInteger() && LDType.isInteger()) {
    switch (LD->getExtensionType()) {
    case ISD::NON_EXTLOAD:
      Val = DAG.getBitcast(LDType, Val);
      return true;
    case ISD::EXTLOAD:
      Val = DAG.getNode(ISD::ANY_EXTEND, SDLoc(LD), LDType, Val);
      return true;
    case ISD::SEXTLOAD:
      Val = DAG.getNode(ISD::SIGN_EXTEND, SDLoc(LD), LDType, Val);
      return true;
    case ISD::ZEXTLOAD:
      Val = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(LD), LDType, Val);
      return true;
    }
  }
  return false;
}

SDValue DAGCombiner::ForwardStoreValueToDirectLoad(LoadSDNode *LD) {
  if (OptLevel == CodeGenOpt::None || !LD->isSimple())
    return SDValue();
  SDValue Chain = LD->getOperand(0);
  StoreSDNode *ST = dyn_cast<StoreSDNode>(Chain.getNode());
  // TODO: Relax this restriction for unordered atomics (see D66309)
  if (!ST || !ST->isSimple())
    return SDValue();

  EVT LDType = LD->getValueType(0);
  EVT LDMemType = LD->getMemoryVT();
  EVT STMemType = ST->getMemoryVT();
  EVT STType = ST->getValue().getValueType();

  // There are two cases to consider here:
  //  1. The store is fixed width and the load is scalable. In this case we
  //     don't know at compile time if the store completely envelops the load
  //     so we abandon the optimisation.
  //  2. The store is scalable and the load is fixed width. We could
  //     potentially support a limited number of cases here, but there has been
  //     no cost-benefit analysis to prove it's worth it.
  bool LdStScalable = LDMemType.isScalableVector();
  if (LdStScalable != STMemType.isScalableVector())
    return SDValue();

  // If we are dealing with scalable vectors on a big endian platform the
  // calculation of offsets below becomes trickier, since we do not know at
  // compile time the absolute size of the vector. Until we've done more
  // analysis on big-endian platforms it seems better to bail out for now.
  if (LdStScalable && DAG.getDataLayout().isBigEndian())
    return SDValue();

  BaseIndexOffset BasePtrLD = BaseIndexOffset::match(LD, DAG);
  BaseIndexOffset BasePtrST = BaseIndexOffset::match(ST, DAG);
  int64_t Offset;
  if (!BasePtrST.equalBaseIndex(BasePtrLD, DAG, Offset))
    return SDValue();

  // Normalize for Endianness. After this Offset=0 will denote that the least
  // significant bit in the loaded value maps to the least significant bit in
  // the stored value). With Offset=n (for n > 0) the loaded value starts at the
  // n:th least significant byte of the stored value.
  if (DAG.getDataLayout().isBigEndian())
    Offset = ((int64_t)STMemType.getStoreSizeInBits().getFixedSize() -
              (int64_t)LDMemType.getStoreSizeInBits().getFixedSize()) /
                 8 -
             Offset;

  // Check that the stored value cover all bits that are loaded.
  bool STCoversLD;

  TypeSize LdMemSize = LDMemType.getSizeInBits();
  TypeSize StMemSize = STMemType.getSizeInBits();
  if (LdStScalable)
    STCoversLD = (Offset == 0) && LdMemSize == StMemSize;
  else
    STCoversLD = (Offset >= 0) && (Offset * 8 + LdMemSize.getFixedSize() <=
                                   StMemSize.getFixedSize());

  auto ReplaceLd = [&](LoadSDNode *LD, SDValue Val, SDValue Chain) -> SDValue {
    if (LD->isIndexed()) {
      // Cannot handle opaque target constants and we must respect the user's
      // request not to split indexes from loads.
      if (!canSplitIdx(LD))
        return SDValue();
      SDValue Idx = SplitIndexingFromLoad(LD);
      SDValue Ops[] = {Val, Idx, Chain};
      return CombineTo(LD, Ops, 3);
    }
    return CombineTo(LD, Val, Chain);
  };

  if (!STCoversLD)
    return SDValue();

  // Memory as copy space (potentially masked).
  if (Offset == 0 && LDType == STType && STMemType == LDMemType) {
    // Simple case: Direct non-truncating forwarding
    if (LDType.getSizeInBits() == LdMemSize)
      return ReplaceLd(LD, ST->getValue(), Chain);
    // Can we model the truncate and extension with an and mask?
    if (STType.isInteger() && LDMemType.isInteger() && !STType.isVector() &&
        !LDMemType.isVector() && LD->getExtensionType() != ISD::SEXTLOAD) {
      // Mask to size of LDMemType
      auto Mask =
          DAG.getConstant(APInt::getLowBitsSet(STType.getFixedSizeInBits(),
                                               StMemSize.getFixedSize()),
                          SDLoc(ST), STType);
      auto Val = DAG.getNode(ISD::AND, SDLoc(LD), LDType, ST->getValue(), Mask);
      return ReplaceLd(LD, Val, Chain);
    }
  }

  // TODO: Deal with nonzero offset.
  if (LD->getBasePtr().isUndef() || Offset != 0)
    return SDValue();
  // Model necessary truncations / extenstions.
  SDValue Val;
  // Truncate Value To Stored Memory Size.
  do {
    if (!getTruncatedStoreValue(ST, Val))
      continue;
    if (!isTypeLegal(LDMemType))
      continue;
    if (STMemType != LDMemType) {
      // TODO: Support vectors? This requires extract_subvector/bitcast.
      if (!STMemType.isVector() && !LDMemType.isVector() &&
          STMemType.isInteger() && LDMemType.isInteger())
        Val = DAG.getNode(ISD::TRUNCATE, SDLoc(LD), LDMemType, Val);
      else
        continue;
    }
    if (!extendLoadedValueToExtension(LD, Val))
      continue;
    return ReplaceLd(LD, Val, Chain);
  } while (false);

  // On failure, cleanup dead nodes we may have created.
  if (Val->use_empty())
    deleteAndRecombine(Val.getNode());
  return SDValue();
}

SDValue DAGCombiner::visitLOAD(SDNode *N) {
  LoadSDNode *LD  = cast<LoadSDNode>(N);
  SDValue Chain = LD->getChain();
  SDValue Ptr   = LD->getBasePtr();

  // If load is not volatile and there are no uses of the loaded value (and
  // the updated indexed value in case of indexed loads), change uses of the
  // chain value into uses of the chain input (i.e. delete the dead load).
  // TODO: Allow this for unordered atomics (see D66309)
  if (LD->isSimple()) {
    if (N->getValueType(1) == MVT::Other) {
      // Unindexed loads.
      if (!N->hasAnyUseOfValue(0)) {
        // It's not safe to use the two value CombineTo variant here. e.g.
        // v1, chain2 = load chain1, loc
        // v2, chain3 = load chain2, loc
        // v3         = add v2, c
        // Now we replace use of chain2 with chain1.  This makes the second load
        // isomorphic to the one we are deleting, and thus makes this load live.
        LLVM_DEBUG(dbgs() << "\nReplacing.6 "; N->dump(&DAG);
                   dbgs() << "\nWith chain: "; Chain.getNode()->dump(&DAG);
                   dbgs() << "\n");
        WorklistRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain);
        AddUsersToWorklist(Chain.getNode());
        if (N->use_empty())
          deleteAndRecombine(N);

        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    } else {
      // Indexed loads.
      assert(N->getValueType(2) == MVT::Other && "Malformed indexed loads?");

      // If this load has an opaque TargetConstant offset, then we cannot split
      // the indexing into an add/sub directly (that TargetConstant may not be
      // valid for a different type of node, and we cannot convert an opaque
      // target constant into a regular constant).
      bool CanSplitIdx = canSplitIdx(LD);

      if (!N->hasAnyUseOfValue(0) && (CanSplitIdx || !N->hasAnyUseOfValue(1))) {
        SDValue Undef = DAG.getUNDEF(N->getValueType(0));
        SDValue Index;
        if (N->hasAnyUseOfValue(1) && CanSplitIdx) {
          Index = SplitIndexingFromLoad(LD);
          // Try to fold the base pointer arithmetic into subsequent loads and
          // stores.
          AddUsersToWorklist(N);
        } else
          Index = DAG.getUNDEF(N->getValueType(1));
        LLVM_DEBUG(dbgs() << "\nReplacing.7 "; N->dump(&DAG);
                   dbgs() << "\nWith: "; Undef.getNode()->dump(&DAG);
                   dbgs() << " and 2 other values\n");
        WorklistRemover DeadNodes(*this);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 0), Undef);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Index);
        DAG.ReplaceAllUsesOfValueWith(SDValue(N, 2), Chain);
        deleteAndRecombine(N);
        return SDValue(N, 0);   // Return N so it doesn't get rechecked!
      }
    }
  }

  // If this load is directly stored, replace the load value with the stored
  // value.
  if (auto V = ForwardStoreValueToDirectLoad(LD))
    return V;

  // Try to infer better alignment information than the load already has.
  if (OptLevel != CodeGenOpt::None && LD->isUnindexed() && !LD->isAtomic()) {
    if (MaybeAlign Alignment = DAG.InferPtrAlign(Ptr)) {
      if (*Alignment > LD->getAlign() &&
          isAligned(*Alignment, LD->getSrcValueOffset())) {
        SDValue NewLoad = DAG.getExtLoad(
            LD->getExtensionType(), SDLoc(N), LD->getValueType(0), Chain, Ptr,
            LD->getPointerInfo(), LD->getMemoryVT(), *Alignment,
            LD->getMemOperand()->getFlags(), LD->getAAInfo());
        // NewLoad will always be N as we are only refining the alignment
        assert(NewLoad.getNode() == N);
        (void)NewLoad;
      }
    }
  }

  if (LD->isUnindexed()) {
    // Walk up chain skipping non-aliasing memory nodes.
    SDValue BetterChain = FindBetterChain(LD, Chain);

    // If there is a better chain.
    if (Chain != BetterChain) {
      SDValue ReplLoad;

      // Replace the chain to void dependency.
      if (LD->getExtensionType() == ISD::NON_EXTLOAD) {
        ReplLoad = DAG.getLoad(N->getValueType(0), SDLoc(LD),
                               BetterChain, Ptr, LD->getMemOperand());
      } else {
        ReplLoad = DAG.getExtLoad(LD->getExtensionType(), SDLoc(LD),
                                  LD->getValueType(0),
                                  BetterChain, Ptr, LD->getMemoryVT(),
                                  LD->getMemOperand());
      }

      // Create token factor to keep old chain connected.
      SDValue Token = DAG.getNode(ISD::TokenFactor, SDLoc(N),
                                  MVT::Other, Chain, ReplLoad.getValue(1));

      // Replace uses with load result and token factor
      return CombineTo(N, ReplLoad.getValue(0), Token);
    }
  }

  // Try transforming N to an indexed load.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  // Try to slice up N to more direct loads if the slices are mapped to
  // different register banks or pairing can take place.
  if (SliceUpLoad(N))
    return SDValue(N, 0);

  return SDValue();
}

namespace {

/// Helper structure used to slice a load in smaller loads.
/// Basically a slice is obtained from the following sequence:
/// Origin = load Ty1, Base
/// Shift = srl Ty1 Origin, CstTy Amount
/// Inst = trunc Shift to Ty2
///
/// Then, it will be rewritten into:
/// Slice = load SliceTy, Base + SliceOffset
/// [Inst = zext Slice to Ty2], only if SliceTy <> Ty2
///
/// SliceTy is deduced from the number of bits that are actually used to
/// build Inst.
struct LoadedSlice {
  /// Helper structure used to compute the cost of a slice.
  struct Cost {
    /// Are we optimizing for code size.
    bool ForCodeSize = false;

    /// Various cost.
    unsigned Loads = 0;
    unsigned Truncates = 0;
    unsigned CrossRegisterBanksCopies = 0;
    unsigned ZExts = 0;
    unsigned Shift = 0;

    explicit Cost(bool ForCodeSize) : ForCodeSize(ForCodeSize) {}

    /// Get the cost of one isolated slice.
    Cost(const LoadedSlice &LS, bool ForCodeSize)
        : ForCodeSize(ForCodeSize), Loads(1) {
      EVT TruncType = LS.Inst->getValueType(0);
      EVT LoadedType = LS.getLoadedType();
      if (TruncType != LoadedType &&
          !LS.DAG->getTargetLoweringInfo().isZExtFree(LoadedType, TruncType))
        ZExts = 1;
    }

    /// Account for slicing gain in the current cost.
    /// Slicing provide a few gains like removing a shift or a
    /// truncate. This method allows to grow the cost of the original
    /// load with the gain from this slice.
    void addSliceGain(const LoadedSlice &LS) {
      // Each slice saves a truncate.
      const TargetLowering &TLI = LS.DAG->getTargetLoweringInfo();
      if (!TLI.isTruncateFree(LS.Inst->getOperand(0).getValueType(),
                              LS.Inst->getValueType(0)))
        ++Truncates;
      // If there is a shift amount, this slice gets rid of it.
      if (LS.Shift)
        ++Shift;
      // If this slice can merge a cross register bank copy, account for it.
      if (LS.canMergeExpensiveCrossRegisterBankCopy())
        ++CrossRegisterBanksCopies;
    }

    Cost &operator+=(const Cost &RHS) {
      Loads += RHS.Loads;
      Truncates += RHS.Truncates;
      CrossRegisterBanksCopies += RHS.CrossRegisterBanksCopies;
      ZExts += RHS.ZExts;
      Shift += RHS.Shift;
      return *this;
    }

    bool operator==(const Cost &RHS) const {
      return Loads == RHS.Loads && Truncates == RHS.Truncates &&
             CrossRegisterBanksCopies == RHS.CrossRegisterBanksCopies &&
             ZExts == RHS.ZExts && Shift == RHS.Shift;
    }

    bool operator!=(const Cost &RHS) const { return !(*this == RHS); }

    bool operator<(const Cost &RHS) const {
      // Assume cross register banks copies are as expensive as loads.
      // FIXME: Do we want some more target hooks?
      unsigned ExpensiveOpsLHS = Loads + CrossRegisterBanksCopies;
      unsigned ExpensiveOpsRHS = RHS.Loads + RHS.CrossRegisterBanksCopies;
      // Unless we are optimizing for code size, consider the
      // expensive operation first.
      if (!ForCodeSize && ExpensiveOpsLHS != ExpensiveOpsRHS)
        return ExpensiveOpsLHS < ExpensiveOpsRHS;
      return (Truncates + ZExts + Shift + ExpensiveOpsLHS) <
             (RHS.Truncates + RHS.ZExts + RHS.Shift + ExpensiveOpsRHS);
    }

    bool operator>(const Cost &RHS) const { return RHS < *this; }

    bool operator<=(const Cost &RHS) const { return !(RHS < *this); }

    bool operator>=(const Cost &RHS) const { return !(*this < RHS); }
  };

  // The last instruction that represent the slice. This should be a
  // truncate instruction.
  SDNode *Inst;

  // The original load instruction.
  LoadSDNode *Origin;

  // The right shift amount in bits from the original load.
  unsigned Shift;

  // The DAG from which Origin came from.
  // This is used to get some contextual information about legal types, etc.
  SelectionDAG *DAG;

  LoadedSlice(SDNode *Inst = nullptr, LoadSDNode *Origin = nullptr,
              unsigned Shift = 0, SelectionDAG *DAG = nullptr)
      : Inst(Inst), Origin(Origin), Shift(Shift), DAG(DAG) {}

  /// Get the bits used in a chunk of bits \p BitWidth large.
  /// \return Result is \p BitWidth and has used bits set to 1 and
  ///         not used bits set to 0.
  APInt getUsedBits() const {
    // Reproduce the trunc(lshr) sequence:
    // - Start from the truncated value.
    // - Zero extend to the desired bit width.
    // - Shift left.
    assert(Origin && "No original load to compare against.");
    unsigned BitWidth = Origin->getValueSizeInBits(0);
    assert(Inst && "This slice is not bound to an instruction");
    assert(Inst->getValueSizeInBits(0) <= BitWidth &&
           "Extracted slice is bigger than the whole type!");
    APInt UsedBits(Inst->getValueSizeInBits(0), 0);
    UsedBits.setAllBits();
    UsedBits = UsedBits.zext(BitWidth);
    UsedBits <<= Shift;
    return UsedBits;
  }

  /// Get the size of the slice to be loaded in bytes.
  unsigned getLoadedSize() const {
    unsigned SliceSize = getUsedBits().countPopulation();
    assert(!(SliceSize & 0x7) && "Size is not a multiple of a byte.");
    return SliceSize / 8;
  }

  /// Get the type that will be loaded for this slice.
  /// Note: This may not be the final type for the slice.
  EVT getLoadedType() const {
    assert(DAG && "Missing context");
    LLVMContext &Ctxt = *DAG->getContext();
    return EVT::getIntegerVT(Ctxt, getLoadedSize() * 8);
  }

  /// Get the alignment of the load used for this slice.
  Align getAlign() const {
    Align Alignment = Origin->getAlign();
    uint64_t Offset = getOffsetFromBase();
    if (Offset != 0)
      Alignment = commonAlignment(Alignment, Alignment.value() + Offset);
    return Alignment;
  }

  /// Check if this slice can be rewritten with legal operations.
  bool isLegal() const {
    // An invalid slice is not legal.
    if (!Origin || !Inst || !DAG)
      return false;

    // Offsets are for indexed load only, we do not handle that.
    if (!Origin->getOffset().isUndef())
      return false;

    const TargetLowering &TLI = DAG->getTargetLoweringInfo();

    // Check that the type is legal.
    EVT SliceType = getLoadedType();
    if (!TLI.isTypeLegal(SliceType))
      return false;

    // Check that the load is legal for this type.
    if (!TLI.isOperationLegal(ISD::LOAD, SliceType))
      return false;

    // Check that the offset can be computed.
    // 1. Check its type.
    EVT PtrType = Origin->getBasePtr().getValueType();
    if (PtrType == MVT::Untyped || PtrType.isExtended())
      return false;

    // 2. Check that it fits in the immediate.
    if (!TLI.isLegalAddImmediate(getOffsetFromBase()))
      return false;

    // 3. Check that the computation is legal.
    if (!TLI.isOperationLegal(ISD::ADD, PtrType))
      return false;

    // Check that the zext is legal if it needs one.
    EVT TruncateType = Inst->getValueType(0);
    if (TruncateType != SliceType &&
        !TLI.isOperationLegal(ISD::ZERO_EXTEND, TruncateType))
      return false;

    return true;
  }

  /// Get the offset in bytes of this slice in the original chunk of
  /// bits.
  /// \pre DAG != nullptr.
  uint64_t getOffsetFromBase() const {
    assert(DAG && "Missing context.");
    bool IsBigEndian = DAG->getDataLayout().isBigEndian();
    assert(!(Shift & 0x7) && "Shifts not aligned on Bytes are not supported.");
    uint64_t Offset = Shift / 8;
    unsigned TySizeInBytes = Origin->getValueSizeInBits(0) / 8;
    assert(!(Origin->getValueSizeInBits(0) & 0x7) &&
           "The size of the original loaded type is not a multiple of a"
           " byte.");
    // If Offset is bigger than TySizeInBytes, it means we are loading all
    // zeros. This should have been optimized before in the process.
    assert(TySizeInBytes > Offset &&
           "Invalid shift amount for given loaded size");
    if (IsBigEndian)
      Offset = TySizeInBytes - Offset - getLoadedSize();
    return Offset;
  }

  /// Generate the sequence of instructions to load the slice
  /// represented by this object and redirect the uses of this slice to
  /// this new sequence of instructions.
  /// \pre this->Inst && this->Origin are valid Instructions and this
  /// object passed the legal check: LoadedSlice::isLegal returned true.
  /// \return The last instruction of the sequence used to load the slice.
  SDValue loadSlice() const {
    assert(Inst && Origin && "Unable to replace a non-existing slice.");
    const SDValue &OldBaseAddr = Origin->getBasePtr();
    SDValue BaseAddr = OldBaseAddr;
    // Get the offset in that chunk of bytes w.r.t. the endianness.
    int64_t Offset = static_cast<int64_t>(getOffsetFromBase());
    assert(Offset >= 0 && "Offset too big to fit in int64_t!");
    if (Offset) {
      // BaseAddr = BaseAddr + Offset.
      EVT ArithType = BaseAddr.getValueType();
      SDLoc DL(Origin);
      BaseAddr = DAG->getNode(ISD::ADD, DL, ArithType, BaseAddr,
                              DAG->getConstant(Offset, DL, ArithType));
    }

    // Create the type of the loaded slice according to its size.
    EVT SliceType = getLoadedType();

    // Create the load for the slice.
    SDValue LastInst =
        DAG->getLoad(SliceType, SDLoc(Origin), Origin->getChain(), BaseAddr,
                     Origin->getPointerInfo().getWithOffset(Offset), getAlign(),
                     Origin->getMemOperand()->getFlags());
    // If the final type is not the same as the loaded type, this means that
    // we have to pad with zero. Create a zero extend for that.
    EVT FinalType = Inst->getValueType(0);
    if (SliceType != FinalType)
      LastInst =
          DAG->getNode(ISD::ZERO_EXTEND, SDLoc(LastInst), FinalType, LastInst);
    return LastInst;
  }

  /// Check if this slice can be merged with an expensive cross register
  /// bank copy. E.g.,
  /// i = load i32
  /// f = bitcast i32 i to float
  bool canMergeExpensiveCrossRegisterBankCopy() const {
    if (!Inst || !Inst->hasOneUse())
      return false;
    SDNode *Use = *Inst->use_begin();
    if (Use->getOpcode() != ISD::BITCAST)
      return false;
    assert(DAG && "Missing context");
    const TargetLowering &TLI = DAG->getTargetLoweringInfo();
    EVT ResVT = Use->getValueType(0);
    const TargetRegisterClass *ResRC =
        TLI.getRegClassFor(ResVT.getSimpleVT(), Use->isDivergent());
    const TargetRegisterClass *ArgRC =
        TLI.getRegClassFor(Use->getOperand(0).getValueType().getSimpleVT(),
                           Use->getOperand(0)->isDivergent());
    if (ArgRC == ResRC || !TLI.isOperationLegal(ISD::LOAD, ResVT))
      return false;

    // At this point, we know that we perform a cross-register-bank copy.
    // Check if it is expensive.
    const TargetRegisterInfo *TRI = DAG->getSubtarget().getRegisterInfo();
    // Assume bitcasts are cheap, unless both register classes do not
    // explicitly share a common sub class.
    if (!TRI || TRI->getCommonSubClass(ArgRC, ResRC))
      return false;

    // Check if it will be merged with the load.
    // 1. Check the alignment constraint.
    Align RequiredAlignment = DAG->getDataLayout().getABITypeAlign(
        ResVT.getTypeForEVT(*DAG->getContext()));

    if (RequiredAlignment > getAlign())
      return false;

    // 2. Check that the load is a legal operation for that type.
    if (!TLI.isOperationLegal(ISD::LOAD, ResVT))
      return false;

    // 3. Check that we do not have a zext in the way.
    if (Inst->getValueType(0) != getLoadedType())
      return false;

    return true;
  }
};

} // end anonymous namespace

/// Check that all bits set in \p UsedBits form a dense region, i.e.,
/// \p UsedBits looks like 0..0 1..1 0..0.
static bool areUsedBitsDense(const APInt &UsedBits) {
  // If all the bits are one, this is dense!
  if (UsedBits.isAllOnesValue())
    return true;

  // Get rid of the unused bits on the right.
  APInt NarrowedUsedBits = UsedBits.lshr(UsedBits.countTrailingZeros());
  // Get rid of the unused bits on the left.
  if (NarrowedUsedBits.countLeadingZeros())
    NarrowedUsedBits = NarrowedUsedBits.trunc(NarrowedUsedBits.getActiveBits());
  // Check that the chunk of bits is completely used.
  return NarrowedUsedBits.isAllOnesValue();
}

/// Check whether or not \p First and \p Second are next to each other
/// in memory. This means that there is no hole between the bits loaded
/// by \p First and the bits loaded by \p Second.
static bool areSlicesNextToEachOther(const LoadedSlice &First,
                                     const LoadedSlice &Second) {
  assert(First.Origin == Second.Origin && First.Origin &&
         "Unable to match different memory origins.");
  APInt UsedBits = First.getUsedBits();
  assert((UsedBits & Second.getUsedBits()) == 0 &&
         "Slices are not supposed to overlap.");
  UsedBits |= Second.getUsedBits();
  return areUsedBitsDense(UsedBits);
}

/// Adjust the \p GlobalLSCost according to the target
/// paring capabilities and the layout of the slices.
/// \pre \p GlobalLSCost should account for at least as many loads as
/// there is in the slices in \p LoadedSlices.
static void adjustCostForPairing(SmallVectorImpl<LoadedSlice> &LoadedSlices,
                                 LoadedSlice::Cost &GlobalLSCost) {
  unsigned NumberOfSlices = LoadedSlices.size();
  // If there is less than 2 elements, no pairing is possible.
  if (NumberOfSlices < 2)
    return;

  // Sort the slices so that elements that are likely to be next to each
  // other in memory are next to each other in the list.
  llvm::sort(LoadedSlices, [](const LoadedSlice &LHS, const LoadedSlice &RHS) {
    assert(LHS.Origin == RHS.Origin && "Different bases not implemented.");
    return LHS.getOffsetFromBase() < RHS.getOffsetFromBase();
  });
  const TargetLowering &TLI = LoadedSlices[0].DAG->getTargetLoweringInfo();
  // First (resp. Second) is the first (resp. Second) potentially candidate
  // to be placed in a paired load.
  const LoadedSlice *First = nullptr;
  const LoadedSlice *Second = nullptr;
  for (unsigned CurrSlice = 0; CurrSlice < NumberOfSlices; ++CurrSlice,
                // Set the beginning of the pair.
                                                           First = Second) {
    Second = &LoadedSlices[CurrSlice];

    // If First is NULL, it means we start a new pair.
    // Get to the next slice.
    if (!First)
      continue;

    EVT LoadedType = First->getLoadedType();

    // If the types of the slices are different, we cannot pair them.
    if (LoadedType != Second->getLoadedType())
      continue;

    // Check if the target supplies paired loads for this type.
    Align RequiredAlignment;
    if (!TLI.hasPairedLoad(LoadedType, RequiredAlignment)) {
      // move to the next pair, this type is hopeless.
      Second = nullptr;
      continue;
    }
    // Check if we meet the alignment requirement.
    if (First->getAlign() < RequiredAlignment)
      continue;

    // Check that both loads are next to each other in memory.
    if (!areSlicesNextToEachOther(*First, *Second))
      continue;

    assert(GlobalLSCost.Loads > 0 && "We save more loads than we created!");
    --GlobalLSCost.Loads;
    // Move to the next pair.
    Second = nullptr;
  }
}

/// Check the profitability of all involved LoadedSlice.
/// Currently, it is considered profitable if there is exactly two
/// involved slices (1) which are (2) next to each other in memory, and
/// whose cost (\see LoadedSlice::Cost) is smaller than the original load (3).
///
/// Note: The order of the elements in \p LoadedSlices may be modified, but not
/// the elements themselves.
///
/// FIXME: When the cost model will be mature enough, we can relax
/// constraints (1) and (2).
static bool isSlicingProfitable(SmallVectorImpl<LoadedSlice> &LoadedSlices,
                                const APInt &UsedBits, bool ForCodeSize) {
  unsigned NumberOfSlices = LoadedSlices.size();
  if (StressLoadSlicing)
    return NumberOfSlices > 1;

  // Check (1).
  if (NumberOfSlices != 2)
    return false;

  // Check (2).
  if (!areUsedBitsDense(UsedBits))
    return false;

  // Check (3).
  LoadedSlice::Cost OrigCost(ForCodeSize), GlobalSlicingCost(ForCodeSize);
  // The original code has one big load.
  OrigCost.Loads = 1;
  for (unsigned CurrSlice = 0; CurrSlice < NumberOfSlices; ++CurrSlice) {
    const LoadedSlice &LS = LoadedSlices[CurrSlice];
    // Accumulate the cost of all the slices.
    LoadedSlice::Cost SliceCost(LS, ForCodeSize);
    GlobalSlicingCost += SliceCost;

    // Account as cost in the original configuration the gain obtained
    // with the current slices.
    OrigCost.addSliceGain(LS);
  }

  // If the target supports paired load, adjust the cost accordingly.
  adjustCostForPairing(LoadedSlices, GlobalSlicingCost);
  return OrigCost > GlobalSlicingCost;
}

/// If the given load, \p LI, is used only by trunc or trunc(lshr)
/// operations, split it in the various pieces being extracted.
///
/// This sort of thing is introduced by SROA.
/// This slicing takes care not to insert overlapping loads.
/// \pre LI is a simple load (i.e., not an atomic or volatile load).
bool DAGCombiner::SliceUpLoad(SDNode *N) {
  if (Level < AfterLegalizeDAG)
    return false;

  LoadSDNode *LD = cast<LoadSDNode>(N);
  if (!LD->isSimple() || !ISD::isNormalLoad(LD) ||
      !LD->getValueType(0).isInteger())
    return false;

  // The algorithm to split up a load of a scalable vector into individual
  // elements currently requires knowing the length of the loaded type,
  // so will need adjusting to work on scalable vectors.
  if (LD->getValueType(0).isScalableVector())
    return false;

  // Keep track of already used bits to detect overlapping values.
  // In that case, we will just abort the transformation.
  APInt UsedBits(LD->getValueSizeInBits(0), 0);

  SmallVector<LoadedSlice, 4> LoadedSlices;

  // Check if this load is used as several smaller chunks of bits.
  // Basically, look for uses in trunc or trunc(lshr) and record a new chain
  // of computation for each trunc.
  for (SDNode::use_iterator UI = LD->use_begin(), UIEnd = LD->use_end();
       UI != UIEnd; ++UI) {
    // Skip the uses of the chain.
    if (UI.getUse().getResNo() != 0)
      continue;

    SDNode *User = *UI;
    unsigned Shift = 0;

    // Check if this is a trunc(lshr).
    if (User->getOpcode() == ISD::SRL && User->hasOneUse() &&
        isa<ConstantSDNode>(User->getOperand(1))) {
      Shift = User->getConstantOperandVal(1);
      User = *User->use_begin();
    }

    // At this point, User is a Truncate, iff we encountered, trunc or
    // trunc(lshr).
    if (User->getOpcode() != ISD::TRUNCATE)
      return false;

    // The width of the type must be a power of 2 and greater than 8-bits.
    // Otherwise the load cannot be represented in LLVM IR.
    // Moreover, if we shifted with a non-8-bits multiple, the slice
    // will be across several bytes. We do not support that.
    unsigned Width = User->getValueSizeInBits(0);
    if (Width < 8 || !isPowerOf2_32(Width) || (Shift & 0x7))
      return false;

    // Build the slice for this chain of computations.
    LoadedSlice LS(User, LD, Shift, &DAG);
    APInt CurrentUsedBits = LS.getUsedBits();

    // Check if this slice overlaps with another.
    if ((CurrentUsedBits & UsedBits) != 0)
      return false;
    // Update the bits used globally.
    UsedBits |= CurrentUsedBits;

    // Check if the new slice would be legal.
    if (!LS.isLegal())
      return false;

    // Record the slice.
    LoadedSlices.push_back(LS);
  }

  // Abort slicing if it does not seem to be profitable.
  if (!isSlicingProfitable(LoadedSlices, UsedBits, ForCodeSize))
    return false;

  ++SlicedLoads;

  // Rewrite each chain to use an independent load.
  // By construction, each chain can be represented by a unique load.

  // Prepare the argument for the new token factor for all the slices.
  SmallVector<SDValue, 8> ArgChains;
  for (const LoadedSlice &LS : LoadedSlices) {
    SDValue SliceInst = LS.loadSlice();
    CombineTo(LS.Inst, SliceInst, true);
    if (SliceInst.getOpcode() != ISD::LOAD)
      SliceInst = SliceInst.getOperand(0);
    assert(SliceInst->getOpcode() == ISD::LOAD &&
           "It takes more than a zext to get to the loaded slice!!");
    ArgChains.push_back(SliceInst.getValue(1));
  }

  SDValue Chain = DAG.getNode(ISD::TokenFactor, SDLoc(LD), MVT::Other,
                              ArgChains);
  DAG.ReplaceAllUsesOfValueWith(SDValue(N, 1), Chain);
  AddToWorklist(Chain.getNode());
  return true;
}

/// Check to see if V is (and load (ptr), imm), where the load is having
/// specific bytes cleared out.  If so, return the byte size being masked out
/// and the shift amount.
static std::pair<unsigned, unsigned>
CheckForMaskedLoad(SDValue V, SDValue Ptr, SDValue Chain) {
  std::pair<unsigned, unsigned> Result(0, 0);

  // Check for the structure we're looking for.
  if (V->getOpcode() != ISD::AND ||
      !isa<ConstantSDNode>(V->getOperand(1)) ||
      !ISD::isNormalLoad(V->getOperand(0).getNode()))
    return Result;

  // Check the chain and pointer.
  LoadSDNode *LD = cast<LoadSDNode>(V->getOperand(0));
  if (LD->getBasePtr() != Ptr) return Result;  // Not from same pointer.

  // This only handles simple types.
  if (V.getValueType() != MVT::i16 &&
      V.getValueType() != MVT::i32 &&
      V.getValueType() != MVT::i64)
    return Result;

  // Check the constant mask.  Invert it so that the bits being masked out are
  // 0 and the bits being kept are 1.  Use getSExtValue so that leading bits
  // follow the sign bit for uniformity.
  uint64_t NotMask = ~cast<ConstantSDNode>(V->getOperand(1))->getSExtValue();
  unsigned NotMaskLZ = countLeadingZeros(NotMask);
  if (NotMaskLZ & 7) return Result;  // Must be multiple of a byte.
  unsigned NotMaskTZ = countTrailingZeros(NotMask);
  if (NotMaskTZ & 7) return Result;  // Must be multiple of a byte.
  if (NotMaskLZ == 64) return Result;  // All zero mask.

  // See if we have a continuous run of bits.  If so, we have 0*1+0*
  if (countTrailingOnes(NotMask >> NotMaskTZ) + NotMaskTZ + NotMaskLZ != 64)
    return Result;

  // Adjust NotMaskLZ down to be from the actual size of the int instead of i64.
  if (V.getValueType() != MVT::i64 && NotMaskLZ)
    NotMaskLZ -= 64-V.getValueSizeInBits();

  unsigned MaskedBytes = (V.getValueSizeInBits()-NotMaskLZ-NotMaskTZ)/8;
  switch (MaskedBytes) {
  case 1:
  case 2:
  case 4: break;
  default: return Result; // All one mask, or 5-byte mask.
  }

  // Verify that the first bit starts at a multiple of mask so that the access
  // is aligned the same as the access width.
  if (NotMaskTZ && NotMaskTZ/8 % MaskedBytes) return Result;

  // For narrowing to be valid, it must be the case that the load the
  // immediately preceding memory operation before the store.
  if (LD == Chain.getNode())
    ; // ok.
  else if (Chain->getOpcode() == ISD::TokenFactor &&
           SDValue(LD, 1).hasOneUse()) {
    // LD has only 1 chain use so they are no indirect dependencies.
    if (!LD->isOperandOf(Chain.getNode()))
      return Result;
  } else
    return Result; // Fail.

  Result.first = MaskedBytes;
  Result.second = NotMaskTZ/8;
  return Result;
}

/// Check to see if IVal is something that provides a value as specified by
/// MaskInfo. If so, replace the specified store with a narrower store of
/// truncated IVal.
static SDValue
ShrinkLoadReplaceStoreWithStore(const std::pair<unsigned, unsigned> &MaskInfo,
                                SDValue IVal, StoreSDNode *St,
                                DAGCombiner *DC) {
  unsigned NumBytes = MaskInfo.first;
  unsigned ByteShift = MaskInfo.second;
  SelectionDAG &DAG = DC->getDAG();

  // Check to see if IVal is all zeros in the part being masked in by the 'or'
  // that uses this.  If not, this is not a replacement.
  APInt Mask = ~APInt::getBitsSet(IVal.getValueSizeInBits(),
                                  ByteShift*8, (ByteShift+NumBytes)*8);
  if (!DAG.MaskedValueIsZero(IVal, Mask)) return SDValue();

  // Check that it is legal on the target to do this.  It is legal if the new
  // VT we're shrinking to (i8/i16/i32) is legal or we're still before type
  // legalization (and the target doesn't explicitly think this is a bad idea).
  MVT VT = MVT::getIntegerVT(NumBytes * 8);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!DC->isTypeLegal(VT))
    return SDValue();
  if (St->getMemOperand() &&
      !TLI.allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), VT,
                              *St->getMemOperand()))
    return SDValue();

  // Okay, we can do this!  Replace the 'St' store with a store of IVal that is
  // shifted by ByteShift and truncated down to NumBytes.
  if (ByteShift) {
    SDLoc DL(IVal);
    IVal = DAG.getNode(ISD::SRL, DL, IVal.getValueType(), IVal,
                       DAG.getConstant(ByteShift*8, DL,
                                    DC->getShiftAmountTy(IVal.getValueType())));
  }

  // Figure out the offset for the store and the alignment of the access.
  unsigned StOffset;
  if (DAG.getDataLayout().isLittleEndian())
    StOffset = ByteShift;
  else
    StOffset = IVal.getValueType().getStoreSize() - ByteShift - NumBytes;

  SDValue Ptr = St->getBasePtr();
  if (StOffset) {
    SDLoc DL(IVal);
    Ptr = DAG.getMemBasePlusOffset(Ptr, TypeSize::Fixed(StOffset), DL);
  }

  // Truncate down to the new size.
  IVal = DAG.getNode(ISD::TRUNCATE, SDLoc(IVal), VT, IVal);

  ++OpsNarrowed;
  return DAG
      .getStore(St->getChain(), SDLoc(St), IVal, Ptr,
                St->getPointerInfo().getWithOffset(StOffset),
                St->getOriginalAlign());
}

/// Look for sequence of load / op / store where op is one of 'or', 'xor', and
/// 'and' of immediates. If 'op' is only touching some of the loaded bits, try
/// narrowing the load and store if it would end up being a win for performance
/// or code size.
SDValue DAGCombiner::ReduceLoadOpStoreWidth(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  if (!ST->isSimple())
    return SDValue();

  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();
  EVT VT = Value.getValueType();

  if (ST->isTruncatingStore() || VT.isVector() || !Value.hasOneUse())
    return SDValue();

  unsigned Opc = Value.getOpcode();

  // If this is "store (or X, Y), P" and X is "(and (load P), cst)", where cst
  // is a byte mask indicating a consecutive number of bytes, check to see if
  // Y is known to provide just those bytes.  If so, we try to replace the
  // load + replace + store sequence with a single (narrower) store, which makes
  // the load dead.
  if (Opc == ISD::OR && EnableShrinkLoadReplaceStoreWithStore) {
    std::pair<unsigned, unsigned> MaskedLoad;
    MaskedLoad = CheckForMaskedLoad(Value.getOperand(0), Ptr, Chain);
    if (MaskedLoad.first)
      if (SDValue NewST = ShrinkLoadReplaceStoreWithStore(MaskedLoad,
                                                  Value.getOperand(1), ST,this))
        return NewST;

    // Or is commutative, so try swapping X and Y.
    MaskedLoad = CheckForMaskedLoad(Value.getOperand(1), Ptr, Chain);
    if (MaskedLoad.first)
      if (SDValue NewST = ShrinkLoadReplaceStoreWithStore(MaskedLoad,
                                                  Value.getOperand(0), ST,this))
        return NewST;
  }

  if (!EnableReduceLoadOpStoreWidth)
    return SDValue();

  if ((Opc != ISD::OR && Opc != ISD::XOR && Opc != ISD::AND) ||
      Value.getOperand(1).getOpcode() != ISD::Constant)
    return SDValue();

  SDValue N0 = Value.getOperand(0);
  if (ISD::isNormalLoad(N0.getNode()) && N0.hasOneUse() &&
      Chain == SDValue(N0.getNode(), 1)) {
    LoadSDNode *LD = cast<LoadSDNode>(N0);
    if (LD->getBasePtr() != Ptr ||
        LD->getPointerInfo().getAddrSpace() !=
        ST->getPointerInfo().getAddrSpace())
      return SDValue();

    // Find the type to narrow it the load / op / store to.
    SDValue N1 = Value.getOperand(1);
    unsigned BitWidth = N1.getValueSizeInBits();
    APInt Imm = cast<ConstantSDNode>(N1)->getAPIntValue();
    if (Opc == ISD::AND)
      Imm ^= APInt::getAllOnesValue(BitWidth);
    if (Imm == 0 || Imm.isAllOnesValue())
      return SDValue();
    unsigned ShAmt = Imm.countTrailingZeros();
    unsigned MSB = BitWidth - Imm.countLeadingZeros() - 1;
    unsigned NewBW = NextPowerOf2(MSB - ShAmt);
    EVT NewVT = EVT::getIntegerVT(*DAG.getContext(), NewBW);
    // The narrowing should be profitable, the load/store operation should be
    // legal (or custom) and the store size should be equal to the NewVT width.
    while (NewBW < BitWidth &&
           (NewVT.getStoreSizeInBits() != NewBW ||
            !TLI.isOperationLegalOrCustom(Opc, NewVT) ||
            !TLI.isNarrowingProfitable(VT, NewVT))) {
      NewBW = NextPowerOf2(NewBW);
      NewVT = EVT::getIntegerVT(*DAG.getContext(), NewBW);
    }
    if (NewBW >= BitWidth)
      return SDValue();

    // If the lsb changed does not start at the type bitwidth boundary,
    // start at the previous one.
    if (ShAmt % NewBW)
      ShAmt = (((ShAmt + NewBW - 1) / NewBW) * NewBW) - NewBW;
    APInt Mask = APInt::getBitsSet(BitWidth, ShAmt,
                                   std::min(BitWidth, ShAmt + NewBW));
    if ((Imm & Mask) == Imm) {
      APInt NewImm = (Imm & Mask).lshr(ShAmt).trunc(NewBW);
      if (Opc == ISD::AND)
        NewImm ^= APInt::getAllOnesValue(NewBW);
      uint64_t PtrOff = ShAmt / 8;
      // For big endian targets, we need to adjust the offset to the pointer to
      // load the correct bytes.
      if (DAG.getDataLayout().isBigEndian())
        PtrOff = (BitWidth + 7 - NewBW) / 8 - PtrOff;

      Align NewAlign = commonAlignment(LD->getAlign(), PtrOff);
      Type *NewVTTy = NewVT.getTypeForEVT(*DAG.getContext());
      if (NewAlign < DAG.getDataLayout().getABITypeAlign(NewVTTy))
        return SDValue();

      SDValue NewPtr =
          DAG.getMemBasePlusOffset(Ptr, TypeSize::Fixed(PtrOff), SDLoc(LD));
      SDValue NewLD =
          DAG.getLoad(NewVT, SDLoc(N0), LD->getChain(), NewPtr,
                      LD->getPointerInfo().getWithOffset(PtrOff), NewAlign,
                      LD->getMemOperand()->getFlags(), LD->getAAInfo());
      SDValue NewVal = DAG.getNode(Opc, SDLoc(Value), NewVT, NewLD,
                                   DAG.getConstant(NewImm, SDLoc(Value),
                                                   NewVT));
      SDValue NewST =
          DAG.getStore(Chain, SDLoc(N), NewVal, NewPtr,
                       ST->getPointerInfo().getWithOffset(PtrOff), NewAlign);

      AddToWorklist(NewPtr.getNode());
      AddToWorklist(NewLD.getNode());
      AddToWorklist(NewVal.getNode());
      WorklistRemover DeadNodes(*this);
      DAG.ReplaceAllUsesOfValueWith(N0.getValue(1), NewLD.getValue(1));
      ++OpsNarrowed;
      return NewST;
    }
  }

  return SDValue();
}

/// For a given floating point load / store pair, if the load value isn't used
/// by any other operations, then consider transforming the pair to integer
/// load / store operations if the target deems the transformation profitable.
SDValue DAGCombiner::TransformFPLoadStorePair(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Value = ST->getValue();
  if (ISD::isNormalStore(ST) && ISD::isNormalLoad(Value.getNode()) &&
      Value.hasOneUse()) {
    LoadSDNode *LD = cast<LoadSDNode>(Value);
    EVT VT = LD->getMemoryVT();
    if (!VT.isFloatingPoint() ||
        VT != ST->getMemoryVT() ||
        LD->isNonTemporal() ||
        ST->isNonTemporal() ||
        LD->getPointerInfo().getAddrSpace() != 0 ||
        ST->getPointerInfo().getAddrSpace() != 0)
      return SDValue();

    TypeSize VTSize = VT.getSizeInBits();

    // We don't know the size of scalable types at compile time so we cannot
    // create an integer of the equivalent size.
    if (VTSize.isScalable())
      return SDValue();

    EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), VTSize.getFixedSize());
    if (!TLI.isOperationLegal(ISD::LOAD, IntVT) ||
        !TLI.isOperationLegal(ISD::STORE, IntVT) ||
        !TLI.isDesirableToTransformToIntegerOp(ISD::LOAD, VT) ||
        !TLI.isDesirableToTransformToIntegerOp(ISD::STORE, VT))
      return SDValue();

    Align LDAlign = LD->getAlign();
    Align STAlign = ST->getAlign();
    Type *IntVTTy = IntVT.getTypeForEVT(*DAG.getContext());
    Align ABIAlign = DAG.getDataLayout().getABITypeAlign(IntVTTy);
    if (LDAlign < ABIAlign || STAlign < ABIAlign)
      return SDValue();

    SDValue NewLD =
        DAG.getLoad(IntVT, SDLoc(Value), LD->getChain(), LD->getBasePtr(),
                    LD->getPointerInfo(), LDAlign);

    SDValue NewST =
        DAG.getStore(ST->getChain(), SDLoc(N), NewLD, ST->getBasePtr(),
                     ST->getPointerInfo(), STAlign);

    AddToWorklist(NewLD.getNode());
    AddToWorklist(NewST.getNode());
    WorklistRemover DeadNodes(*this);
    DAG.ReplaceAllUsesOfValueWith(Value.getValue(1), NewLD.getValue(1));
    ++LdStFP2Int;
    return NewST;
  }

  return SDValue();
}

// This is a helper function for visitMUL to check the profitability
// of folding (mul (add x, c1), c2) -> (add (mul x, c2), c1*c2).
// MulNode is the original multiply, AddNode is (add x, c1),
// and ConstNode is c2.
//
// If the (add x, c1) has multiple uses, we could increase
// the number of adds if we make this transformation.
// It would only be worth doing this if we can remove a
// multiply in the process. Check for that here.
// To illustrate:
//     (A + c1) * c3
//     (A + c2) * c3
// We're checking for cases where we have common "c3 * A" expressions.
bool DAGCombiner::isMulAddWithConstProfitable(SDNode *MulNode,
                                              SDValue &AddNode,
                                              SDValue &ConstNode) {
  APInt Val;

  // If the add only has one use, this would be OK to do.
  if (AddNode.getNode()->hasOneUse())
    return true;

  // Walk all the users of the constant with which we're multiplying.
  for (SDNode *Use : ConstNode->uses()) {
    if (Use == MulNode) // This use is the one we're on right now. Skip it.
      continue;

    if (Use->getOpcode() == ISD::MUL) { // We have another multiply use.
      SDNode *OtherOp;
      SDNode *MulVar = AddNode.getOperand(0).getNode();

      // OtherOp is what we're multiplying against the constant.
      if (Use->getOperand(0) == ConstNode)
        OtherOp = Use->getOperand(1).getNode();
      else
        OtherOp = Use->getOperand(0).getNode();

      // Check to see if multiply is with the same operand of our "add".
      //
      //     ConstNode  = CONST
      //     Use = ConstNode * A  <-- visiting Use. OtherOp is A.
      //     ...
      //     AddNode  = (A + c1)  <-- MulVar is A.
      //         = AddNode * ConstNode   <-- current visiting instruction.
      //
      // If we make this transformation, we will have a common
      // multiply (ConstNode * A) that we can save.
      if (OtherOp == MulVar)
        return true;

      // Now check to see if a future expansion will give us a common
      // multiply.
      //
      //     ConstNode  = CONST
      //     AddNode    = (A + c1)
      //     ...   = AddNode * ConstNode <-- current visiting instruction.
      //     ...
      //     OtherOp = (A + c2)
      //     Use     = OtherOp * ConstNode <-- visiting Use.
      //
      // If we make this transformation, we will have a common
      // multiply (CONST * A) after we also do the same transformation
      // to the "t2" instruction.
      if (OtherOp->getOpcode() == ISD::ADD &&
          DAG.isConstantIntBuildVectorOrConstantInt(OtherOp->getOperand(1)) &&
          OtherOp->getOperand(0).getNode() == MulVar)
        return true;
    }
  }

  // Didn't find a case where this would be profitable.
  return false;
}

SDValue DAGCombiner::getMergeStoreChains(SmallVectorImpl<MemOpLink> &StoreNodes,
                                         unsigned NumStores) {
  SmallVector<SDValue, 8> Chains;
  SmallPtrSet<const SDNode *, 8> Visited;
  SDLoc StoreDL(StoreNodes[0].MemNode);

  for (unsigned i = 0; i < NumStores; ++i) {
    Visited.insert(StoreNodes[i].MemNode);
  }

  // don't include nodes that are children or repeated nodes.
  for (unsigned i = 0; i < NumStores; ++i) {
    if (Visited.insert(StoreNodes[i].MemNode->getChain().getNode()).second)
      Chains.push_back(StoreNodes[i].MemNode->getChain());
  }

  assert(Chains.size() > 0 && "Chain should have generated a chain");
  return DAG.getTokenFactor(StoreDL, Chains);
}

bool DAGCombiner::mergeStoresOfConstantsOrVecElts(
    SmallVectorImpl<MemOpLink> &StoreNodes, EVT MemVT, unsigned NumStores,
    bool IsConstantSrc, bool UseVector, bool UseTrunc) {
  // Make sure we have something to merge.
  if (NumStores < 2)
    return false;

  assert((!UseTrunc || !UseVector) &&
         "This optimization cannot emit a vector truncating store");

  // The latest Node in the DAG.
  SDLoc DL(StoreNodes[0].MemNode);

  TypeSize ElementSizeBits = MemVT.getStoreSizeInBits();
  unsigned SizeInBits = NumStores * ElementSizeBits;
  unsigned NumMemElts = MemVT.isVector() ? MemVT.getVectorNumElements() : 1;

  EVT StoreTy;
  if (UseVector) {
    unsigned Elts = NumStores * NumMemElts;
    // Get the type for the merged vector store.
    StoreTy = EVT::getVectorVT(*DAG.getContext(), MemVT.getScalarType(), Elts);
  } else
    StoreTy = EVT::getIntegerVT(*DAG.getContext(), SizeInBits);

  SDValue StoredVal;
  if (UseVector) {
    if (IsConstantSrc) {
      SmallVector<SDValue, 8> BuildVector;
      for (unsigned I = 0; I != NumStores; ++I) {
        StoreSDNode *St = cast<StoreSDNode>(StoreNodes[I].MemNode);
        SDValue Val = St->getValue();
        // If constant is of the wrong type, convert it now.
        if (MemVT != Val.getValueType()) {
          Val = peekThroughBitcasts(Val);
          // Deal with constants of wrong size.
          if (ElementSizeBits != Val.getValueSizeInBits()) {
            EVT IntMemVT =
                EVT::getIntegerVT(*DAG.getContext(), MemVT.getSizeInBits());
            if (isa<ConstantFPSDNode>(Val)) {
              // Not clear how to truncate FP values.
              return false;
            } else if (auto *C = dyn_cast<ConstantSDNode>(Val))
              Val = DAG.getConstant(C->getAPIntValue()
                                        .zextOrTrunc(Val.getValueSizeInBits())
                                        .zextOrTrunc(ElementSizeBits),
                                    SDLoc(C), IntMemVT);
          }
          // Make sure correctly size type is the correct type.
          Val = DAG.getBitcast(MemVT, Val);
        }
        BuildVector.push_back(Val);
      }
      StoredVal = DAG.getNode(MemVT.isVector() ? ISD::CONCAT_VECTORS
                                               : ISD::BUILD_VECTOR,
                              DL, StoreTy, BuildVector);
    } else {
      SmallVector<SDValue, 8> Ops;
      for (unsigned i = 0; i < NumStores; ++i) {
        StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
        SDValue Val = peekThroughBitcasts(St->getValue());
        // All operands of BUILD_VECTOR / CONCAT_VECTOR must be of
        // type MemVT. If the underlying value is not the correct
        // type, but it is an extraction of an appropriate vector we
        // can recast Val to be of the correct type. This may require
        // converting between EXTRACT_VECTOR_ELT and
        // EXTRACT_SUBVECTOR.
        if ((MemVT != Val.getValueType()) &&
            (Val.getOpcode() == ISD::EXTRACT_VECTOR_ELT ||
             Val.getOpcode() == ISD::EXTRACT_SUBVECTOR)) {
          EVT MemVTScalarTy = MemVT.getScalarType();
          // We may need to add a bitcast here to get types to line up.
          if (MemVTScalarTy != Val.getValueType().getScalarType()) {
            Val = DAG.getBitcast(MemVT, Val);
          } else {
            unsigned OpC = MemVT.isVector() ? ISD::EXTRACT_SUBVECTOR
                                            : ISD::EXTRACT_VECTOR_ELT;
            SDValue Vec = Val.getOperand(0);
            SDValue Idx = Val.getOperand(1);
            Val = DAG.getNode(OpC, SDLoc(Val), MemVT, Vec, Idx);
          }
        }
        Ops.push_back(Val);
      }

      // Build the extracted vector elements back into a vector.
      StoredVal = DAG.getNode(MemVT.isVector() ? ISD::CONCAT_VECTORS
                                               : ISD::BUILD_VECTOR,
                              DL, StoreTy, Ops);
    }
  } else {
    // We should always use a vector store when merging extracted vector
    // elements, so this path implies a store of constants.
    assert(IsConstantSrc && "Merged vector elements should use vector store");

    APInt StoreInt(SizeInBits, 0);

    // Construct a single integer constant which is made of the smaller
    // constant inputs.
    bool IsLE = DAG.getDataLayout().isLittleEndian();
    for (unsigned i = 0; i < NumStores; ++i) {
      unsigned Idx = IsLE ? (NumStores - 1 - i) : i;
      StoreSDNode *St  = cast<StoreSDNode>(StoreNodes[Idx].MemNode);

      SDValue Val = St->getValue();
      Val = peekThroughBitcasts(Val);
      StoreInt <<= ElementSizeBits;
      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Val)) {
        StoreInt |= C->getAPIntValue()
                        .zextOrTrunc(ElementSizeBits)
                        .zextOrTrunc(SizeInBits);
      } else if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(Val)) {
        StoreInt |= C->getValueAPF()
                        .bitcastToAPInt()
                        .zextOrTrunc(ElementSizeBits)
                        .zextOrTrunc(SizeInBits);
        // If fp truncation is necessary give up for now.
        if (MemVT.getSizeInBits() != ElementSizeBits)
          return false;
      } else {
        llvm_unreachable("Invalid constant element type");
      }
    }

    // Create the new Load and Store operations.
    StoredVal = DAG.getConstant(StoreInt, DL, StoreTy);
  }

  LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;
  SDValue NewChain = getMergeStoreChains(StoreNodes, NumStores);

  // make sure we use trunc store if it's necessary to be legal.
  SDValue NewStore;
  if (!UseTrunc) {
    NewStore =
        DAG.getStore(NewChain, DL, StoredVal, FirstInChain->getBasePtr(),
                     FirstInChain->getPointerInfo(), FirstInChain->getAlign());
  } else { // Must be realized as a trunc store
    EVT LegalizedStoredValTy =
        TLI.getTypeToTransformTo(*DAG.getContext(), StoredVal.getValueType());
    unsigned LegalizedStoreSize = LegalizedStoredValTy.getSizeInBits();
    ConstantSDNode *C = cast<ConstantSDNode>(StoredVal);
    SDValue ExtendedStoreVal =
        DAG.getConstant(C->getAPIntValue().zextOrTrunc(LegalizedStoreSize), DL,
                        LegalizedStoredValTy);
    NewStore = DAG.getTruncStore(
        NewChain, DL, ExtendedStoreVal, FirstInChain->getBasePtr(),
        FirstInChain->getPointerInfo(), StoredVal.getValueType() /*TVT*/,
        FirstInChain->getAlign(), FirstInChain->getMemOperand()->getFlags());
  }

  // Replace all merged stores with the new store.
  for (unsigned i = 0; i < NumStores; ++i)
    CombineTo(StoreNodes[i].MemNode, NewStore);

  AddToWorklist(NewChain.getNode());
  return true;
}

void DAGCombiner::getStoreMergeCandidates(
    StoreSDNode *St, SmallVectorImpl<MemOpLink> &StoreNodes,
    SDNode *&RootNode) {
  // This holds the base pointer, index, and the offset in bytes from the base
  // pointer. We must have a base and an offset. Do not handle stores to undef
  // base pointers.
  BaseIndexOffset BasePtr = BaseIndexOffset::match(St, DAG);
  if (!BasePtr.getBase().getNode() || BasePtr.getBase().isUndef())
    return;

  SDValue Val = peekThroughBitcasts(St->getValue());
  StoreSource StoreSrc = getStoreSource(Val);
  assert(StoreSrc != StoreSource::Unknown && "Expected known source for store");

  // Match on loadbaseptr if relevant.
  EVT MemVT = St->getMemoryVT();
  BaseIndexOffset LBasePtr;
  EVT LoadVT;
  if (StoreSrc == StoreSource::Load) {
    auto *Ld = cast<LoadSDNode>(Val);
    LBasePtr = BaseIndexOffset::match(Ld, DAG);
    LoadVT = Ld->getMemoryVT();
    // Load and store should be the same type.
    if (MemVT != LoadVT)
      return;
    // Loads must only have one use.
    if (!Ld->hasNUsesOfValue(1, 0))
      return;
    // The memory operands must not be volatile/indexed/atomic.
    // TODO: May be able to relax for unordered atomics (see D66309)
    if (!Ld->isSimple() || Ld->isIndexed())
      return;
  }
  auto CandidateMatch = [&](StoreSDNode *Other, BaseIndexOffset &Ptr,
                            int64_t &Offset) -> bool {
    // The memory operands must not be volatile/indexed/atomic.
    // TODO: May be able to relax for unordered atomics (see D66309)
    if (!Other->isSimple() || Other->isIndexed())
      return false;
    // Don't mix temporal stores with non-temporal stores.
    if (St->isNonTemporal() != Other->isNonTemporal())
      return false;
    SDValue OtherBC = peekThroughBitcasts(Other->getValue());
    // Allow merging constants of different types as integers.
    bool NoTypeMatch = (MemVT.isInteger()) ? !MemVT.bitsEq(Other->getMemoryVT())
                                           : Other->getMemoryVT() != MemVT;
    switch (StoreSrc) {
    case StoreSource::Load: {
      if (NoTypeMatch)
        return false;
      // The Load's Base Ptr must also match.
      auto *OtherLd = dyn_cast<LoadSDNode>(OtherBC);
      if (!OtherLd)
        return false;
      BaseIndexOffset LPtr = BaseIndexOffset::match(OtherLd, DAG);
      if (LoadVT != OtherLd->getMemoryVT())
        return false;
      // Loads must only have one use.
      if (!OtherLd->hasNUsesOfValue(1, 0))
        return false;
      // The memory operands must not be volatile/indexed/atomic.
      // TODO: May be able to relax for unordered atomics (see D66309)
      if (!OtherLd->isSimple() || OtherLd->isIndexed())
        return false;
      // Don't mix temporal loads with non-temporal loads.
      if (cast<LoadSDNode>(Val)->isNonTemporal() != OtherLd->isNonTemporal())
        return false;
      if (!(LBasePtr.equalBaseIndex(LPtr, DAG)))
        return false;
      break;
    }
    case StoreSource::Constant:
      if (NoTypeMatch)
        return false;
      if (!isIntOrFPConstant(OtherBC))
        return false;
      break;
    case StoreSource::Extract:
      // Do not merge truncated stores here.
      if (Other->isTruncatingStore())
        return false;
      if (!MemVT.bitsEq(OtherBC.getValueType()))
        return false;
      if (OtherBC.getOpcode() != ISD::EXTRACT_VECTOR_ELT &&
          OtherBC.getOpcode() != ISD::EXTRACT_SUBVECTOR)
        return false;
      break;
    default:
      llvm_unreachable("Unhandled store source for merging");
    }
    Ptr = BaseIndexOffset::match(Other, DAG);
    return (BasePtr.equalBaseIndex(Ptr, DAG, Offset));
  };

  // Check if the pair of StoreNode and the RootNode already bail out many
  // times which is over the limit in dependence check.
  auto OverLimitInDependenceCheck = [&](SDNode *StoreNode,
                                        SDNode *RootNode) -> bool {
    auto RootCount = StoreRootCountMap.find(StoreNode);
    return RootCount != StoreRootCountMap.end() &&
           RootCount->second.first == RootNode &&
           RootCount->second.second > StoreMergeDependenceLimit;
  };

  auto TryToAddCandidate = [&](SDNode::use_iterator UseIter) {
    // This must be a chain use.
    if (UseIter.getOperandNo() != 0)
      return;
    if (auto *OtherStore = dyn_cast<StoreSDNode>(*UseIter)) {
      BaseIndexOffset Ptr;
      int64_t PtrDiff;
      if (CandidateMatch(OtherStore, Ptr, PtrDiff) &&
          !OverLimitInDependenceCheck(OtherStore, RootNode))
        StoreNodes.push_back(MemOpLink(OtherStore, PtrDiff));
    }
  };

  // We looking for a root node which is an ancestor to all mergable
  // stores. We search up through a load, to our root and then down
  // through all children. For instance we will find Store{1,2,3} if
  // St is Store1, Store2. or Store3 where the root is not a load
  // which always true for nonvolatile ops. TODO: Expand
  // the search to find all valid candidates through multiple layers of loads.
  //
  // Root
  // |-------|-------|
  // Load    Load    Store3
  // |       |
  // Store1   Store2
  //
  // FIXME: We should be able to climb and
  // descend TokenFactors to find candidates as well.

  RootNode = St->getChain().getNode();

  unsigned NumNodesExplored = 0;
  const unsigned MaxSearchNodes = 1024;
  if (auto *Ldn = dyn_cast<LoadSDNode>(RootNode)) {
    RootNode = Ldn->getChain().getNode();
    for (auto I = RootNode->use_begin(), E = RootNode->use_end();
         I != E && NumNodesExplored < MaxSearchNodes; ++I, ++NumNodesExplored) {
      if (I.getOperandNo() == 0 && isa<LoadSDNode>(*I)) { // walk down chain
        for (auto I2 = (*I)->use_begin(), E2 = (*I)->use_end(); I2 != E2; ++I2)
          TryToAddCandidate(I2);
      }
    }
  } else {
    for (auto I = RootNode->use_begin(), E = RootNode->use_end();
         I != E && NumNodesExplored < MaxSearchNodes; ++I, ++NumNodesExplored)
      TryToAddCandidate(I);
  }
}

// We need to check that merging these stores does not cause a loop in
// the DAG. Any store candidate may depend on another candidate
// indirectly through its operand (we already consider dependencies
// through the chain). Check in parallel by searching up from
// non-chain operands of candidates.
bool DAGCombiner::checkMergeStoreCandidatesForDependencies(
    SmallVectorImpl<MemOpLink> &StoreNodes, unsigned NumStores,
    SDNode *RootNode) {
  // FIXME: We should be able to truncate a full search of
  // predecessors by doing a BFS and keeping tabs the originating
  // stores from which worklist nodes come from in a similar way to
  // TokenFactor simplfication.

  SmallPtrSet<const SDNode *, 32> Visited;
  SmallVector<const SDNode *, 8> Worklist;

  // RootNode is a predecessor to all candidates so we need not search
  // past it. Add RootNode (peeking through TokenFactors). Do not count
  // these towards size check.

  Worklist.push_back(RootNode);
  while (!Worklist.empty()) {
    auto N = Worklist.pop_back_val();
    if (!Visited.insert(N).second)
      continue; // Already present in Visited.
    if (N->getOpcode() == ISD::TokenFactor) {
      for (SDValue Op : N->ops())
        Worklist.push_back(Op.getNode());
    }
  }

  // Don't count pruning nodes towards max.
  unsigned int Max = 1024 + Visited.size();
  // Search Ops of store candidates.
  for (unsigned i = 0; i < NumStores; ++i) {
    SDNode *N = StoreNodes[i].MemNode;
    // Of the 4 Store Operands:
    //   * Chain (Op 0) -> We have already considered these
    //                    in candidate selection and can be
    //                    safely ignored
    //   * Value (Op 1) -> Cycles may happen (e.g. through load chains)
    //   * Address (Op 2) -> Merged addresses may only vary by a fixed constant,
    //                       but aren't necessarily fromt the same base node, so
    //                       cycles possible (e.g. via indexed store).
    //   * (Op 3) -> Represents the pre or post-indexing offset (or undef for
    //               non-indexed stores). Not constant on all targets (e.g. ARM)
    //               and so can participate in a cycle.
    for (unsigned j = 1; j < N->getNumOperands(); ++j)
      Worklist.push_back(N->getOperand(j).getNode());
  }
  // Search through DAG. We can stop early if we find a store node.
  for (unsigned i = 0; i < NumStores; ++i)
    if (SDNode::hasPredecessorHelper(StoreNodes[i].MemNode, Visited, Worklist,
                                     Max)) {
      // If the searching bail out, record the StoreNode and RootNode in the
      // StoreRootCountMap. If we have seen the pair many times over a limit,
      // we won't add the StoreNode into StoreNodes set again.
      if (Visited.size() >= Max) {
        auto &RootCount = StoreRootCountMap[StoreNodes[i].MemNode];
        if (RootCount.first == RootNode)
          RootCount.second++;
        else
          RootCount = {RootNode, 1};
      }
      return false;
    }
  return true;
}

unsigned
DAGCombiner::getConsecutiveStores(SmallVectorImpl<MemOpLink> &StoreNodes,
                                  int64_t ElementSizeBytes) const {
  while (true) {
    // Find a store past the width of the first store.
    size_t StartIdx = 0;
    while ((StartIdx + 1 < StoreNodes.size()) &&
           StoreNodes[StartIdx].OffsetFromBase + ElementSizeBytes !=
              StoreNodes[StartIdx + 1].OffsetFromBase)
      ++StartIdx;

    // Bail if we don't have enough candidates to merge.
    if (StartIdx + 1 >= StoreNodes.size())
      return 0;

    // Trim stores that overlapped with the first store.
    if (StartIdx)
      StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + StartIdx);

    // Scan the memory operations on the chain and find the first
    // non-consecutive store memory address.
    unsigned NumConsecutiveStores = 1;
    int64_t StartAddress = StoreNodes[0].OffsetFromBase;
    // Check that the addresses are consecutive starting from the second
    // element in the list of stores.
    for (unsigned i = 1, e = StoreNodes.size(); i < e; ++i) {
      int64_t CurrAddress = StoreNodes[i].OffsetFromBase;
      if (CurrAddress - StartAddress != (ElementSizeBytes * i))
        break;
      NumConsecutiveStores = i + 1;
    }
    if (NumConsecutiveStores > 1)
      return NumConsecutiveStores;

    // There are no consecutive stores at the start of the list.
    // Remove the first store and try again.
    StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + 1);
  }
}

bool DAGCombiner::tryStoreMergeOfConstants(
    SmallVectorImpl<MemOpLink> &StoreNodes, unsigned NumConsecutiveStores,
    EVT MemVT, SDNode *RootNode, bool AllowVectors) {
  LLVMContext &Context = *DAG.getContext();
  const DataLayout &DL = DAG.getDataLayout();
  int64_t ElementSizeBytes = MemVT.getStoreSize();
  unsigned NumMemElts = MemVT.isVector() ? MemVT.getVectorNumElements() : 1;
  bool MadeChange = false;

  // Store the constants into memory as one consecutive store.
  while (NumConsecutiveStores >= 2) {
    LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;
    unsigned FirstStoreAS = FirstInChain->getAddressSpace();
    unsigned FirstStoreAlign = FirstInChain->getAlignment();
    unsigned LastLegalType = 1;
    unsigned LastLegalVectorType = 1;
    bool LastIntegerTrunc = false;
    bool NonZero = false;
    unsigned FirstZeroAfterNonZero = NumConsecutiveStores;
    for (unsigned i = 0; i < NumConsecutiveStores; ++i) {
      StoreSDNode *ST = cast<StoreSDNode>(StoreNodes[i].MemNode);
      SDValue StoredVal = ST->getValue();
      bool IsElementZero = false;
      if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(StoredVal))
        IsElementZero = C->isNullValue();
      else if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(StoredVal))
        IsElementZero = C->getConstantFPValue()->isNullValue();
      if (IsElementZero) {
        if (NonZero && FirstZeroAfterNonZero == NumConsecutiveStores)
          FirstZeroAfterNonZero = i;
      }
      NonZero |= !IsElementZero;

      // Find a legal type for the constant store.
      unsigned SizeInBits = (i + 1) * ElementSizeBytes * 8;
      EVT StoreTy = EVT::getIntegerVT(Context, SizeInBits);
      bool IsFast = false;

      // Break early when size is too large to be legal.
      if (StoreTy.getSizeInBits() > MaximumLegalStoreInBits)
        break;

      if (TLI.isTypeLegal(StoreTy) &&
          TLI.canMergeStoresTo(FirstStoreAS, StoreTy,
                               DAG.getMachineFunction()) &&
          TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                 *FirstInChain->getMemOperand(), &IsFast) &&
          IsFast) {
        LastIntegerTrunc = false;
        LastLegalType = i + 1;
        // Or check whether a truncstore is legal.
      } else if (TLI.getTypeAction(Context, StoreTy) ==
                 TargetLowering::TypePromoteInteger) {
        EVT LegalizedStoredValTy =
            TLI.getTypeToTransformTo(Context, StoredVal.getValueType());
        if (TLI.isTruncStoreLegal(LegalizedStoredValTy, StoreTy) &&
            TLI.canMergeStoresTo(FirstStoreAS, LegalizedStoredValTy,
                                 DAG.getMachineFunction()) &&
            TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                   *FirstInChain->getMemOperand(), &IsFast) &&
            IsFast) {
          LastIntegerTrunc = true;
          LastLegalType = i + 1;
        }
      }

      // We only use vectors if the constant is known to be zero or the
      // target allows it and the function is not marked with the
      // noimplicitfloat attribute.
      if ((!NonZero ||
           TLI.storeOfVectorConstantIsCheap(MemVT, i + 1, FirstStoreAS)) &&
          AllowVectors) {
        // Find a legal type for the vector store.
        unsigned Elts = (i + 1) * NumMemElts;
        EVT Ty = EVT::getVectorVT(Context, MemVT.getScalarType(), Elts);
        if (TLI.isTypeLegal(Ty) && TLI.isTypeLegal(MemVT) &&
            TLI.canMergeStoresTo(FirstStoreAS, Ty, DAG.getMachineFunction()) &&
            TLI.allowsMemoryAccess(Context, DL, Ty,
                                   *FirstInChain->getMemOperand(), &IsFast) &&
            IsFast)
          LastLegalVectorType = i + 1;
      }
    }

    bool UseVector = (LastLegalVectorType > LastLegalType) && AllowVectors;
    unsigned NumElem = (UseVector) ? LastLegalVectorType : LastLegalType;
    bool UseTrunc = LastIntegerTrunc && !UseVector;

    // Check if we found a legal integer type that creates a meaningful
    // merge.
    if (NumElem < 2) {
      // We know that candidate stores are in order and of correct
      // shape. While there is no mergeable sequence from the
      // beginning one may start later in the sequence. The only
      // reason a merge of size N could have failed where another of
      // the same size would not have, is if the alignment has
      // improved or we've dropped a non-zero value. Drop as many
      // candidates as we can here.
      unsigned NumSkip = 1;
      while ((NumSkip < NumConsecutiveStores) &&
             (NumSkip < FirstZeroAfterNonZero) &&
             (StoreNodes[NumSkip].MemNode->getAlignment() <= FirstStoreAlign))
        NumSkip++;

      StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumSkip);
      NumConsecutiveStores -= NumSkip;
      continue;
    }

    // Check that we can merge these candidates without causing a cycle.
    if (!checkMergeStoreCandidatesForDependencies(StoreNodes, NumElem,
                                                  RootNode)) {
      StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumElem);
      NumConsecutiveStores -= NumElem;
      continue;
    }

    MadeChange |= mergeStoresOfConstantsOrVecElts(StoreNodes, MemVT, NumElem,
                                                  /*IsConstantSrc*/ true,
                                                  UseVector, UseTrunc);

    // Remove merged stores for next iteration.
    StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumElem);
    NumConsecutiveStores -= NumElem;
  }
  return MadeChange;
}

bool DAGCombiner::tryStoreMergeOfExtracts(
    SmallVectorImpl<MemOpLink> &StoreNodes, unsigned NumConsecutiveStores,
    EVT MemVT, SDNode *RootNode) {
  LLVMContext &Context = *DAG.getContext();
  const DataLayout &DL = DAG.getDataLayout();
  unsigned NumMemElts = MemVT.isVector() ? MemVT.getVectorNumElements() : 1;
  bool MadeChange = false;

  // Loop on Consecutive Stores on success.
  while (NumConsecutiveStores >= 2) {
    LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;
    unsigned FirstStoreAS = FirstInChain->getAddressSpace();
    unsigned FirstStoreAlign = FirstInChain->getAlignment();
    unsigned NumStoresToMerge = 1;
    for (unsigned i = 0; i < NumConsecutiveStores; ++i) {
      // Find a legal type for the vector store.
      unsigned Elts = (i + 1) * NumMemElts;
      EVT Ty = EVT::getVectorVT(*DAG.getContext(), MemVT.getScalarType(), Elts);
      bool IsFast = false;

      // Break early when size is too large to be legal.
      if (Ty.getSizeInBits() > MaximumLegalStoreInBits)
        break;

      if (TLI.isTypeLegal(Ty) &&
          TLI.canMergeStoresTo(FirstStoreAS, Ty, DAG.getMachineFunction()) &&
          TLI.allowsMemoryAccess(Context, DL, Ty,
                                 *FirstInChain->getMemOperand(), &IsFast) &&
          IsFast)
        NumStoresToMerge = i + 1;
    }

    // Check if we found a legal integer type creating a meaningful
    // merge.
    if (NumStoresToMerge < 2) {
      // We know that candidate stores are in order and of correct
      // shape. While there is no mergeable sequence from the
      // beginning one may start later in the sequence. The only
      // reason a merge of size N could have failed where another of
      // the same size would not have, is if the alignment has
      // improved. Drop as many candidates as we can here.
      unsigned NumSkip = 1;
      while ((NumSkip < NumConsecutiveStores) &&
             (StoreNodes[NumSkip].MemNode->getAlignment() <= FirstStoreAlign))
        NumSkip++;

      StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumSkip);
      NumConsecutiveStores -= NumSkip;
      continue;
    }

    // Check that we can merge these candidates without causing a cycle.
    if (!checkMergeStoreCandidatesForDependencies(StoreNodes, NumStoresToMerge,
                                                  RootNode)) {
      StoreNodes.erase(StoreNodes.begin(),
                       StoreNodes.begin() + NumStoresToMerge);
      NumConsecutiveStores -= NumStoresToMerge;
      continue;
    }

    MadeChange |= mergeStoresOfConstantsOrVecElts(
        StoreNodes, MemVT, NumStoresToMerge, /*IsConstantSrc*/ false,
        /*UseVector*/ true, /*UseTrunc*/ false);

    StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumStoresToMerge);
    NumConsecutiveStores -= NumStoresToMerge;
  }
  return MadeChange;
}

bool DAGCombiner::tryStoreMergeOfLoads(SmallVectorImpl<MemOpLink> &StoreNodes,
                                       unsigned NumConsecutiveStores, EVT MemVT,
                                       SDNode *RootNode, bool AllowVectors,
                                       bool IsNonTemporalStore,
                                       bool IsNonTemporalLoad) {
  LLVMContext &Context = *DAG.getContext();
  const DataLayout &DL = DAG.getDataLayout();
  int64_t ElementSizeBytes = MemVT.getStoreSize();
  unsigned NumMemElts = MemVT.isVector() ? MemVT.getVectorNumElements() : 1;
  bool MadeChange = false;

  // Look for load nodes which are used by the stored values.
  SmallVector<MemOpLink, 8> LoadNodes;

  // Find acceptable loads. Loads need to have the same chain (token factor),
  // must not be zext, volatile, indexed, and they must be consecutive.
  BaseIndexOffset LdBasePtr;

  for (unsigned i = 0; i < NumConsecutiveStores; ++i) {
    StoreSDNode *St = cast<StoreSDNode>(StoreNodes[i].MemNode);
    SDValue Val = peekThroughBitcasts(St->getValue());
    LoadSDNode *Ld = cast<LoadSDNode>(Val);

    BaseIndexOffset LdPtr = BaseIndexOffset::match(Ld, DAG);
    // If this is not the first ptr that we check.
    int64_t LdOffset = 0;
    if (LdBasePtr.getBase().getNode()) {
      // The base ptr must be the same.
      if (!LdBasePtr.equalBaseIndex(LdPtr, DAG, LdOffset))
        break;
    } else {
      // Check that all other base pointers are the same as this one.
      LdBasePtr = LdPtr;
    }

    // We found a potential memory operand to merge.
    LoadNodes.push_back(MemOpLink(Ld, LdOffset));
  }

  while (NumConsecutiveStores >= 2 && LoadNodes.size() >= 2) {
    Align RequiredAlignment;
    bool NeedRotate = false;
    if (LoadNodes.size() == 2) {
      // If we have load/store pair instructions and we only have two values,
      // don't bother merging.
      if (TLI.hasPairedLoad(MemVT, RequiredAlignment) &&
          StoreNodes[0].MemNode->getAlign() >= RequiredAlignment) {
        StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + 2);
        LoadNodes.erase(LoadNodes.begin(), LoadNodes.begin() + 2);
        break;
      }
      // If the loads are reversed, see if we can rotate the halves into place.
      int64_t Offset0 = LoadNodes[0].OffsetFromBase;
      int64_t Offset1 = LoadNodes[1].OffsetFromBase;
      EVT PairVT = EVT::getIntegerVT(Context, ElementSizeBytes * 8 * 2);
      if (Offset0 - Offset1 == ElementSizeBytes &&
          (hasOperation(ISD::ROTL, PairVT) ||
           hasOperation(ISD::ROTR, PairVT))) {
        std::swap(LoadNodes[0], LoadNodes[1]);
        NeedRotate = true;
      }
    }
    LSBaseSDNode *FirstInChain = StoreNodes[0].MemNode;
    unsigned FirstStoreAS = FirstInChain->getAddressSpace();
    Align FirstStoreAlign = FirstInChain->getAlign();
    LoadSDNode *FirstLoad = cast<LoadSDNode>(LoadNodes[0].MemNode);

    // Scan the memory operations on the chain and find the first
    // non-consecutive load memory address. These variables hold the index in
    // the store node array.

    unsigned LastConsecutiveLoad = 1;

    // This variable refers to the size and not index in the array.
    unsigned LastLegalVectorType = 1;
    unsigned LastLegalIntegerType = 1;
    bool isDereferenceable = true;
    bool DoIntegerTruncate = false;
    int64_t StartAddress = LoadNodes[0].OffsetFromBase;
    SDValue LoadChain = FirstLoad->getChain();
    for (unsigned i = 1; i < LoadNodes.size(); ++i) {
      // All loads must share the same chain.
      if (LoadNodes[i].MemNode->getChain() != LoadChain)
        break;

      int64_t CurrAddress = LoadNodes[i].OffsetFromBase;
      if (CurrAddress - StartAddress != (ElementSizeBytes * i))
        break;
      LastConsecutiveLoad = i;

      if (isDereferenceable && !LoadNodes[i].MemNode->isDereferenceable())
        isDereferenceable = false;

      // Find a legal type for the vector store.
      unsigned Elts = (i + 1) * NumMemElts;
      EVT StoreTy = EVT::getVectorVT(Context, MemVT.getScalarType(), Elts);

      // Break early when size is too large to be legal.
      if (StoreTy.getSizeInBits() > MaximumLegalStoreInBits)
        break;

      bool IsFastSt = false;
      bool IsFastLd = false;
      if (TLI.isTypeLegal(StoreTy) &&
          TLI.canMergeStoresTo(FirstStoreAS, StoreTy,
                               DAG.getMachineFunction()) &&
          TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                 *FirstInChain->getMemOperand(), &IsFastSt) &&
          IsFastSt &&
          TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                 *FirstLoad->getMemOperand(), &IsFastLd) &&
          IsFastLd) {
        LastLegalVectorType = i + 1;
      }

      // Find a legal type for the integer store.
      unsigned SizeInBits = (i + 1) * ElementSizeBytes * 8;
      StoreTy = EVT::getIntegerVT(Context, SizeInBits);
      if (TLI.isTypeLegal(StoreTy) &&
          TLI.canMergeStoresTo(FirstStoreAS, StoreTy,
                               DAG.getMachineFunction()) &&
          TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                 *FirstInChain->getMemOperand(), &IsFastSt) &&
          IsFastSt &&
          TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                 *FirstLoad->getMemOperand(), &IsFastLd) &&
          IsFastLd) {
        LastLegalIntegerType = i + 1;
        DoIntegerTruncate = false;
        // Or check whether a truncstore and extload is legal.
      } else if (TLI.getTypeAction(Context, StoreTy) ==
                 TargetLowering::TypePromoteInteger) {
        EVT LegalizedStoredValTy = TLI.getTypeToTransformTo(Context, StoreTy);
        if (TLI.isTruncStoreLegal(LegalizedStoredValTy, StoreTy) &&
            TLI.canMergeStoresTo(FirstStoreAS, LegalizedStoredValTy,
                                 DAG.getMachineFunction()) &&
            TLI.isLoadExtLegal(ISD::ZEXTLOAD, LegalizedStoredValTy, StoreTy) &&
            TLI.isLoadExtLegal(ISD::SEXTLOAD, LegalizedStoredValTy, StoreTy) &&
            TLI.isLoadExtLegal(ISD::EXTLOAD, LegalizedStoredValTy, StoreTy) &&
            TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                   *FirstInChain->getMemOperand(), &IsFastSt) &&
            IsFastSt &&
            TLI.allowsMemoryAccess(Context, DL, StoreTy,
                                   *FirstLoad->getMemOperand(), &IsFastLd) &&
            IsFastLd) {
          LastLegalIntegerType = i + 1;
          DoIntegerTruncate = true;
        }
      }
    }

    // Only use vector types if the vector type is larger than the integer
    // type. If they are the same, use integers.
    bool UseVectorTy =
        LastLegalVectorType > LastLegalIntegerType && AllowVectors;
    unsigned LastLegalType =
        std::max(LastLegalVectorType, LastLegalIntegerType);

    // We add +1 here because the LastXXX variables refer to location while
    // the NumElem refers to array/index size.
    unsigned NumElem = std::min(NumConsecutiveStores, LastConsecutiveLoad + 1);
    NumElem = std::min(LastLegalType, NumElem);
    Align FirstLoadAlign = FirstLoad->getAlign();

    if (NumElem < 2) {
      // We know that candidate stores are in order and of correct
      // shape. While there is no mergeable sequence from the
      // beginning one may start later in the sequence. The only
      // reason a merge of size N could have failed where another of
      // the same size would not have is if the alignment or either
      // the load or store has improved. Drop as many candidates as we
      // can here.
      unsigned NumSkip = 1;
      while ((NumSkip < LoadNodes.size()) &&
             (LoadNodes[NumSkip].MemNode->getAlign() <= FirstLoadAlign) &&
             (StoreNodes[NumSkip].MemNode->getAlign() <= FirstStoreAlign))
        NumSkip++;
      StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumSkip);
      LoadNodes.erase(LoadNodes.begin(), LoadNodes.begin() + NumSkip);
      NumConsecutiveStores -= NumSkip;
      continue;
    }

    // Check that we can merge these candidates without causing a cycle.
    if (!checkMergeStoreCandidatesForDependencies(StoreNodes, NumElem,
                                                  RootNode)) {
      StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumElem);
      LoadNodes.erase(LoadNodes.begin(), LoadNodes.begin() + NumElem);
      NumConsecutiveStores -= NumElem;
      continue;
    }

    // Find if it is better to use vectors or integers to load and store
    // to memory.
    EVT JointMemOpVT;
    if (UseVectorTy) {
      // Find a legal type for the vector store.
      unsigned Elts = NumElem * NumMemElts;
      JointMemOpVT = EVT::getVectorVT(Context, MemVT.getScalarType(), Elts);
    } else {
      unsigned SizeInBits = NumElem * ElementSizeBytes * 8;
      JointMemOpVT = EVT::getIntegerVT(Context, SizeInBits);
    }

    SDLoc LoadDL(LoadNodes[0].MemNode);
    SDLoc StoreDL(StoreNodes[0].MemNode);

    // The merged loads are required to have the same incoming chain, so
    // using the first's chain is acceptable.

    SDValue NewStoreChain = getMergeStoreChains(StoreNodes, NumElem);
    AddToWorklist(NewStoreChain.getNode());

    MachineMemOperand::Flags LdMMOFlags =
        isDereferenceable ? MachineMemOperand::MODereferenceable
                          : MachineMemOperand::MONone;
    if (IsNonTemporalLoad)
      LdMMOFlags |= MachineMemOperand::MONonTemporal;

    MachineMemOperand::Flags StMMOFlags = IsNonTemporalStore
                                              ? MachineMemOperand::MONonTemporal
                                              : MachineMemOperand::MONone;

    SDValue NewLoad, NewStore;
    if (UseVectorTy || !DoIntegerTruncate) {
      NewLoad = DAG.getLoad(
          JointMemOpVT, LoadDL, FirstLoad->getChain(), FirstLoad->getBasePtr(),
          FirstLoad->getPointerInfo(), FirstLoadAlign, LdMMOFlags);
      SDValue StoreOp = NewLoad;
      if (NeedRotate) {
        unsigned LoadWidth = ElementSizeBytes * 8 * 2;
        assert(JointMemOpVT == EVT::getIntegerVT(Context, LoadWidth) &&
               "Unexpected type for rotate-able load pair");
        SDValue RotAmt =
            DAG.getShiftAmountConstant(LoadWidth / 2, JointMemOpVT, LoadDL);
        // Target can convert to the identical ROTR if it does not have ROTL.
        StoreOp = DAG.getNode(ISD::ROTL, LoadDL, JointMemOpVT, NewLoad, RotAmt);
      }
      NewStore = DAG.getStore(
          NewStoreChain, StoreDL, StoreOp, FirstInChain->getBasePtr(),
          FirstInChain->getPointerInfo(), FirstStoreAlign, StMMOFlags);
    } else { // This must be the truncstore/extload case
      EVT ExtendedTy =
          TLI.getTypeToTransformTo(*DAG.getContext(), JointMemOpVT);
      NewLoad = DAG.getExtLoad(ISD::EXTLOAD, LoadDL, ExtendedTy,
                               FirstLoad->getChain(), FirstLoad->getBasePtr(),
                               FirstLoad->getPointerInfo(), JointMemOpVT,
                               FirstLoadAlign, LdMMOFlags);
      NewStore = DAG.getTruncStore(
          NewStoreChain, StoreDL, NewLoad, FirstInChain->getBasePtr(),
          FirstInChain->getPointerInfo(), JointMemOpVT,
          FirstInChain->getAlign(), FirstInChain->getMemOperand()->getFlags());
    }

    // Transfer chain users from old loads to the new load.
    for (unsigned i = 0; i < NumElem; ++i) {
      LoadSDNode *Ld = cast<LoadSDNode>(LoadNodes[i].MemNode);
      DAG.ReplaceAllUsesOfValueWith(SDValue(Ld, 1),
                                    SDValue(NewLoad.getNode(), 1));
    }

    // Replace all stores with the new store. Recursively remove corresponding
    // values if they are no longer used.
    for (unsigned i = 0; i < NumElem; ++i) {
      SDValue Val = StoreNodes[i].MemNode->getOperand(1);
      CombineTo(StoreNodes[i].MemNode, NewStore);
      if (Val.getNode()->use_empty())
        recursivelyDeleteUnusedNodes(Val.getNode());
    }

    MadeChange = true;
    StoreNodes.erase(StoreNodes.begin(), StoreNodes.begin() + NumElem);
    LoadNodes.erase(LoadNodes.begin(), LoadNodes.begin() + NumElem);
    NumConsecutiveStores -= NumElem;
  }
  return MadeChange;
}

bool DAGCombiner::mergeConsecutiveStores(StoreSDNode *St) {
  if (OptLevel == CodeGenOpt::None || !EnableStoreMerging)
    return false;

  // TODO: Extend this function to merge stores of scalable vectors.
  // (i.e. two <vscale x 8 x i8> stores can be merged to one <vscale x 16 x i8>
  // store since we know <vscale x 16 x i8> is exactly twice as large as
  // <vscale x 8 x i8>). Until then, bail out for scalable vectors.
  EVT MemVT = St->getMemoryVT();
  if (MemVT.isScalableVector())
    return false;
  if (!MemVT.isSimple() || MemVT.getSizeInBits() * 2 > MaximumLegalStoreInBits)
    return false;

  // This function cannot currently deal with non-byte-sized memory sizes.
  int64_t ElementSizeBytes = MemVT.getStoreSize();
  if (ElementSizeBytes * 8 != (int64_t)MemVT.getSizeInBits())
    return false;

  // Do not bother looking at stored values that are not constants, loads, or
  // extracted vector elements.
  SDValue StoredVal = peekThroughBitcasts(St->getValue());
  const StoreSource StoreSrc = getStoreSource(StoredVal);
  if (StoreSrc == StoreSource::Unknown)
    return false;

  SmallVector<MemOpLink, 8> StoreNodes;
  SDNode *RootNode;
  // Find potential store merge candidates by searching through chain sub-DAG
  getStoreMergeCandidates(St, StoreNodes, RootNode);

  // Check if there is anything to merge.
  if (StoreNodes.size() < 2)
    return false;

  // Sort the memory operands according to their distance from the
  // base pointer.
  llvm::sort(StoreNodes, [](MemOpLink LHS, MemOpLink RHS) {
    return LHS.OffsetFromBase < RHS.OffsetFromBase;
  });

  bool AllowVectors = !DAG.getMachineFunction().getFunction().hasFnAttribute(
      Attribute::NoImplicitFloat);
  bool IsNonTemporalStore = St->isNonTemporal();
  bool IsNonTemporalLoad = StoreSrc == StoreSource::Load &&
                           cast<LoadSDNode>(StoredVal)->isNonTemporal();

  // Store Merge attempts to merge the lowest stores. This generally
  // works out as if successful, as the remaining stores are checked
  // after the first collection of stores is merged. However, in the
  // case that a non-mergeable store is found first, e.g., {p[-2],
  // p[0], p[1], p[2], p[3]}, we would fail and miss the subsequent
  // mergeable cases. To prevent this, we prune such stores from the
  // front of StoreNodes here.
  bool MadeChange = false;
  while (StoreNodes.size() > 1) {
    unsigned NumConsecutiveStores =
        getConsecutiveStores(StoreNodes, ElementSizeBytes);
    // There are no more stores in the list to examine.
    if (NumConsecutiveStores == 0)
      return MadeChange;

    // We have at least 2 consecutive stores. Try to merge them.
    assert(NumConsecutiveStores >= 2 && "Expected at least 2 stores");
    switch (StoreSrc) {
    case StoreSource::Constant:
      MadeChange |= tryStoreMergeOfConstants(StoreNodes, NumConsecutiveStores,
                                             MemVT, RootNode, AllowVectors);
      break;

    case StoreSource::Extract:
      MadeChange |= tryStoreMergeOfExtracts(StoreNodes, NumConsecutiveStores,
                                            MemVT, RootNode);
      break;

    case StoreSource::Load:
      MadeChange |= tryStoreMergeOfLoads(StoreNodes, NumConsecutiveStores,
                                         MemVT, RootNode, AllowVectors,
                                         IsNonTemporalStore, IsNonTemporalLoad);
      break;

    default:
      llvm_unreachable("Unhandled store source type");
    }
  }
  return MadeChange;
}

SDValue DAGCombiner::replaceStoreChain(StoreSDNode *ST, SDValue BetterChain) {
  SDLoc SL(ST);
  SDValue ReplStore;

  // Replace the chain to avoid dependency.
  if (ST->isTruncatingStore()) {
    ReplStore = DAG.getTruncStore(BetterChain, SL, ST->getValue(),
                                  ST->getBasePtr(), ST->getMemoryVT(),
                                  ST->getMemOperand());
  } else {
    ReplStore = DAG.getStore(BetterChain, SL, ST->getValue(), ST->getBasePtr(),
                             ST->getMemOperand());
  }

  // Create token to keep both nodes around.
  SDValue Token = DAG.getNode(ISD::TokenFactor, SL,
                              MVT::Other, ST->getChain(), ReplStore);

  // Make sure the new and old chains are cleaned up.
  AddToWorklist(Token.getNode());

  // Don't add users to work list.
  return CombineTo(ST, Token, false);
}

SDValue DAGCombiner::replaceStoreOfFPConstant(StoreSDNode *ST) {
  SDValue Value = ST->getValue();
  if (Value.getOpcode() == ISD::TargetConstantFP)
    return SDValue();

  if (!ISD::isNormalStore(ST))
    return SDValue();

  SDLoc DL(ST);

  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();

  const ConstantFPSDNode *CFP = cast<ConstantFPSDNode>(Value);

  // NOTE: If the original store is volatile, this transform must not increase
  // the number of stores.  For example, on x86-32 an f64 can be stored in one
  // processor operation but an i64 (which is not legal) requires two.  So the
  // transform should not be done in this case.

  SDValue Tmp;
  switch (CFP->getSimpleValueType(0).SimpleTy) {
  default:
    llvm_unreachable("Unknown FP type");
  case MVT::f16:    // We don't do this for these yet.
  case MVT::f80:
  case MVT::f128:
  case MVT::ppcf128:
    return SDValue();
  case MVT::f32:
    if ((isTypeLegal(MVT::i32) && !LegalOperations && ST->isSimple()) ||
        TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
      ;
      Tmp = DAG.getConstant((uint32_t)CFP->getValueAPF().
                            bitcastToAPInt().getZExtValue(), SDLoc(CFP),
                            MVT::i32);
      return DAG.getStore(Chain, DL, Tmp, Ptr, ST->getMemOperand());
    }

    return SDValue();
  case MVT::f64:
    if ((TLI.isTypeLegal(MVT::i64) && !LegalOperations &&
         ST->isSimple()) ||
        TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i64)) {
      ;
      Tmp = DAG.getConstant(CFP->getValueAPF().bitcastToAPInt().
                            getZExtValue(), SDLoc(CFP), MVT::i64);
      return DAG.getStore(Chain, DL, Tmp,
                          Ptr, ST->getMemOperand());
    }

    if (ST->isSimple() &&
        TLI.isOperationLegalOrCustom(ISD::STORE, MVT::i32)) {
      // Many FP stores are not made apparent until after legalize, e.g. for
      // argument passing.  Since this is so common, custom legalize the
      // 64-bit integer store into two 32-bit stores.
      uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
      SDValue Lo = DAG.getConstant(Val & 0xFFFFFFFF, SDLoc(CFP), MVT::i32);
      SDValue Hi = DAG.getConstant(Val >> 32, SDLoc(CFP), MVT::i32);
      if (DAG.getDataLayout().isBigEndian())
        std::swap(Lo, Hi);

      MachineMemOperand::Flags MMOFlags = ST->getMemOperand()->getFlags();
      AAMDNodes AAInfo = ST->getAAInfo();

      SDValue St0 = DAG.getStore(Chain, DL, Lo, Ptr, ST->getPointerInfo(),
                                 ST->getOriginalAlign(), MMOFlags, AAInfo);
      Ptr = DAG.getMemBasePlusOffset(Ptr, TypeSize::Fixed(4), DL);
      SDValue St1 = DAG.getStore(Chain, DL, Hi, Ptr,
                                 ST->getPointerInfo().getWithOffset(4),
                                 ST->getOriginalAlign(), MMOFlags, AAInfo);
      return DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                         St0, St1);
    }

    return SDValue();
  }
}

SDValue DAGCombiner::visitSTORE(SDNode *N) {
  StoreSDNode *ST  = cast<StoreSDNode>(N);
  SDValue Chain = ST->getChain();
  SDValue Value = ST->getValue();
  SDValue Ptr   = ST->getBasePtr();

  // If this is a store of a bit convert, store the input value if the
  // resultant store does not need a higher alignment than the original.
  if (Value.getOpcode() == ISD::BITCAST && !ST->isTruncatingStore() &&
      ST->isUnindexed()) {
    EVT SVT = Value.getOperand(0).getValueType();
    // If the store is volatile, we only want to change the store type if the
    // resulting store is legal. Otherwise we might increase the number of
    // memory accesses. We don't care if the original type was legal or not
    // as we assume software couldn't rely on the number of accesses of an
    // illegal type.
    // TODO: May be able to relax for unordered atomics (see D66309)
    if (((!LegalOperations && ST->isSimple()) ||
         TLI.isOperationLegal(ISD::STORE, SVT)) &&
        TLI.isStoreBitCastBeneficial(Value.getValueType(), SVT,
                                     DAG, *ST->getMemOperand())) {
      return DAG.getStore(Chain, SDLoc(N), Value.getOperand(0), Ptr,
                          ST->getMemOperand());
    }
  }

  // Turn 'store undef, Ptr' -> nothing.
  if (Value.isUndef() && ST->isUnindexed())
    return Chain;

  // Try to infer better alignment information than the store already has.
  if (OptLevel != CodeGenOpt::None && ST->isUnindexed() && !ST->isAtomic()) {
    if (MaybeAlign Alignment = DAG.InferPtrAlign(Ptr)) {
      if (*Alignment > ST->getAlign() &&
          isAligned(*Alignment, ST->getSrcValueOffset())) {
        SDValue NewStore =
            DAG.getTruncStore(Chain, SDLoc(N), Value, Ptr, ST->getPointerInfo(),
                              ST->getMemoryVT(), *Alignment,
                              ST->getMemOperand()->getFlags(), ST->getAAInfo());
        // NewStore will always be N as we are only refining the alignment
        assert(NewStore.getNode() == N);
        (void)NewStore;
      }
    }
  }

  // Try transforming a pair floating point load / store ops to integer
  // load / store ops.
  if (SDValue NewST = TransformFPLoadStorePair(N))
    return NewST;

  // Try transforming several stores into STORE (BSWAP).
  if (SDValue Store = mergeTruncStores(ST))
    return Store;

  if (ST->isUnindexed()) {
    // Walk up chain skipping non-aliasing memory nodes, on this store and any
    // adjacent stores.
    if (findBetterNeighborChains(ST)) {
      // replaceStoreChain uses CombineTo, which handled all of the worklist
      // manipulation. Return the original node to not do anything else.
      return SDValue(ST, 0);
    }
    Chain = ST->getChain();
  }

  // FIXME: is there such a thing as a truncating indexed store?
  if (ST->isTruncatingStore() && ST->isUnindexed() &&
      Value.getValueType().isInteger() &&
      (!isa<ConstantSDNode>(Value) ||
       !cast<ConstantSDNode>(Value)->isOpaque())) {
    APInt TruncDemandedBits =
        APInt::getLowBitsSet(Value.getScalarValueSizeInBits(),
                             ST->getMemoryVT().getScalarSizeInBits());

    // See if we can simplify the input to this truncstore with knowledge that
    // only the low bits are being used.  For example:
    // "truncstore (or (shl x, 8), y), i8"  -> "truncstore y, i8"
    AddToWorklist(Value.getNode());
    if (SDValue Shorter = DAG.GetDemandedBits(Value, TruncDemandedBits))
      return DAG.getTruncStore(Chain, SDLoc(N), Shorter, Ptr, ST->getMemoryVT(),
                               ST->getMemOperand());

    // Otherwise, see if we can simplify the operation with
    // SimplifyDemandedBits, which only works if the value has a single use.
    if (SimplifyDemandedBits(Value, TruncDemandedBits)) {
      // Re-visit the store if anything changed and the store hasn't been merged
      // with another node (N is deleted) SimplifyDemandedBits will add Value's
      // node back to the worklist if necessary, but we also need to re-visit
      // the Store node itself.
      if (N->getOpcode() != ISD::DELETED_NODE)
        AddToWorklist(N);
      return SDValue(N, 0);
    }
  }

  // If this is a load followed by a store to the same location, then the store
  // is dead/noop.
  // TODO: Can relax for unordered atomics (see D66309)
  if (LoadSDNode *Ld = dyn_cast<LoadSDNode>(Value)) {
    if (Ld->getBasePtr() == Ptr && ST->getMemoryVT() == Ld->getMemoryVT() &&
        ST->isUnindexed() && ST->isSimple() &&
        Ld->getAddressSpace() == ST->getAddressSpace() &&
        // There can't be any side effects between the load and store, such as
        // a call or store.
        Chain.reachesChainWithoutSideEffects(SDValue(Ld, 1))) {
      // The store is dead, remove it.
      return Chain;
    }
  }

  // TODO: Can relax for unordered atomics (see D66309)
  if (StoreSDNode *ST1 = dyn_cast<StoreSDNode>(Chain)) {
    if (ST->isUnindexed() && ST->isSimple() &&
        ST1->isUnindexed() && ST1->isSimple()) {
      if (ST1->getBasePtr() == Ptr && ST1->getValue() == Value &&
          ST->getMemoryVT() == ST1->getMemoryVT() &&
          ST->getAddressSpace() == ST1->getAddressSpace()) {
        // If this is a store followed by a store with the same value to the
        // same location, then the store is dead/noop.
        return Chain;
      }

      if (OptLevel != CodeGenOpt::None && ST1->hasOneUse() &&
          !ST1->getBasePtr().isUndef() &&
          // BaseIndexOffset and the code below requires knowing the size
          // of a vector, so bail out if MemoryVT is scalable.
          !ST->getMemoryVT().isScalableVector() &&
          !ST1->getMemoryVT().isScalableVector() &&
          ST->getAddressSpace() == ST1->getAddressSpace()) {
        const BaseIndexOffset STBase = BaseIndexOffset::match(ST, DAG);
        const BaseIndexOffset ChainBase = BaseIndexOffset::match(ST1, DAG);
        unsigned STBitSize = ST->getMemoryVT().getFixedSizeInBits();
        unsigned ChainBitSize = ST1->getMemoryVT().getFixedSizeInBits();
        // If this is a store who's preceding store to a subset of the current
        // location and no one other node is chained to that store we can
        // effectively drop the store. Do not remove stores to undef as they may
        // be used as data sinks.
        if (STBase.contains(DAG, STBitSize, ChainBase, ChainBitSize)) {
          CombineTo(ST1, ST1->getChain());
          return SDValue();
        }
      }
    }
  }

  // If this is an FP_ROUND or TRUNC followed by a store, fold this into a
  // truncating store.  We can do this even if this is already a truncstore.
  if ((Value.getOpcode() == ISD::FP_ROUND ||
       Value.getOpcode() == ISD::TRUNCATE) &&
      Value.getNode()->hasOneUse() && ST->isUnindexed() &&
      TLI.canCombineTruncStore(Value.getOperand(0).getValueType(),
                               ST->getMemoryVT(), LegalOperations)) {
    return DAG.getTruncStore(Chain, SDLoc(N), Value.getOperand(0),
                             Ptr, ST->getMemoryVT(), ST->getMemOperand());
  }

  // Always perform this optimization before types are legal. If the target
  // prefers, also try this after legalization to catch stores that were created
  // by intrinsics or other nodes.
  if (!LegalTypes || (TLI.mergeStoresAfterLegalization(ST->getMemoryVT()))) {
    while (true) {
      // There can be multiple store sequences on the same chain.
      // Keep trying to merge store sequences until we are unable to do so
      // or until we merge the last store on the chain.
      bool Changed = mergeConsecutiveStores(ST);
      if (!Changed) break;
      // Return N as merge only uses CombineTo and no worklist clean
      // up is necessary.
      if (N->getOpcode() == ISD::DELETED_NODE || !isa<StoreSDNode>(N))
        return SDValue(N, 0);
    }
  }

  // Try transforming N to an indexed store.
  if (CombineToPreIndexedLoadStore(N) || CombineToPostIndexedLoadStore(N))
    return SDValue(N, 0);

  // Turn 'store float 1.0, Ptr' -> 'store int 0x12345678, Ptr'
  //
  // Make sure to do this only after attempting to merge stores in order to
  //  avoid changing the types of some subset of stores due to visit order,
  //  preventing their merging.
  if (isa<ConstantFPSDNode>(ST->getValue())) {
    if (SDValue NewSt = replaceStoreOfFPConstant(ST))
      return NewSt;
  }

  if (SDValue NewSt = splitMergedValStore(ST))
    return NewSt;

  return ReduceLoadOpStoreWidth(N);
}

SDValue DAGCombiner::visitLIFETIME_END(SDNode *N) {
  const auto *LifetimeEnd = cast<LifetimeSDNode>(N);
  if (!LifetimeEnd->hasOffset())
    return SDValue();

  const BaseIndexOffset LifetimeEndBase(N->getOperand(1), SDValue(),
                                        LifetimeEnd->getOffset(), false);

  // We walk up the chains to find stores.
  SmallVector<SDValue, 8> Chains = {N->getOperand(0)};
  while (!Chains.empty()) {
    SDValue Chain = Chains.pop_back_val();
    if (!Chain.hasOneUse())
      continue;
    switch (Chain.getOpcode()) {
    case ISD::TokenFactor:
      for (unsigned Nops = Chain.getNumOperands(); Nops;)
        Chains.push_back(Chain.getOperand(--Nops));
      break;
    case ISD::LIFETIME_START:
    case ISD::LIFETIME_END:
      // We can forward past any lifetime start/end that can be proven not to
      // alias the node.
      if (!isAlias(Chain.getNode(), N))
        Chains.push_back(Chain.getOperand(0));
      break;
    case ISD::STORE: {
      StoreSDNode *ST = dyn_cast<StoreSDNode>(Chain);
      // TODO: Can relax for unordered atomics (see D66309)
      if (!ST->isSimple() || ST->isIndexed())
        continue;
      const TypeSize StoreSize = ST->getMemoryVT().getStoreSize();
      // The bounds of a scalable store are not known until runtime, so this
      // store cannot be elided.
      if (StoreSize.isScalable())
        continue;
      const BaseIndexOffset StoreBase = BaseIndexOffset::match(ST, DAG);
      // If we store purely within object bounds just before its lifetime ends,
      // we can remove the store.
      if (LifetimeEndBase.contains(DAG, LifetimeEnd->getSize() * 8, StoreBase,
                                   StoreSize.getFixedSize() * 8)) {
        LLVM_DEBUG(dbgs() << "\nRemoving store:"; StoreBase.dump();
                   dbgs() << "\nwithin LIFETIME_END of : ";
                   LifetimeEndBase.dump(); dbgs() << "\n");
        CombineTo(ST, ST->getChain());
        return SDValue(N, 0);
      }
    }
    }
  }
  return SDValue();
}

/// For the instruction sequence of store below, F and I values
/// are bundled together as an i64 value before being stored into memory.
/// Sometimes it is more efficent to generate separate stores for F and I,
/// which can remove the bitwise instructions or sink them to colder places.
///
///   (store (or (zext (bitcast F to i32) to i64),
///              (shl (zext I to i64), 32)), addr)  -->
///   (store F, addr) and (store I, addr+4)
///
/// Similarly, splitting for other merged store can also be beneficial, like:
/// For pair of {i32, i32}, i64 store --> two i32 stores.
/// For pair of {i32, i16}, i64 store --> two i32 stores.
/// For pair of {i16, i16}, i32 store --> two i16 stores.
/// For pair of {i16, i8},  i32 store --> two i16 stores.
/// For pair of {i8, i8},   i16 store --> two i8 stores.
///
/// We allow each target to determine specifically which kind of splitting is
/// supported.
///
/// The store patterns are commonly seen from the simple code snippet below
/// if only std::make_pair(...) is sroa transformed before inlined into hoo.
///   void goo(const std::pair<int, float> &);
///   hoo() {
///     ...
///     goo(std::make_pair(tmp, ftmp));
///     ...
///   }
///
SDValue DAGCombiner::splitMergedValStore(StoreSDNode *ST) {
  if (OptLevel == CodeGenOpt::None)
    return SDValue();

  // Can't change the number of memory accesses for a volatile store or break
  // atomicity for an atomic one.
  if (!ST->isSimple())
    return SDValue();

  SDValue Val = ST->getValue();
  SDLoc DL(ST);

  // Match OR operand.
  if (!Val.getValueType().isScalarInteger() || Val.getOpcode() != ISD::OR)
    return SDValue();

  // Match SHL operand and get Lower and Higher parts of Val.
  SDValue Op1 = Val.getOperand(0);
  SDValue Op2 = Val.getOperand(1);
  SDValue Lo, Hi;
  if (Op1.getOpcode() != ISD::SHL) {
    std::swap(Op1, Op2);
    if (Op1.getOpcode() != ISD::SHL)
      return SDValue();
  }
  Lo = Op2;
  Hi = Op1.getOperand(0);
  if (!Op1.hasOneUse())
    return SDValue();

  // Match shift amount to HalfValBitSize.
  unsigned HalfValBitSize = Val.getValueSizeInBits() / 2;
  ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op1.getOperand(1));
  if (!ShAmt || ShAmt->getAPIntValue() != HalfValBitSize)
    return SDValue();

  // Lo and Hi are zero-extended from int with size less equal than 32
  // to i64.
  if (Lo.getOpcode() != ISD::ZERO_EXTEND || !Lo.hasOneUse() ||
      !Lo.getOperand(0).getValueType().isScalarInteger() ||
      Lo.getOperand(0).getValueSizeInBits() > HalfValBitSize ||
      Hi.getOpcode() != ISD::ZERO_EXTEND || !Hi.hasOneUse() ||
      !Hi.getOperand(0).getValueType().isScalarInteger() ||
      Hi.getOperand(0).getValueSizeInBits() > HalfValBitSize)
    return SDValue();

  // Use the EVT of low and high parts before bitcast as the input
  // of target query.
  EVT LowTy = (Lo.getOperand(0).getOpcode() == ISD::BITCAST)
                  ? Lo.getOperand(0).getValueType()
                  : Lo.getValueType();
  EVT HighTy = (Hi.getOperand(0).getOpcode() == ISD::BITCAST)
                   ? Hi.getOperand(0).getValueType()
                   : Hi.getValueType();
  if (!TLI.isMultiStoresCheaperThanBitsMerge(LowTy, HighTy))
    return SDValue();

  // Start to split store.
  MachineMemOperand::Flags MMOFlags = ST->getMemOperand()->getFlags();
  AAMDNodes AAInfo = ST->getAAInfo();

  // Change the sizes of Lo and Hi's value types to HalfValBitSize.
  EVT VT = EVT::getIntegerVT(*DAG.getContext(), HalfValBitSize);
  Lo = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Lo.getOperand(0));
  Hi = DAG.getNode(ISD::ZERO_EXTEND, DL, VT, Hi.getOperand(0));

  SDValue Chain = ST->getChain();
  SDValue Ptr = ST->getBasePtr();
  // Lower value store.
  SDValue St0 = DAG.getStore(Chain, DL, Lo, Ptr, ST->getPointerInfo(),
                             ST->getOriginalAlign(), MMOFlags, AAInfo);
  Ptr = DAG.getMemBasePlusOffset(Ptr, TypeSize::Fixed(HalfValBitSize / 8), DL);
  // Higher value store.
  SDValue St1 = DAG.getStore(
      St0, DL, Hi, Ptr, ST->getPointerInfo().getWithOffset(HalfValBitSize / 8),
      ST->getOriginalAlign(), MMOFlags, AAInfo);
  return St1;
}

/// Convert a disguised subvector insertion into a shuffle:
SDValue DAGCombiner::combineInsertEltToShuffle(SDNode *N, unsigned InsIndex) {
  assert(N->getOpcode() == ISD::INSERT_VECTOR_ELT &&
         "Expected extract_vector_elt");
  SDValue InsertVal = N->getOperand(1);
  SDValue Vec = N->getOperand(0);

  // (insert_vector_elt (vector_shuffle X, Y), (extract_vector_elt X, N),
  // InsIndex)
  //   --> (vector_shuffle X, Y) and variations where shuffle operands may be
  //   CONCAT_VECTORS.
  if (Vec.getOpcode() == ISD::VECTOR_SHUFFLE && Vec.hasOneUse() &&
      InsertVal.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      isa<ConstantSDNode>(InsertVal.getOperand(1))) {
    ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Vec.getNode());
    ArrayRef<int> Mask = SVN->getMask();

    SDValue X = Vec.getOperand(0);
    SDValue Y = Vec.getOperand(1);

    // Vec's operand 0 is using indices from 0 to N-1 and
    // operand 1 from N to 2N - 1, where N is the number of
    // elements in the vectors.
    SDValue InsertVal0 = InsertVal.getOperand(0);
    int ElementOffset = -1;

    // We explore the inputs of the shuffle in order to see if we find the
    // source of the extract_vector_elt. If so, we can use it to modify the
    // shuffle rather than perform an insert_vector_elt.
    SmallVector<std::pair<int, SDValue>, 8> ArgWorkList;
    ArgWorkList.emplace_back(Mask.size(), Y);
    ArgWorkList.emplace_back(0, X);

    while (!ArgWorkList.empty()) {
      int ArgOffset;
      SDValue ArgVal;
      std::tie(ArgOffset, ArgVal) = ArgWorkList.pop_back_val();

      if (ArgVal == InsertVal0) {
        ElementOffset = ArgOffset;
        break;
      }

      // Peek through concat_vector.
      if (ArgVal.getOpcode() == ISD::CONCAT_VECTORS) {
        int CurrentArgOffset =
            ArgOffset + ArgVal.getValueType().getVectorNumElements();
        int Step = ArgVal.getOperand(0).getValueType().getVectorNumElements();
        for (SDValue Op : reverse(ArgVal->ops())) {
          CurrentArgOffset -= Step;
          ArgWorkList.emplace_back(CurrentArgOffset, Op);
        }

        // Make sure we went through all the elements and did not screw up index
        // computation.
        assert(CurrentArgOffset == ArgOffset);
      }
    }

    if (ElementOffset != -1) {
      SmallVector<int, 16> NewMask(Mask.begin(), Mask.end());

      auto *ExtrIndex = cast<ConstantSDNode>(InsertVal.getOperand(1));
      NewMask[InsIndex] = ElementOffset + ExtrIndex->getZExtValue();
      assert(NewMask[InsIndex] <
                 (int)(2 * Vec.getValueType().getVectorNumElements()) &&
             NewMask[InsIndex] >= 0 && "NewMask[InsIndex] is out of bound");

      SDValue LegalShuffle =
              TLI.buildLegalVectorShuffle(Vec.getValueType(), SDLoc(N), X,
                                          Y, NewMask, DAG);
      if (LegalShuffle)
        return LegalShuffle;
    }
  }

  // insert_vector_elt V, (bitcast X from vector type), IdxC -->
  // bitcast(shuffle (bitcast V), (extended X), Mask)
  // Note: We do not use an insert_subvector node because that requires a
  // legal subvector type.
  if (InsertVal.getOpcode() != ISD::BITCAST || !InsertVal.hasOneUse() ||
      !InsertVal.getOperand(0).getValueType().isVector())
    return SDValue();

  SDValue SubVec = InsertVal.getOperand(0);
  SDValue DestVec = N->getOperand(0);
  EVT SubVecVT = SubVec.getValueType();
  EVT VT = DestVec.getValueType();
  unsigned NumSrcElts = SubVecVT.getVectorNumElements();
  // If the source only has a single vector element, the cost of creating adding
  // it to a vector is likely to exceed the cost of a insert_vector_elt.
  if (NumSrcElts == 1)
    return SDValue();
  unsigned ExtendRatio = VT.getSizeInBits() / SubVecVT.getSizeInBits();
  unsigned NumMaskVals = ExtendRatio * NumSrcElts;

  // Step 1: Create a shuffle mask that implements this insert operation. The
  // vector that we are inserting into will be operand 0 of the shuffle, so
  // those elements are just 'i'. The inserted subvector is in the first
  // positions of operand 1 of the shuffle. Example:
  // insert v4i32 V, (v2i16 X), 2 --> shuffle v8i16 V', X', {0,1,2,3,8,9,6,7}
  SmallVector<int, 16> Mask(NumMaskVals);
  for (unsigned i = 0; i != NumMaskVals; ++i) {
    if (i / NumSrcElts == InsIndex)
      Mask[i] = (i % NumSrcElts) + NumMaskVals;
    else
      Mask[i] = i;
  }

  // Bail out if the target can not handle the shuffle we want to create.
  EVT SubVecEltVT = SubVecVT.getVectorElementType();
  EVT ShufVT = EVT::getVectorVT(*DAG.getContext(), SubVecEltVT, NumMaskVals);
  if (!TLI.isShuffleMaskLegal(Mask, ShufVT))
    return SDValue();

  // Step 2: Create a wide vector from the inserted source vector by appending
  // undefined elements. This is the same size as our destination vector.
  SDLoc DL(N);
  SmallVector<SDValue, 8> ConcatOps(ExtendRatio, DAG.getUNDEF(SubVecVT));
  ConcatOps[0] = SubVec;
  SDValue PaddedSubV = DAG.getNode(ISD::CONCAT_VECTORS, DL, ShufVT, ConcatOps);

  // Step 3: Shuffle in the padded subvector.
  SDValue DestVecBC = DAG.getBitcast(ShufVT, DestVec);
  SDValue Shuf = DAG.getVectorShuffle(ShufVT, DL, DestVecBC, PaddedSubV, Mask);
  AddToWorklist(PaddedSubV.getNode());
  AddToWorklist(DestVecBC.getNode());
  AddToWorklist(Shuf.getNode());
  return DAG.getBitcast(VT, Shuf);
}

SDValue DAGCombiner::visitINSERT_VECTOR_ELT(SDNode *N) {
  SDValue InVec = N->getOperand(0);
  SDValue InVal = N->getOperand(1);
  SDValue EltNo = N->getOperand(2);
  SDLoc DL(N);

  EVT VT = InVec.getValueType();
  auto *IndexC = dyn_cast<ConstantSDNode>(EltNo);

  // Insert into out-of-bounds element is undefined.
  if (IndexC && VT.isFixedLengthVector() &&
      IndexC->getZExtValue() >= VT.getVectorNumElements())
    return DAG.getUNDEF(VT);

  // Remove redundant insertions:
  // (insert_vector_elt x (extract_vector_elt x idx) idx) -> x
  if (InVal.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      InVec == InVal.getOperand(0) && EltNo == InVal.getOperand(1))
    return InVec;

  if (!IndexC) {
    // If this is variable insert to undef vector, it might be better to splat:
    // inselt undef, InVal, EltNo --> build_vector < InVal, InVal, ... >
    if (InVec.isUndef() && TLI.shouldSplatInsEltVarIndex(VT)) {
      if (VT.isScalableVector())
        return DAG.getSplatVector(VT, DL, InVal);
      else {
        SmallVector<SDValue, 8> Ops(VT.getVectorNumElements(), InVal);
        return DAG.getBuildVector(VT, DL, Ops);
      }
    }
    return SDValue();
  }

  if (VT.isScalableVector())
    return SDValue();

  unsigned NumElts = VT.getVectorNumElements();

  // We must know which element is being inserted for folds below here.
  unsigned Elt = IndexC->getZExtValue();
  if (SDValue Shuf = combineInsertEltToShuffle(N, Elt))
    return Shuf;

  // Canonicalize insert_vector_elt dag nodes.
  // Example:
  // (insert_vector_elt (insert_vector_elt A, Idx0), Idx1)
  // -> (insert_vector_elt (insert_vector_elt A, Idx1), Idx0)
  //
  // Do this only if the child insert_vector node has one use; also
  // do this only if indices are both constants and Idx1 < Idx0.
  if (InVec.getOpcode() == ISD::INSERT_VECTOR_ELT && InVec.hasOneUse()
      && isa<ConstantSDNode>(InVec.getOperand(2))) {
    unsigned OtherElt = InVec.getConstantOperandVal(2);
    if (Elt < OtherElt) {
      // Swap nodes.
      SDValue NewOp = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, VT,
                                  InVec.getOperand(0), InVal, EltNo);
      AddToWorklist(NewOp.getNode());
      return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(InVec.getNode()),
                         VT, NewOp, InVec.getOperand(1), InVec.getOperand(2));
    }
  }

  // If we can't generate a legal BUILD_VECTOR, exit
  if (LegalOperations && !TLI.isOperationLegal(ISD::BUILD_VECTOR, VT))
    return SDValue();

  // Check that the operand is a BUILD_VECTOR (or UNDEF, which can essentially
  // be converted to a BUILD_VECTOR).  Fill in the Ops vector with the
  // vector elements.
  SmallVector<SDValue, 8> Ops;
  // Do not combine these two vectors if the output vector will not replace
  // the input vector.
  if (InVec.getOpcode() == ISD::BUILD_VECTOR && InVec.hasOneUse()) {
    Ops.append(InVec.getNode()->op_begin(),
               InVec.getNode()->op_end());
  } else if (InVec.isUndef()) {
    Ops.append(NumElts, DAG.getUNDEF(InVal.getValueType()));
  } else {
    return SDValue();
  }
  assert(Ops.size() == NumElts && "Unexpected vector size");

  // Insert the element
  if (Elt < Ops.size()) {
    // All the operands of BUILD_VECTOR must have the same type;
    // we enforce that here.
    EVT OpVT = Ops[0].getValueType();
    Ops[Elt] = OpVT.isInteger() ? DAG.getAnyExtOrTrunc(InVal, DL, OpVT) : InVal;
  }

  // Return the new vector
  return DAG.getBuildVector(VT, DL, Ops);
}

SDValue DAGCombiner::scalarizeExtractedVectorLoad(SDNode *EVE, EVT InVecVT,
                                                  SDValue EltNo,
                                                  LoadSDNode *OriginalLoad) {
  assert(OriginalLoad->isSimple());

  EVT ResultVT = EVE->getValueType(0);
  EVT VecEltVT = InVecVT.getVectorElementType();

  // If the vector element type is not a multiple of a byte then we are unable
  // to correctly compute an address to load only the extracted element as a
  // scalar.
  if (!VecEltVT.isByteSized())
    return SDValue();

  Align Alignment = OriginalLoad->getAlign();
  Align NewAlign = DAG.getDataLayout().getABITypeAlign(
      VecEltVT.getTypeForEVT(*DAG.getContext()));

  if (NewAlign > Alignment ||
      !TLI.isOperationLegalOrCustom(ISD::LOAD, VecEltVT))
    return SDValue();

  ISD::LoadExtType ExtTy = ResultVT.bitsGT(VecEltVT) ?
    ISD::NON_EXTLOAD : ISD::EXTLOAD;
  if (!TLI.shouldReduceLoadWidth(OriginalLoad, ExtTy, VecEltVT))
    return SDValue();

  Alignment = NewAlign;

  MachinePointerInfo MPI;
  SDLoc DL(EVE);
  if (auto *ConstEltNo = dyn_cast<ConstantSDNode>(EltNo)) {
    int Elt = ConstEltNo->getZExtValue();
    unsigned PtrOff = VecEltVT.getSizeInBits() * Elt / 8;
    MPI = OriginalLoad->getPointerInfo().getWithOffset(PtrOff);
  } else {
    // Discard the pointer info except the address space because the memory
    // operand can't represent this new access since the offset is variable.
    MPI = MachinePointerInfo(OriginalLoad->getPointerInfo().getAddrSpace());
  }
  SDValue NewPtr = TLI.getVectorElementPointer(DAG, OriginalLoad->getBasePtr(),
                                               InVecVT, EltNo);

  // The replacement we need to do here is a little tricky: we need to
  // replace an extractelement of a load with a load.
  // Use ReplaceAllUsesOfValuesWith to do the replacement.
  // Note that this replacement assumes that the extractvalue is the only
  // use of the load; that's okay because we don't want to perform this
  // transformation in other cases anyway.
  SDValue Load;
  SDValue Chain;
  if (ResultVT.bitsGT(VecEltVT)) {
    // If the result type of vextract is wider than the load, then issue an
    // extending load instead.
    ISD::LoadExtType ExtType = TLI.isLoadExtLegal(ISD::ZEXTLOAD, ResultVT,
                                                  VecEltVT)
                                   ? ISD::ZEXTLOAD
                                   : ISD::EXTLOAD;
    Load = DAG.getExtLoad(ExtType, SDLoc(EVE), ResultVT,
                          OriginalLoad->getChain(), NewPtr, MPI, VecEltVT,
                          Alignment, OriginalLoad->getMemOperand()->getFlags(),
                          OriginalLoad->getAAInfo());
    Chain = Load.getValue(1);
  } else {
    Load = DAG.getLoad(
        VecEltVT, SDLoc(EVE), OriginalLoad->getChain(), NewPtr, MPI, Alignment,
        OriginalLoad->getMemOperand()->getFlags(), OriginalLoad->getAAInfo());
    Chain = Load.getValue(1);
    if (ResultVT.bitsLT(VecEltVT))
      Load = DAG.getNode(ISD::TRUNCATE, SDLoc(EVE), ResultVT, Load);
    else
      Load = DAG.getBitcast(ResultVT, Load);
  }
  WorklistRemover DeadNodes(*this);
  SDValue From[] = { SDValue(EVE, 0), SDValue(OriginalLoad, 1) };
  SDValue To[] = { Load, Chain };
  DAG.ReplaceAllUsesOfValuesWith(From, To, 2);
  // Make sure to revisit this node to clean it up; it will usually be dead.
  AddToWorklist(EVE);
  // Since we're explicitly calling ReplaceAllUses, add the new node to the
  // worklist explicitly as well.
  AddToWorklistWithUsers(Load.getNode());
  ++OpsNarrowed;
  return SDValue(EVE, 0);
}

/// Transform a vector binary operation into a scalar binary operation by moving
/// the math/logic after an extract element of a vector.
static SDValue scalarizeExtractedBinop(SDNode *ExtElt, SelectionDAG &DAG,
                                       bool LegalOperations) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Vec = ExtElt->getOperand(0);
  SDValue Index = ExtElt->getOperand(1);
  auto *IndexC = dyn_cast<ConstantSDNode>(Index);
  if (!IndexC || !TLI.isBinOp(Vec.getOpcode()) || !Vec.hasOneUse() ||
      Vec.getNode()->getNumValues() != 1)
    return SDValue();

  // Targets may want to avoid this to prevent an expensive register transfer.
  if (!TLI.shouldScalarizeBinop(Vec))
    return SDValue();

  // Extracting an element of a vector constant is constant-folded, so this
  // transform is just replacing a vector op with a scalar op while moving the
  // extract.
  SDValue Op0 = Vec.getOperand(0);
  SDValue Op1 = Vec.getOperand(1);
  if (isAnyConstantBuildVector(Op0, true) ||
      isAnyConstantBuildVector(Op1, true)) {
    // extractelt (binop X, C), IndexC --> binop (extractelt X, IndexC), C'
    // extractelt (binop C, X), IndexC --> binop C', (extractelt X, IndexC)
    SDLoc DL(ExtElt);
    EVT VT = ExtElt->getValueType(0);
    SDValue Ext0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, Op0, Index);
    SDValue Ext1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, VT, Op1, Index);
    return DAG.getNode(Vec.getOpcode(), DL, VT, Ext0, Ext1);
  }

  return SDValue();
}

SDValue DAGCombiner::visitEXTRACT_VECTOR_ELT(SDNode *N) {
  SDValue VecOp = N->getOperand(0);
  SDValue Index = N->getOperand(1);
  EVT ScalarVT = N->getValueType(0);
  EVT VecVT = VecOp.getValueType();
  if (VecOp.isUndef())
    return DAG.getUNDEF(ScalarVT);

  // extract_vector_elt (insert_vector_elt vec, val, idx), idx) -> val
  //
  // This only really matters if the index is non-constant since other combines
  // on the constant elements already work.
  SDLoc DL(N);
  if (VecOp.getOpcode() == ISD::INSERT_VECTOR_ELT &&
      Index == VecOp.getOperand(2)) {
    SDValue Elt = VecOp.getOperand(1);
    return VecVT.isInteger() ? DAG.getAnyExtOrTrunc(Elt, DL, ScalarVT) : Elt;
  }

  // (vextract (scalar_to_vector val, 0) -> val
  if (VecOp.getOpcode() == ISD::SCALAR_TO_VECTOR) {
    // Only 0'th element of SCALAR_TO_VECTOR is defined.
    if (DAG.isKnownNeverZero(Index))
      return DAG.getUNDEF(ScalarVT);

    // Check if the result type doesn't match the inserted element type. A
    // SCALAR_TO_VECTOR may truncate the inserted element and the
    // EXTRACT_VECTOR_ELT may widen the extracted vector.
    SDValue InOp = VecOp.getOperand(0);
    if (InOp.getValueType() != ScalarVT) {
      assert(InOp.getValueType().isInteger() && ScalarVT.isInteger());
      return DAG.getSExtOrTrunc(InOp, DL, ScalarVT);
    }
    return InOp;
  }

  // extract_vector_elt of out-of-bounds element -> UNDEF
  auto *IndexC = dyn_cast<ConstantSDNode>(Index);
  if (IndexC && VecVT.isFixedLengthVector() &&
      IndexC->getAPIntValue().uge(VecVT.getVectorNumElements()))
    return DAG.getUNDEF(ScalarVT);

  // extract_vector_elt (build_vector x, y), 1 -> y
  if (((IndexC && VecOp.getOpcode() == ISD::BUILD_VECTOR) ||
       VecOp.getOpcode() == ISD::SPLAT_VECTOR) &&
      TLI.isTypeLegal(VecVT) &&
      (VecOp.hasOneUse() || TLI.aggressivelyPreferBuildVectorSources(VecVT))) {
    assert((VecOp.getOpcode() != ISD::BUILD_VECTOR ||
            VecVT.isFixedLengthVector()) &&
           "BUILD_VECTOR used for scalable vectors");
    unsigned IndexVal =
        VecOp.getOpcode() == ISD::BUILD_VECTOR ? IndexC->getZExtValue() : 0;
    SDValue Elt = VecOp.getOperand(IndexVal);
    EVT InEltVT = Elt.getValueType();

    // Sometimes build_vector's scalar input types do not match result type.
    if (ScalarVT == InEltVT)
      return Elt;

    // TODO: It may be useful to truncate if free if the build_vector implicitly
    // converts.
  }

  if (VecVT.isScalableVector())
    return SDValue();

  // All the code from this point onwards assumes fixed width vectors, but it's
  // possible that some of the combinations could be made to work for scalable
  // vectors too.
  unsigned NumElts = VecVT.getVectorNumElements();
  unsigned VecEltBitWidth = VecVT.getScalarSizeInBits();

  // TODO: These transforms should not require the 'hasOneUse' restriction, but
  // there are regressions on multiple targets without it. We can end up with a
  // mess of scalar and vector code if we reduce only part of the DAG to scalar.
  if (IndexC && VecOp.getOpcode() == ISD::BITCAST && VecVT.isInteger() &&
      VecOp.hasOneUse()) {
    // The vector index of the LSBs of the source depend on the endian-ness.
    bool IsLE = DAG.getDataLayout().isLittleEndian();
    unsigned ExtractIndex = IndexC->getZExtValue();
    // extract_elt (v2i32 (bitcast i64:x)), BCTruncElt -> i32 (trunc i64:x)
    unsigned BCTruncElt = IsLE ? 0 : NumElts - 1;
    SDValue BCSrc = VecOp.getOperand(0);
    if (ExtractIndex == BCTruncElt && BCSrc.getValueType().isScalarInteger())
      return DAG.getNode(ISD::TRUNCATE, DL, ScalarVT, BCSrc);

    if (LegalTypes && BCSrc.getValueType().isInteger() &&
        BCSrc.getOpcode() == ISD::SCALAR_TO_VECTOR) {
      // ext_elt (bitcast (scalar_to_vec i64 X to v2i64) to v4i32), TruncElt -->
      // trunc i64 X to i32
      SDValue X = BCSrc.getOperand(0);
      assert(X.getValueType().isScalarInteger() && ScalarVT.isScalarInteger() &&
             "Extract element and scalar to vector can't change element type "
             "from FP to integer.");
      unsigned XBitWidth = X.getValueSizeInBits();
      BCTruncElt = IsLE ? 0 : XBitWidth / VecEltBitWidth - 1;

      // An extract element return value type can be wider than its vector
      // operand element type. In that case, the high bits are undefined, so
      // it's possible that we may need to extend rather than truncate.
      if (ExtractIndex == BCTruncElt && XBitWidth > VecEltBitWidth) {
        assert(XBitWidth % VecEltBitWidth == 0 &&
               "Scalar bitwidth must be a multiple of vector element bitwidth");
        return DAG.getAnyExtOrTrunc(X, DL, ScalarVT);
      }
    }
  }

  if (SDValue BO = scalarizeExtractedBinop(N, DAG, LegalOperations))
    return BO;

  // Transform: (EXTRACT_VECTOR_ELT( VECTOR_SHUFFLE )) -> EXTRACT_VECTOR_ELT.
  // We only perform this optimization before the op legalization phase because
  // we may introduce new vector instructions which are not backed by TD
  // patterns. For example on AVX, extracting elements from a wide vector
  // without using extract_subvector. However, if we can find an underlying
  // scalar value, then we can always use that.
  if (IndexC && VecOp.getOpcode() == ISD::VECTOR_SHUFFLE) {
    auto *Shuf = cast<ShuffleVectorSDNode>(VecOp);
    // Find the new index to extract from.
    int OrigElt = Shuf->getMaskElt(IndexC->getZExtValue());

    // Extracting an undef index is undef.
    if (OrigElt == -1)
      return DAG.getUNDEF(ScalarVT);

    // Select the right vector half to extract from.
    SDValue SVInVec;
    if (OrigElt < (int)NumElts) {
      SVInVec = VecOp.getOperand(0);
    } else {
      SVInVec = VecOp.getOperand(1);
      OrigElt -= NumElts;
    }

    if (SVInVec.getOpcode() == ISD::BUILD_VECTOR) {
      SDValue InOp = SVInVec.getOperand(OrigElt);
      if (InOp.getValueType() != ScalarVT) {
        assert(InOp.getValueType().isInteger() && ScalarVT.isInteger());
        InOp = DAG.getSExtOrTrunc(InOp, DL, ScalarVT);
      }

      return InOp;
    }

    // FIXME: We should handle recursing on other vector shuffles and
    // scalar_to_vector here as well.

    if (!LegalOperations ||
        // FIXME: Should really be just isOperationLegalOrCustom.
        TLI.isOperationLegal(ISD::EXTRACT_VECTOR_ELT, VecVT) ||
        TLI.isOperationExpand(ISD::VECTOR_SHUFFLE, VecVT)) {
      return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ScalarVT, SVInVec,
                         DAG.getVectorIdxConstant(OrigElt, DL));
    }
  }

  // If only EXTRACT_VECTOR_ELT nodes use the source vector we can
  // simplify it based on the (valid) extraction indices.
  if (llvm::all_of(VecOp->uses(), [&](SDNode *Use) {
        return Use->getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
               Use->getOperand(0) == VecOp &&
               isa<ConstantSDNode>(Use->getOperand(1));
      })) {
    APInt DemandedElts = APInt::getNullValue(NumElts);
    for (SDNode *Use : VecOp->uses()) {
      auto *CstElt = cast<ConstantSDNode>(Use->getOperand(1));
      if (CstElt->getAPIntValue().ult(NumElts))
        DemandedElts.setBit(CstElt->getZExtValue());
    }
    if (SimplifyDemandedVectorElts(VecOp, DemandedElts, true)) {
      // We simplified the vector operand of this extract element. If this
      // extract is not dead, visit it again so it is folded properly.
      if (N->getOpcode() != ISD::DELETED_NODE)
        AddToWorklist(N);
      return SDValue(N, 0);
    }
    APInt DemandedBits = APInt::getAllOnesValue(VecEltBitWidth);
    if (SimplifyDemandedBits(VecOp, DemandedBits, DemandedElts, true)) {
      // We simplified the vector operand of this extract element. If this
      // extract is not dead, visit it again so it is folded properly.
      if (N->getOpcode() != ISD::DELETED_NODE)
        AddToWorklist(N);
      return SDValue(N, 0);
    }
  }

  // Everything under here is trying to match an extract of a loaded value.
  // If the result of load has to be truncated, then it's not necessarily
  // profitable.
  bool BCNumEltsChanged = false;
  EVT ExtVT = VecVT.getVectorElementType();
  EVT LVT = ExtVT;
  if (ScalarVT.bitsLT(LVT) && !TLI.isTruncateFree(LVT, ScalarVT))
    return SDValue();

  if (VecOp.getOpcode() == ISD::BITCAST) {
    // Don't duplicate a load with other uses.
    if (!VecOp.hasOneUse())
      return SDValue();

    EVT BCVT = VecOp.getOperand(0).getValueType();
    if (!BCVT.isVector() || ExtVT.bitsGT(BCVT.getVectorElementType()))
      return SDValue();
    if (NumElts != BCVT.getVectorNumElements())
      BCNumEltsChanged = true;
    VecOp = VecOp.getOperand(0);
    ExtVT = BCVT.getVectorElementType();
  }

  // extract (vector load $addr), i --> load $addr + i * size
  if (!LegalOperations && !IndexC && VecOp.hasOneUse() &&
      ISD::isNormalLoad(VecOp.getNode()) &&
      !Index->hasPredecessor(VecOp.getNode())) {
    auto *VecLoad = dyn_cast<LoadSDNode>(VecOp);
    if (VecLoad && VecLoad->isSimple())
      return scalarizeExtractedVectorLoad(N, VecVT, Index, VecLoad);
  }

  // Perform only after legalization to ensure build_vector / vector_shuffle
  // optimizations have already been done.
  if (!LegalOperations || !IndexC)
    return SDValue();

  // (vextract (v4f32 load $addr), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 s2v (f32 load $addr)), c) -> (f32 load $addr+c*size)
  // (vextract (v4f32 shuffle (load $addr), <1,u,u,u>), 0) -> (f32 load $addr)
  int Elt = IndexC->getZExtValue();
  LoadSDNode *LN0 = nullptr;
  if (ISD::isNormalLoad(VecOp.getNode())) {
    LN0 = cast<LoadSDNode>(VecOp);
  } else if (VecOp.getOpcode() == ISD::SCALAR_TO_VECTOR &&
             VecOp.getOperand(0).getValueType() == ExtVT &&
             ISD::isNormalLoad(VecOp.getOperand(0).getNode())) {
    // Don't duplicate a load with other uses.
    if (!VecOp.hasOneUse())
      return SDValue();

    LN0 = cast<LoadSDNode>(VecOp.getOperand(0));
  }
  if (auto *Shuf = dyn_cast<ShuffleVectorSDNode>(VecOp)) {
    // (vextract (vector_shuffle (load $addr), v2, <1, u, u, u>), 1)
    // =>
    // (load $addr+1*size)

    // Don't duplicate a load with other uses.
    if (!VecOp.hasOneUse())
      return SDValue();

    // If the bit convert changed the number of elements, it is unsafe
    // to examine the mask.
    if (BCNumEltsChanged)
      return SDValue();

    // Select the input vector, guarding against out of range extract vector.
    int Idx = (Elt > (int)NumElts) ? -1 : Shuf->getMaskElt(Elt);
    VecOp = (Idx < (int)NumElts) ? VecOp.getOperand(0) : VecOp.getOperand(1);

    if (VecOp.getOpcode() == ISD::BITCAST) {
      // Don't duplicate a load with other uses.
      if (!VecOp.hasOneUse())
        return SDValue();

      VecOp = VecOp.getOperand(0);
    }
    if (ISD::isNormalLoad(VecOp.getNode())) {
      LN0 = cast<LoadSDNode>(VecOp);
      Elt = (Idx < (int)NumElts) ? Idx : Idx - (int)NumElts;
      Index = DAG.getConstant(Elt, DL, Index.getValueType());
    }
  } else if (VecOp.getOpcode() == ISD::CONCAT_VECTORS && !BCNumEltsChanged &&
             VecVT.getVectorElementType() == ScalarVT &&
             (!LegalTypes ||
              TLI.isTypeLegal(
                  VecOp.getOperand(0).getValueType().getVectorElementType()))) {
    // extract_vector_elt (concat_vectors v2i16:a, v2i16:b), 0
    //      -> extract_vector_elt a, 0
    // extract_vector_elt (concat_vectors v2i16:a, v2i16:b), 1
    //      -> extract_vector_elt a, 1
    // extract_vector_elt (concat_vectors v2i16:a, v2i16:b), 2
    //      -> extract_vector_elt b, 0
    // extract_vector_elt (concat_vectors v2i16:a, v2i16:b), 3
    //      -> extract_vector_elt b, 1
    SDLoc SL(N);
    EVT ConcatVT = VecOp.getOperand(0).getValueType();
    unsigned ConcatNumElts = ConcatVT.getVectorNumElements();
    SDValue NewIdx = DAG.getConstant(Elt % ConcatNumElts, SL,
                                     Index.getValueType());

    SDValue ConcatOp = VecOp.getOperand(Elt / ConcatNumElts);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL,
                              ConcatVT.getVectorElementType(),
                              ConcatOp, NewIdx);
    return DAG.getNode(ISD::BITCAST, SL, ScalarVT, Elt);
  }

  // Make sure we found a non-volatile load and the extractelement is
  // the only use.
  if (!LN0 || !LN0->hasNUsesOfValue(1,0) || !LN0->isSimple())
    return SDValue();

  // If Idx was -1 above, Elt is going to be -1, so just return undef.
  if (Elt == -1)
    return DAG.getUNDEF(LVT);

  return scalarizeExtractedVectorLoad(N, VecVT, Index, LN0);
}

// Simplify (build_vec (ext )) to (bitcast (build_vec ))
SDValue DAGCombiner::reduceBuildVecExtToExtBuildVec(SDNode *N) {
  // We perform this optimization post type-legalization because
  // the type-legalizer often scalarizes integer-promoted vectors.
  // Performing this optimization before may create bit-casts which
  // will be type-legalized to complex code sequences.
  // We perform this optimization only before the operation legalizer because we
  // may introduce illegal operations.
  if (Level != AfterLegalizeVectorOps && Level != AfterLegalizeTypes)
    return SDValue();

  unsigned NumInScalars = N->getNumOperands();
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // Check to see if this is a BUILD_VECTOR of a bunch of values
  // which come from any_extend or zero_extend nodes. If so, we can create
  // a new BUILD_VECTOR using bit-casts which may enable other BUILD_VECTOR
  // optimizations. We do not handle sign-extend because we can't fill the sign
  // using shuffles.
  EVT SourceType = MVT::Other;
  bool AllAnyExt = true;

  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = N->getOperand(i);
    // Ignore undef inputs.
    if (In.isUndef()) continue;

    bool AnyExt  = In.getOpcode() == ISD::ANY_EXTEND;
    bool ZeroExt = In.getOpcode() == ISD::ZERO_EXTEND;

    // Abort if the element is not an extension.
    if (!ZeroExt && !AnyExt) {
      SourceType = MVT::Other;
      break;
    }

    // The input is a ZeroExt or AnyExt. Check the original type.
    EVT InTy = In.getOperand(0).getValueType();

    // Check that all of the widened source types are the same.
    if (SourceType == MVT::Other)
      // First time.
      SourceType = InTy;
    else if (InTy != SourceType) {
      // Multiple income types. Abort.
      SourceType = MVT::Other;
      break;
    }

    // Check if all of the extends are ANY_EXTENDs.
    AllAnyExt &= AnyExt;
  }

  // In order to have valid types, all of the inputs must be extended from the
  // same source type and all of the inputs must be any or zero extend.
  // Scalar sizes must be a power of two.
  EVT OutScalarTy = VT.getScalarType();
  bool ValidTypes = SourceType != MVT::Other &&
                 isPowerOf2_32(OutScalarTy.getSizeInBits()) &&
                 isPowerOf2_32(SourceType.getSizeInBits());

  // Create a new simpler BUILD_VECTOR sequence which other optimizations can
  // turn into a single shuffle instruction.
  if (!ValidTypes)
    return SDValue();

  // If we already have a splat buildvector, then don't fold it if it means
  // introducing zeros.
  if (!AllAnyExt && DAG.isSplatValue(SDValue(N, 0), /*AllowUndefs*/ true))
    return SDValue();

  bool isLE = DAG.getDataLayout().isLittleEndian();
  unsigned ElemRatio = OutScalarTy.getSizeInBits()/SourceType.getSizeInBits();
  assert(ElemRatio > 1 && "Invalid element size ratio");
  SDValue Filler = AllAnyExt ? DAG.getUNDEF(SourceType):
                               DAG.getConstant(0, DL, SourceType);

  unsigned NewBVElems = ElemRatio * VT.getVectorNumElements();
  SmallVector<SDValue, 8> Ops(NewBVElems, Filler);

  // Populate the new build_vector
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue Cast = N->getOperand(i);
    assert((Cast.getOpcode() == ISD::ANY_EXTEND ||
            Cast.getOpcode() == ISD::ZERO_EXTEND ||
            Cast.isUndef()) && "Invalid cast opcode");
    SDValue In;
    if (Cast.isUndef())
      In = DAG.getUNDEF(SourceType);
    else
      In = Cast->getOperand(0);
    unsigned Index = isLE ? (i * ElemRatio) :
                            (i * ElemRatio + (ElemRatio - 1));

    assert(Index < Ops.size() && "Invalid index");
    Ops[Index] = In;
  }

  // The type of the new BUILD_VECTOR node.
  EVT VecVT = EVT::getVectorVT(*DAG.getContext(), SourceType, NewBVElems);
  assert(VecVT.getSizeInBits() == VT.getSizeInBits() &&
         "Invalid vector size");
  // Check if the new vector type is legal.
  if (!isTypeLegal(VecVT) ||
      (!TLI.isOperationLegal(ISD::BUILD_VECTOR, VecVT) &&
       TLI.isOperationLegal(ISD::BUILD_VECTOR, VT)))
    return SDValue();

  // Make the new BUILD_VECTOR.
  SDValue BV = DAG.getBuildVector(VecVT, DL, Ops);

  // The new BUILD_VECTOR node has the potential to be further optimized.
  AddToWorklist(BV.getNode());
  // Bitcast to the desired type.
  return DAG.getBitcast(VT, BV);
}

// Simplify (build_vec (trunc $1)
//                     (trunc (srl $1 half-width))
//                     (trunc (srl $1 (2 * half-width))) …)
// to (bitcast $1)
SDValue DAGCombiner::reduceBuildVecTruncToBitCast(SDNode *N) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR && "Expected build vector");

  // Only for little endian
  if (!DAG.getDataLayout().isLittleEndian())
    return SDValue();

  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  EVT OutScalarTy = VT.getScalarType();
  uint64_t ScalarTypeBitsize = OutScalarTy.getSizeInBits();

  // Only for power of two types to be sure that bitcast works well
  if (!isPowerOf2_64(ScalarTypeBitsize))
    return SDValue();

  unsigned NumInScalars = N->getNumOperands();

  // Look through bitcasts
  auto PeekThroughBitcast = [](SDValue Op) {
    if (Op.getOpcode() == ISD::BITCAST)
      return Op.getOperand(0);
    return Op;
  };

  // The source value where all the parts are extracted.
  SDValue Src;
  for (unsigned i = 0; i != NumInScalars; ++i) {
    SDValue In = PeekThroughBitcast(N->getOperand(i));
    // Ignore undef inputs.
    if (In.isUndef()) continue;

    if (In.getOpcode() != ISD::TRUNCATE)
      return SDValue();

    In = PeekThroughBitcast(In.getOperand(0));

    if (In.getOpcode() != ISD::SRL) {
      // For now only build_vec without shuffling, handle shifts here in the
      // future.
      if (i != 0)
        return SDValue();

      Src = In;
    } else {
      // In is SRL
      SDValue part = PeekThroughBitcast(In.getOperand(0));

      if (!Src) {
        Src = part;
      } else if (Src != part) {
        // Vector parts do not stem from the same variable
        return SDValue();
      }

      SDValue ShiftAmtVal = In.getOperand(1);
      if (!isa<ConstantSDNode>(ShiftAmtVal))
        return SDValue();

      uint64_t ShiftAmt = In.getNode()->getConstantOperandVal(1);

      // The extracted value is not extracted at the right position
      if (ShiftAmt != i * ScalarTypeBitsize)
        return SDValue();
    }
  }

  // Only cast if the size is the same
  if (Src.getValueType().getSizeInBits() != VT.getSizeInBits())
    return SDValue();

  return DAG.getBitcast(VT, Src);
}

SDValue DAGCombiner::createBuildVecShuffle(const SDLoc &DL, SDNode *N,
                                           ArrayRef<int> VectorMask,
                                           SDValue VecIn1, SDValue VecIn2,
                                           unsigned LeftIdx, bool DidSplitVec) {
  SDValue ZeroIdx = DAG.getVectorIdxConstant(0, DL);

  EVT VT = N->getValueType(0);
  EVT InVT1 = VecIn1.getValueType();
  EVT InVT2 = VecIn2.getNode() ? VecIn2.getValueType() : InVT1;

  unsigned NumElems = VT.getVectorNumElements();
  unsigned ShuffleNumElems = NumElems;

  // If we artificially split a vector in two already, then the offsets in the
  // operands will all be based off of VecIn1, even those in VecIn2.
  unsigned Vec2Offset = DidSplitVec ? 0 : InVT1.getVectorNumElements();

  uint64_t VTSize = VT.getFixedSizeInBits();
  uint64_t InVT1Size = InVT1.getFixedSizeInBits();
  uint64_t InVT2Size = InVT2.getFixedSizeInBits();

  assert(InVT2Size <= InVT1Size &&
         "Inputs must be sorted to be in non-increasing vector size order.");

  // We can't generate a shuffle node with mismatched input and output types.
  // Try to make the types match the type of the output.
  if (InVT1 != VT || InVT2 != VT) {
    if ((VTSize % InVT1Size == 0) && InVT1 == InVT2) {
      // If the output vector length is a multiple of both input lengths,
      // we can concatenate them and pad the rest with undefs.
      unsigned NumConcats = VTSize / InVT1Size;
      assert(NumConcats >= 2 && "Concat needs at least two inputs!");
      SmallVector<SDValue, 2> ConcatOps(NumConcats, DAG.getUNDEF(InVT1));
      ConcatOps[0] = VecIn1;
      ConcatOps[1] = VecIn2 ? VecIn2 : DAG.getUNDEF(InVT1);
      VecIn1 = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, ConcatOps);
      VecIn2 = SDValue();
    } else if (InVT1Size == VTSize * 2) {
      if (!TLI.isExtractSubvectorCheap(VT, InVT1, NumElems))
        return SDValue();

      if (!VecIn2.getNode()) {
        // If we only have one input vector, and it's twice the size of the
        // output, split it in two.
        VecIn2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, VecIn1,
                             DAG.getVectorIdxConstant(NumElems, DL));
        VecIn1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, VecIn1, ZeroIdx);
        // Since we now have shorter input vectors, adjust the offset of the
        // second vector's start.
        Vec2Offset = NumElems;
      } else {
        assert(InVT2Size <= InVT1Size &&
               "Second input is not going to be larger than the first one.");

        // VecIn1 is wider than the output, and we have another, possibly
        // smaller input. Pad the smaller input with undefs, shuffle at the
        // input vector width, and extract the output.
        // The shuffle type is different than VT, so check legality again.
        if (LegalOperations &&
            !TLI.isOperationLegal(ISD::VECTOR_SHUFFLE, InVT1))
          return SDValue();

        // Legalizing INSERT_SUBVECTOR is tricky - you basically have to
        // lower it back into a BUILD_VECTOR. So if the inserted type is
        // illegal, don't even try.
        if (InVT1 != InVT2) {
          if (!TLI.isTypeLegal(InVT2))
            return SDValue();
          VecIn2 = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, InVT1,
                               DAG.getUNDEF(InVT1), VecIn2, ZeroIdx);
        }
        ShuffleNumElems = NumElems * 2;
      }
    } else if (InVT2Size * 2 == VTSize && InVT1Size == VTSize) {
      SmallVector<SDValue, 2> ConcatOps(2, DAG.getUNDEF(InVT2));
      ConcatOps[0] = VecIn2;
      VecIn2 = DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, ConcatOps);
    } else {
      // TODO: Support cases where the length mismatch isn't exactly by a
      // factor of 2.
      // TODO: Move this check upwards, so that if we have bad type
      // mismatches, we don't create any DAG nodes.
      return SDValue();
    }
  }

  // Initialize mask to undef.
  SmallVector<int, 8> Mask(ShuffleNumElems, -1);

  // Only need to run up to the number of elements actually used, not the
  // total number of elements in the shuffle - if we are shuffling a wider
  // vector, the high lanes should be set to undef.
  for (unsigned i = 0; i != NumElems; ++i) {
    if (VectorMask[i] <= 0)
      continue;

    unsigned ExtIndex = N->getOperand(i).getConstantOperandVal(1);
    if (VectorMask[i] == (int)LeftIdx) {
      Mask[i] = ExtIndex;
    } else if (VectorMask[i] == (int)LeftIdx + 1) {
      Mask[i] = Vec2Offset + ExtIndex;
    }
  }

  // The type the input vectors may have changed above.
  InVT1 = VecIn1.getValueType();

  // If we already have a VecIn2, it should have the same type as VecIn1.
  // If we don't, get an undef/zero vector of the appropriate type.
  VecIn2 = VecIn2.getNode() ? VecIn2 : DAG.getUNDEF(InVT1);
  assert(InVT1 == VecIn2.getValueType() && "Unexpected second input type.");

  SDValue Shuffle = DAG.getVectorShuffle(InVT1, DL, VecIn1, VecIn2, Mask);
  if (ShuffleNumElems > NumElems)
    Shuffle = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, Shuffle, ZeroIdx);

  return Shuffle;
}

static SDValue reduceBuildVecToShuffleWithZero(SDNode *BV, SelectionDAG &DAG) {
  assert(BV->getOpcode() == ISD::BUILD_VECTOR && "Expected build vector");

  // First, determine where the build vector is not undef.
  // TODO: We could extend this to handle zero elements as well as undefs.
  int NumBVOps = BV->getNumOperands();
  int ZextElt = -1;
  for (int i = 0; i != NumBVOps; ++i) {
    SDValue Op = BV->getOperand(i);
    if (Op.isUndef())
      continue;
    if (ZextElt == -1)
      ZextElt = i;
    else
      return SDValue();
  }
  // Bail out if there's no non-undef element.
  if (ZextElt == -1)
    return SDValue();

  // The build vector contains some number of undef elements and exactly
  // one other element. That other element must be a zero-extended scalar
  // extracted from a vector at a constant index to turn this into a shuffle.
  // Also, require that the build vector does not implicitly truncate/extend
  // its elements.
  // TODO: This could be enhanced to allow ANY_EXTEND as well as ZERO_EXTEND.
  EVT VT = BV->getValueType(0);
  SDValue Zext = BV->getOperand(ZextElt);
  if (Zext.getOpcode() != ISD::ZERO_EXTEND || !Zext.hasOneUse() ||
      Zext.getOperand(0).getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
      !isa<ConstantSDNode>(Zext.getOperand(0).getOperand(1)) ||
      Zext.getValueSizeInBits() != VT.getScalarSizeInBits())
    return SDValue();

  // The zero-extend must be a multiple of the source size, and we must be
  // building a vector of the same size as the source of the extract element.
  SDValue Extract = Zext.getOperand(0);
  unsigned DestSize = Zext.getValueSizeInBits();
  unsigned SrcSize = Extract.getValueSizeInBits();
  if (DestSize % SrcSize != 0 ||
      Extract.getOperand(0).getValueSizeInBits() != VT.getSizeInBits())
    return SDValue();

  // Create a shuffle mask that will combine the extracted element with zeros
  // and undefs.
  int ZextRatio = DestSize / SrcSize;
  int NumMaskElts = NumBVOps * ZextRatio;
  SmallVector<int, 32> ShufMask(NumMaskElts, -1);
  for (int i = 0; i != NumMaskElts; ++i) {
    if (i / ZextRatio == ZextElt) {
      // The low bits of the (potentially translated) extracted element map to
      // the source vector. The high bits map to zero. We will use a zero vector
      // as the 2nd source operand of the shuffle, so use the 1st element of
      // that vector (mask value is number-of-elements) for the high bits.
      if (i % ZextRatio == 0)
        ShufMask[i] = Extract.getConstantOperandVal(1);
      else
        ShufMask[i] = NumMaskElts;
    }

    // Undef elements of the build vector remain undef because we initialize
    // the shuffle mask with -1.
  }

  // buildvec undef, ..., (zext (extractelt V, IndexC)), undef... -->
  // bitcast (shuffle V, ZeroVec, VectorMask)
  SDLoc DL(BV);
  EVT VecVT = Extract.getOperand(0).getValueType();
  SDValue ZeroVec = DAG.getConstant(0, DL, VecVT);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Shuf = TLI.buildLegalVectorShuffle(VecVT, DL, Extract.getOperand(0),
                                             ZeroVec, ShufMask, DAG);
  if (!Shuf)
    return SDValue();
  return DAG.getBitcast(VT, Shuf);
}

// FIXME: promote to STLExtras.
template <typename R, typename T>
static auto getFirstIndexOf(R &&Range, const T &Val) {
  auto I = find(Range, Val);
  if (I == Range.end())
    return static_cast<decltype(std::distance(Range.begin(), I))>(-1);
  return std::distance(Range.begin(), I);
}

// Check to see if this is a BUILD_VECTOR of a bunch of EXTRACT_VECTOR_ELT
// operations. If the types of the vectors we're extracting from allow it,
// turn this into a vector_shuffle node.
SDValue DAGCombiner::reduceBuildVecToShuffle(SDNode *N) {
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // Only type-legal BUILD_VECTOR nodes are converted to shuffle nodes.
  if (!isTypeLegal(VT))
    return SDValue();

  if (SDValue V = reduceBuildVecToShuffleWithZero(N, DAG))
    return V;

  // May only combine to shuffle after legalize if shuffle is legal.
  if (LegalOperations && !TLI.isOperationLegal(ISD::VECTOR_SHUFFLE, VT))
    return SDValue();

  bool UsesZeroVector = false;
  unsigned NumElems = N->getNumOperands();

  // Record, for each element of the newly built vector, which input vector
  // that element comes from. -1 stands for undef, 0 for the zero vector,
  // and positive values for the input vectors.
  // VectorMask maps each element to its vector number, and VecIn maps vector
  // numbers to their initial SDValues.

  SmallVector<int, 8> VectorMask(NumElems, -1);
  SmallVector<SDValue, 8> VecIn;
  VecIn.push_back(SDValue());

  for (unsigned i = 0; i != NumElems; ++i) {
    SDValue Op = N->getOperand(i);

    if (Op.isUndef())
      continue;

    // See if we can use a blend with a zero vector.
    // TODO: Should we generalize this to a blend with an arbitrary constant
    // vector?
    if (isNullConstant(Op) || isNullFPConstant(Op)) {
      UsesZeroVector = true;
      VectorMask[i] = 0;
      continue;
    }

    // Not an undef or zero. If the input is something other than an
    // EXTRACT_VECTOR_ELT with an in-range constant index, bail out.
    if (Op.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        !isa<ConstantSDNode>(Op.getOperand(1)))
      return SDValue();
    SDValue ExtractedFromVec = Op.getOperand(0);

    if (ExtractedFromVec.getValueType().isScalableVector())
      return SDValue();

    const APInt &ExtractIdx = Op.getConstantOperandAPInt(1);
    if (ExtractIdx.uge(ExtractedFromVec.getValueType().getVectorNumElements()))
      return SDValue();

    // All inputs must have the same element type as the output.
    if (VT.getVectorElementType() !=
        ExtractedFromVec.getValueType().getVectorElementType())
      return SDValue();

    // Have we seen this input vector before?
    // The vectors are expected to be tiny (usually 1 or 2 elements), so using
    // a map back from SDValues to numbers isn't worth it.
    int Idx = getFirstIndexOf(VecIn, ExtractedFromVec);
    if (Idx == -1) { // A new source vector?
      Idx = VecIn.size();
      VecIn.push_back(ExtractedFromVec);
    }

    VectorMask[i] = Idx;
  }

  // If we didn't find at least one input vector, bail out.
  if (VecIn.size() < 2)
    return SDValue();

  // If all the Operands of BUILD_VECTOR extract from same
  // vector, then split the vector efficiently based on the maximum
  // vector access index and adjust the VectorMask and
  // VecIn accordingly.
  bool DidSplitVec = false;
  if (VecIn.size() == 2) {
    unsigned MaxIndex = 0;
    unsigned NearestPow2 = 0;
    SDValue Vec = VecIn.back();
    EVT InVT = Vec.getValueType();
    SmallVector<unsigned, 8> IndexVec(NumElems, 0);

    for (unsigned i = 0; i < NumElems; i++) {
      if (VectorMask[i] <= 0)
        continue;
      unsigned Index = N->getOperand(i).getConstantOperandVal(1);
      IndexVec[i] = Index;
      MaxIndex = std::max(MaxIndex, Index);
    }

    NearestPow2 = PowerOf2Ceil(MaxIndex);
    if (InVT.isSimple() && NearestPow2 > 2 && MaxIndex < NearestPow2 &&
        NumElems * 2 < NearestPow2) {
      unsigned SplitSize = NearestPow2 / 2;
      EVT SplitVT = EVT::getVectorVT(*DAG.getContext(),
                                     InVT.getVectorElementType(), SplitSize);
      if (TLI.isTypeLegal(SplitVT) &&
          SplitSize + SplitVT.getVectorNumElements() <=
              InVT.getVectorNumElements()) {
        SDValue VecIn2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SplitVT, Vec,
                                     DAG.getVectorIdxConstant(SplitSize, DL));
        SDValue VecIn1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, SplitVT, Vec,
                                     DAG.getVectorIdxConstant(0, DL));
        VecIn.pop_back();
        VecIn.push_back(VecIn1);
        VecIn.push_back(VecIn2);
        DidSplitVec = true;

        for (unsigned i = 0; i < NumElems; i++) {
          if (VectorMask[i] <= 0)
            continue;
          VectorMask[i] = (IndexVec[i] < SplitSize) ? 1 : 2;
        }
      }
    }
  }

  // Sort input vectors by decreasing vector element count,
  // while preserving the relative order of equally-sized vectors.
  // Note that we keep the first "implicit zero vector as-is.
  SmallVector<SDValue, 8> SortedVecIn(VecIn);
  llvm::stable_sort(MutableArrayRef<SDValue>(SortedVecIn).drop_front(),
                    [](const SDValue &a, const SDValue &b) {
                      return a.getValueType().getVectorNumElements() >
                             b.getValueType().getVectorNumElements();
                    });

  // We now also need to rebuild the VectorMask, because it referenced element
  // order in VecIn, and we just sorted them.
  for (int &SourceVectorIndex : VectorMask) {
    if (SourceVectorIndex <= 0)
      continue;
    unsigned Idx = getFirstIndexOf(SortedVecIn, VecIn[SourceVectorIndex]);
    assert(Idx > 0 && Idx < SortedVecIn.size() &&
           VecIn[SourceVectorIndex] == SortedVecIn[Idx] && "Remapping failure");
    SourceVectorIndex = Idx;
  }

  VecIn = std::move(SortedVecIn);

  // TODO: Should this fire if some of the input vectors has illegal type (like
  // it does now), or should we let legalization run its course first?

  // Shuffle phase:
  // Take pairs of vectors, and shuffle them so that the result has elements
  // from these vectors in the correct places.
  // For example, given:
  // t10: i32 = extract_vector_elt t1, Constant:i64<0>
  // t11: i32 = extract_vector_elt t2, Constant:i64<0>
  // t12: i32 = extract_vector_elt t3, Constant:i64<0>
  // t13: i32 = extract_vector_elt t1, Constant:i64<1>
  // t14: v4i32 = BUILD_VECTOR t10, t11, t12, t13
  // We will generate:
  // t20: v4i32 = vector_shuffle<0,4,u,1> t1, t2
  // t21: v4i32 = vector_shuffle<u,u,0,u> t3, undef
  SmallVector<SDValue, 4> Shuffles;
  for (unsigned In = 0, Len = (VecIn.size() / 2); In < Len; ++In) {
    unsigned LeftIdx = 2 * In + 1;
    SDValue VecLeft = VecIn[LeftIdx];
    SDValue VecRight =
        (LeftIdx + 1) < VecIn.size() ? VecIn[LeftIdx + 1] : SDValue();

    if (SDValue Shuffle = createBuildVecShuffle(DL, N, VectorMask, VecLeft,
                                                VecRight, LeftIdx, DidSplitVec))
      Shuffles.push_back(Shuffle);
    else
      return SDValue();
  }

  // If we need the zero vector as an "ingredient" in the blend tree, add it
  // to the list of shuffles.
  if (UsesZeroVector)
    Shuffles.push_back(VT.isInteger() ? DAG.getConstant(0, DL, VT)
                                      : DAG.getConstantFP(0.0, DL, VT));

  // If we only have one shuffle, we're done.
  if (Shuffles.size() == 1)
    return Shuffles[0];

  // Update the vector mask to point to the post-shuffle vectors.
  for (int &Vec : VectorMask)
    if (Vec == 0)
      Vec = Shuffles.size() - 1;
    else
      Vec = (Vec - 1) / 2;

  // More than one shuffle. Generate a binary tree of blends, e.g. if from
  // the previous step we got the set of shuffles t10, t11, t12, t13, we will
  // generate:
  // t10: v8i32 = vector_shuffle<0,8,u,u,u,u,u,u> t1, t2
  // t11: v8i32 = vector_shuffle<u,u,0,8,u,u,u,u> t3, t4
  // t12: v8i32 = vector_shuffle<u,u,u,u,0,8,u,u> t5, t6
  // t13: v8i32 = vector_shuffle<u,u,u,u,u,u,0,8> t7, t8
  // t20: v8i32 = vector_shuffle<0,1,10,11,u,u,u,u> t10, t11
  // t21: v8i32 = vector_shuffle<u,u,u,u,4,5,14,15> t12, t13
  // t30: v8i32 = vector_shuffle<0,1,2,3,12,13,14,15> t20, t21

  // Make sure the initial size of the shuffle list is even.
  if (Shuffles.size() % 2)
    Shuffles.push_back(DAG.getUNDEF(VT));

  for (unsigned CurSize = Shuffles.size(); CurSize > 1; CurSize /= 2) {
    if (CurSize % 2) {
      Shuffles[CurSize] = DAG.getUNDEF(VT);
      CurSize++;
    }
    for (unsigned In = 0, Len = CurSize / 2; In < Len; ++In) {
      int Left = 2 * In;
      int Right = 2 * In + 1;
      SmallVector<int, 8> Mask(NumElems, -1);
      for (unsigned i = 0; i != NumElems; ++i) {
        if (VectorMask[i] == Left) {
          Mask[i] = i;
          VectorMask[i] = In;
        } else if (VectorMask[i] == Right) {
          Mask[i] = i + NumElems;
          VectorMask[i] = In;
        }
      }

      Shuffles[In] =
          DAG.getVectorShuffle(VT, DL, Shuffles[Left], Shuffles[Right], Mask);
    }
  }
  return Shuffles[0];
}

// Try to turn a build vector of zero extends of extract vector elts into a
// a vector zero extend and possibly an extract subvector.
// TODO: Support sign extend?
// TODO: Allow undef elements?
SDValue DAGCombiner::convertBuildVecZextToZext(SDNode *N) {
  if (LegalOperations)
    return SDValue();

  EVT VT = N->getValueType(0);

  bool FoundZeroExtend = false;
  SDValue Op0 = N->getOperand(0);
  auto checkElem = [&](SDValue Op) -> int64_t {
    unsigned Opc = Op.getOpcode();
    FoundZeroExtend |= (Opc == ISD::ZERO_EXTEND);
    if ((Opc == ISD::ZERO_EXTEND || Opc == ISD::ANY_EXTEND) &&
        Op.getOperand(0).getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
        Op0.getOperand(0).getOperand(0) == Op.getOperand(0).getOperand(0))
      if (auto *C = dyn_cast<ConstantSDNode>(Op.getOperand(0).getOperand(1)))
        return C->getZExtValue();
    return -1;
  };

  // Make sure the first element matches
  // (zext (extract_vector_elt X, C))
  int64_t Offset = checkElem(Op0);
  if (Offset < 0)
    return SDValue();

  unsigned NumElems = N->getNumOperands();
  SDValue In = Op0.getOperand(0).getOperand(0);
  EVT InSVT = In.getValueType().getScalarType();
  EVT InVT = EVT::getVectorVT(*DAG.getContext(), InSVT, NumElems);

  // Don't create an illegal input type after type legalization.
  if (LegalTypes && !TLI.isTypeLegal(InVT))
    return SDValue();

  // Ensure all the elements come from the same vector and are adjacent.
  for (unsigned i = 1; i != NumElems; ++i) {
    if ((Offset + i) != checkElem(N->getOperand(i)))
      return SDValue();
  }

  SDLoc DL(N);
  In = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, InVT, In,
                   Op0.getOperand(0).getOperand(1));
  return DAG.getNode(FoundZeroExtend ? ISD::ZERO_EXTEND : ISD::ANY_EXTEND, DL,
                     VT, In);
}

SDValue DAGCombiner::visitBUILD_VECTOR(SDNode *N) {
  EVT VT = N->getValueType(0);

  // A vector built entirely of undefs is undef.
  if (ISD::allOperandsUndef(N))
    return DAG.getUNDEF(VT);

  // If this is a splat of a bitcast from another vector, change to a
  // concat_vector.
  // For example:
  //   (build_vector (i64 (bitcast (v2i32 X))), (i64 (bitcast (v2i32 X)))) ->
  //     (v2i64 (bitcast (concat_vectors (v2i32 X), (v2i32 X))))
  //
  // If X is a build_vector itself, the concat can become a larger build_vector.
  // TODO: Maybe this is useful for non-splat too?
  if (!LegalOperations) {
    if (SDValue Splat = cast<BuildVectorSDNode>(N)->getSplatValue()) {
      Splat = peekThroughBitcasts(Splat);
      EVT SrcVT = Splat.getValueType();
      if (SrcVT.isVector()) {
        unsigned NumElts = N->getNumOperands() * SrcVT.getVectorNumElements();
        EVT NewVT = EVT::getVectorVT(*DAG.getContext(),
                                     SrcVT.getVectorElementType(), NumElts);
        if (!LegalTypes || TLI.isTypeLegal(NewVT)) {
          SmallVector<SDValue, 8> Ops(N->getNumOperands(), Splat);
          SDValue Concat = DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N),
                                       NewVT, Ops);
          return DAG.getBitcast(VT, Concat);
        }
      }
    }
  }

  // Check if we can express BUILD VECTOR via subvector extract.
  if (!LegalTypes && (N->getNumOperands() > 1)) {
    SDValue Op0 = N->getOperand(0);
    auto checkElem = [&](SDValue Op) -> uint64_t {
      if ((Op.getOpcode() == ISD::EXTRACT_VECTOR_ELT) &&
          (Op0.getOperand(0) == Op.getOperand(0)))
        if (auto CNode = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
          return CNode->getZExtValue();
      return -1;
    };

    int Offset = checkElem(Op0);
    for (unsigned i = 0; i < N->getNumOperands(); ++i) {
      if (Offset + i != checkElem(N->getOperand(i))) {
        Offset = -1;
        break;
      }
    }

    if ((Offset == 0) &&
        (Op0.getOperand(0).getValueType() == N->getValueType(0)))
      return Op0.getOperand(0);
    if ((Offset != -1) &&
        ((Offset % N->getValueType(0).getVectorNumElements()) ==
         0)) // IDX must be multiple of output size.
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SDLoc(N), N->getValueType(0),
                         Op0.getOperand(0), Op0.getOperand(1));
  }

  if (SDValue V = convertBuildVecZextToZext(N))
    return V;

  if (SDValue V = reduceBuildVecExtToExtBuildVec(N))
    return V;

  if (SDValue V = reduceBuildVecTruncToBitCast(N))
    return V;

  if (SDValue V = reduceBuildVecToShuffle(N))
    return V;

  // A splat of a single element is a SPLAT_VECTOR if supported on the target.
  // Do this late as some of the above may replace the splat.
  if (TLI.getOperationAction(ISD::SPLAT_VECTOR, VT) != TargetLowering::Expand)
    if (SDValue V = cast<BuildVectorSDNode>(N)->getSplatValue()) {
      assert(!V.isUndef() && "Splat of undef should have been handled earlier");
      return DAG.getNode(ISD::SPLAT_VECTOR, SDLoc(N), VT, V);
    }

  return SDValue();
}

static SDValue combineConcatVectorOfScalars(SDNode *N, SelectionDAG &DAG) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT OpVT = N->getOperand(0).getValueType();

  // If the operands are legal vectors, leave them alone.
  if (TLI.isTypeLegal(OpVT))
    return SDValue();

  SDLoc DL(N);
  EVT VT = N->getValueType(0);
  SmallVector<SDValue, 8> Ops;

  EVT SVT = EVT::getIntegerVT(*DAG.getContext(), OpVT.getSizeInBits());
  SDValue ScalarUndef = DAG.getNode(ISD::UNDEF, DL, SVT);

  // Keep track of what we encounter.
  bool AnyInteger = false;
  bool AnyFP = false;
  for (const SDValue &Op : N->ops()) {
    if (ISD::BITCAST == Op.getOpcode() &&
        !Op.getOperand(0).getValueType().isVector())
      Ops.push_back(Op.getOperand(0));
    else if (ISD::UNDEF == Op.getOpcode())
      Ops.push_back(ScalarUndef);
    else
      return SDValue();

    // Note whether we encounter an integer or floating point scalar.
    // If it's neither, bail out, it could be something weird like x86mmx.
    EVT LastOpVT = Ops.back().getValueType();
    if (LastOpVT.isFloatingPoint())
      AnyFP = true;
    else if (LastOpVT.isInteger())
      AnyInteger = true;
    else
      return SDValue();
  }

  // If any of the operands is a floating point scalar bitcast to a vector,
  // use floating point types throughout, and bitcast everything.
  // Replace UNDEFs by another scalar UNDEF node, of the final desired type.
  if (AnyFP) {
    SVT = EVT::getFloatingPointVT(OpVT.getSizeInBits());
    ScalarUndef = DAG.getNode(ISD::UNDEF, DL, SVT);
    if (AnyInteger) {
      for (SDValue &Op : Ops) {
        if (Op.getValueType() == SVT)
          continue;
        if (Op.isUndef())
          Op = ScalarUndef;
        else
          Op = DAG.getBitcast(SVT, Op);
      }
    }
  }

  EVT VecVT = EVT::getVectorVT(*DAG.getContext(), SVT,
                               VT.getSizeInBits() / SVT.getSizeInBits());
  return DAG.getBitcast(VT, DAG.getBuildVector(VecVT, DL, Ops));
}

// Check to see if this is a CONCAT_VECTORS of a bunch of EXTRACT_SUBVECTOR
// operations. If so, and if the EXTRACT_SUBVECTOR vector inputs come from at
// most two distinct vectors the same size as the result, attempt to turn this
// into a legal shuffle.
static SDValue combineConcatVectorOfExtracts(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  EVT OpVT = N->getOperand(0).getValueType();

  // We currently can't generate an appropriate shuffle for a scalable vector.
  if (VT.isScalableVector())
    return SDValue();

  int NumElts = VT.getVectorNumElements();
  int NumOpElts = OpVT.getVectorNumElements();

  SDValue SV0 = DAG.getUNDEF(VT), SV1 = DAG.getUNDEF(VT);
  SmallVector<int, 8> Mask;

  for (SDValue Op : N->ops()) {
    Op = peekThroughBitcasts(Op);

    // UNDEF nodes convert to UNDEF shuffle mask values.
    if (Op.isUndef()) {
      Mask.append((unsigned)NumOpElts, -1);
      continue;
    }

    if (Op.getOpcode() != ISD::EXTRACT_SUBVECTOR)
      return SDValue();

    // What vector are we extracting the subvector from and at what index?
    SDValue ExtVec = Op.getOperand(0);
    int ExtIdx = Op.getConstantOperandVal(1);

    // We want the EVT of the original extraction to correctly scale the
    // extraction index.
    EVT ExtVT = ExtVec.getValueType();
    ExtVec = peekThroughBitcasts(ExtVec);

    // UNDEF nodes convert to UNDEF shuffle mask values.
    if (ExtVec.isUndef()) {
      Mask.append((unsigned)NumOpElts, -1);
      continue;
    }

    // Ensure that we are extracting a subvector from a vector the same
    // size as the result.
    if (ExtVT.getSizeInBits() != VT.getSizeInBits())
      return SDValue();

    // Scale the subvector index to account for any bitcast.
    int NumExtElts = ExtVT.getVectorNumElements();
    if (0 == (NumExtElts % NumElts))
      ExtIdx /= (NumExtElts / NumElts);
    else if (0 == (NumElts % NumExtElts))
      ExtIdx *= (NumElts / NumExtElts);
    else
      return SDValue();

    // At most we can reference 2 inputs in the final shuffle.
    if (SV0.isUndef() || SV0 == ExtVec) {
      SV0 = ExtVec;
      for (int i = 0; i != NumOpElts; ++i)
        Mask.push_back(i + ExtIdx);
    } else if (SV1.isUndef() || SV1 == ExtVec) {
      SV1 = ExtVec;
      for (int i = 0; i != NumOpElts; ++i)
        Mask.push_back(i + ExtIdx + NumElts);
    } else {
      return SDValue();
    }
  }

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  return TLI.buildLegalVectorShuffle(VT, SDLoc(N), DAG.getBitcast(VT, SV0),
                                     DAG.getBitcast(VT, SV1), Mask, DAG);
}

static SDValue combineConcatVectorOfCasts(SDNode *N, SelectionDAG &DAG) {
  unsigned CastOpcode = N->getOperand(0).getOpcode();
  switch (CastOpcode) {
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    // TODO: Allow more opcodes?
    //  case ISD::BITCAST:
    //  case ISD::TRUNCATE:
    //  case ISD::ZERO_EXTEND:
    //  case ISD::SIGN_EXTEND:
    //  case ISD::FP_EXTEND:
    break;
  default:
    return SDValue();
  }

  EVT SrcVT = N->getOperand(0).getOperand(0).getValueType();
  if (!SrcVT.isVector())
    return SDValue();

  // All operands of the concat must be the same kind of cast from the same
  // source type.
  SmallVector<SDValue, 4> SrcOps;
  for (SDValue Op : N->ops()) {
    if (Op.getOpcode() != CastOpcode || !Op.hasOneUse() ||
        Op.getOperand(0).getValueType() != SrcVT)
      return SDValue();
    SrcOps.push_back(Op.getOperand(0));
  }

  // The wider cast must be supported by the target. This is unusual because
  // the operation support type parameter depends on the opcode. In addition,
  // check the other type in the cast to make sure this is really legal.
  EVT VT = N->getValueType(0);
  EVT SrcEltVT = SrcVT.getVectorElementType();
  ElementCount NumElts = SrcVT.getVectorElementCount() * N->getNumOperands();
  EVT ConcatSrcVT = EVT::getVectorVT(*DAG.getContext(), SrcEltVT, NumElts);
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  switch (CastOpcode) {
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    if (!TLI.isOperationLegalOrCustom(CastOpcode, ConcatSrcVT) ||
        !TLI.isTypeLegal(VT))
      return SDValue();
    break;
  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    if (!TLI.isOperationLegalOrCustom(CastOpcode, VT) ||
        !TLI.isTypeLegal(ConcatSrcVT))
      return SDValue();
    break;
  default:
    llvm_unreachable("Unexpected cast opcode");
  }

  // concat (cast X), (cast Y)... -> cast (concat X, Y...)
  SDLoc DL(N);
  SDValue NewConcat = DAG.getNode(ISD::CONCAT_VECTORS, DL, ConcatSrcVT, SrcOps);
  return DAG.getNode(CastOpcode, DL, VT, NewConcat);
}

SDValue DAGCombiner::visitCONCAT_VECTORS(SDNode *N) {
  // If we only have one input vector, we don't need to do any concatenation.
  if (N->getNumOperands() == 1)
    return N->getOperand(0);

  // Check if all of the operands are undefs.
  EVT VT = N->getValueType(0);
  if (ISD::allOperandsUndef(N))
    return DAG.getUNDEF(VT);

  // Optimize concat_vectors where all but the first of the vectors are undef.
  if (all_of(drop_begin(N->ops()),
             [](const SDValue &Op) { return Op.isUndef(); })) {
    SDValue In = N->getOperand(0);
    assert(In.getValueType().isVector() && "Must concat vectors");

    // If the input is a concat_vectors, just make a larger concat by padding
    // with smaller undefs.
    if (In.getOpcode() == ISD::CONCAT_VECTORS && In.hasOneUse()) {
      unsigned NumOps = N->getNumOperands() * In.getNumOperands();
      SmallVector<SDValue, 4> Ops(In->op_begin(), In->op_end());
      Ops.resize(NumOps, DAG.getUNDEF(Ops[0].getValueType()));
      return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Ops);
    }

    SDValue Scalar = peekThroughOneUseBitcasts(In);

    // concat_vectors(scalar_to_vector(scalar), undef) ->
    //     scalar_to_vector(scalar)
    if (!LegalOperations && Scalar.getOpcode() == ISD::SCALAR_TO_VECTOR &&
         Scalar.hasOneUse()) {
      EVT SVT = Scalar.getValueType().getVectorElementType();
      if (SVT == Scalar.getOperand(0).getValueType())
        Scalar = Scalar.getOperand(0);
    }

    // concat_vectors(scalar, undef) -> scalar_to_vector(scalar)
    if (!Scalar.getValueType().isVector()) {
      // If the bitcast type isn't legal, it might be a trunc of a legal type;
      // look through the trunc so we can still do the transform:
      //   concat_vectors(trunc(scalar), undef) -> scalar_to_vector(scalar)
      if (Scalar->getOpcode() == ISD::TRUNCATE &&
          !TLI.isTypeLegal(Scalar.getValueType()) &&
          TLI.isTypeLegal(Scalar->getOperand(0).getValueType()))
        Scalar = Scalar->getOperand(0);

      EVT SclTy = Scalar.getValueType();

      if (!SclTy.isFloatingPoint() && !SclTy.isInteger())
        return SDValue();

      // Bail out if the vector size is not a multiple of the scalar size.
      if (VT.getSizeInBits() % SclTy.getSizeInBits())
        return SDValue();

      unsigned VNTNumElms = VT.getSizeInBits() / SclTy.getSizeInBits();
      if (VNTNumElms < 2)
        return SDValue();

      EVT NVT = EVT::getVectorVT(*DAG.getContext(), SclTy, VNTNumElms);
      if (!TLI.isTypeLegal(NVT) || !TLI.isTypeLegal(Scalar.getValueType()))
        return SDValue();

      SDValue Res = DAG.getNode(ISD::SCALAR_TO_VECTOR, SDLoc(N), NVT, Scalar);
      return DAG.getBitcast(VT, Res);
    }
  }

  // Fold any combination of BUILD_VECTOR or UNDEF nodes into one BUILD_VECTOR.
  // We have already tested above for an UNDEF only concatenation.
  // fold (concat_vectors (BUILD_VECTOR A, B, ...), (BUILD_VECTOR C, D, ...))
  // -> (BUILD_VECTOR A, B, ..., C, D, ...)
  auto IsBuildVectorOrUndef = [](const SDValue &Op) {
    return ISD::UNDEF == Op.getOpcode() || ISD::BUILD_VECTOR == Op.getOpcode();
  };
  if (llvm::all_of(N->ops(), IsBuildVectorOrUndef)) {
    SmallVector<SDValue, 8> Opnds;
    EVT SVT = VT.getScalarType();

    EVT MinVT = SVT;
    if (!SVT.isFloatingPoint()) {
      // If BUILD_VECTOR are from built from integer, they may have different
      // operand types. Get the smallest type and truncate all operands to it.
      bool FoundMinVT = false;
      for (const SDValue &Op : N->ops())
        if (ISD::BUILD_VECTOR == Op.getOpcode()) {
          EVT OpSVT = Op.getOperand(0).getValueType();
          MinVT = (!FoundMinVT || OpSVT.bitsLE(MinVT)) ? OpSVT : MinVT;
          FoundMinVT = true;
        }
      assert(FoundMinVT && "Concat vector type mismatch");
    }

    for (const SDValue &Op : N->ops()) {
      EVT OpVT = Op.getValueType();
      unsigned NumElts = OpVT.getVectorNumElements();

      if (ISD::UNDEF == Op.getOpcode())
        Opnds.append(NumElts, DAG.getUNDEF(MinVT));

      if (ISD::BUILD_VECTOR == Op.getOpcode()) {
        if (SVT.isFloatingPoint()) {
          assert(SVT == OpVT.getScalarType() && "Concat vector type mismatch");
          Opnds.append(Op->op_begin(), Op->op_begin() + NumElts);
        } else {
          for (unsigned i = 0; i != NumElts; ++i)
            Opnds.push_back(
                DAG.getNode(ISD::TRUNCATE, SDLoc(N), MinVT, Op.getOperand(i)));
        }
      }
    }

    assert(VT.getVectorNumElements() == Opnds.size() &&
           "Concat vector type mismatch");
    return DAG.getBuildVector(VT, SDLoc(N), Opnds);
  }

  // Fold CONCAT_VECTORS of only bitcast scalars (or undef) to BUILD_VECTOR.
  if (SDValue V = combineConcatVectorOfScalars(N, DAG))
    return V;

  // Fold CONCAT_VECTORS of EXTRACT_SUBVECTOR (or undef) to VECTOR_SHUFFLE.
  if (Level < AfterLegalizeVectorOps && TLI.isTypeLegal(VT))
    if (SDValue V = combineConcatVectorOfExtracts(N, DAG))
      return V;

  if (SDValue V = combineConcatVectorOfCasts(N, DAG))
    return V;

  // Type legalization of vectors and DAG canonicalization of SHUFFLE_VECTOR
  // nodes often generate nop CONCAT_VECTOR nodes. Scan the CONCAT_VECTOR
  // operands and look for a CONCAT operations that place the incoming vectors
  // at the exact same location.
  //
  // For scalable vectors, EXTRACT_SUBVECTOR indexes are implicitly scaled.
  SDValue SingleSource = SDValue();
  unsigned PartNumElem =
      N->getOperand(0).getValueType().getVectorMinNumElements();

  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDValue Op = N->getOperand(i);

    if (Op.isUndef())
      continue;

    // Check if this is the identity extract:
    if (Op.getOpcode() != ISD::EXTRACT_SUBVECTOR)
      return SDValue();

    // Find the single incoming vector for the extract_subvector.
    if (SingleSource.getNode()) {
      if (Op.getOperand(0) != SingleSource)
        return SDValue();
    } else {
      SingleSource = Op.getOperand(0);

      // Check the source type is the same as the type of the result.
      // If not, this concat may extend the vector, so we can not
      // optimize it away.
      if (SingleSource.getValueType() != N->getValueType(0))
        return SDValue();
    }

    // Check that we are reading from the identity index.
    unsigned IdentityIndex = i * PartNumElem;
    if (Op.getConstantOperandAPInt(1) != IdentityIndex)
      return SDValue();
  }

  if (SingleSource.getNode())
    return SingleSource;

  return SDValue();
}

// Helper that peeks through INSERT_SUBVECTOR/CONCAT_VECTORS to find
// if the subvector can be sourced for free.
static SDValue getSubVectorSrc(SDValue V, SDValue Index, EVT SubVT) {
  if (V.getOpcode() == ISD::INSERT_SUBVECTOR &&
      V.getOperand(1).getValueType() == SubVT && V.getOperand(2) == Index) {
    return V.getOperand(1);
  }
  auto *IndexC = dyn_cast<ConstantSDNode>(Index);
  if (IndexC && V.getOpcode() == ISD::CONCAT_VECTORS &&
      V.getOperand(0).getValueType() == SubVT &&
      (IndexC->getZExtValue() % SubVT.getVectorMinNumElements()) == 0) {
    uint64_t SubIdx = IndexC->getZExtValue() / SubVT.getVectorMinNumElements();
    return V.getOperand(SubIdx);
  }
  return SDValue();
}

static SDValue narrowInsertExtractVectorBinOp(SDNode *Extract,
                                              SelectionDAG &DAG,
                                              bool LegalOperations) {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue BinOp = Extract->getOperand(0);
  unsigned BinOpcode = BinOp.getOpcode();
  if (!TLI.isBinOp(BinOpcode) || BinOp.getNode()->getNumValues() != 1)
    return SDValue();

  EVT VecVT = BinOp.getValueType();
  SDValue Bop0 = BinOp.getOperand(0), Bop1 = BinOp.getOperand(1);
  if (VecVT != Bop0.getValueType() || VecVT != Bop1.getValueType())
    return SDValue();

  SDValue Index = Extract->getOperand(1);
  EVT SubVT = Extract->getValueType(0);
  if (!TLI.isOperationLegalOrCustom(BinOpcode, SubVT, LegalOperations))
    return SDValue();

  SDValue Sub0 = getSubVectorSrc(Bop0, Index, SubVT);
  SDValue Sub1 = getSubVectorSrc(Bop1, Index, SubVT);

  // TODO: We could handle the case where only 1 operand is being inserted by
  //       creating an extract of the other operand, but that requires checking
  //       number of uses and/or costs.
  if (!Sub0 || !Sub1)
    return SDValue();

  // We are inserting both operands of the wide binop only to extract back
  // to the narrow vector size. Eliminate all of the insert/extract:
  // ext (binop (ins ?, X, Index), (ins ?, Y, Index)), Index --> binop X, Y
  return DAG.getNode(BinOpcode, SDLoc(Extract), SubVT, Sub0, Sub1,
                     BinOp->getFlags());
}

/// If we are extracting a subvector produced by a wide binary operator try
/// to use a narrow binary operator and/or avoid concatenation and extraction.
static SDValue narrowExtractedVectorBinOp(SDNode *Extract, SelectionDAG &DAG,
                                          bool LegalOperations) {
  // TODO: Refactor with the caller (visitEXTRACT_SUBVECTOR), so we can share
  // some of these bailouts with other transforms.

  if (SDValue V = narrowInsertExtractVectorBinOp(Extract, DAG, LegalOperations))
    return V;

  // The extract index must be a constant, so we can map it to a concat operand.
  auto *ExtractIndexC = dyn_cast<ConstantSDNode>(Extract->getOperand(1));
  if (!ExtractIndexC)
    return SDValue();

  // We are looking for an optionally bitcasted wide vector binary operator
  // feeding an extract subvector.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue BinOp = peekThroughBitcasts(Extract->getOperand(0));
  unsigned BOpcode = BinOp.getOpcode();
  if (!TLI.isBinOp(BOpcode) || BinOp.getNode()->getNumValues() != 1)
    return SDValue();

  // Exclude the fake form of fneg (fsub -0.0, x) because that is likely to be
  // reduced to the unary fneg when it is visited, and we probably want to deal
  // with fneg in a target-specific way.
  if (BOpcode == ISD::FSUB) {
    auto *C = isConstOrConstSplatFP(BinOp.getOperand(0), /*AllowUndefs*/ true);
    if (C && C->getValueAPF().isNegZero())
      return SDValue();
  }

  // The binop must be a vector type, so we can extract some fraction of it.
  EVT WideBVT = BinOp.getValueType();
  // The optimisations below currently assume we are dealing with fixed length
  // vectors. It is possible to add support for scalable vectors, but at the
  // moment we've done no analysis to prove whether they are profitable or not.
  if (!WideBVT.isFixedLengthVector())
    return SDValue();

  EVT VT = Extract->getValueType(0);
  unsigned ExtractIndex = ExtractIndexC->getZExtValue();
  assert(ExtractIndex % VT.getVectorNumElements() == 0 &&
         "Extract index is not a multiple of the vector length.");

  // Bail out if this is not a proper multiple width extraction.
  unsigned WideWidth = WideBVT.getSizeInBits();
  unsigned NarrowWidth = VT.getSizeInBits();
  if (WideWidth % NarrowWidth != 0)
    return SDValue();

  // Bail out if we are extracting a fraction of a single operation. This can
  // occur because we potentially looked through a bitcast of the binop.
  unsigned NarrowingRatio = WideWidth / NarrowWidth;
  unsigned WideNumElts = WideBVT.getVectorNumElements();
  if (WideNumElts % NarrowingRatio != 0)
    return SDValue();

  // Bail out if the target does not support a narrower version of the binop.
  EVT NarrowBVT = EVT::getVectorVT(*DAG.getContext(), WideBVT.getScalarType(),
                                   WideNumElts / NarrowingRatio);
  if (!TLI.isOperationLegalOrCustomOrPromote(BOpcode, NarrowBVT))
    return SDValue();

  // If extraction is cheap, we don't need to look at the binop operands
  // for concat ops. The narrow binop alone makes this transform profitable.
  // We can't just reuse the original extract index operand because we may have
  // bitcasted.
  unsigned ConcatOpNum = ExtractIndex / VT.getVectorNumElements();
  unsigned ExtBOIdx = ConcatOpNum * NarrowBVT.getVectorNumElements();
  if (TLI.isExtractSubvectorCheap(NarrowBVT, WideBVT, ExtBOIdx) &&
      BinOp.hasOneUse() && Extract->getOperand(0)->hasOneUse()) {
    // extract (binop B0, B1), N --> binop (extract B0, N), (extract B1, N)
    SDLoc DL(Extract);
    SDValue NewExtIndex = DAG.getVectorIdxConstant(ExtBOIdx, DL);
    SDValue X = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NarrowBVT,
                            BinOp.getOperand(0), NewExtIndex);
    SDValue Y = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NarrowBVT,
                            BinOp.getOperand(1), NewExtIndex);
    SDValue NarrowBinOp = DAG.getNode(BOpcode, DL, NarrowBVT, X, Y,
                                      BinOp.getNode()->getFlags());
    return DAG.getBitcast(VT, NarrowBinOp);
  }

  // Only handle the case where we are doubling and then halving. A larger ratio
  // may require more than two narrow binops to replace the wide binop.
  if (NarrowingRatio != 2)
    return SDValue();

  // TODO: The motivating case for this transform is an x86 AVX1 target. That
  // target has temptingly almost legal versions of bitwise logic ops in 256-bit
  // flavors, but no other 256-bit integer support. This could be extended to
  // handle any binop, but that may require fixing/adding other folds to avoid
  // codegen regressions.
  if (BOpcode != ISD::AND && BOpcode != ISD::OR && BOpcode != ISD::XOR)
    return SDValue();

  // We need at least one concatenation operation of a binop operand to make
  // this transform worthwhile. The concat must double the input vector sizes.
  auto GetSubVector = [ConcatOpNum](SDValue V) -> SDValue {
    if (V.getOpcode() == ISD::CONCAT_VECTORS && V.getNumOperands() == 2)
      return V.getOperand(ConcatOpNum);
    return SDValue();
  };
  SDValue SubVecL = GetSubVector(peekThroughBitcasts(BinOp.getOperand(0)));
  SDValue SubVecR = GetSubVector(peekThroughBitcasts(BinOp.getOperand(1)));

  if (SubVecL || SubVecR) {
    // If a binop operand was not the result of a concat, we must extract a
    // half-sized operand for our new narrow binop:
    // extract (binop (concat X1, X2), (concat Y1, Y2)), N --> binop XN, YN
    // extract (binop (concat X1, X2), Y), N --> binop XN, (extract Y, IndexC)
    // extract (binop X, (concat Y1, Y2)), N --> binop (extract X, IndexC), YN
    SDLoc DL(Extract);
    SDValue IndexC = DAG.getVectorIdxConstant(ExtBOIdx, DL);
    SDValue X = SubVecL ? DAG.getBitcast(NarrowBVT, SubVecL)
                        : DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NarrowBVT,
                                      BinOp.getOperand(0), IndexC);

    SDValue Y = SubVecR ? DAG.getBitcast(NarrowBVT, SubVecR)
                        : DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NarrowBVT,
                                      BinOp.getOperand(1), IndexC);

    SDValue NarrowBinOp = DAG.getNode(BOpcode, DL, NarrowBVT, X, Y);
    return DAG.getBitcast(VT, NarrowBinOp);
  }

  return SDValue();
}

/// If we are extracting a subvector from a wide vector load, convert to a
/// narrow load to eliminate the extraction:
/// (extract_subvector (load wide vector)) --> (load narrow vector)
static SDValue narrowExtractedVectorLoad(SDNode *Extract, SelectionDAG &DAG) {
  // TODO: Add support for big-endian. The offset calculation must be adjusted.
  if (DAG.getDataLayout().isBigEndian())
    return SDValue();

  auto *Ld = dyn_cast<LoadSDNode>(Extract->getOperand(0));
  auto *ExtIdx = dyn_cast<ConstantSDNode>(Extract->getOperand(1));
  if (!Ld || Ld->getExtensionType() || !Ld->isSimple() ||
      !ExtIdx)
    return SDValue();

  // Allow targets to opt-out.
  EVT VT = Extract->getValueType(0);

  // We can only create byte sized loads.
  if (!VT.isByteSized())
    return SDValue();

  unsigned Index = ExtIdx->getZExtValue();
  unsigned NumElts = VT.getVectorMinNumElements();

  // The definition of EXTRACT_SUBVECTOR states that the index must be a
  // multiple of the minimum number of elements in the result type.
  assert(Index % NumElts == 0 && "The extract subvector index is not a "
                                 "multiple of the result's element count");

  // It's fine to use TypeSize here as we know the offset will not be negative.
  TypeSize Offset = VT.getStoreSize() * (Index / NumElts);

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (!TLI.shouldReduceLoadWidth(Ld, Ld->getExtensionType(), VT))
    return SDValue();

  // The narrow load will be offset from the base address of the old load if
  // we are extracting from something besides index 0 (little-endian).
  SDLoc DL(Extract);

  // TODO: Use "BaseIndexOffset" to make this more effective.
  SDValue NewAddr = DAG.getMemBasePlusOffset(Ld->getBasePtr(), Offset, DL);

  uint64_t StoreSize = MemoryLocation::getSizeOrUnknown(VT.getStoreSize());
  MachineFunction &MF = DAG.getMachineFunction();
  MachineMemOperand *MMO;
  if (Offset.isScalable()) {
    MachinePointerInfo MPI =
        MachinePointerInfo(Ld->getPointerInfo().getAddrSpace());
    MMO = MF.getMachineMemOperand(Ld->getMemOperand(), MPI, StoreSize);
  } else
    MMO = MF.getMachineMemOperand(Ld->getMemOperand(), Offset.getFixedSize(),
                                  StoreSize);

  SDValue NewLd = DAG.getLoad(VT, DL, Ld->getChain(), NewAddr, MMO);
  DAG.makeEquivalentMemoryOrdering(Ld, NewLd);
  return NewLd;
}

SDValue DAGCombiner::visitEXTRACT_SUBVECTOR(SDNode *N) {
  EVT NVT = N->getValueType(0);
  SDValue V = N->getOperand(0);
  uint64_t ExtIdx = N->getConstantOperandVal(1);

  // Extract from UNDEF is UNDEF.
  if (V.isUndef())
    return DAG.getUNDEF(NVT);

  if (TLI.isOperationLegalOrCustomOrPromote(ISD::LOAD, NVT))
    if (SDValue NarrowLoad = narrowExtractedVectorLoad(N, DAG))
      return NarrowLoad;

  // Combine an extract of an extract into a single extract_subvector.
  // ext (ext X, C), 0 --> ext X, C
  if (ExtIdx == 0 && V.getOpcode() == ISD::EXTRACT_SUBVECTOR && V.hasOneUse()) {
    if (TLI.isExtractSubvectorCheap(NVT, V.getOperand(0).getValueType(),
                                    V.getConstantOperandVal(1)) &&
        TLI.isOperationLegalOrCustom(ISD::EXTRACT_SUBVECTOR, NVT)) {
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SDLoc(N), NVT, V.getOperand(0),
                         V.getOperand(1));
    }
  }

  // Try to move vector bitcast after extract_subv by scaling extraction index:
  // extract_subv (bitcast X), Index --> bitcast (extract_subv X, Index')
  if (V.getOpcode() == ISD::BITCAST &&
      V.getOperand(0).getValueType().isVector() &&
      (!LegalOperations || TLI.isOperationLegal(ISD::BITCAST, NVT))) {
    SDValue SrcOp = V.getOperand(0);
    EVT SrcVT = SrcOp.getValueType();
    unsigned SrcNumElts = SrcVT.getVectorMinNumElements();
    unsigned DestNumElts = V.getValueType().getVectorMinNumElements();
    if ((SrcNumElts % DestNumElts) == 0) {
      unsigned SrcDestRatio = SrcNumElts / DestNumElts;
      ElementCount NewExtEC = NVT.getVectorElementCount() * SrcDestRatio;
      EVT NewExtVT = EVT::getVectorVT(*DAG.getContext(), SrcVT.getScalarType(),
                                      NewExtEC);
      if (TLI.isOperationLegalOrCustom(ISD::EXTRACT_SUBVECTOR, NewExtVT)) {
        SDLoc DL(N);
        SDValue NewIndex = DAG.getVectorIdxConstant(ExtIdx * SrcDestRatio, DL);
        SDValue NewExtract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NewExtVT,
                                         V.getOperand(0), NewIndex);
        return DAG.getBitcast(NVT, NewExtract);
      }
    }
    if ((DestNumElts % SrcNumElts) == 0) {
      unsigned DestSrcRatio = DestNumElts / SrcNumElts;
      if (NVT.getVectorElementCount().isKnownMultipleOf(DestSrcRatio)) {
        ElementCount NewExtEC =
            NVT.getVectorElementCount().divideCoefficientBy(DestSrcRatio);
        EVT ScalarVT = SrcVT.getScalarType();
        if ((ExtIdx % DestSrcRatio) == 0) {
          SDLoc DL(N);
          unsigned IndexValScaled = ExtIdx / DestSrcRatio;
          EVT NewExtVT =
              EVT::getVectorVT(*DAG.getContext(), ScalarVT, NewExtEC);
          if (TLI.isOperationLegalOrCustom(ISD::EXTRACT_SUBVECTOR, NewExtVT)) {
            SDValue NewIndex = DAG.getVectorIdxConstant(IndexValScaled, DL);
            SDValue NewExtract =
                DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NewExtVT,
                            V.getOperand(0), NewIndex);
            return DAG.getBitcast(NVT, NewExtract);
          }
          if (NewExtEC.isScalar() &&
              TLI.isOperationLegalOrCustom(ISD::EXTRACT_VECTOR_ELT, ScalarVT)) {
            SDValue NewIndex = DAG.getVectorIdxConstant(IndexValScaled, DL);
            SDValue NewExtract =
                DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, ScalarVT,
                            V.getOperand(0), NewIndex);
            return DAG.getBitcast(NVT, NewExtract);
          }
        }
      }
    }
  }

  if (V.getOpcode() == ISD::CONCAT_VECTORS) {
    unsigned ExtNumElts = NVT.getVectorMinNumElements();
    EVT ConcatSrcVT = V.getOperand(0).getValueType();
    assert(ConcatSrcVT.getVectorElementType() == NVT.getVectorElementType() &&
           "Concat and extract subvector do not change element type");
    assert((ExtIdx % ExtNumElts) == 0 &&
           "Extract index is not a multiple of the input vector length.");

    unsigned ConcatSrcNumElts = ConcatSrcVT.getVectorMinNumElements();
    unsigned ConcatOpIdx = ExtIdx / ConcatSrcNumElts;

    // If the concatenated source types match this extract, it's a direct
    // simplification:
    // extract_subvec (concat V1, V2, ...), i --> Vi
    if (ConcatSrcNumElts == ExtNumElts)
      return V.getOperand(ConcatOpIdx);

    // If the concatenated source vectors are a multiple length of this extract,
    // then extract a fraction of one of those source vectors directly from a
    // concat operand. Example:
    //   v2i8 extract_subvec (v16i8 concat (v8i8 X), (v8i8 Y), 14 -->
    //   v2i8 extract_subvec v8i8 Y, 6
    if (NVT.isFixedLengthVector() && ConcatSrcNumElts % ExtNumElts == 0) {
      SDLoc DL(N);
      unsigned NewExtIdx = ExtIdx - ConcatOpIdx * ConcatSrcNumElts;
      assert(NewExtIdx + ExtNumElts <= ConcatSrcNumElts &&
             "Trying to extract from >1 concat operand?");
      assert(NewExtIdx % ExtNumElts == 0 &&
             "Extract index is not a multiple of the input vector length.");
      SDValue NewIndexC = DAG.getVectorIdxConstant(NewExtIdx, DL);
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, NVT,
                         V.getOperand(ConcatOpIdx), NewIndexC);
    }
  }

  V = peekThroughBitcasts(V);

  // If the input is a build vector. Try to make a smaller build vector.
  if (V.getOpcode() == ISD::BUILD_VECTOR) {
    EVT InVT = V.getValueType();
    unsigned ExtractSize = NVT.getSizeInBits();
    unsigned EltSize = InVT.getScalarSizeInBits();
    // Only do this if we won't split any elements.
    if (ExtractSize % EltSize == 0) {
      unsigned NumElems = ExtractSize / EltSize;
      EVT EltVT = InVT.getVectorElementType();
      EVT ExtractVT =
          NumElems == 1 ? EltVT
                        : EVT::getVectorVT(*DAG.getContext(), EltVT, NumElems);
      if ((Level < AfterLegalizeDAG ||
           (NumElems == 1 ||
            TLI.isOperationLegal(ISD::BUILD_VECTOR, ExtractVT))) &&
          (!LegalTypes || TLI.isTypeLegal(ExtractVT))) {
        unsigned IdxVal = (ExtIdx * NVT.getScalarSizeInBits()) / EltSize;

        if (NumElems == 1) {
          SDValue Src = V->getOperand(IdxVal);
          if (EltVT != Src.getValueType())
            Src = DAG.getNode(ISD::TRUNCATE, SDLoc(N), InVT, Src);
          return DAG.getBitcast(NVT, Src);
        }

        // Extract the pieces from the original build_vector.
        SDValue BuildVec = DAG.getBuildVector(ExtractVT, SDLoc(N),
                                              V->ops().slice(IdxVal, NumElems));
        return DAG.getBitcast(NVT, BuildVec);
      }
    }
  }

  if (V.getOpcode() == ISD::INSERT_SUBVECTOR) {
    // Handle only simple case where vector being inserted and vector
    // being extracted are of same size.
    EVT SmallVT = V.getOperand(1).getValueType();
    if (!NVT.bitsEq(SmallVT))
      return SDValue();

    // Combine:
    //    (extract_subvec (insert_subvec V1, V2, InsIdx), ExtIdx)
    // Into:
    //    indices are equal or bit offsets are equal => V1
    //    otherwise => (extract_subvec V1, ExtIdx)
    uint64_t InsIdx = V.getConstantOperandVal(2);
    if (InsIdx * SmallVT.getScalarSizeInBits() ==
        ExtIdx * NVT.getScalarSizeInBits())
      return DAG.getBitcast(NVT, V.getOperand(1));
    return DAG.getNode(
        ISD::EXTRACT_SUBVECTOR, SDLoc(N), NVT,
        DAG.getBitcast(N->getOperand(0).getValueType(), V.getOperand(0)),
        N->getOperand(1));
  }

  if (SDValue NarrowBOp = narrowExtractedVectorBinOp(N, DAG, LegalOperations))
    return NarrowBOp;

  if (SimplifyDemandedVectorElts(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

/// Try to convert a wide shuffle of concatenated vectors into 2 narrow shuffles
/// followed by concatenation. Narrow vector ops may have better performance
/// than wide ops, and this can unlock further narrowing of other vector ops.
/// Targets can invert this transform later if it is not profitable.
static SDValue foldShuffleOfConcatUndefs(ShuffleVectorSDNode *Shuf,
                                         SelectionDAG &DAG) {
  SDValue N0 = Shuf->getOperand(0), N1 = Shuf->getOperand(1);
  if (N0.getOpcode() != ISD::CONCAT_VECTORS || N0.getNumOperands() != 2 ||
      N1.getOpcode() != ISD::CONCAT_VECTORS || N1.getNumOperands() != 2 ||
      !N0.getOperand(1).isUndef() || !N1.getOperand(1).isUndef())
    return SDValue();

  // Split the wide shuffle mask into halves. Any mask element that is accessing
  // operand 1 is offset down to account for narrowing of the vectors.
  ArrayRef<int> Mask = Shuf->getMask();
  EVT VT = Shuf->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  unsigned HalfNumElts = NumElts / 2;
  SmallVector<int, 16> Mask0(HalfNumElts, -1);
  SmallVector<int, 16> Mask1(HalfNumElts, -1);
  for (unsigned i = 0; i != NumElts; ++i) {
    if (Mask[i] == -1)
      continue;
    // If we reference the upper (undef) subvector then the element is undef.
    if ((Mask[i] % NumElts) >= HalfNumElts)
      continue;
    int M = Mask[i] < (int)NumElts ? Mask[i] : Mask[i] - (int)HalfNumElts;
    if (i < HalfNumElts)
      Mask0[i] = M;
    else
      Mask1[i - HalfNumElts] = M;
  }

  // Ask the target if this is a valid transform.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  EVT HalfVT = EVT::getVectorVT(*DAG.getContext(), VT.getScalarType(),
                                HalfNumElts);
  if (!TLI.isShuffleMaskLegal(Mask0, HalfVT) ||
      !TLI.isShuffleMaskLegal(Mask1, HalfVT))
    return SDValue();

  // shuffle (concat X, undef), (concat Y, undef), Mask -->
  // concat (shuffle X, Y, Mask0), (shuffle X, Y, Mask1)
  SDValue X = N0.getOperand(0), Y = N1.getOperand(0);
  SDLoc DL(Shuf);
  SDValue Shuf0 = DAG.getVectorShuffle(HalfVT, DL, X, Y, Mask0);
  SDValue Shuf1 = DAG.getVectorShuffle(HalfVT, DL, X, Y, Mask1);
  return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, Shuf0, Shuf1);
}

// Tries to turn a shuffle of two CONCAT_VECTORS into a single concat,
// or turn a shuffle of a single concat into simpler shuffle then concat.
static SDValue partitionShuffleOfConcats(SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);
  ArrayRef<int> Mask = SVN->getMask();

  SmallVector<SDValue, 4> Ops;
  EVT ConcatVT = N0.getOperand(0).getValueType();
  unsigned NumElemsPerConcat = ConcatVT.getVectorNumElements();
  unsigned NumConcats = NumElts / NumElemsPerConcat;

  auto IsUndefMaskElt = [](int i) { return i == -1; };

  // Special case: shuffle(concat(A,B)) can be more efficiently represented
  // as concat(shuffle(A,B),UNDEF) if the shuffle doesn't set any of the high
  // half vector elements.
  if (NumElemsPerConcat * 2 == NumElts && N1.isUndef() &&
      llvm::all_of(Mask.slice(NumElemsPerConcat, NumElemsPerConcat),
                   IsUndefMaskElt)) {
    N0 = DAG.getVectorShuffle(ConcatVT, SDLoc(N), N0.getOperand(0),
                              N0.getOperand(1),
                              Mask.slice(0, NumElemsPerConcat));
    N1 = DAG.getUNDEF(ConcatVT);
    return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, N0, N1);
  }

  // Look at every vector that's inserted. We're looking for exact
  // subvector-sized copies from a concatenated vector
  for (unsigned I = 0; I != NumConcats; ++I) {
    unsigned Begin = I * NumElemsPerConcat;
    ArrayRef<int> SubMask = Mask.slice(Begin, NumElemsPerConcat);

    // Make sure we're dealing with a copy.
    if (llvm::all_of(SubMask, IsUndefMaskElt)) {
      Ops.push_back(DAG.getUNDEF(ConcatVT));
      continue;
    }

    int OpIdx = -1;
    for (int i = 0; i != (int)NumElemsPerConcat; ++i) {
      if (IsUndefMaskElt(SubMask[i]))
        continue;
      if ((SubMask[i] % (int)NumElemsPerConcat) != i)
        return SDValue();
      int EltOpIdx = SubMask[i] / NumElemsPerConcat;
      if (0 <= OpIdx && EltOpIdx != OpIdx)
        return SDValue();
      OpIdx = EltOpIdx;
    }
    assert(0 <= OpIdx && "Unknown concat_vectors op");

    if (OpIdx < (int)N0.getNumOperands())
      Ops.push_back(N0.getOperand(OpIdx));
    else
      Ops.push_back(N1.getOperand(OpIdx - N0.getNumOperands()));
  }

  return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Ops);
}

// Attempt to combine a shuffle of 2 inputs of 'scalar sources' -
// BUILD_VECTOR or SCALAR_TO_VECTOR into a single BUILD_VECTOR.
//
// SHUFFLE(BUILD_VECTOR(), BUILD_VECTOR()) -> BUILD_VECTOR() is always
// a simplification in some sense, but it isn't appropriate in general: some
// BUILD_VECTORs are substantially cheaper than others. The general case
// of a BUILD_VECTOR requires inserting each element individually (or
// performing the equivalent in a temporary stack variable). A BUILD_VECTOR of
// all constants is a single constant pool load.  A BUILD_VECTOR where each
// element is identical is a splat.  A BUILD_VECTOR where most of the operands
// are undef lowers to a small number of element insertions.
//
// To deal with this, we currently use a bunch of mostly arbitrary heuristics.
// We don't fold shuffles where one side is a non-zero constant, and we don't
// fold shuffles if the resulting (non-splat) BUILD_VECTOR would have duplicate
// non-constant operands. This seems to work out reasonably well in practice.
static SDValue combineShuffleOfScalars(ShuffleVectorSDNode *SVN,
                                       SelectionDAG &DAG,
                                       const TargetLowering &TLI) {
  EVT VT = SVN->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  SDValue N0 = SVN->getOperand(0);
  SDValue N1 = SVN->getOperand(1);

  if (!N0->hasOneUse())
    return SDValue();

  // If only one of N1,N2 is constant, bail out if it is not ALL_ZEROS as
  // discussed above.
  if (!N1.isUndef()) {
    if (!N1->hasOneUse())
      return SDValue();

    bool N0AnyConst = isAnyConstantBuildVector(N0);
    bool N1AnyConst = isAnyConstantBuildVector(N1);
    if (N0AnyConst && !N1AnyConst && !ISD::isBuildVectorAllZeros(N0.getNode()))
      return SDValue();
    if (!N0AnyConst && N1AnyConst && !ISD::isBuildVectorAllZeros(N1.getNode()))
      return SDValue();
  }

  // If both inputs are splats of the same value then we can safely merge this
  // to a single BUILD_VECTOR with undef elements based on the shuffle mask.
  bool IsSplat = false;
  auto *BV0 = dyn_cast<BuildVectorSDNode>(N0);
  auto *BV1 = dyn_cast<BuildVectorSDNode>(N1);
  if (BV0 && BV1)
    if (SDValue Splat0 = BV0->getSplatValue())
      IsSplat = (Splat0 == BV1->getSplatValue());

  SmallVector<SDValue, 8> Ops;
  SmallSet<SDValue, 16> DuplicateOps;
  for (int M : SVN->getMask()) {
    SDValue Op = DAG.getUNDEF(VT.getScalarType());
    if (M >= 0) {
      int Idx = M < (int)NumElts ? M : M - NumElts;
      SDValue &S = (M < (int)NumElts ? N0 : N1);
      if (S.getOpcode() == ISD::BUILD_VECTOR) {
        Op = S.getOperand(Idx);
      } else if (S.getOpcode() == ISD::SCALAR_TO_VECTOR) {
        SDValue Op0 = S.getOperand(0);
        Op = Idx == 0 ? Op0 : DAG.getUNDEF(Op0.getValueType());
      } else {
        // Operand can't be combined - bail out.
        return SDValue();
      }
    }

    // Don't duplicate a non-constant BUILD_VECTOR operand unless we're
    // generating a splat; semantically, this is fine, but it's likely to
    // generate low-quality code if the target can't reconstruct an appropriate
    // shuffle.
    if (!Op.isUndef() && !isIntOrFPConstant(Op))
      if (!IsSplat && !DuplicateOps.insert(Op).second)
        return SDValue();

    Ops.push_back(Op);
  }

  // BUILD_VECTOR requires all inputs to be of the same type, find the
  // maximum type and extend them all.
  EVT SVT = VT.getScalarType();
  if (SVT.isInteger())
    for (SDValue &Op : Ops)
      SVT = (SVT.bitsLT(Op.getValueType()) ? Op.getValueType() : SVT);
  if (SVT != VT.getScalarType())
    for (SDValue &Op : Ops)
      Op = TLI.isZExtFree(Op.getValueType(), SVT)
               ? DAG.getZExtOrTrunc(Op, SDLoc(SVN), SVT)
               : DAG.getSExtOrTrunc(Op, SDLoc(SVN), SVT);
  return DAG.getBuildVector(VT, SDLoc(SVN), Ops);
}

// Match shuffles that can be converted to any_vector_extend_in_reg.
// This is often generated during legalization.
// e.g. v4i32 <0,u,1,u> -> (v2i64 any_vector_extend_in_reg(v4i32 src))
// TODO Add support for ZERO_EXTEND_VECTOR_INREG when we have a test case.
static SDValue combineShuffleToVectorExtend(ShuffleVectorSDNode *SVN,
                                            SelectionDAG &DAG,
                                            const TargetLowering &TLI,
                                            bool LegalOperations) {
  EVT VT = SVN->getValueType(0);
  bool IsBigEndian = DAG.getDataLayout().isBigEndian();

  // TODO Add support for big-endian when we have a test case.
  if (!VT.isInteger() || IsBigEndian)
    return SDValue();

  unsigned NumElts = VT.getVectorNumElements();
  unsigned EltSizeInBits = VT.getScalarSizeInBits();
  ArrayRef<int> Mask = SVN->getMask();
  SDValue N0 = SVN->getOperand(0);

  // shuffle<0,-1,1,-1> == (v2i64 anyextend_vector_inreg(v4i32))
  auto isAnyExtend = [&Mask, &NumElts](unsigned Scale) {
    for (unsigned i = 0; i != NumElts; ++i) {
      if (Mask[i] < 0)
        continue;
      if ((i % Scale) == 0 && Mask[i] == (int)(i / Scale))
        continue;
      return false;
    }
    return true;
  };

  // Attempt to match a '*_extend_vector_inreg' shuffle, we just search for
  // power-of-2 extensions as they are the most likely.
  for (unsigned Scale = 2; Scale < NumElts; Scale *= 2) {
    // Check for non power of 2 vector sizes
    if (NumElts % Scale != 0)
      continue;
    if (!isAnyExtend(Scale))
      continue;

    EVT OutSVT = EVT::getIntegerVT(*DAG.getContext(), EltSizeInBits * Scale);
    EVT OutVT = EVT::getVectorVT(*DAG.getContext(), OutSVT, NumElts / Scale);
    // Never create an illegal type. Only create unsupported operations if we
    // are pre-legalization.
    if (TLI.isTypeLegal(OutVT))
      if (!LegalOperations ||
          TLI.isOperationLegalOrCustom(ISD::ANY_EXTEND_VECTOR_INREG, OutVT))
        return DAG.getBitcast(VT,
                              DAG.getNode(ISD::ANY_EXTEND_VECTOR_INREG,
                                          SDLoc(SVN), OutVT, N0));
  }

  return SDValue();
}

// Detect 'truncate_vector_inreg' style shuffles that pack the lower parts of
// each source element of a large type into the lowest elements of a smaller
// destination type. This is often generated during legalization.
// If the source node itself was a '*_extend_vector_inreg' node then we should
// then be able to remove it.
static SDValue combineTruncationShuffle(ShuffleVectorSDNode *SVN,
                                        SelectionDAG &DAG) {
  EVT VT = SVN->getValueType(0);
  bool IsBigEndian = DAG.getDataLayout().isBigEndian();

  // TODO Add support for big-endian when we have a test case.
  if (!VT.isInteger() || IsBigEndian)
    return SDValue();

  SDValue N0 = peekThroughBitcasts(SVN->getOperand(0));

  unsigned Opcode = N0.getOpcode();
  if (Opcode != ISD::ANY_EXTEND_VECTOR_INREG &&
      Opcode != ISD::SIGN_EXTEND_VECTOR_INREG &&
      Opcode != ISD::ZERO_EXTEND_VECTOR_INREG)
    return SDValue();

  SDValue N00 = N0.getOperand(0);
  ArrayRef<int> Mask = SVN->getMask();
  unsigned NumElts = VT.getVectorNumElements();
  unsigned EltSizeInBits = VT.getScalarSizeInBits();
  unsigned ExtSrcSizeInBits = N00.getScalarValueSizeInBits();
  unsigned ExtDstSizeInBits = N0.getScalarValueSizeInBits();

  if (ExtDstSizeInBits % ExtSrcSizeInBits != 0)
    return SDValue();
  unsigned ExtScale = ExtDstSizeInBits / ExtSrcSizeInBits;

  // (v4i32 truncate_vector_inreg(v2i64)) == shuffle<0,2-1,-1>
  // (v8i16 truncate_vector_inreg(v4i32)) == shuffle<0,2,4,6,-1,-1,-1,-1>
  // (v8i16 truncate_vector_inreg(v2i64)) == shuffle<0,4,-1,-1,-1,-1,-1,-1>
  auto isTruncate = [&Mask, &NumElts](unsigned Scale) {
    for (unsigned i = 0; i != NumElts; ++i) {
      if (Mask[i] < 0)
        continue;
      if ((i * Scale) < NumElts && Mask[i] == (int)(i * Scale))
        continue;
      return false;
    }
    return true;
  };

  // At the moment we just handle the case where we've truncated back to the
  // same size as before the extension.
  // TODO: handle more extension/truncation cases as cases arise.
  if (EltSizeInBits != ExtSrcSizeInBits)
    return SDValue();

  // We can remove *extend_vector_inreg only if the truncation happens at
  // the same scale as the extension.
  if (isTruncate(ExtScale))
    return DAG.getBitcast(VT, N00);

  return SDValue();
}

// Combine shuffles of splat-shuffles of the form:
// shuffle (shuffle V, undef, splat-mask), undef, M
// If splat-mask contains undef elements, we need to be careful about
// introducing undef's in the folded mask which are not the result of composing
// the masks of the shuffles.
static SDValue combineShuffleOfSplatVal(ShuffleVectorSDNode *Shuf,
                                        SelectionDAG &DAG) {
  if (!Shuf->getOperand(1).isUndef())
    return SDValue();
  auto *Splat = dyn_cast<ShuffleVectorSDNode>(Shuf->getOperand(0));
  if (!Splat || !Splat->isSplat())
    return SDValue();

  ArrayRef<int> ShufMask = Shuf->getMask();
  ArrayRef<int> SplatMask = Splat->getMask();
  assert(ShufMask.size() == SplatMask.size() && "Mask length mismatch");

  // Prefer simplifying to the splat-shuffle, if possible. This is legal if
  // every undef mask element in the splat-shuffle has a corresponding undef
  // element in the user-shuffle's mask or if the composition of mask elements
  // would result in undef.
  // Examples for (shuffle (shuffle v, undef, SplatMask), undef, UserMask):
  // * UserMask=[0,2,u,u], SplatMask=[2,u,2,u] -> [2,2,u,u]
  //   In this case it is not legal to simplify to the splat-shuffle because we
  //   may be exposing the users of the shuffle an undef element at index 1
  //   which was not there before the combine.
  // * UserMask=[0,u,2,u], SplatMask=[2,u,2,u] -> [2,u,2,u]
  //   In this case the composition of masks yields SplatMask, so it's ok to
  //   simplify to the splat-shuffle.
  // * UserMask=[3,u,2,u], SplatMask=[2,u,2,u] -> [u,u,2,u]
  //   In this case the composed mask includes all undef elements of SplatMask
  //   and in addition sets element zero to undef. It is safe to simplify to
  //   the splat-shuffle.
  auto CanSimplifyToExistingSplat = [](ArrayRef<int> UserMask,
                                       ArrayRef<int> SplatMask) {
    for (unsigned i = 0, e = UserMask.size(); i != e; ++i)
      if (UserMask[i] != -1 && SplatMask[i] == -1 &&
          SplatMask[UserMask[i]] != -1)
        return false;
    return true;
  };
  if (CanSimplifyToExistingSplat(ShufMask, SplatMask))
    return Shuf->getOperand(0);

  // Create a new shuffle with a mask that is composed of the two shuffles'
  // masks.
  SmallVector<int, 32> NewMask;
  for (int Idx : ShufMask)
    NewMask.push_back(Idx == -1 ? -1 : SplatMask[Idx]);

  return DAG.getVectorShuffle(Splat->getValueType(0), SDLoc(Splat),
                              Splat->getOperand(0), Splat->getOperand(1),
                              NewMask);
}

/// Combine shuffle of shuffle of the form:
/// shuf (shuf X, undef, InnerMask), undef, OuterMask --> splat X
static SDValue formSplatFromShuffles(ShuffleVectorSDNode *OuterShuf,
                                     SelectionDAG &DAG) {
  if (!OuterShuf->getOperand(1).isUndef())
    return SDValue();
  auto *InnerShuf = dyn_cast<ShuffleVectorSDNode>(OuterShuf->getOperand(0));
  if (!InnerShuf || !InnerShuf->getOperand(1).isUndef())
    return SDValue();

  ArrayRef<int> OuterMask = OuterShuf->getMask();
  ArrayRef<int> InnerMask = InnerShuf->getMask();
  unsigned NumElts = OuterMask.size();
  assert(NumElts == InnerMask.size() && "Mask length mismatch");
  SmallVector<int, 32> CombinedMask(NumElts, -1);
  int SplatIndex = -1;
  for (unsigned i = 0; i != NumElts; ++i) {
    // Undef lanes remain undef.
    int OuterMaskElt = OuterMask[i];
    if (OuterMaskElt == -1)
      continue;

    // Peek through the shuffle masks to get the underlying source element.
    int InnerMaskElt = InnerMask[OuterMaskElt];
    if (InnerMaskElt == -1)
      continue;

    // Initialize the splatted element.
    if (SplatIndex == -1)
      SplatIndex = InnerMaskElt;

    // Non-matching index - this is not a splat.
    if (SplatIndex != InnerMaskElt)
      return SDValue();

    CombinedMask[i] = InnerMaskElt;
  }
  assert((all_of(CombinedMask, [](int M) { return M == -1; }) ||
          getSplatIndex(CombinedMask) != -1) &&
         "Expected a splat mask");

  // TODO: The transform may be a win even if the mask is not legal.
  EVT VT = OuterShuf->getValueType(0);
  assert(VT == InnerShuf->getValueType(0) && "Expected matching shuffle types");
  if (!DAG.getTargetLoweringInfo().isShuffleMaskLegal(CombinedMask, VT))
    return SDValue();

  return DAG.getVectorShuffle(VT, SDLoc(OuterShuf), InnerShuf->getOperand(0),
                              InnerShuf->getOperand(1), CombinedMask);
}

/// If the shuffle mask is taking exactly one element from the first vector
/// operand and passing through all other elements from the second vector
/// operand, return the index of the mask element that is choosing an element
/// from the first operand. Otherwise, return -1.
static int getShuffleMaskIndexOfOneElementFromOp0IntoOp1(ArrayRef<int> Mask) {
  int MaskSize = Mask.size();
  int EltFromOp0 = -1;
  // TODO: This does not match if there are undef elements in the shuffle mask.
  // Should we ignore undefs in the shuffle mask instead? The trade-off is
  // removing an instruction (a shuffle), but losing the knowledge that some
  // vector lanes are not needed.
  for (int i = 0; i != MaskSize; ++i) {
    if (Mask[i] >= 0 && Mask[i] < MaskSize) {
      // We're looking for a shuffle of exactly one element from operand 0.
      if (EltFromOp0 != -1)
        return -1;
      EltFromOp0 = i;
    } else if (Mask[i] != i + MaskSize) {
      // Nothing from operand 1 can change lanes.
      return -1;
    }
  }
  return EltFromOp0;
}

/// If a shuffle inserts exactly one element from a source vector operand into
/// another vector operand and we can access the specified element as a scalar,
/// then we can eliminate the shuffle.
static SDValue replaceShuffleOfInsert(ShuffleVectorSDNode *Shuf,
                                      SelectionDAG &DAG) {
  // First, check if we are taking one element of a vector and shuffling that
  // element into another vector.
  ArrayRef<int> Mask = Shuf->getMask();
  SmallVector<int, 16> CommutedMask(Mask.begin(), Mask.end());
  SDValue Op0 = Shuf->getOperand(0);
  SDValue Op1 = Shuf->getOperand(1);
  int ShufOp0Index = getShuffleMaskIndexOfOneElementFromOp0IntoOp1(Mask);
  if (ShufOp0Index == -1) {
    // Commute mask and check again.
    ShuffleVectorSDNode::commuteMask(CommutedMask);
    ShufOp0Index = getShuffleMaskIndexOfOneElementFromOp0IntoOp1(CommutedMask);
    if (ShufOp0Index == -1)
      return SDValue();
    // Commute operands to match the commuted shuffle mask.
    std::swap(Op0, Op1);
    Mask = CommutedMask;
  }

  // The shuffle inserts exactly one element from operand 0 into operand 1.
  // Now see if we can access that element as a scalar via a real insert element
  // instruction.
  // TODO: We can try harder to locate the element as a scalar. Examples: it
  // could be an operand of SCALAR_TO_VECTOR, BUILD_VECTOR, or a constant.
  assert(Mask[ShufOp0Index] >= 0 && Mask[ShufOp0Index] < (int)Mask.size() &&
         "Shuffle mask value must be from operand 0");
  if (Op0.getOpcode() != ISD::INSERT_VECTOR_ELT)
    return SDValue();

  auto *InsIndexC = dyn_cast<ConstantSDNode>(Op0.getOperand(2));
  if (!InsIndexC || InsIndexC->getSExtValue() != Mask[ShufOp0Index])
    return SDValue();

  // There's an existing insertelement with constant insertion index, so we
  // don't need to check the legality/profitability of a replacement operation
  // that differs at most in the constant value. The target should be able to
  // lower any of those in a similar way. If not, legalization will expand this
  // to a scalar-to-vector plus shuffle.
  //
  // Note that the shuffle may move the scalar from the position that the insert
  // element used. Therefore, our new insert element occurs at the shuffle's
  // mask index value, not the insert's index value.
  // shuffle (insertelt v1, x, C), v2, mask --> insertelt v2, x, C'
  SDValue NewInsIndex = DAG.getVectorIdxConstant(ShufOp0Index, SDLoc(Shuf));
  return DAG.getNode(ISD::INSERT_VECTOR_ELT, SDLoc(Shuf), Op0.getValueType(),
                     Op1, Op0.getOperand(1), NewInsIndex);
}

/// If we have a unary shuffle of a shuffle, see if it can be folded away
/// completely. This has the potential to lose undef knowledge because the first
/// shuffle may not have an undef mask element where the second one does. So
/// only call this after doing simplifications based on demanded elements.
static SDValue simplifyShuffleOfShuffle(ShuffleVectorSDNode *Shuf) {
  // shuf (shuf0 X, Y, Mask0), undef, Mask
  auto *Shuf0 = dyn_cast<ShuffleVectorSDNode>(Shuf->getOperand(0));
  if (!Shuf0 || !Shuf->getOperand(1).isUndef())
    return SDValue();

  ArrayRef<int> Mask = Shuf->getMask();
  ArrayRef<int> Mask0 = Shuf0->getMask();
  for (int i = 0, e = (int)Mask.size(); i != e; ++i) {
    // Ignore undef elements.
    if (Mask[i] == -1)
      continue;
    assert(Mask[i] >= 0 && Mask[i] < e && "Unexpected shuffle mask value");

    // Is the element of the shuffle operand chosen by this shuffle the same as
    // the element chosen by the shuffle operand itself?
    if (Mask0[Mask[i]] != Mask0[i])
      return SDValue();
  }
  // Every element of this shuffle is identical to the result of the previous
  // shuffle, so we can replace this value.
  return Shuf->getOperand(0);
}

SDValue DAGCombiner::visitVECTOR_SHUFFLE(SDNode *N) {
  EVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  assert(N0.getValueType() == VT && "Vector shuffle must be normalized in DAG");

  // Canonicalize shuffle undef, undef -> undef
  if (N0.isUndef() && N1.isUndef())
    return DAG.getUNDEF(VT);

  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(N);

  // Canonicalize shuffle v, v -> v, undef
  if (N0 == N1) {
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= (int)NumElts) Idx -= NumElts;
      NewMask.push_back(Idx);
    }
    return DAG.getVectorShuffle(VT, SDLoc(N), N0, DAG.getUNDEF(VT), NewMask);
  }

  // Canonicalize shuffle undef, v -> v, undef.  Commute the shuffle mask.
  if (N0.isUndef())
    return DAG.getCommutedVectorShuffle(*SVN);

  // Remove references to rhs if it is undef
  if (N1.isUndef()) {
    bool Changed = false;
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= (int)NumElts) {
        Idx = -1;
        Changed = true;
      }
      NewMask.push_back(Idx);
    }
    if (Changed)
      return DAG.getVectorShuffle(VT, SDLoc(N), N0, N1, NewMask);
  }

  if (SDValue InsElt = replaceShuffleOfInsert(SVN, DAG))
    return InsElt;

  // A shuffle of a single vector that is a splatted value can always be folded.
  if (SDValue V = combineShuffleOfSplatVal(SVN, DAG))
    return V;

  if (SDValue V = formSplatFromShuffles(SVN, DAG))
    return V;

  // If it is a splat, check if the argument vector is another splat or a
  // build_vector.
  if (SVN->isSplat() && SVN->getSplatIndex() < (int)NumElts) {
    int SplatIndex = SVN->getSplatIndex();
    if (N0.hasOneUse() && TLI.isExtractVecEltCheap(VT, SplatIndex) &&
        TLI.isBinOp(N0.getOpcode()) && N0.getNode()->getNumValues() == 1) {
      // splat (vector_bo L, R), Index -->
      // splat (scalar_bo (extelt L, Index), (extelt R, Index))
      SDValue L = N0.getOperand(0), R = N0.getOperand(1);
      SDLoc DL(N);
      EVT EltVT = VT.getScalarType();
      SDValue Index = DAG.getVectorIdxConstant(SplatIndex, DL);
      SDValue ExtL = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, L, Index);
      SDValue ExtR = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, R, Index);
      SDValue NewBO = DAG.getNode(N0.getOpcode(), DL, EltVT, ExtL, ExtR,
                                  N0.getNode()->getFlags());
      SDValue Insert = DAG.getNode(ISD::SCALAR_TO_VECTOR, DL, VT, NewBO);
      SmallVector<int, 16> ZeroMask(VT.getVectorNumElements(), 0);
      return DAG.getVectorShuffle(VT, DL, Insert, DAG.getUNDEF(VT), ZeroMask);
    }

    // If this is a bit convert that changes the element type of the vector but
    // not the number of vector elements, look through it.  Be careful not to
    // look though conversions that change things like v4f32 to v2f64.
    SDNode *V = N0.getNode();
    if (V->getOpcode() == ISD::BITCAST) {
      SDValue ConvInput = V->getOperand(0);
      if (ConvInput.getValueType().isVector() &&
          ConvInput.getValueType().getVectorNumElements() == NumElts)
        V = ConvInput.getNode();
    }

    if (V->getOpcode() == ISD::BUILD_VECTOR) {
      assert(V->getNumOperands() == NumElts &&
             "BUILD_VECTOR has wrong number of operands");
      SDValue Base;
      bool AllSame = true;
      for (unsigned i = 0; i != NumElts; ++i) {
        if (!V->getOperand(i).isUndef()) {
          Base = V->getOperand(i);
          break;
        }
      }
      // Splat of <u, u, u, u>, return <u, u, u, u>
      if (!Base.getNode())
        return N0;
      for (unsigned i = 0; i != NumElts; ++i) {
        if (V->getOperand(i) != Base) {
          AllSame = false;
          break;
        }
      }
      // Splat of <x, x, x, x>, return <x, x, x, x>
      if (AllSame)
        return N0;

      // Canonicalize any other splat as a build_vector.
      SDValue Splatted = V->getOperand(SplatIndex);
      SmallVector<SDValue, 8> Ops(NumElts, Splatted);
      SDValue NewBV = DAG.getBuildVector(V->getValueType(0), SDLoc(N), Ops);

      // We may have jumped through bitcasts, so the type of the
      // BUILD_VECTOR may not match the type of the shuffle.
      if (V->getValueType(0) != VT)
        NewBV = DAG.getBitcast(VT, NewBV);
      return NewBV;
    }
  }

  // Simplify source operands based on shuffle mask.
  if (SimplifyDemandedVectorElts(SDValue(N, 0)))
    return SDValue(N, 0);

  // This is intentionally placed after demanded elements simplification because
  // it could eliminate knowledge of undef elements created by this shuffle.
  if (SDValue ShufOp = simplifyShuffleOfShuffle(SVN))
    return ShufOp;

  // Match shuffles that can be converted to any_vector_extend_in_reg.
  if (SDValue V = combineShuffleToVectorExtend(SVN, DAG, TLI, LegalOperations))
    return V;

  // Combine "truncate_vector_in_reg" style shuffles.
  if (SDValue V = combineTruncationShuffle(SVN, DAG))
    return V;

  if (N0.getOpcode() == ISD::CONCAT_VECTORS &&
      Level < AfterLegalizeVectorOps &&
      (N1.isUndef() ||
      (N1.getOpcode() == ISD::CONCAT_VECTORS &&
       N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()))) {
    if (SDValue V = partitionShuffleOfConcats(N, DAG))
      return V;
  }

  // A shuffle of a concat of the same narrow vector can be reduced to use
  // only low-half elements of a concat with undef:
  // shuf (concat X, X), undef, Mask --> shuf (concat X, undef), undef, Mask'
  if (N0.getOpcode() == ISD::CONCAT_VECTORS && N1.isUndef() &&
      N0.getNumOperands() == 2 &&
      N0.getOperand(0) == N0.getOperand(1)) {
    int HalfNumElts = (int)NumElts / 2;
    SmallVector<int, 8> NewMask;
    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx >= HalfNumElts) {
        assert(Idx < (int)NumElts && "Shuffle mask chooses undef op");
        Idx -= HalfNumElts;
      }
      NewMask.push_back(Idx);
    }
    if (TLI.isShuffleMaskLegal(NewMask, VT)) {
      SDValue UndefVec = DAG.getUNDEF(N0.getOperand(0).getValueType());
      SDValue NewCat = DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT,
                                   N0.getOperand(0), UndefVec);
      return DAG.getVectorShuffle(VT, SDLoc(N), NewCat, N1, NewMask);
    }
  }

  // See if we can replace a shuffle with an insert_subvector.
  // e.g. v2i32 into v8i32:
  // shuffle(lhs,concat(rhs0,rhs1,rhs2,rhs3),0,1,2,3,10,11,6,7).
  // --> insert_subvector(lhs,rhs1,4).
  if (Level < AfterLegalizeVectorOps && TLI.isTypeLegal(VT) &&
      TLI.isOperationLegalOrCustom(ISD::INSERT_SUBVECTOR, VT)) {
    auto ShuffleToInsert = [&](SDValue LHS, SDValue RHS, ArrayRef<int> Mask) {
      // Ensure RHS subvectors are legal.
      assert(RHS.getOpcode() == ISD::CONCAT_VECTORS && "Can't find subvectors");
      EVT SubVT = RHS.getOperand(0).getValueType();
      int NumSubVecs = RHS.getNumOperands();
      int NumSubElts = SubVT.getVectorNumElements();
      assert((NumElts % NumSubElts) == 0 && "Subvector mismatch");
      if (!TLI.isTypeLegal(SubVT))
        return SDValue();

      // Don't bother if we have an unary shuffle (matches undef + LHS elts).
      if (all_of(Mask, [NumElts](int M) { return M < (int)NumElts; }))
        return SDValue();

      // Search [NumSubElts] spans for RHS sequence.
      // TODO: Can we avoid nested loops to increase performance?
      SmallVector<int> InsertionMask(NumElts);
      for (int SubVec = 0; SubVec != NumSubVecs; ++SubVec) {
        for (int SubIdx = 0; SubIdx != (int)NumElts; SubIdx += NumSubElts) {
          // Reset mask to identity.
          std::iota(InsertionMask.begin(), InsertionMask.end(), 0);

          // Add subvector insertion.
          std::iota(InsertionMask.begin() + SubIdx,
                    InsertionMask.begin() + SubIdx + NumSubElts,
                    NumElts + (SubVec * NumSubElts));

          // See if the shuffle mask matches the reference insertion mask.
          bool MatchingShuffle = true;
          for (int i = 0; i != (int)NumElts; ++i) {
            int ExpectIdx = InsertionMask[i];
            int ActualIdx = Mask[i];
            if (0 <= ActualIdx && ExpectIdx != ActualIdx) {
              MatchingShuffle = false;
              break;
            }
          }

          if (MatchingShuffle)
            return DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), VT, LHS,
                               RHS.getOperand(SubVec),
                               DAG.getVectorIdxConstant(SubIdx, SDLoc(N)));
        }
      }
      return SDValue();
    };
    ArrayRef<int> Mask = SVN->getMask();
    if (N1.getOpcode() == ISD::CONCAT_VECTORS)
      if (SDValue InsertN1 = ShuffleToInsert(N0, N1, Mask))
        return InsertN1;
    if (N0.getOpcode() == ISD::CONCAT_VECTORS) {
      SmallVector<int> CommuteMask(Mask.begin(), Mask.end());
      ShuffleVectorSDNode::commuteMask(CommuteMask);
      if (SDValue InsertN0 = ShuffleToInsert(N1, N0, CommuteMask))
        return InsertN0;
    }
  }

  // Attempt to combine a shuffle of 2 inputs of 'scalar sources' -
  // BUILD_VECTOR or SCALAR_TO_VECTOR into a single BUILD_VECTOR.
  if (Level < AfterLegalizeDAG && TLI.isTypeLegal(VT))
    if (SDValue Res = combineShuffleOfScalars(SVN, DAG, TLI))
      return Res;

  // If this shuffle only has a single input that is a bitcasted shuffle,
  // attempt to merge the 2 shuffles and suitably bitcast the inputs/output
  // back to their original types.
  if (N0.getOpcode() == ISD::BITCAST && N0.hasOneUse() &&
      N1.isUndef() && Level < AfterLegalizeVectorOps &&
      TLI.isTypeLegal(VT)) {

    SDValue BC0 = peekThroughOneUseBitcasts(N0);
    if (BC0.getOpcode() == ISD::VECTOR_SHUFFLE && BC0.hasOneUse()) {
      EVT SVT = VT.getScalarType();
      EVT InnerVT = BC0->getValueType(0);
      EVT InnerSVT = InnerVT.getScalarType();

      // Determine which shuffle works with the smaller scalar type.
      EVT ScaleVT = SVT.bitsLT(InnerSVT) ? VT : InnerVT;
      EVT ScaleSVT = ScaleVT.getScalarType();

      if (TLI.isTypeLegal(ScaleVT) &&
          0 == (InnerSVT.getSizeInBits() % ScaleSVT.getSizeInBits()) &&
          0 == (SVT.getSizeInBits() % ScaleSVT.getSizeInBits())) {
        int InnerScale = InnerSVT.getSizeInBits() / ScaleSVT.getSizeInBits();
        int OuterScale = SVT.getSizeInBits() / ScaleSVT.getSizeInBits();

        // Scale the shuffle masks to the smaller scalar type.
        ShuffleVectorSDNode *InnerSVN = cast<ShuffleVectorSDNode>(BC0);
        SmallVector<int, 8> InnerMask;
        SmallVector<int, 8> OuterMask;
        narrowShuffleMaskElts(InnerScale, InnerSVN->getMask(), InnerMask);
        narrowShuffleMaskElts(OuterScale, SVN->getMask(), OuterMask);

        // Merge the shuffle masks.
        SmallVector<int, 8> NewMask;
        for (int M : OuterMask)
          NewMask.push_back(M < 0 ? -1 : InnerMask[M]);

        // Test for shuffle mask legality over both commutations.
        SDValue SV0 = BC0->getOperand(0);
        SDValue SV1 = BC0->getOperand(1);
        bool LegalMask = TLI.isShuffleMaskLegal(NewMask, ScaleVT);
        if (!LegalMask) {
          std::swap(SV0, SV1);
          ShuffleVectorSDNode::commuteMask(NewMask);
          LegalMask = TLI.isShuffleMaskLegal(NewMask, ScaleVT);
        }

        if (LegalMask) {
          SV0 = DAG.getBitcast(ScaleVT, SV0);
          SV1 = DAG.getBitcast(ScaleVT, SV1);
          return DAG.getBitcast(
              VT, DAG.getVectorShuffle(ScaleVT, SDLoc(N), SV0, SV1, NewMask));
        }
      }
    }
  }

  // Compute the combined shuffle mask for a shuffle with SV0 as the first
  // operand, and SV1 as the second operand.
  // i.e. Merge SVN(OtherSVN, N1) -> shuffle(SV0, SV1, Mask) iff Commute = false
  //      Merge SVN(N1, OtherSVN) -> shuffle(SV0, SV1, Mask') iff Commute = true
  auto MergeInnerShuffle =
      [NumElts, &VT](bool Commute, ShuffleVectorSDNode *SVN,
                     ShuffleVectorSDNode *OtherSVN, SDValue N1,
                     const TargetLowering &TLI, SDValue &SV0, SDValue &SV1,
                     SmallVectorImpl<int> &Mask) -> bool {
    // Don't try to fold splats; they're likely to simplify somehow, or they
    // might be free.
    if (OtherSVN->isSplat())
      return false;

    SV0 = SV1 = SDValue();
    Mask.clear();

    for (unsigned i = 0; i != NumElts; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (Idx < 0) {
        // Propagate Undef.
        Mask.push_back(Idx);
        continue;
      }

      if (Commute)
        Idx = (Idx < (int)NumElts) ? (Idx + NumElts) : (Idx - NumElts);

      SDValue CurrentVec;
      if (Idx < (int)NumElts) {
        // This shuffle index refers to the inner shuffle N0. Lookup the inner
        // shuffle mask to identify which vector is actually referenced.
        Idx = OtherSVN->getMaskElt(Idx);
        if (Idx < 0) {
          // Propagate Undef.
          Mask.push_back(Idx);
          continue;
        }
        CurrentVec = (Idx < (int)NumElts) ? OtherSVN->getOperand(0)
                                          : OtherSVN->getOperand(1);
      } else {
        // This shuffle index references an element within N1.
        CurrentVec = N1;
      }

      // Simple case where 'CurrentVec' is UNDEF.
      if (CurrentVec.isUndef()) {
        Mask.push_back(-1);
        continue;
      }

      // Canonicalize the shuffle index. We don't know yet if CurrentVec
      // will be the first or second operand of the combined shuffle.
      Idx = Idx % NumElts;
      if (!SV0.getNode() || SV0 == CurrentVec) {
        // Ok. CurrentVec is the left hand side.
        // Update the mask accordingly.
        SV0 = CurrentVec;
        Mask.push_back(Idx);
        continue;
      }
      if (!SV1.getNode() || SV1 == CurrentVec) {
        // Ok. CurrentVec is the right hand side.
        // Update the mask accordingly.
        SV1 = CurrentVec;
        Mask.push_back(Idx + NumElts);
        continue;
      }

      // Last chance - see if the vector is another shuffle and if it
      // uses one of the existing candidate shuffle ops.
      if (auto *CurrentSVN = dyn_cast<ShuffleVectorSDNode>(CurrentVec)) {
        int InnerIdx = CurrentSVN->getMaskElt(Idx);
        if (InnerIdx < 0) {
          Mask.push_back(-1);
          continue;
        }
        SDValue InnerVec = (InnerIdx < (int)NumElts)
                               ? CurrentSVN->getOperand(0)
                               : CurrentSVN->getOperand(1);
        if (InnerVec.isUndef()) {
          Mask.push_back(-1);
          continue;
        }
        InnerIdx %= NumElts;
        if (InnerVec == SV0) {
          Mask.push_back(InnerIdx);
          continue;
        }
        if (InnerVec == SV1) {
          Mask.push_back(InnerIdx + NumElts);
          continue;
        }
      }

      // Bail out if we cannot convert the shuffle pair into a single shuffle.
      return false;
    }

    if (llvm::all_of(Mask, [](int M) { return M < 0; }))
      return true;

    // Avoid introducing shuffles with illegal mask.
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, B, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, C, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(B, C, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(B, A, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(C, A, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(C, B, M2)
    if (TLI.isShuffleMaskLegal(Mask, VT))
      return true;

    std::swap(SV0, SV1);
    ShuffleVectorSDNode::commuteMask(Mask);
    return TLI.isShuffleMaskLegal(Mask, VT);
  };

  if (Level < AfterLegalizeDAG && TLI.isTypeLegal(VT)) {
    // Canonicalize shuffles according to rules:
    //  shuffle(A, shuffle(A, B)) -> shuffle(shuffle(A,B), A)
    //  shuffle(B, shuffle(A, B)) -> shuffle(shuffle(A,B), B)
    //  shuffle(B, shuffle(A, Undef)) -> shuffle(shuffle(A, Undef), B)
    if (N1.getOpcode() == ISD::VECTOR_SHUFFLE &&
        N0.getOpcode() != ISD::VECTOR_SHUFFLE) {
      // The incoming shuffle must be of the same type as the result of the
      // current shuffle.
      assert(N1->getOperand(0).getValueType() == VT &&
             "Shuffle types don't match");

      SDValue SV0 = N1->getOperand(0);
      SDValue SV1 = N1->getOperand(1);
      bool HasSameOp0 = N0 == SV0;
      bool IsSV1Undef = SV1.isUndef();
      if (HasSameOp0 || IsSV1Undef || N0 == SV1)
        // Commute the operands of this shuffle so merging below will trigger.
        return DAG.getCommutedVectorShuffle(*SVN);
    }

    // Canonicalize splat shuffles to the RHS to improve merging below.
    //  shuffle(splat(A,u), shuffle(C,D)) -> shuffle'(shuffle(C,D), splat(A,u))
    if (N0.getOpcode() == ISD::VECTOR_SHUFFLE &&
        N1.getOpcode() == ISD::VECTOR_SHUFFLE &&
        cast<ShuffleVectorSDNode>(N0)->isSplat() &&
        !cast<ShuffleVectorSDNode>(N1)->isSplat()) {
      return DAG.getCommutedVectorShuffle(*SVN);
    }

    // Try to fold according to rules:
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, B, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(A, C, M2)
    //   shuffle(shuffle(A, B, M0), C, M1) -> shuffle(B, C, M2)
    // Don't try to fold shuffles with illegal type.
    // Only fold if this shuffle is the only user of the other shuffle.
    // Try matching shuffle(C,shuffle(A,B)) commutted patterns as well.
    for (int i = 0; i != 2; ++i) {
      if (N->getOperand(i).getOpcode() == ISD::VECTOR_SHUFFLE &&
          N->isOnlyUserOf(N->getOperand(i).getNode())) {
        // The incoming shuffle must be of the same type as the result of the
        // current shuffle.
        auto *OtherSV = cast<ShuffleVectorSDNode>(N->getOperand(i));
        assert(OtherSV->getOperand(0).getValueType() == VT &&
               "Shuffle types don't match");

        SDValue SV0, SV1;
        SmallVector<int, 4> Mask;
        if (MergeInnerShuffle(i != 0, SVN, OtherSV, N->getOperand(1 - i), TLI,
                              SV0, SV1, Mask)) {
          // Check if all indices in Mask are Undef. In case, propagate Undef.
          if (llvm::all_of(Mask, [](int M) { return M < 0; }))
            return DAG.getUNDEF(VT);

          return DAG.getVectorShuffle(VT, SDLoc(N),
                                      SV0 ? SV0 : DAG.getUNDEF(VT),
                                      SV1 ? SV1 : DAG.getUNDEF(VT), Mask);
        }
      }
    }

    // Merge shuffles through binops if we are able to merge it with at least
    // one other shuffles.
    // shuffle(bop(shuffle(x,y),shuffle(z,w)),undef)
    // shuffle(bop(shuffle(x,y),shuffle(z,w)),bop(shuffle(a,b),shuffle(c,d)))
    unsigned SrcOpcode = N0.getOpcode();
    if (TLI.isBinOp(SrcOpcode) && N->isOnlyUserOf(N0.getNode()) &&
        (N1.isUndef() ||
         (SrcOpcode == N1.getOpcode() && N->isOnlyUserOf(N1.getNode())))) {
      // Get binop source ops, or just pass on the undef.
      SDValue Op00 = N0.getOperand(0);
      SDValue Op01 = N0.getOperand(1);
      SDValue Op10 = N1.isUndef() ? N1 : N1.getOperand(0);
      SDValue Op11 = N1.isUndef() ? N1 : N1.getOperand(1);
      // TODO: We might be able to relax the VT check but we don't currently
      // have any isBinOp() that has different result/ops VTs so play safe until
      // we have test coverage.
      if (Op00.getValueType() == VT && Op10.getValueType() == VT &&
          Op01.getValueType() == VT && Op11.getValueType() == VT &&
          (Op00.getOpcode() == ISD::VECTOR_SHUFFLE ||
           Op10.getOpcode() == ISD::VECTOR_SHUFFLE ||
           Op01.getOpcode() == ISD::VECTOR_SHUFFLE ||
           Op11.getOpcode() == ISD::VECTOR_SHUFFLE)) {
        auto CanMergeInnerShuffle = [&](SDValue &SV0, SDValue &SV1,
                                        SmallVectorImpl<int> &Mask, bool LeftOp,
                                        bool Commute) {
          SDValue InnerN = Commute ? N1 : N0;
          SDValue Op0 = LeftOp ? Op00 : Op01;
          SDValue Op1 = LeftOp ? Op10 : Op11;
          if (Commute)
            std::swap(Op0, Op1);
          // Only accept the merged shuffle if we don't introduce undef elements,
          // or the inner shuffle already contained undef elements.
          auto *SVN0 = dyn_cast<ShuffleVectorSDNode>(Op0);
          return SVN0 && InnerN->isOnlyUserOf(SVN0) &&
                 MergeInnerShuffle(Commute, SVN, SVN0, Op1, TLI, SV0, SV1,
                                   Mask) &&
                 (llvm::any_of(SVN0->getMask(), [](int M) { return M < 0; }) ||
                  llvm::none_of(Mask, [](int M) { return M < 0; }));
        };

        // Ensure we don't increase the number of shuffles - we must merge a
        // shuffle from at least one of the LHS and RHS ops.
        bool MergedLeft = false;
        SDValue LeftSV0, LeftSV1;
        SmallVector<int, 4> LeftMask;
        if (CanMergeInnerShuffle(LeftSV0, LeftSV1, LeftMask, true, false) ||
            CanMergeInnerShuffle(LeftSV0, LeftSV1, LeftMask, true, true)) {
          MergedLeft = true;
        } else {
          LeftMask.assign(SVN->getMask().begin(), SVN->getMask().end());
          LeftSV0 = Op00, LeftSV1 = Op10;
        }

        bool MergedRight = false;
        SDValue RightSV0, RightSV1;
        SmallVector<int, 4> RightMask;
        if (CanMergeInnerShuffle(RightSV0, RightSV1, RightMask, false, false) ||
            CanMergeInnerShuffle(RightSV0, RightSV1, RightMask, false, true)) {
          MergedRight = true;
        } else {
          RightMask.assign(SVN->getMask().begin(), SVN->getMask().end());
          RightSV0 = Op01, RightSV1 = Op11;
        }

        if (MergedLeft || MergedRight) {
          SDLoc DL(N);
          SDValue LHS = DAG.getVectorShuffle(
              VT, DL, LeftSV0 ? LeftSV0 : DAG.getUNDEF(VT),
              LeftSV1 ? LeftSV1 : DAG.getUNDEF(VT), LeftMask);
          SDValue RHS = DAG.getVectorShuffle(
              VT, DL, RightSV0 ? RightSV0 : DAG.getUNDEF(VT),
              RightSV1 ? RightSV1 : DAG.getUNDEF(VT), RightMask);
          return DAG.getNode(SrcOpcode, DL, VT, LHS, RHS);
        }
      }
    }
  }

  if (SDValue V = foldShuffleOfConcatUndefs(SVN, DAG))
    return V;

  return SDValue();
}

SDValue DAGCombiner::visitSCALAR_TO_VECTOR(SDNode *N) {
  SDValue InVal = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // Replace a SCALAR_TO_VECTOR(EXTRACT_VECTOR_ELT(V,C0)) pattern
  // with a VECTOR_SHUFFLE and possible truncate.
  if (InVal.getOpcode() == ISD::EXTRACT_VECTOR_ELT &&
      VT.isFixedLengthVector() &&
      InVal->getOperand(0).getValueType().isFixedLengthVector()) {
    SDValue InVec = InVal->getOperand(0);
    SDValue EltNo = InVal->getOperand(1);
    auto InVecT = InVec.getValueType();
    if (ConstantSDNode *C0 = dyn_cast<ConstantSDNode>(EltNo)) {
      SmallVector<int, 8> NewMask(InVecT.getVectorNumElements(), -1);
      int Elt = C0->getZExtValue();
      NewMask[0] = Elt;
      // If we have an implict truncate do truncate here as long as it's legal.
      // if it's not legal, this should
      if (VT.getScalarType() != InVal.getValueType() &&
          InVal.getValueType().isScalarInteger() &&
          isTypeLegal(VT.getScalarType())) {
        SDValue Val =
            DAG.getNode(ISD::TRUNCATE, SDLoc(InVal), VT.getScalarType(), InVal);
        return DAG.getNode(ISD::SCALAR_TO_VECTOR, SDLoc(N), VT, Val);
      }
      if (VT.getScalarType() == InVecT.getScalarType() &&
          VT.getVectorNumElements() <= InVecT.getVectorNumElements()) {
        SDValue LegalShuffle =
          TLI.buildLegalVectorShuffle(InVecT, SDLoc(N), InVec,
                                      DAG.getUNDEF(InVecT), NewMask, DAG);
        if (LegalShuffle) {
          // If the initial vector is the correct size this shuffle is a
          // valid result.
          if (VT == InVecT)
            return LegalShuffle;
          // If not we must truncate the vector.
          if (VT.getVectorNumElements() != InVecT.getVectorNumElements()) {
            SDValue ZeroIdx = DAG.getVectorIdxConstant(0, SDLoc(N));
            EVT SubVT = EVT::getVectorVT(*DAG.getContext(),
                                         InVecT.getVectorElementType(),
                                         VT.getVectorNumElements());
            return DAG.getNode(ISD::EXTRACT_SUBVECTOR, SDLoc(N), SubVT,
                               LegalShuffle, ZeroIdx);
          }
        }
      }
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitINSERT_SUBVECTOR(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  uint64_t InsIdx = N->getConstantOperandVal(2);

  // If inserting an UNDEF, just return the original vector.
  if (N1.isUndef())
    return N0;

  // If this is an insert of an extracted vector into an undef vector, we can
  // just use the input to the extract.
  if (N0.isUndef() && N1.getOpcode() == ISD::EXTRACT_SUBVECTOR &&
      N1.getOperand(1) == N2 && N1.getOperand(0).getValueType() == VT)
    return N1.getOperand(0);

  // If we are inserting a bitcast value into an undef, with the same
  // number of elements, just use the bitcast input of the extract.
  // i.e. INSERT_SUBVECTOR UNDEF (BITCAST N1) N2 ->
  //        BITCAST (INSERT_SUBVECTOR UNDEF N1 N2)
  if (N0.isUndef() && N1.getOpcode() == ISD::BITCAST &&
      N1.getOperand(0).getOpcode() == ISD::EXTRACT_SUBVECTOR &&
      N1.getOperand(0).getOperand(1) == N2 &&
      N1.getOperand(0).getOperand(0).getValueType().getVectorElementCount() ==
          VT.getVectorElementCount() &&
      N1.getOperand(0).getOperand(0).getValueType().getSizeInBits() ==
          VT.getSizeInBits()) {
    return DAG.getBitcast(VT, N1.getOperand(0).getOperand(0));
  }

  // If both N1 and N2 are bitcast values on which insert_subvector
  // would makes sense, pull the bitcast through.
  // i.e. INSERT_SUBVECTOR (BITCAST N0) (BITCAST N1) N2 ->
  //        BITCAST (INSERT_SUBVECTOR N0 N1 N2)
  if (N0.getOpcode() == ISD::BITCAST && N1.getOpcode() == ISD::BITCAST) {
    SDValue CN0 = N0.getOperand(0);
    SDValue CN1 = N1.getOperand(0);
    EVT CN0VT = CN0.getValueType();
    EVT CN1VT = CN1.getValueType();
    if (CN0VT.isVector() && CN1VT.isVector() &&
        CN0VT.getVectorElementType() == CN1VT.getVectorElementType() &&
        CN0VT.getVectorElementCount() == VT.getVectorElementCount()) {
      SDValue NewINSERT = DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N),
                                      CN0.getValueType(), CN0, CN1, N2);
      return DAG.getBitcast(VT, NewINSERT);
    }
  }

  // Combine INSERT_SUBVECTORs where we are inserting to the same index.
  // INSERT_SUBVECTOR( INSERT_SUBVECTOR( Vec, SubOld, Idx ), SubNew, Idx )
  // --> INSERT_SUBVECTOR( Vec, SubNew, Idx )
  if (N0.getOpcode() == ISD::INSERT_SUBVECTOR &&
      N0.getOperand(1).getValueType() == N1.getValueType() &&
      N0.getOperand(2) == N2)
    return DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), VT, N0.getOperand(0),
                       N1, N2);

  // Eliminate an intermediate insert into an undef vector:
  // insert_subvector undef, (insert_subvector undef, X, 0), N2 -->
  // insert_subvector undef, X, N2
  if (N0.isUndef() && N1.getOpcode() == ISD::INSERT_SUBVECTOR &&
      N1.getOperand(0).isUndef() && isNullConstant(N1.getOperand(2)))
    return DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), VT, N0,
                       N1.getOperand(1), N2);

  // Push subvector bitcasts to the output, adjusting the index as we go.
  // insert_subvector(bitcast(v), bitcast(s), c1)
  // -> bitcast(insert_subvector(v, s, c2))
  if ((N0.isUndef() || N0.getOpcode() == ISD::BITCAST) &&
      N1.getOpcode() == ISD::BITCAST) {
    SDValue N0Src = peekThroughBitcasts(N0);
    SDValue N1Src = peekThroughBitcasts(N1);
    EVT N0SrcSVT = N0Src.getValueType().getScalarType();
    EVT N1SrcSVT = N1Src.getValueType().getScalarType();
    if ((N0.isUndef() || N0SrcSVT == N1SrcSVT) &&
        N0Src.getValueType().isVector() && N1Src.getValueType().isVector()) {
      EVT NewVT;
      SDLoc DL(N);
      SDValue NewIdx;
      LLVMContext &Ctx = *DAG.getContext();
      ElementCount NumElts = VT.getVectorElementCount();
      unsigned EltSizeInBits = VT.getScalarSizeInBits();
      if ((EltSizeInBits % N1SrcSVT.getSizeInBits()) == 0) {
        unsigned Scale = EltSizeInBits / N1SrcSVT.getSizeInBits();
        NewVT = EVT::getVectorVT(Ctx, N1SrcSVT, NumElts * Scale);
        NewIdx = DAG.getVectorIdxConstant(InsIdx * Scale, DL);
      } else if ((N1SrcSVT.getSizeInBits() % EltSizeInBits) == 0) {
        unsigned Scale = N1SrcSVT.getSizeInBits() / EltSizeInBits;
        if (NumElts.isKnownMultipleOf(Scale) && (InsIdx % Scale) == 0) {
          NewVT = EVT::getVectorVT(Ctx, N1SrcSVT,
                                   NumElts.divideCoefficientBy(Scale));
          NewIdx = DAG.getVectorIdxConstant(InsIdx / Scale, DL);
        }
      }
      if (NewIdx && hasOperation(ISD::INSERT_SUBVECTOR, NewVT)) {
        SDValue Res = DAG.getBitcast(NewVT, N0Src);
        Res = DAG.getNode(ISD::INSERT_SUBVECTOR, DL, NewVT, Res, N1Src, NewIdx);
        return DAG.getBitcast(VT, Res);
      }
    }
  }

  // Canonicalize insert_subvector dag nodes.
  // Example:
  // (insert_subvector (insert_subvector A, Idx0), Idx1)
  // -> (insert_subvector (insert_subvector A, Idx1), Idx0)
  if (N0.getOpcode() == ISD::INSERT_SUBVECTOR && N0.hasOneUse() &&
      N1.getValueType() == N0.getOperand(1).getValueType()) {
    unsigned OtherIdx = N0.getConstantOperandVal(2);
    if (InsIdx < OtherIdx) {
      // Swap nodes.
      SDValue NewOp = DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N), VT,
                                  N0.getOperand(0), N1, N2);
      AddToWorklist(NewOp.getNode());
      return DAG.getNode(ISD::INSERT_SUBVECTOR, SDLoc(N0.getNode()),
                         VT, NewOp, N0.getOperand(1), N0.getOperand(2));
    }
  }

  // If the input vector is a concatenation, and the insert replaces
  // one of the pieces, we can optimize into a single concat_vectors.
  if (N0.getOpcode() == ISD::CONCAT_VECTORS && N0.hasOneUse() &&
      N0.getOperand(0).getValueType() == N1.getValueType() &&
      N0.getOperand(0).getValueType().isScalableVector() ==
          N1.getValueType().isScalableVector()) {
    unsigned Factor = N1.getValueType().getVectorMinNumElements();
    SmallVector<SDValue, 8> Ops(N0->op_begin(), N0->op_end());
    Ops[InsIdx / Factor] = N1;
    return DAG.getNode(ISD::CONCAT_VECTORS, SDLoc(N), VT, Ops);
  }

  // Simplify source operands based on insertion.
  if (SimplifyDemandedVectorElts(SDValue(N, 0)))
    return SDValue(N, 0);

  return SDValue();
}

SDValue DAGCombiner::visitFP_TO_FP16(SDNode *N) {
  SDValue N0 = N->getOperand(0);

  // fold (fp_to_fp16 (fp16_to_fp op)) -> op
  if (N0->getOpcode() == ISD::FP16_TO_FP)
    return N0->getOperand(0);

  return SDValue();
}

SDValue DAGCombiner::visitFP16_TO_FP(SDNode *N) {
  SDValue N0 = N->getOperand(0);

  // fold fp16_to_fp(op & 0xffff) -> fp16_to_fp(op)
  if (!TLI.shouldKeepZExtForFP16Conv() && N0->getOpcode() == ISD::AND) {
    ConstantSDNode *AndConst = getAsNonOpaqueConstant(N0.getOperand(1));
    if (AndConst && AndConst->getAPIntValue() == 0xffff) {
      return DAG.getNode(ISD::FP16_TO_FP, SDLoc(N), N->getValueType(0),
                         N0.getOperand(0));
    }
  }

  return SDValue();
}

SDValue DAGCombiner::visitVECREDUCE(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N0.getValueType();
  unsigned Opcode = N->getOpcode();

  // VECREDUCE over 1-element vector is just an extract.
  if (VT.getVectorElementCount().isScalar()) {
    SDLoc dl(N);
    SDValue Res =
        DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, VT.getVectorElementType(), N0,
                    DAG.getVectorIdxConstant(0, dl));
    if (Res.getValueType() != N->getValueType(0))
      Res = DAG.getNode(ISD::ANY_EXTEND, dl, N->getValueType(0), Res);
    return Res;
  }

  // On an boolean vector an and/or reduction is the same as a umin/umax
  // reduction. Convert them if the latter is legal while the former isn't.
  if (Opcode == ISD::VECREDUCE_AND || Opcode == ISD::VECREDUCE_OR) {
    unsigned NewOpcode = Opcode == ISD::VECREDUCE_AND
        ? ISD::VECREDUCE_UMIN : ISD::VECREDUCE_UMAX;
    if (!TLI.isOperationLegalOrCustom(Opcode, VT) &&
        TLI.isOperationLegalOrCustom(NewOpcode, VT) &&
        DAG.ComputeNumSignBits(N0) == VT.getScalarSizeInBits())
      return DAG.getNode(NewOpcode, SDLoc(N), N->getValueType(0), N0);
  }

  return SDValue();
}

/// Returns a vector_shuffle if it able to transform an AND to a vector_shuffle
/// with the destination vector and a zero vector.
/// e.g. AND V, <0xffffffff, 0, 0xffffffff, 0>. ==>
///      vector_shuffle V, Zero, <0, 4, 2, 4>
SDValue DAGCombiner::XformToShuffleWithZero(SDNode *N) {
  assert(N->getOpcode() == ISD::AND && "Unexpected opcode!");

  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = peekThroughBitcasts(N->getOperand(1));
  SDLoc DL(N);

  // Make sure we're not running after operation legalization where it
  // may have custom lowered the vector shuffles.
  if (LegalOperations)
    return SDValue();

  if (RHS.getOpcode() != ISD::BUILD_VECTOR)
    return SDValue();

  EVT RVT = RHS.getValueType();
  unsigned NumElts = RHS.getNumOperands();

  // Attempt to create a valid clear mask, splitting the mask into
  // sub elements and checking to see if each is
  // all zeros or all ones - suitable for shuffle masking.
  auto BuildClearMask = [&](int Split) {
    int NumSubElts = NumElts * Split;
    int NumSubBits = RVT.getScalarSizeInBits() / Split;

    SmallVector<int, 8> Indices;
    for (int i = 0; i != NumSubElts; ++i) {
      int EltIdx = i / Split;
      int SubIdx = i % Split;
      SDValue Elt = RHS.getOperand(EltIdx);
      // X & undef --> 0 (not undef). So this lane must be converted to choose
      // from the zero constant vector (same as if the element had all 0-bits).
      if (Elt.isUndef()) {
        Indices.push_back(i + NumSubElts);
        continue;
      }

      APInt Bits;
      if (isa<ConstantSDNode>(Elt))
        Bits = cast<ConstantSDNode>(Elt)->getAPIntValue();
      else if (isa<ConstantFPSDNode>(Elt))
        Bits = cast<ConstantFPSDNode>(Elt)->getValueAPF().bitcastToAPInt();
      else
        return SDValue();

      // Extract the sub element from the constant bit mask.
      if (DAG.getDataLayout().isBigEndian())
        Bits = Bits.extractBits(NumSubBits, (Split - SubIdx - 1) * NumSubBits);
      else
        Bits = Bits.extractBits(NumSubBits, SubIdx * NumSubBits);

      if (Bits.isAllOnesValue())
        Indices.push_back(i);
      else if (Bits == 0)
        Indices.push_back(i + NumSubElts);
      else
        return SDValue();
    }

    // Let's see if the target supports this vector_shuffle.
    EVT ClearSVT = EVT::getIntegerVT(*DAG.getContext(), NumSubBits);
    EVT ClearVT = EVT::getVectorVT(*DAG.getContext(), ClearSVT, NumSubElts);
    if (!TLI.isVectorClearMaskLegal(Indices, ClearVT))
      return SDValue();

    SDValue Zero = DAG.getConstant(0, DL, ClearVT);
    return DAG.getBitcast(VT, DAG.getVectorShuffle(ClearVT, DL,
                                                   DAG.getBitcast(ClearVT, LHS),
                                                   Zero, Indices));
  };

  // Determine maximum split level (byte level masking).
  int MaxSplit = 1;
  if (RVT.getScalarSizeInBits() % 8 == 0)
    MaxSplit = RVT.getScalarSizeInBits() / 8;

  for (int Split = 1; Split <= MaxSplit; ++Split)
    if (RVT.getScalarSizeInBits() % Split == 0)
      if (SDValue S = BuildClearMask(Split))
        return S;

  return SDValue();
}

/// If a vector binop is performed on splat values, it may be profitable to
/// extract, scalarize, and insert/splat.
static SDValue scalarizeBinOpOfSplats(SDNode *N, SelectionDAG &DAG) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  unsigned Opcode = N->getOpcode();
  EVT VT = N->getValueType(0);
  EVT EltVT = VT.getVectorElementType();
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // TODO: Remove/replace the extract cost check? If the elements are available
  //       as scalars, then there may be no extract cost. Should we ask if
  //       inserting a scalar back into a vector is cheap instead?
  int Index0, Index1;
  SDValue Src0 = DAG.getSplatSourceVector(N0, Index0);
  SDValue Src1 = DAG.getSplatSourceVector(N1, Index1);
  if (!Src0 || !Src1 || Index0 != Index1 ||
      Src0.getValueType().getVectorElementType() != EltVT ||
      Src1.getValueType().getVectorElementType() != EltVT ||
      !TLI.isExtractVecEltCheap(VT, Index0) ||
      !TLI.isOperationLegalOrCustom(Opcode, EltVT))
    return SDValue();

  SDLoc DL(N);
  SDValue IndexC = DAG.getVectorIdxConstant(Index0, DL);
  SDValue X = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Src0, IndexC);
  SDValue Y = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, EltVT, Src1, IndexC);
  SDValue ScalarBO = DAG.getNode(Opcode, DL, EltVT, X, Y, N->getFlags());

  // If all lanes but 1 are undefined, no need to splat the scalar result.
  // TODO: Keep track of undefs and use that info in the general case.
  if (N0.getOpcode() == ISD::BUILD_VECTOR && N0.getOpcode() == N1.getOpcode() &&
      count_if(N0->ops(), [](SDValue V) { return !V.isUndef(); }) == 1 &&
      count_if(N1->ops(), [](SDValue V) { return !V.isUndef(); }) == 1) {
    // bo (build_vec ..undef, X, undef...), (build_vec ..undef, Y, undef...) -->
    // build_vec ..undef, (bo X, Y), undef...
    SmallVector<SDValue, 8> Ops(VT.getVectorNumElements(), DAG.getUNDEF(EltVT));
    Ops[Index0] = ScalarBO;
    return DAG.getBuildVector(VT, DL, Ops);
  }

  // bo (splat X, Index), (splat Y, Index) --> splat (bo X, Y), Index
  SmallVector<SDValue, 8> Ops(VT.getVectorNumElements(), ScalarBO);
  return DAG.getBuildVector(VT, DL, Ops);
}

/// Visit a binary vector operation, like ADD.
SDValue DAGCombiner::SimplifyVBinOp(SDNode *N) {
  assert(N->getValueType(0).isVector() &&
         "SimplifyVBinOp only works on vectors!");

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue Ops[] = {LHS, RHS};
  EVT VT = N->getValueType(0);
  unsigned Opcode = N->getOpcode();
  SDNodeFlags Flags = N->getFlags();

  // See if we can constant fold the vector operation.
  if (SDValue Fold = DAG.FoldConstantVectorArithmetic(
          Opcode, SDLoc(LHS), LHS.getValueType(), Ops, N->getFlags()))
    return Fold;

  // Move unary shuffles with identical masks after a vector binop:
  // VBinOp (shuffle A, Undef, Mask), (shuffle B, Undef, Mask))
  //   --> shuffle (VBinOp A, B), Undef, Mask
  // This does not require type legality checks because we are creating the
  // same types of operations that are in the original sequence. We do have to
  // restrict ops like integer div that have immediate UB (eg, div-by-zero)
  // though. This code is adapted from the identical transform in instcombine.
  if (Opcode != ISD::UDIV && Opcode != ISD::SDIV &&
      Opcode != ISD::UREM && Opcode != ISD::SREM &&
      Opcode != ISD::UDIVREM && Opcode != ISD::SDIVREM) {
    auto *Shuf0 = dyn_cast<ShuffleVectorSDNode>(LHS);
    auto *Shuf1 = dyn_cast<ShuffleVectorSDNode>(RHS);
    if (Shuf0 && Shuf1 && Shuf0->getMask().equals(Shuf1->getMask()) &&
        LHS.getOperand(1).isUndef() && RHS.getOperand(1).isUndef() &&
        (LHS.hasOneUse() || RHS.hasOneUse() || LHS == RHS)) {
      SDLoc DL(N);
      SDValue NewBinOp = DAG.getNode(Opcode, DL, VT, LHS.getOperand(0),
                                     RHS.getOperand(0), Flags);
      SDValue UndefV = LHS.getOperand(1);
      return DAG.getVectorShuffle(VT, DL, NewBinOp, UndefV, Shuf0->getMask());
    }

    // Try to sink a splat shuffle after a binop with a uniform constant.
    // This is limited to cases where neither the shuffle nor the constant have
    // undefined elements because that could be poison-unsafe or inhibit
    // demanded elements analysis. It is further limited to not change a splat
    // of an inserted scalar because that may be optimized better by
    // load-folding or other target-specific behaviors.
    if (isConstOrConstSplat(RHS) && Shuf0 && is_splat(Shuf0->getMask()) &&
        Shuf0->hasOneUse() && Shuf0->getOperand(1).isUndef() &&
        Shuf0->getOperand(0).getOpcode() != ISD::INSERT_VECTOR_ELT) {
      // binop (splat X), (splat C) --> splat (binop X, C)
      SDLoc DL(N);
      SDValue X = Shuf0->getOperand(0);
      SDValue NewBinOp = DAG.getNode(Opcode, DL, VT, X, RHS, Flags);
      return DAG.getVectorShuffle(VT, DL, NewBinOp, DAG.getUNDEF(VT),
                                  Shuf0->getMask());
    }
    if (isConstOrConstSplat(LHS) && Shuf1 && is_splat(Shuf1->getMask()) &&
        Shuf1->hasOneUse() && Shuf1->getOperand(1).isUndef() &&
        Shuf1->getOperand(0).getOpcode() != ISD::INSERT_VECTOR_ELT) {
      // binop (splat C), (splat X) --> splat (binop C, X)
      SDLoc DL(N);
      SDValue X = Shuf1->getOperand(0);
      SDValue NewBinOp = DAG.getNode(Opcode, DL, VT, LHS, X, Flags);
      return DAG.getVectorShuffle(VT, DL, NewBinOp, DAG.getUNDEF(VT),
                                  Shuf1->getMask());
    }
  }

  // The following pattern is likely to emerge with vector reduction ops. Moving
  // the binary operation ahead of insertion may allow using a narrower vector
  // instruction that has better performance than the wide version of the op:
  // VBinOp (ins undef, X, Z), (ins undef, Y, Z) --> ins VecC, (VBinOp X, Y), Z
  if (LHS.getOpcode() == ISD::INSERT_SUBVECTOR && LHS.getOperand(0).isUndef() &&
      RHS.getOpcode() == ISD::INSERT_SUBVECTOR && RHS.getOperand(0).isUndef() &&
      LHS.getOperand(2) == RHS.getOperand(2) &&
      (LHS.hasOneUse() || RHS.hasOneUse())) {
    SDValue X = LHS.getOperand(1);
    SDValue Y = RHS.getOperand(1);
    SDValue Z = LHS.getOperand(2);
    EVT NarrowVT = X.getValueType();
    if (NarrowVT == Y.getValueType() &&
        TLI.isOperationLegalOrCustomOrPromote(Opcode, NarrowVT,
                                              LegalOperations)) {
      // (binop undef, undef) may not return undef, so compute that result.
      SDLoc DL(N);
      SDValue VecC =
          DAG.getNode(Opcode, DL, VT, DAG.getUNDEF(VT), DAG.getUNDEF(VT));
      SDValue NarrowBO = DAG.getNode(Opcode, DL, NarrowVT, X, Y);
      return DAG.getNode(ISD::INSERT_SUBVECTOR, DL, VT, VecC, NarrowBO, Z);
    }
  }

  // Make sure all but the first op are undef or constant.
  auto ConcatWithConstantOrUndef = [](SDValue Concat) {
    return Concat.getOpcode() == ISD::CONCAT_VECTORS &&
           all_of(drop_begin(Concat->ops()), [](const SDValue &Op) {
             return Op.isUndef() ||
                    ISD::isBuildVectorOfConstantSDNodes(Op.getNode());
           });
  };

  // The following pattern is likely to emerge with vector reduction ops. Moving
  // the binary operation ahead of the concat may allow using a narrower vector
  // instruction that has better performance than the wide version of the op:
  // VBinOp (concat X, undef/constant), (concat Y, undef/constant) -->
  //   concat (VBinOp X, Y), VecC
  if (ConcatWithConstantOrUndef(LHS) && ConcatWithConstantOrUndef(RHS) &&
      (LHS.hasOneUse() || RHS.hasOneUse())) {
    EVT NarrowVT = LHS.getOperand(0).getValueType();
    if (NarrowVT == RHS.getOperand(0).getValueType() &&
        TLI.isOperationLegalOrCustomOrPromote(Opcode, NarrowVT)) {
      SDLoc DL(N);
      unsigned NumOperands = LHS.getNumOperands();
      SmallVector<SDValue, 4> ConcatOps;
      for (unsigned i = 0; i != NumOperands; ++i) {
        // This constant fold for operands 1 and up.
        ConcatOps.push_back(DAG.getNode(Opcode, DL, NarrowVT, LHS.getOperand(i),
                                        RHS.getOperand(i)));
      }

      return DAG.getNode(ISD::CONCAT_VECTORS, DL, VT, ConcatOps);
    }
  }

  if (SDValue V = scalarizeBinOpOfSplats(N, DAG))
    return V;

  return SDValue();
}

SDValue DAGCombiner::SimplifySelect(const SDLoc &DL, SDValue N0, SDValue N1,
                                    SDValue N2) {
  assert(N0.getOpcode() ==ISD::SETCC && "First argument must be a SetCC node!");

  SDValue SCC = SimplifySelectCC(DL, N0.getOperand(0), N0.getOperand(1), N1, N2,
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get());

  // If we got a simplified select_cc node back from SimplifySelectCC, then
  // break it down into a new SETCC node, and a new SELECT node, and then return
  // the SELECT node, since we were called with a SELECT node.
  if (SCC.getNode()) {
    // Check to see if we got a select_cc back (to turn into setcc/select).
    // Otherwise, just return whatever node we got back, like fabs.
    if (SCC.getOpcode() == ISD::SELECT_CC) {
      const SDNodeFlags Flags = N0.getNode()->getFlags();
      SDValue SETCC = DAG.getNode(ISD::SETCC, SDLoc(N0),
                                  N0.getValueType(),
                                  SCC.getOperand(0), SCC.getOperand(1),
                                  SCC.getOperand(4), Flags);
      AddToWorklist(SETCC.getNode());
      SDValue SelectNode = DAG.getSelect(SDLoc(SCC), SCC.getValueType(), SETCC,
                                         SCC.getOperand(2), SCC.getOperand(3));
      SelectNode->setFlags(Flags);
      return SelectNode;
    }

    return SCC;
  }
  return SDValue();
}

/// Given a SELECT or a SELECT_CC node, where LHS and RHS are the two values
/// being selected between, see if we can simplify the select.  Callers of this
/// should assume that TheSelect is deleted if this returns true.  As such, they
/// should return the appropriate thing (e.g. the node) back to the top-level of
/// the DAG combiner loop to avoid it being looked at.
bool DAGCombiner::SimplifySelectOps(SDNode *TheSelect, SDValue LHS,
                                    SDValue RHS) {
  // fold (select (setcc x, [+-]0.0, *lt), NaN, (fsqrt x))
  // The select + setcc is redundant, because fsqrt returns NaN for X < 0.
  if (const ConstantFPSDNode *NaN = isConstOrConstSplatFP(LHS)) {
    if (NaN->isNaN() && RHS.getOpcode() == ISD::FSQRT) {
      // We have: (select (setcc ?, ?, ?), NaN, (fsqrt ?))
      SDValue Sqrt = RHS;
      ISD::CondCode CC;
      SDValue CmpLHS;
      const ConstantFPSDNode *Zero = nullptr;

      if (TheSelect->getOpcode() == ISD::SELECT_CC) {
        CC = cast<CondCodeSDNode>(TheSelect->getOperand(4))->get();
        CmpLHS = TheSelect->getOperand(0);
        Zero = isConstOrConstSplatFP(TheSelect->getOperand(1));
      } else {
        // SELECT or VSELECT
        SDValue Cmp = TheSelect->getOperand(0);
        if (Cmp.getOpcode() == ISD::SETCC) {
          CC = cast<CondCodeSDNode>(Cmp.getOperand(2))->get();
          CmpLHS = Cmp.getOperand(0);
          Zero = isConstOrConstSplatFP(Cmp.getOperand(1));
        }
      }
      if (Zero && Zero->isZero() &&
          Sqrt.getOperand(0) == CmpLHS && (CC == ISD::SETOLT ||
          CC == ISD::SETULT || CC == ISD::SETLT)) {
        // We have: (select (setcc x, [+-]0.0, *lt), NaN, (fsqrt x))
        CombineTo(TheSelect, Sqrt);
        return true;
      }
    }
  }
  // Cannot simplify select with vector condition
  if (TheSelect->getOperand(0).getValueType().isVector()) return false;

  // If this is a select from two identical things, try to pull the operation
  // through the select.
  if (LHS.getOpcode() != RHS.getOpcode() ||
      !LHS.hasOneUse() || !RHS.hasOneUse())
    return false;

  // If this is a load and the token chain is identical, replace the select
  // of two loads with a load through a select of the address to load from.
  // This triggers in things like "select bool X, 10.0, 123.0" after the FP
  // constants have been dropped into the constant pool.
  if (LHS.getOpcode() == ISD::LOAD) {
    LoadSDNode *LLD = cast<LoadSDNode>(LHS);
    LoadSDNode *RLD = cast<LoadSDNode>(RHS);

    // Token chains must be identical.
    if (LHS.getOperand(0) != RHS.getOperand(0) ||
        // Do not let this transformation reduce the number of volatile loads.
        // Be conservative for atomics for the moment
        // TODO: This does appear to be legal for unordered atomics (see D66309)
        !LLD->isSimple() || !RLD->isSimple() ||
        // FIXME: If either is a pre/post inc/dec load,
        // we'd need to split out the address adjustment.
        LLD->isIndexed() || RLD->isIndexed() ||
        // If this is an EXTLOAD, the VT's must match.
        LLD->getMemoryVT() != RLD->getMemoryVT() ||
        // If this is an EXTLOAD, the kind of extension must match.
        (LLD->getExtensionType() != RLD->getExtensionType() &&
         // The only exception is if one of the extensions is anyext.
         LLD->getExtensionType() != ISD::EXTLOAD &&
         RLD->getExtensionType() != ISD::EXTLOAD) ||
        // FIXME: this discards src value information.  This is
        // over-conservative. It would be beneficial to be able to remember
        // both potential memory locations.  Since we are discarding
        // src value info, don't do the transformation if the memory
        // locations are not in the default address space.
        LLD->getPointerInfo().getAddrSpace() != 0 ||
        RLD->getPointerInfo().getAddrSpace() != 0 ||
        // We can't produce a CMOV of a TargetFrameIndex since we won't
        // generate the address generation required.
        LLD->getBasePtr().getOpcode() == ISD::TargetFrameIndex ||
        RLD->getBasePtr().getOpcode() == ISD::TargetFrameIndex ||
        !TLI.isOperationLegalOrCustom(TheSelect->getOpcode(),
                                      LLD->getBasePtr().getValueType()))
      return false;

    // The loads must not depend on one another.
    if (LLD->isPredecessorOf(RLD) || RLD->isPredecessorOf(LLD))
      return false;

    // Check that the select condition doesn't reach either load.  If so,
    // folding this will induce a cycle into the DAG.  If not, this is safe to
    // xform, so create a select of the addresses.

    SmallPtrSet<const SDNode *, 32> Visited;
    SmallVector<const SDNode *, 16> Worklist;

    // Always fail if LLD and RLD are not independent. TheSelect is a
    // predecessor to all Nodes in question so we need not search past it.

    Visited.insert(TheSelect);
    Worklist.push_back(LLD);
    Worklist.push_back(RLD);

    if (SDNode::hasPredecessorHelper(LLD, Visited, Worklist) ||
        SDNode::hasPredecessorHelper(RLD, Visited, Worklist))
      return false;

    SDValue Addr;
    if (TheSelect->getOpcode() == ISD::SELECT) {
      // We cannot do this optimization if any pair of {RLD, LLD} is a
      // predecessor to {RLD, LLD, CondNode}. As we've already compared the
      // Loads, we only need to check if CondNode is a successor to one of the
      // loads. We can further avoid this if there's no use of their chain
      // value.
      SDNode *CondNode = TheSelect->getOperand(0).getNode();
      Worklist.push_back(CondNode);

      if ((LLD->hasAnyUseOfValue(1) &&
           SDNode::hasPredecessorHelper(LLD, Visited, Worklist)) ||
          (RLD->hasAnyUseOfValue(1) &&
           SDNode::hasPredecessorHelper(RLD, Visited, Worklist)))
        return false;

      Addr = DAG.getSelect(SDLoc(TheSelect),
                           LLD->getBasePtr().getValueType(),
                           TheSelect->getOperand(0), LLD->getBasePtr(),
                           RLD->getBasePtr());
    } else {  // Otherwise SELECT_CC
      // We cannot do this optimization if any pair of {RLD, LLD} is a
      // predecessor to {RLD, LLD, CondLHS, CondRHS}. As we've already compared
      // the Loads, we only need to check if CondLHS/CondRHS is a successor to
      // one of the loads. We can further avoid this if there's no use of their
      // chain value.

      SDNode *CondLHS = TheSelect->getOperand(0).getNode();
      SDNode *CondRHS = TheSelect->getOperand(1).getNode();
      Worklist.push_back(CondLHS);
      Worklist.push_back(CondRHS);

      if ((LLD->hasAnyUseOfValue(1) &&
           SDNode::hasPredecessorHelper(LLD, Visited, Worklist)) ||
          (RLD->hasAnyUseOfValue(1) &&
           SDNode::hasPredecessorHelper(RLD, Visited, Worklist)))
        return false;

      Addr = DAG.getNode(ISD::SELECT_CC, SDLoc(TheSelect),
                         LLD->getBasePtr().getValueType(),
                         TheSelect->getOperand(0),
                         TheSelect->getOperand(1),
                         LLD->getBasePtr(), RLD->getBasePtr(),
                         TheSelect->getOperand(4));
    }

    SDValue Load;
    // It is safe to replace the two loads if they have different alignments,
    // but the new load must be the minimum (most restrictive) alignment of the
    // inputs.
    Align Alignment = std::min(LLD->getAlign(), RLD->getAlign());
    MachineMemOperand::Flags MMOFlags = LLD->getMemOperand()->getFlags();
    if (!RLD->isInvariant())
      MMOFlags &= ~MachineMemOperand::MOInvariant;
    if (!RLD->isDereferenceable())
      MMOFlags &= ~MachineMemOperand::MODereferenceable;
    if (LLD->getExtensionType() == ISD::NON_EXTLOAD) {
      // FIXME: Discards pointer and AA info.
      Load = DAG.getLoad(TheSelect->getValueType(0), SDLoc(TheSelect),
                         LLD->getChain(), Addr, MachinePointerInfo(), Alignment,
                         MMOFlags);
    } else {
      // FIXME: Discards pointer and AA info.
      Load = DAG.getExtLoad(
          LLD->getExtensionType() == ISD::EXTLOAD ? RLD->getExtensionType()
                                                  : LLD->getExtensionType(),
          SDLoc(TheSelect), TheSelect->getValueType(0), LLD->getChain(), Addr,
          MachinePointerInfo(), LLD->getMemoryVT(), Alignment, MMOFlags);
    }

    // Users of the select now use the result of the load.
    CombineTo(TheSelect, Load);

    // Users of the old loads now use the new load's chain.  We know the
    // old-load value is dead now.
    CombineTo(LHS.getNode(), Load.getValue(0), Load.getValue(1));
    CombineTo(RHS.getNode(), Load.getValue(0), Load.getValue(1));
    return true;
  }

  return false;
}

/// Try to fold an expression of the form (N0 cond N1) ? N2 : N3 to a shift and
/// bitwise 'and'.
SDValue DAGCombiner::foldSelectCCToShiftAnd(const SDLoc &DL, SDValue N0,
                                            SDValue N1, SDValue N2, SDValue N3,
                                            ISD::CondCode CC) {
  // If this is a select where the false operand is zero and the compare is a
  // check of the sign bit, see if we can perform the "gzip trick":
  // select_cc setlt X, 0, A, 0 -> and (sra X, size(X)-1), A
  // select_cc setgt X, 0, A, 0 -> and (not (sra X, size(X)-1)), A
  EVT XType = N0.getValueType();
  EVT AType = N2.getValueType();
  if (!isNullConstant(N3) || !XType.bitsGE(AType))
    return SDValue();

  // If the comparison is testing for a positive value, we have to invert
  // the sign bit mask, so only do that transform if the target has a bitwise
  // 'and not' instruction (the invert is free).
  if (CC == ISD::SETGT && TLI.hasAndNot(N2)) {
    // (X > -1) ? A : 0
    // (X >  0) ? X : 0 <-- This is canonical signed max.
    if (!(isAllOnesConstant(N1) || (isNullConstant(N1) && N0 == N2)))
      return SDValue();
  } else if (CC == ISD::SETLT) {
    // (X <  0) ? A : 0
    // (X <  1) ? X : 0 <-- This is un-canonicalized signed min.
    if (!(isNullConstant(N1) || (isOneConstant(N1) && N0 == N2)))
      return SDValue();
  } else {
    return SDValue();
  }

  // and (sra X, size(X)-1), A -> "and (srl X, C2), A" iff A is a single-bit
  // constant.
  EVT ShiftAmtTy = getShiftAmountTy(N0.getValueType());
  auto *N2C = dyn_cast<ConstantSDNode>(N2.getNode());
  if (N2C && ((N2C->getAPIntValue() & (N2C->getAPIntValue() - 1)) == 0)) {
    unsigned ShCt = XType.getSizeInBits() - N2C->getAPIntValue().logBase2() - 1;
    if (!TLI.shouldAvoidTransformToShift(XType, ShCt)) {
      SDValue ShiftAmt = DAG.getConstant(ShCt, DL, ShiftAmtTy);
      SDValue Shift = DAG.getNode(ISD::SRL, DL, XType, N0, ShiftAmt);
      AddToWorklist(Shift.getNode());

      if (XType.bitsGT(AType)) {
        Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
        AddToWorklist(Shift.getNode());
      }

      if (CC == ISD::SETGT)
        Shift = DAG.getNOT(DL, Shift, AType);

      return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
    }
  }

  unsigned ShCt = XType.getSizeInBits() - 1;
  if (TLI.shouldAvoidTransformToShift(XType, ShCt))
    return SDValue();

  SDValue ShiftAmt = DAG.getConstant(ShCt, DL, ShiftAmtTy);
  SDValue Shift = DAG.getNode(ISD::SRA, DL, XType, N0, ShiftAmt);
  AddToWorklist(Shift.getNode());

  if (XType.bitsGT(AType)) {
    Shift = DAG.getNode(ISD::TRUNCATE, DL, AType, Shift);
    AddToWorklist(Shift.getNode());
  }

  if (CC == ISD::SETGT)
    Shift = DAG.getNOT(DL, Shift, AType);

  return DAG.getNode(ISD::AND, DL, AType, Shift, N2);
}

// Fold select(cc, binop(), binop()) -> binop(select(), select()) etc.
SDValue DAGCombiner::foldSelectOfBinops(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  EVT VT = N->getValueType(0);
  SDLoc DL(N);

  unsigned BinOpc = N1.getOpcode();
  if (!TLI.isBinOp(BinOpc) || (N2.getOpcode() != BinOpc))
    return SDValue();

  if (!N->isOnlyUserOf(N0.getNode()) || !N->isOnlyUserOf(N1.getNode()))
    return SDValue();

  // Fold select(cond, binop(x, y), binop(z, y))
  //  --> binop(select(cond, x, z), y)
  if (N1.getOperand(1) == N2.getOperand(1)) {
    SDValue NewSel =
        DAG.getSelect(DL, VT, N0, N1.getOperand(0), N2.getOperand(0));
    SDValue NewBinOp = DAG.getNode(BinOpc, DL, VT, NewSel, N1.getOperand(1));
    NewBinOp->setFlags(N1->getFlags());
    NewBinOp->intersectFlagsWith(N2->getFlags());
    return NewBinOp;
  }

  // Fold select(cond, binop(x, y), binop(x, z))
  //  --> binop(x, select(cond, y, z))
  // Second op VT might be different (e.g. shift amount type)
  if (N1.getOperand(0) == N2.getOperand(0) &&
      VT == N1.getOperand(1).getValueType() &&
      VT == N2.getOperand(1).getValueType()) {
    SDValue NewSel =
        DAG.getSelect(DL, VT, N0, N1.getOperand(1), N2.getOperand(1));
    SDValue NewBinOp = DAG.getNode(BinOpc, DL, VT, N1.getOperand(0), NewSel);
    NewBinOp->setFlags(N1->getFlags());
    NewBinOp->intersectFlagsWith(N2->getFlags());
    return NewBinOp;
  }

  // TODO: Handle isCommutativeBinOp patterns as well?
  return SDValue();
}

// Transform (fneg/fabs (bitconvert x)) to avoid loading constant pool values.
SDValue DAGCombiner::foldSignChangeInBitcast(SDNode *N) {
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);
  bool IsFabs = N->getOpcode() == ISD::FABS;
  bool IsFree = IsFabs ? TLI.isFAbsFree(VT) : TLI.isFNegFree(VT);

  if (IsFree || N0.getOpcode() != ISD::BITCAST || !N0.hasOneUse())
    return SDValue();

  SDValue Int = N0.getOperand(0);
  EVT IntVT = Int.getValueType();

  // The operand to cast should be integer.
  if (!IntVT.isInteger() || IntVT.isVector())
    return SDValue();

  // (fneg (bitconvert x)) -> (bitconvert (xor x sign))
  // (fabs (bitconvert x)) -> (bitconvert (and x ~sign))
  APInt SignMask;
  if (N0.getValueType().isVector()) {
    // For vector, create a sign mask (0x80...) or its inverse (for fabs,
    // 0x7f...) per element and splat it.
    SignMask = APInt::getSignMask(N0.getScalarValueSizeInBits());
    if (IsFabs)
      SignMask = ~SignMask;
    SignMask = APInt::getSplat(IntVT.getSizeInBits(), SignMask);
  } else {
    // For scalar, just use the sign mask (0x80... or the inverse, 0x7f...)
    SignMask = APInt::getSignMask(IntVT.getSizeInBits());
    if (IsFabs)
      SignMask = ~SignMask;
  }
  SDLoc DL(N0);
  Int = DAG.getNode(IsFabs ? ISD::AND : ISD::XOR, DL, IntVT, Int,
                    DAG.getConstant(SignMask, DL, IntVT));
  AddToWorklist(Int.getNode());
  return DAG.getBitcast(VT, Int);
}

/// Turn "(a cond b) ? 1.0f : 2.0f" into "load (tmp + ((a cond b) ? 0 : 4)"
/// where "tmp" is a constant pool entry containing an array with 1.0 and 2.0
/// in it. This may be a win when the constant is not otherwise available
/// because it replaces two constant pool loads with one.
SDValue DAGCombiner::convertSelectOfFPConstantsToLoadOffset(
    const SDLoc &DL, SDValue N0, SDValue N1, SDValue N2, SDValue N3,
    ISD::CondCode CC) {
  if (!TLI.reduceSelectOfFPConstantLoads(N0.getValueType()))
    return SDValue();

  // If we are before legalize types, we want the other legalization to happen
  // first (for example, to avoid messing with soft float).
  auto *TV = dyn_cast<ConstantFPSDNode>(N2);
  auto *FV = dyn_cast<ConstantFPSDNode>(N3);
  EVT VT = N2.getValueType();
  if (!TV || !FV || !TLI.isTypeLegal(VT))
    return SDValue();

  // If a constant can be materialized without loads, this does not make sense.
  if (TLI.getOperationAction(ISD::ConstantFP, VT) == TargetLowering::Legal ||
      TLI.isFPImmLegal(TV->getValueAPF(), TV->getValueType(0), ForCodeSize) ||
      TLI.isFPImmLegal(FV->getValueAPF(), FV->getValueType(0), ForCodeSize))
    return SDValue();

  // If both constants have multiple uses, then we won't need to do an extra
  // load. The values are likely around in registers for other users.
  if (!TV->hasOneUse() && !FV->hasOneUse())
    return SDValue();

  Constant *Elts[] = { const_cast<ConstantFP*>(FV->getConstantFPValue()),
                       const_cast<ConstantFP*>(TV->getConstantFPValue()) };
  Type *FPTy = Elts[0]->getType();
  const DataLayout &TD = DAG.getDataLayout();

  // Create a ConstantArray of the two constants.
  Constant *CA = ConstantArray::get(ArrayType::get(FPTy, 2), Elts);
  SDValue CPIdx = DAG.getConstantPool(CA, TLI.getPointerTy(DAG.getDataLayout()),
                                      TD.getPrefTypeAlign(FPTy));
  Align Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlign();

  // Get offsets to the 0 and 1 elements of the array, so we can select between
  // them.
  SDValue Zero = DAG.getIntPtrConstant(0, DL);
  unsigned EltSize = (unsigned)TD.getTypeAllocSize(Elts[0]->getType());
  SDValue One = DAG.getIntPtrConstant(EltSize, SDLoc(FV));
  SDValue Cond =
      DAG.getSetCC(DL, getSetCCResultType(N0.getValueType()), N0, N1, CC);
  AddToWorklist(Cond.getNode());
  SDValue CstOffset = DAG.getSelect(DL, Zero.getValueType(), Cond, One, Zero);
  AddToWorklist(CstOffset.getNode());
  CPIdx = DAG.getNode(ISD::ADD, DL, CPIdx.getValueType(), CPIdx, CstOffset);
  AddToWorklist(CPIdx.getNode());
  return DAG.getLoad(TV->getValueType(0), DL, DAG.getEntryNode(), CPIdx,
                     MachinePointerInfo::getConstantPool(
                         DAG.getMachineFunction()), Alignment);
}

/// Simplify an expression of the form (N0 cond N1) ? N2 : N3
/// where 'cond' is the comparison specified by CC.
SDValue DAGCombiner::SimplifySelectCC(const SDLoc &DL, SDValue N0, SDValue N1,
                                      SDValue N2, SDValue N3, ISD::CondCode CC,
                                      bool NotExtCompare) {
  // (x ? y : y) -> y.
  if (N2 == N3) return N2;

  EVT CmpOpVT = N0.getValueType();
  EVT CmpResVT = getSetCCResultType(CmpOpVT);
  EVT VT = N2.getValueType();
  auto *N1C = dyn_cast<ConstantSDNode>(N1.getNode());
  auto *N2C = dyn_cast<ConstantSDNode>(N2.getNode());
  auto *N3C = dyn_cast<ConstantSDNode>(N3.getNode());

  // Determine if the condition we're dealing with is constant.
  if (SDValue SCC = DAG.FoldSetCC(CmpResVT, N0, N1, CC, DL)) {
    AddToWorklist(SCC.getNode());
    if (auto *SCCC = dyn_cast<ConstantSDNode>(SCC)) {
      // fold select_cc true, x, y -> x
      // fold select_cc false, x, y -> y
      return !(SCCC->isNullValue()) ? N2 : N3;
    }
  }

  if (SDValue V =
          convertSelectOfFPConstantsToLoadOffset(DL, N0, N1, N2, N3, CC))
    return V;

  if (SDValue V = foldSelectCCToShiftAnd(DL, N0, N1, N2, N3, CC))
    return V;

  // fold (select_cc seteq (and x, y), 0, 0, A) -> (and (shr (shl x)) A)
  // where y is has a single bit set.
  // A plaintext description would be, we can turn the SELECT_CC into an AND
  // when the condition can be materialized as an all-ones register.  Any
  // single bit-test can be materialized as an all-ones register with
  // shift-left and shift-right-arith.
  if (CC == ISD::SETEQ && N0->getOpcode() == ISD::AND &&
      N0->getValueType(0) == VT && isNullConstant(N1) && isNullConstant(N2)) {
    SDValue AndLHS = N0->getOperand(0);
    auto *ConstAndRHS = dyn_cast<ConstantSDNode>(N0->getOperand(1));
    if (ConstAndRHS && ConstAndRHS->getAPIntValue().countPopulation() == 1) {
      // Shift the tested bit over the sign bit.
      const APInt &AndMask = ConstAndRHS->getAPIntValue();
      unsigned ShCt = AndMask.getBitWidth() - 1;
      if (!TLI.shouldAvoidTransformToShift(VT, ShCt)) {
        SDValue ShlAmt =
          DAG.getConstant(AndMask.countLeadingZeros(), SDLoc(AndLHS),
                          getShiftAmountTy(AndLHS.getValueType()));
        SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(N0), VT, AndLHS, ShlAmt);

        // Now arithmetic right shift it all the way over, so the result is
        // either all-ones, or zero.
        SDValue ShrAmt =
          DAG.getConstant(ShCt, SDLoc(Shl),
                          getShiftAmountTy(Shl.getValueType()));
        SDValue Shr = DAG.getNode(ISD::SRA, SDLoc(N0), VT, Shl, ShrAmt);

        return DAG.getNode(ISD::AND, DL, VT, Shr, N3);
      }
    }
  }

  // fold select C, 16, 0 -> shl C, 4
  bool Fold = N2C && isNullConstant(N3) && N2C->getAPIntValue().isPowerOf2();
  bool Swap = N3C && isNullConstant(N2) && N3C->getAPIntValue().isPowerOf2();

  if ((Fold || Swap) &&
      TLI.getBooleanContents(CmpOpVT) ==
          TargetLowering::ZeroOrOneBooleanContent &&
      (!LegalOperations || TLI.isOperationLegal(ISD::SETCC, CmpOpVT))) {

    if (Swap) {
      CC = ISD::getSetCCInverse(CC, CmpOpVT);
      std::swap(N2C, N3C);
    }

    // If the caller doesn't want us to simplify this into a zext of a compare,
    // don't do it.
    if (NotExtCompare && N2C->isOne())
      return SDValue();

    SDValue Temp, SCC;
    // zext (setcc n0, n1)
    if (LegalTypes) {
      SCC = DAG.getSetCC(DL, CmpResVT, N0, N1, CC);
      if (VT.bitsLT(SCC.getValueType()))
        Temp = DAG.getZeroExtendInReg(SCC, SDLoc(N2), VT);
      else
        Temp = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N2), VT, SCC);
    } else {
      SCC = DAG.getSetCC(SDLoc(N0), MVT::i1, N0, N1, CC);
      Temp = DAG.getNode(ISD::ZERO_EXTEND, SDLoc(N2), VT, SCC);
    }

    AddToWorklist(SCC.getNode());
    AddToWorklist(Temp.getNode());

    if (N2C->isOne())
      return Temp;

    unsigned ShCt = N2C->getAPIntValue().logBase2();
    if (TLI.shouldAvoidTransformToShift(VT, ShCt))
      return SDValue();

    // shl setcc result by log2 n2c
    return DAG.getNode(ISD::SHL, DL, N2.getValueType(), Temp,
                       DAG.getConstant(ShCt, SDLoc(Temp),
                                       getShiftAmountTy(Temp.getValueType())));
  }

  // select_cc seteq X, 0, sizeof(X), ctlz(X) -> ctlz(X)
  // select_cc seteq X, 0, sizeof(X), ctlz_zero_undef(X) -> ctlz(X)
  // select_cc seteq X, 0, sizeof(X), cttz(X) -> cttz(X)
  // select_cc seteq X, 0, sizeof(X), cttz_zero_undef(X) -> cttz(X)
  // select_cc setne X, 0, ctlz(X), sizeof(X) -> ctlz(X)
  // select_cc setne X, 0, ctlz_zero_undef(X), sizeof(X) -> ctlz(X)
  // select_cc setne X, 0, cttz(X), sizeof(X) -> cttz(X)
  // select_cc setne X, 0, cttz_zero_undef(X), sizeof(X) -> cttz(X)
  if (N1C && N1C->isNullValue() && (CC == ISD::SETEQ || CC == ISD::SETNE)) {
    SDValue ValueOnZero = N2;
    SDValue Count = N3;
    // If the condition is NE instead of E, swap the operands.
    if (CC == ISD::SETNE)
      std::swap(ValueOnZero, Count);
    // Check if the value on zero is a constant equal to the bits in the type.
    if (auto *ValueOnZeroC = dyn_cast<ConstantSDNode>(ValueOnZero)) {
      if (ValueOnZeroC->getAPIntValue() == VT.getSizeInBits()) {
        // If the other operand is cttz/cttz_zero_undef of N0, and cttz is
        // legal, combine to just cttz.
        if ((Count.getOpcode() == ISD::CTTZ ||
             Count.getOpcode() == ISD::CTTZ_ZERO_UNDEF) &&
            N0 == Count.getOperand(0) &&
            (!LegalOperations || TLI.isOperationLegal(ISD::CTTZ, VT)))
          return DAG.getNode(ISD::CTTZ, DL, VT, N0);
        // If the other operand is ctlz/ctlz_zero_undef of N0, and ctlz is
        // legal, combine to just ctlz.
        if ((Count.getOpcode() == ISD::CTLZ ||
             Count.getOpcode() == ISD::CTLZ_ZERO_UNDEF) &&
            N0 == Count.getOperand(0) &&
            (!LegalOperations || TLI.isOperationLegal(ISD::CTLZ, VT)))
          return DAG.getNode(ISD::CTLZ, DL, VT, N0);
      }
    }
  }

  return SDValue();
}

/// This is a stub for TargetLowering::SimplifySetCC.
SDValue DAGCombiner::SimplifySetCC(EVT VT, SDValue N0, SDValue N1,
                                   ISD::CondCode Cond, const SDLoc &DL,
                                   bool foldBooleans) {
  TargetLowering::DAGCombinerInfo
    DagCombineInfo(DAG, Level, false, this);
  return TLI.SimplifySetCC(VT, N0, N1, Cond, foldBooleans, DagCombineInfo, DL);
}

/// Given an ISD::SDIV node expressing a divide by constant, return
/// a DAG expression to select that will generate the same value by multiplying
/// by a magic number.
/// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
SDValue DAGCombiner::BuildSDIV(SDNode *N) {
  // when optimising for minimum size, we don't want to expand a div to a mul
  // and a shift.
  if (DAG.getMachineFunction().getFunction().hasMinSize())
    return SDValue();

  SmallVector<SDNode *, 8> Built;
  if (SDValue S = TLI.BuildSDIV(N, DAG, LegalOperations, Built)) {
    for (SDNode *N : Built)
      AddToWorklist(N);
    return S;
  }

  return SDValue();
}

/// Given an ISD::SDIV node expressing a divide by constant power of 2, return a
/// DAG expression that will generate the same value by right shifting.
SDValue DAGCombiner::BuildSDIVPow2(SDNode *N) {
  ConstantSDNode *C = isConstOrConstSplat(N->getOperand(1));
  if (!C)
    return SDValue();

  // Avoid division by zero.
  if (C->isNullValue())
    return SDValue();

  SmallVector<SDNode *, 8> Built;
  if (SDValue S = TLI.BuildSDIVPow2(N, C->getAPIntValue(), DAG, Built)) {
    for (SDNode *N : Built)
      AddToWorklist(N);
    return S;
  }

  return SDValue();
}

/// Given an ISD::UDIV node expressing a divide by constant, return a DAG
/// expression that will generate the same value by multiplying by a magic
/// number.
/// Ref: "Hacker's Delight" or "The PowerPC Compiler Writer's Guide".
SDValue DAGCombiner::BuildUDIV(SDNode *N) {
  // when optimising for minimum size, we don't want to expand a div to a mul
  // and a shift.
  if (DAG.getMachineFunction().getFunction().hasMinSize())
    return SDValue();

  SmallVector<SDNode *, 8> Built;
  if (SDValue S = TLI.BuildUDIV(N, DAG, LegalOperations, Built)) {
    for (SDNode *N : Built)
      AddToWorklist(N);
    return S;
  }

  return SDValue();
}

/// Determines the LogBase2 value for a non-null input value using the
/// transform: LogBase2(V) = (EltBits - 1) - ctlz(V).
SDValue DAGCombiner::BuildLogBase2(SDValue V, const SDLoc &DL) {
  EVT VT = V.getValueType();
  SDValue Ctlz = DAG.getNode(ISD::CTLZ, DL, VT, V);
  SDValue Base = DAG.getConstant(VT.getScalarSizeInBits() - 1, DL, VT);
  SDValue LogBase2 = DAG.getNode(ISD::SUB, DL, VT, Base, Ctlz);
  return LogBase2;
}

/// Newton iteration for a function: F(X) is X_{i+1} = X_i - F(X_i)/F'(X_i)
/// For the reciprocal, we need to find the zero of the function:
///   F(X) = 1/X - A [which has a zero at X = 1/A]
///     =>
///   X_{i+1} = X_i (2 - A X_i) = X_i + X_i (1 - A X_i) [this second form
///     does not require additional intermediate precision]
/// For the last iteration, put numerator N into it to gain more precision:
///   Result = N X_i + X_i (N - N A X_i)
SDValue DAGCombiner::BuildDivEstimate(SDValue N, SDValue Op,
                                      SDNodeFlags Flags) {
  if (LegalDAG)
    return SDValue();

  // TODO: Handle half and/or extended types?
  EVT VT = Op.getValueType();
  if (VT.getScalarType() != MVT::f32 && VT.getScalarType() != MVT::f64)
    return SDValue();

  // If estimates are explicitly disabled for this function, we're done.
  MachineFunction &MF = DAG.getMachineFunction();
  int Enabled = TLI.getRecipEstimateDivEnabled(VT, MF);
  if (Enabled == TLI.ReciprocalEstimate::Disabled)
    return SDValue();

  // Estimates may be explicitly enabled for this type with a custom number of
  // refinement steps.
  int Iterations = TLI.getDivRefinementSteps(VT, MF);
  if (SDValue Est = TLI.getRecipEstimate(Op, DAG, Enabled, Iterations)) {
    AddToWorklist(Est.getNode());

    SDLoc DL(Op);
    if (Iterations) {
      SDValue FPOne = DAG.getConstantFP(1.0, DL, VT);

      // Newton iterations: Est = Est + Est (N - Arg * Est)
      // If this is the last iteration, also multiply by the numerator.
      for (int i = 0; i < Iterations; ++i) {
        SDValue MulEst = Est;

        if (i == Iterations - 1) {
          MulEst = DAG.getNode(ISD::FMUL, DL, VT, N, Est, Flags);
          AddToWorklist(MulEst.getNode());
        }

        SDValue NewEst = DAG.getNode(ISD::FMUL, DL, VT, Op, MulEst, Flags);
        AddToWorklist(NewEst.getNode());

        NewEst = DAG.getNode(ISD::FSUB, DL, VT,
                             (i == Iterations - 1 ? N : FPOne), NewEst, Flags);
        AddToWorklist(NewEst.getNode());

        NewEst = DAG.getNode(ISD::FMUL, DL, VT, Est, NewEst, Flags);
        AddToWorklist(NewEst.getNode());

        Est = DAG.getNode(ISD::FADD, DL, VT, MulEst, NewEst, Flags);
        AddToWorklist(Est.getNode());
      }
    } else {
      // If no iterations are available, multiply with N.
      Est = DAG.getNode(ISD::FMUL, DL, VT, Est, N, Flags);
      AddToWorklist(Est.getNode());
    }

    return Est;
  }

  return SDValue();
}

/// Newton iteration for a function: F(X) is X_{i+1} = X_i - F(X_i)/F'(X_i)
/// For the reciprocal sqrt, we need to find the zero of the function:
///   F(X) = 1/X^2 - A [which has a zero at X = 1/sqrt(A)]
///     =>
///   X_{i+1} = X_i (1.5 - A X_i^2 / 2)
/// As a result, we precompute A/2 prior to the iteration loop.
SDValue DAGCombiner::buildSqrtNROneConst(SDValue Arg, SDValue Est,
                                         unsigned Iterations,
                                         SDNodeFlags Flags, bool Reciprocal) {
  EVT VT = Arg.getValueType();
  SDLoc DL(Arg);
  SDValue ThreeHalves = DAG.getConstantFP(1.5, DL, VT);

  // We now need 0.5 * Arg which we can write as (1.5 * Arg - Arg) so that
  // this entire sequence requires only one FP constant.
  SDValue HalfArg = DAG.getNode(ISD::FMUL, DL, VT, ThreeHalves, Arg, Flags);
  HalfArg = DAG.getNode(ISD::FSUB, DL, VT, HalfArg, Arg, Flags);

  // Newton iterations: Est = Est * (1.5 - HalfArg * Est * Est)
  for (unsigned i = 0; i < Iterations; ++i) {
    SDValue NewEst = DAG.getNode(ISD::FMUL, DL, VT, Est, Est, Flags);
    NewEst = DAG.getNode(ISD::FMUL, DL, VT, HalfArg, NewEst, Flags);
    NewEst = DAG.getNode(ISD::FSUB, DL, VT, ThreeHalves, NewEst, Flags);
    Est = DAG.getNode(ISD::FMUL, DL, VT, Est, NewEst, Flags);
  }

  // If non-reciprocal square root is requested, multiply the result by Arg.
  if (!Reciprocal)
    Est = DAG.getNode(ISD::FMUL, DL, VT, Est, Arg, Flags);

  return Est;
}

/// Newton iteration for a function: F(X) is X_{i+1} = X_i - F(X_i)/F'(X_i)
/// For the reciprocal sqrt, we need to find the zero of the function:
///   F(X) = 1/X^2 - A [which has a zero at X = 1/sqrt(A)]
///     =>
///   X_{i+1} = (-0.5 * X_i) * (A * X_i * X_i + (-3.0))
SDValue DAGCombiner::buildSqrtNRTwoConst(SDValue Arg, SDValue Est,
                                         unsigned Iterations,
                                         SDNodeFlags Flags, bool Reciprocal) {
  EVT VT = Arg.getValueType();
  SDLoc DL(Arg);
  SDValue MinusThree = DAG.getConstantFP(-3.0, DL, VT);
  SDValue MinusHalf = DAG.getConstantFP(-0.5, DL, VT);

  // This routine must enter the loop below to work correctly
  // when (Reciprocal == false).
  assert(Iterations > 0);

  // Newton iterations for reciprocal square root:
  // E = (E * -0.5) * ((A * E) * E + -3.0)
  for (unsigned i = 0; i < Iterations; ++i) {
    SDValue AE = DAG.getNode(ISD::FMUL, DL, VT, Arg, Est, Flags);
    SDValue AEE = DAG.getNode(ISD::FMUL, DL, VT, AE, Est, Flags);
    SDValue RHS = DAG.getNode(ISD::FADD, DL, VT, AEE, MinusThree, Flags);

    // When calculating a square root at the last iteration build:
    // S = ((A * E) * -0.5) * ((A * E) * E + -3.0)
    // (notice a common subexpression)
    SDValue LHS;
    if (Reciprocal || (i + 1) < Iterations) {
      // RSQRT: LHS = (E * -0.5)
      LHS = DAG.getNode(ISD::FMUL, DL, VT, Est, MinusHalf, Flags);
    } else {
      // SQRT: LHS = (A * E) * -0.5
      LHS = DAG.getNode(ISD::FMUL, DL, VT, AE, MinusHalf, Flags);
    }

    Est = DAG.getNode(ISD::FMUL, DL, VT, LHS, RHS, Flags);
  }

  return Est;
}

/// Build code to calculate either rsqrt(Op) or sqrt(Op). In the latter case
/// Op*rsqrt(Op) is actually computed, so additional postprocessing is needed if
/// Op can be zero.
SDValue DAGCombiner::buildSqrtEstimateImpl(SDValue Op, SDNodeFlags Flags,
                                           bool Reciprocal) {
  if (LegalDAG)
    return SDValue();

  // TODO: Handle half and/or extended types?
  EVT VT = Op.getValueType();
  if (VT.getScalarType() != MVT::f32 && VT.getScalarType() != MVT::f64)
    return SDValue();

  // If estimates are explicitly disabled for this function, we're done.
  MachineFunction &MF = DAG.getMachineFunction();
  int Enabled = TLI.getRecipEstimateSqrtEnabled(VT, MF);
  if (Enabled == TLI.ReciprocalEstimate::Disabled)
    return SDValue();

  // Estimates may be explicitly enabled for this type with a custom number of
  // refinement steps.
  int Iterations = TLI.getSqrtRefinementSteps(VT, MF);

  bool UseOneConstNR = false;
  if (SDValue Est =
      TLI.getSqrtEstimate(Op, DAG, Enabled, Iterations, UseOneConstNR,
                          Reciprocal)) {
    AddToWorklist(Est.getNode());

    if (Iterations)
      Est = UseOneConstNR
            ? buildSqrtNROneConst(Op, Est, Iterations, Flags, Reciprocal)
            : buildSqrtNRTwoConst(Op, Est, Iterations, Flags, Reciprocal);
    if (!Reciprocal) {
      SDLoc DL(Op);
      // Try the target specific test first.
      SDValue Test = TLI.getSqrtInputTest(Op, DAG, DAG.getDenormalMode(VT));

      // The estimate is now completely wrong if the input was exactly 0.0 or
      // possibly a denormal. Force the answer to 0.0 or value provided by
      // target for those cases.
      Est = DAG.getNode(
          Test.getValueType().isVector() ? ISD::VSELECT : ISD::SELECT, DL, VT,
          Test, TLI.getSqrtResultForDenormInput(Op, DAG), Est);
    }
    return Est;
  }

  return SDValue();
}

SDValue DAGCombiner::buildRsqrtEstimate(SDValue Op, SDNodeFlags Flags) {
  return buildSqrtEstimateImpl(Op, Flags, true);
}

SDValue DAGCombiner::buildSqrtEstimate(SDValue Op, SDNodeFlags Flags) {
  return buildSqrtEstimateImpl(Op, Flags, false);
}

/// Return true if there is any possibility that the two addresses overlap.
bool DAGCombiner::isAlias(SDNode *Op0, SDNode *Op1) const {

  struct MemUseCharacteristics {
    bool IsVolatile;
    bool IsAtomic;
    SDValue BasePtr;
    int64_t Offset;
    Optional<int64_t> NumBytes;
    MachineMemOperand *MMO;
  };

  auto getCharacteristics = [](SDNode *N) -> MemUseCharacteristics {
    if (const auto *LSN = dyn_cast<LSBaseSDNode>(N)) {
      int64_t Offset = 0;
      if (auto *C = dyn_cast<ConstantSDNode>(LSN->getOffset()))
        Offset = (LSN->getAddressingMode() == ISD::PRE_INC)
                     ? C->getSExtValue()
                     : (LSN->getAddressingMode() == ISD::PRE_DEC)
                           ? -1 * C->getSExtValue()
                           : 0;
      uint64_t Size =
          MemoryLocation::getSizeOrUnknown(LSN->getMemoryVT().getStoreSize());
      return {LSN->isVolatile(), LSN->isAtomic(), LSN->getBasePtr(),
              Offset /*base offset*/,
              Optional<int64_t>(Size),
              LSN->getMemOperand()};
    }
    if (const auto *LN = cast<LifetimeSDNode>(N))
      return {false /*isVolatile*/, /*isAtomic*/ false, LN->getOperand(1),
              (LN->hasOffset()) ? LN->getOffset() : 0,
              (LN->hasOffset()) ? Optional<int64_t>(LN->getSize())
                                : Optional<int64_t>(),
              (MachineMemOperand *)nullptr};
    // Default.
    return {false /*isvolatile*/, /*isAtomic*/ false, SDValue(),
            (int64_t)0 /*offset*/,
            Optional<int64_t>() /*size*/, (MachineMemOperand *)nullptr};
  };

  MemUseCharacteristics MUC0 = getCharacteristics(Op0),
                        MUC1 = getCharacteristics(Op1);

  // If they are to the same address, then they must be aliases.
  if (MUC0.BasePtr.getNode() && MUC0.BasePtr == MUC1.BasePtr &&
      MUC0.Offset == MUC1.Offset)
    return true;

  // If they are both volatile then they cannot be reordered.
  if (MUC0.IsVolatile && MUC1.IsVolatile)
    return true;

  // Be conservative about atomics for the moment
  // TODO: This is way overconservative for unordered atomics (see D66309)
  if (MUC0.IsAtomic && MUC1.IsAtomic)
    return true;

  if (MUC0.MMO && MUC1.MMO) {
    if ((MUC0.MMO->isInvariant() && MUC1.MMO->isStore()) ||
        (MUC1.MMO->isInvariant() && MUC0.MMO->isStore()))
      return false;
  }

  // Try to prove that there is aliasing, or that there is no aliasing. Either
  // way, we can return now. If nothing can be proved, proceed with more tests.
  bool IsAlias;
  if (BaseIndexOffset::computeAliasing(Op0, MUC0.NumBytes, Op1, MUC1.NumBytes,
                                       DAG, IsAlias))
    return IsAlias;

  // The following all rely on MMO0 and MMO1 being valid. Fail conservatively if
  // either are not known.
  if (!MUC0.MMO || !MUC1.MMO)
    return true;

  // If one operation reads from invariant memory, and the other may store, they
  // cannot alias. These should really be checking the equivalent of mayWrite,
  // but it only matters for memory nodes other than load /store.
  if ((MUC0.MMO->isInvariant() && MUC1.MMO->isStore()) ||
      (MUC1.MMO->isInvariant() && MUC0.MMO->isStore()))
    return false;

  // If we know required SrcValue1 and SrcValue2 have relatively large
  // alignment compared to the size and offset of the access, we may be able
  // to prove they do not alias. This check is conservative for now to catch
  // cases created by splitting vector types, it only works when the offsets are
  // multiples of the size of the data.
  int64_t SrcValOffset0 = MUC0.MMO->getOffset();
  int64_t SrcValOffset1 = MUC1.MMO->getOffset();
  Align OrigAlignment0 = MUC0.MMO->getBaseAlign();
  Align OrigAlignment1 = MUC1.MMO->getBaseAlign();
  auto &Size0 = MUC0.NumBytes;
  auto &Size1 = MUC1.NumBytes;
  if (OrigAlignment0 == OrigAlignment1 && SrcValOffset0 != SrcValOffset1 &&
      Size0.hasValue() && Size1.hasValue() && *Size0 == *Size1 &&
      OrigAlignment0 > *Size0 && SrcValOffset0 % *Size0 == 0 &&
      SrcValOffset1 % *Size1 == 0) {
    int64_t OffAlign0 = SrcValOffset0 % OrigAlignment0.value();
    int64_t OffAlign1 = SrcValOffset1 % OrigAlignment1.value();

    // There is no overlap between these relatively aligned accesses of
    // similar size. Return no alias.
    if ((OffAlign0 + *Size0) <= OffAlign1 || (OffAlign1 + *Size1) <= OffAlign0)
      return false;
  }

  bool UseAA = CombinerGlobalAA.getNumOccurrences() > 0
                   ? CombinerGlobalAA
                   : DAG.getSubtarget().useAA();
#ifndef NDEBUG
  if (CombinerAAOnlyFunc.getNumOccurrences() &&
      CombinerAAOnlyFunc != DAG.getMachineFunction().getName())
    UseAA = false;
#endif

  if (UseAA && AA && MUC0.MMO->getValue() && MUC1.MMO->getValue() &&
      Size0.hasValue() && Size1.hasValue()) {
    // Use alias analysis information.
    int64_t MinOffset = std::min(SrcValOffset0, SrcValOffset1);
    int64_t Overlap0 = *Size0 + SrcValOffset0 - MinOffset;
    int64_t Overlap1 = *Size1 + SrcValOffset1 - MinOffset;
    if (AA->isNoAlias(
            MemoryLocation(MUC0.MMO->getValue(), Overlap0,
                           UseTBAA ? MUC0.MMO->getAAInfo() : AAMDNodes()),
            MemoryLocation(MUC1.MMO->getValue(), Overlap1,
                           UseTBAA ? MUC1.MMO->getAAInfo() : AAMDNodes())))
      return false;
  }

  // Otherwise we have to assume they alias.
  return true;
}

/// Walk up chain skipping non-aliasing memory nodes,
/// looking for aliasing nodes and adding them to the Aliases vector.
void DAGCombiner::GatherAllAliases(SDNode *N, SDValue OriginalChain,
                                   SmallVectorImpl<SDValue> &Aliases) {
  SmallVector<SDValue, 8> Chains;     // List of chains to visit.
  SmallPtrSet<SDNode *, 16> Visited;  // Visited node set.

  // Get alias information for node.
  // TODO: relax aliasing for unordered atomics (see D66309)
  const bool IsLoad = isa<LoadSDNode>(N) && cast<LoadSDNode>(N)->isSimple();

  // Starting off.
  Chains.push_back(OriginalChain);
  unsigned Depth = 0;

  // Attempt to improve chain by a single step
  std::function<bool(SDValue &)> ImproveChain = [&](SDValue &C) -> bool {
    switch (C.getOpcode()) {
    case ISD::EntryToken:
      // No need to mark EntryToken.
      C = SDValue();
      return true;
    case ISD::LOAD:
    case ISD::STORE: {
      // Get alias information for C.
      // TODO: Relax aliasing for unordered atomics (see D66309)
      bool IsOpLoad = isa<LoadSDNode>(C.getNode()) &&
                      cast<LSBaseSDNode>(C.getNode())->isSimple();
      if ((IsLoad && IsOpLoad) || !isAlias(N, C.getNode())) {
        // Look further up the chain.
        C = C.getOperand(0);
        return true;
      }
      // Alias, so stop here.
      return false;
    }

    case ISD::CopyFromReg:
      // Always forward past past CopyFromReg.
      C = C.getOperand(0);
      return true;

    case ISD::LIFETIME_START:
    case ISD::LIFETIME_END: {
      // We can forward past any lifetime start/end that can be proven not to
      // alias the memory access.
      if (!isAlias(N, C.getNode())) {
        // Look further up the chain.
        C = C.getOperand(0);
        return true;
      }
      return false;
    }
    default:
      return false;
    }
  };

  // Look at each chain and determine if it is an alias.  If so, add it to the
  // aliases list.  If not, then continue up the chain looking for the next
  // candidate.
  while (!Chains.empty()) {
    SDValue Chain = Chains.pop_back_val();

    // Don't bother if we've seen Chain before.
    if (!Visited.insert(Chain.getNode()).second)
      continue;

    // For TokenFactor nodes, look at each operand and only continue up the
    // chain until we reach the depth limit.
    //
    // FIXME: The depth check could be made to return the last non-aliasing
    // chain we found before we hit a tokenfactor rather than the original
    // chain.
    if (Depth > TLI.getGatherAllAliasesMaxDepth()) {
      Aliases.clear();
      Aliases.push_back(OriginalChain);
      return;
    }

    if (Chain.getOpcode() == ISD::TokenFactor) {
      // We have to check each of the operands of the token factor for "small"
      // token factors, so we queue them up.  Adding the operands to the queue
      // (stack) in reverse order maintains the original order and increases the
      // likelihood that getNode will find a matching token factor (CSE.)
      if (Chain.getNumOperands() > 16) {
        Aliases.push_back(Chain);
        continue;
      }
      for (unsigned n = Chain.getNumOperands(); n;)
        Chains.push_back(Chain.getOperand(--n));
      ++Depth;
      continue;
    }
    // Everything else
    if (ImproveChain(Chain)) {
      // Updated Chain Found, Consider new chain if one exists.
      if (Chain.getNode())
        Chains.push_back(Chain);
      ++Depth;
      continue;
    }
    // No Improved Chain Possible, treat as Alias.
    Aliases.push_back(Chain);
  }
}

/// Walk up chain skipping non-aliasing memory nodes, looking for a better chain
/// (aliasing node.)
SDValue DAGCombiner::FindBetterChain(SDNode *N, SDValue OldChain) {
  if (OptLevel == CodeGenOpt::None)
    return OldChain;

  // Ops for replacing token factor.
  SmallVector<SDValue, 8> Aliases;

  // Accumulate all the aliases to this node.
  GatherAllAliases(N, OldChain, Aliases);

  // If no operands then chain to entry token.
  if (Aliases.size() == 0)
    return DAG.getEntryNode();

  // If a single operand then chain to it.  We don't need to revisit it.
  if (Aliases.size() == 1)
    return Aliases[0];

  // Construct a custom tailored token factor.
  return DAG.getTokenFactor(SDLoc(N), Aliases);
}

namespace {
// TODO: Replace with with std::monostate when we move to C++17.
struct UnitT { } Unit;
bool operator==(const UnitT &, const UnitT &) { return true; }
bool operator!=(const UnitT &, const UnitT &) { return false; }
} // namespace

// This function tries to collect a bunch of potentially interesting
// nodes to improve the chains of, all at once. This might seem
// redundant, as this function gets called when visiting every store
// node, so why not let the work be done on each store as it's visited?
//
// I believe this is mainly important because mergeConsecutiveStores
// is unable to deal with merging stores of different sizes, so unless
// we improve the chains of all the potential candidates up-front
// before running mergeConsecutiveStores, it might only see some of
// the nodes that will eventually be candidates, and then not be able
// to go from a partially-merged state to the desired final
// fully-merged state.

bool DAGCombiner::parallelizeChainedStores(StoreSDNode *St) {
  SmallVector<StoreSDNode *, 8> ChainedStores;
  StoreSDNode *STChain = St;
  // Intervals records which offsets from BaseIndex have been covered. In
  // the common case, every store writes to the immediately previous address
  // space and thus merged with the previous interval at insertion time.

  using IMap =
      llvm::IntervalMap<int64_t, UnitT, 8, IntervalMapHalfOpenInfo<int64_t>>;
  IMap::Allocator A;
  IMap Intervals(A);

  // This holds the base pointer, index, and the offset in bytes from the base
  // pointer.
  const BaseIndexOffset BasePtr = BaseIndexOffset::match(St, DAG);

  // We must have a base and an offset.
  if (!BasePtr.getBase().getNode())
    return false;

  // Do not handle stores to undef base pointers.
  if (BasePtr.getBase().isUndef())
    return false;

  // Do not handle stores to opaque types
  if (St->getMemoryVT().isZeroSized())
    return false;

  // BaseIndexOffset assumes that offsets are fixed-size, which
  // is not valid for scalable vectors where the offsets are
  // scaled by `vscale`, so bail out early.
  if (St->getMemoryVT().isScalableVector())
    return false;

  // Add ST's interval.
  Intervals.insert(0, (St->getMemoryVT().getSizeInBits() + 7) / 8, Unit);

  while (StoreSDNode *Chain = dyn_cast<StoreSDNode>(STChain->getChain())) {
    if (Chain->getMemoryVT().isScalableVector())
      return false;

    // If the chain has more than one use, then we can't reorder the mem ops.
    if (!SDValue(Chain, 0)->hasOneUse())
      break;
    // TODO: Relax for unordered atomics (see D66309)
    if (!Chain->isSimple() || Chain->isIndexed())
      break;

    // Find the base pointer and offset for this memory node.
    const BaseIndexOffset Ptr = BaseIndexOffset::match(Chain, DAG);
    // Check that the base pointer is the same as the original one.
    int64_t Offset;
    if (!BasePtr.equalBaseIndex(Ptr, DAG, Offset))
      break;
    int64_t Length = (Chain->getMemoryVT().getSizeInBits() + 7) / 8;
    // Make sure we don't overlap with other intervals by checking the ones to
    // the left or right before inserting.
    auto I = Intervals.find(Offset);
    // If there's a next interval, we should end before it.
    if (I != Intervals.end() && I.start() < (Offset + Length))
      break;
    // If there's a previous interval, we should start after it.
    if (I != Intervals.begin() && (--I).stop() <= Offset)
      break;
    Intervals.insert(Offset, Offset + Length, Unit);

    ChainedStores.push_back(Chain);
    STChain = Chain;
  }

  // If we didn't find a chained store, exit.
  if (ChainedStores.size() == 0)
    return false;

  // Improve all chained stores (St and ChainedStores members) starting from
  // where the store chain ended and return single TokenFactor.
  SDValue NewChain = STChain->getChain();
  SmallVector<SDValue, 8> TFOps;
  for (unsigned I = ChainedStores.size(); I;) {
    StoreSDNode *S = ChainedStores[--I];
    SDValue BetterChain = FindBetterChain(S, NewChain);
    S = cast<StoreSDNode>(DAG.UpdateNodeOperands(
        S, BetterChain, S->getOperand(1), S->getOperand(2), S->getOperand(3)));
    TFOps.push_back(SDValue(S, 0));
    ChainedStores[I] = S;
  }

  // Improve St's chain. Use a new node to avoid creating a loop from CombineTo.
  SDValue BetterChain = FindBetterChain(St, NewChain);
  SDValue NewST;
  if (St->isTruncatingStore())
    NewST = DAG.getTruncStore(BetterChain, SDLoc(St), St->getValue(),
                              St->getBasePtr(), St->getMemoryVT(),
                              St->getMemOperand());
  else
    NewST = DAG.getStore(BetterChain, SDLoc(St), St->getValue(),
                         St->getBasePtr(), St->getMemOperand());

  TFOps.push_back(NewST);

  // If we improved every element of TFOps, then we've lost the dependence on
  // NewChain to successors of St and we need to add it back to TFOps. Do so at
  // the beginning to keep relative order consistent with FindBetterChains.
  auto hasImprovedChain = [&](SDValue ST) -> bool {
    return ST->getOperand(0) != NewChain;
  };
  bool AddNewChain = llvm::all_of(TFOps, hasImprovedChain);
  if (AddNewChain)
    TFOps.insert(TFOps.begin(), NewChain);

  SDValue TF = DAG.getTokenFactor(SDLoc(STChain), TFOps);
  CombineTo(St, TF);

  // Add TF and its operands to the worklist.
  AddToWorklist(TF.getNode());
  for (const SDValue &Op : TF->ops())
    AddToWorklist(Op.getNode());
  AddToWorklist(STChain);
  return true;
}

bool DAGCombiner::findBetterNeighborChains(StoreSDNode *St) {
  if (OptLevel == CodeGenOpt::None)
    return false;

  const BaseIndexOffset BasePtr = BaseIndexOffset::match(St, DAG);

  // We must have a base and an offset.
  if (!BasePtr.getBase().getNode())
    return false;

  // Do not handle stores to undef base pointers.
  if (BasePtr.getBase().isUndef())
    return false;

  // Directly improve a chain of disjoint stores starting at St.
  if (parallelizeChainedStores(St))
    return true;

  // Improve St's Chain..
  SDValue BetterChain = FindBetterChain(St, St->getChain());
  if (St->getChain() != BetterChain) {
    replaceStoreChain(St, BetterChain);
    return true;
  }
  return false;
}

/// This is the entry point for the file.
void SelectionDAG::Combine(CombineLevel Level, AliasAnalysis *AA,
                           CodeGenOpt::Level OptLevel) {
  /// This is the main entry point to this class.
  DAGCombiner(*this, AA, OptLevel).Run(Level);
}
