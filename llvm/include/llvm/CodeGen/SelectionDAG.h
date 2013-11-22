//===-- llvm/CodeGen/SelectionDAG.h - InstSelection DAG ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SelectionDAG class, and transitively defines the
// SDNode class and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SELECTIONDAG_H
#define LLVM_CODEGEN_SELECTIONDAG_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/ilist.h"
#include "llvm/CodeGen/DAGCombine.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <map>
#include <string>
#include <vector>

namespace llvm {

class AliasAnalysis;
class MachineConstantPoolValue;
class MachineFunction;
class MDNode;
class SDDbgValue;
class TargetLowering;
class TargetSelectionDAGInfo;
class TargetTransformInfo;

class SDVTListNode : public FoldingSetNode {
  friend struct FoldingSetTrait<SDVTListNode>;
  /// FastID - A reference to an Interned FoldingSetNodeID for this node.
  /// The Allocator in SelectionDAG holds the data.
  /// SDVTList contains all types which are frequently accessed in SelectionDAG.
  /// The size of this list is not expected big so it won't introduce memory penalty.
  FoldingSetNodeIDRef FastID;
  const EVT *VTs;
  unsigned int NumVTs;
  /// The hash value for SDVTList is fixed so cache it to avoid hash calculation
  unsigned HashValue;
public:
  SDVTListNode(const FoldingSetNodeIDRef ID, const EVT *VT, unsigned int Num) :
      FastID(ID), VTs(VT), NumVTs(Num) {
    HashValue = ID.ComputeHash();
  }
  SDVTList getSDVTList() {
    SDVTList result = {VTs, NumVTs};
    return result;
  }
};

// Specialize FoldingSetTrait for SDVTListNode
// To avoid computing temp FoldingSetNodeID and hash value.
template<> struct FoldingSetTrait<SDVTListNode> : DefaultFoldingSetTrait<SDVTListNode> {
  static void Profile(const SDVTListNode &X, FoldingSetNodeID& ID) {
    ID = X.FastID;
  }
  static bool Equals(const SDVTListNode &X, const FoldingSetNodeID &ID,
                     unsigned IDHash, FoldingSetNodeID &TempID) {
    if (X.HashValue != IDHash)
      return false;
    return ID == X.FastID;
  }
  static unsigned ComputeHash(const SDVTListNode &X, FoldingSetNodeID &TempID) {
    return X.HashValue;
  }
};

template<> struct ilist_traits<SDNode> : public ilist_default_traits<SDNode> {
private:
  mutable ilist_half_node<SDNode> Sentinel;
public:
  SDNode *createSentinel() const {
    return static_cast<SDNode*>(&Sentinel);
  }
  static void destroySentinel(SDNode *) {}

  SDNode *provideInitialHead() const { return createSentinel(); }
  SDNode *ensureHead(SDNode*) const { return createSentinel(); }
  static void noteHead(SDNode*, SDNode*) {}

  static void deleteNode(SDNode *) {
    llvm_unreachable("ilist_traits<SDNode> shouldn't see a deleteNode call!");
  }
private:
  static void createNode(const SDNode &);
};

/// SDDbgInfo - Keeps track of dbg_value information through SDISel.  We do
/// not build SDNodes for these so as not to perturb the generated code;
/// instead the info is kept off to the side in this structure. Each SDNode may
/// have one or more associated dbg_value entries. This information is kept in
/// DbgValMap.
/// Byval parameters are handled separately because they don't use alloca's,
/// which busts the normal mechanism.  There is good reason for handling all
/// parameters separately:  they may not have code generated for them, they
/// should always go at the beginning of the function regardless of other code
/// motion, and debug info for them is potentially useful even if the parameter
/// is unused.  Right now only byval parameters are handled separately.
class SDDbgInfo {
  SmallVector<SDDbgValue*, 32> DbgValues;
  SmallVector<SDDbgValue*, 32> ByvalParmDbgValues;
  typedef DenseMap<const SDNode*, SmallVector<SDDbgValue*, 2> > DbgValMapType;
  DbgValMapType DbgValMap;

  void operator=(const SDDbgInfo&) LLVM_DELETED_FUNCTION;
  SDDbgInfo(const SDDbgInfo&) LLVM_DELETED_FUNCTION;
public:
  SDDbgInfo() {}

  void add(SDDbgValue *V, const SDNode *Node, bool isParameter) {
    if (isParameter) {
      ByvalParmDbgValues.push_back(V);
    } else     DbgValues.push_back(V);
    if (Node)
      DbgValMap[Node].push_back(V);
  }

  void clear() {
    DbgValMap.clear();
    DbgValues.clear();
    ByvalParmDbgValues.clear();
  }

  bool empty() const {
    return DbgValues.empty() && ByvalParmDbgValues.empty();
  }

  ArrayRef<SDDbgValue*> getSDDbgValues(const SDNode *Node) {
    DbgValMapType::iterator I = DbgValMap.find(Node);
    if (I != DbgValMap.end())
      return I->second;
    return ArrayRef<SDDbgValue*>();
  }

  typedef SmallVectorImpl<SDDbgValue*>::iterator DbgIterator;
  DbgIterator DbgBegin() { return DbgValues.begin(); }
  DbgIterator DbgEnd()   { return DbgValues.end(); }
  DbgIterator ByvalParmDbgBegin() { return ByvalParmDbgValues.begin(); }
  DbgIterator ByvalParmDbgEnd()   { return ByvalParmDbgValues.end(); }
};

class SelectionDAG;
void checkForCycles(const SDNode *N);
void checkForCycles(const SelectionDAG *DAG);

/// SelectionDAG class - This is used to represent a portion of an LLVM function
/// in a low-level Data Dependence DAG representation suitable for instruction
/// selection.  This DAG is constructed as the first step of instruction
/// selection in order to allow implementation of machine specific optimizations
/// and code simplifications.
///
/// The representation used by the SelectionDAG is a target-independent
/// representation, which has some similarities to the GCC RTL representation,
/// but is significantly more simple, powerful, and is a graph form instead of a
/// linear form.
///
class SelectionDAG {
  const TargetMachine &TM;
  const TargetSelectionDAGInfo &TSI;
  const TargetTransformInfo *TTI;
  const TargetLowering *TLI;
  MachineFunction *MF;
  LLVMContext *Context;
  CodeGenOpt::Level OptLevel;

  /// EntryNode - The starting token.
  SDNode EntryNode;

  /// Root - The root of the entire DAG.
  SDValue Root;

  /// AllNodes - A linked list of nodes in the current DAG.
  ilist<SDNode> AllNodes;

  /// NodeAllocatorType - The AllocatorType for allocating SDNodes. We use
  /// pool allocation with recycling.
  typedef RecyclingAllocator<BumpPtrAllocator, SDNode, sizeof(LargestSDNode),
                             AlignOf<MostAlignedSDNode>::Alignment>
    NodeAllocatorType;

  /// NodeAllocator - Pool allocation for nodes.
  NodeAllocatorType NodeAllocator;

  /// CSEMap - This structure is used to memoize nodes, automatically performing
  /// CSE with existing nodes when a duplicate is requested.
  FoldingSet<SDNode> CSEMap;

  /// OperandAllocator - Pool allocation for machine-opcode SDNode operands.
  BumpPtrAllocator OperandAllocator;

  /// Allocator - Pool allocation for misc. objects that are created once per
  /// SelectionDAG.
  BumpPtrAllocator Allocator;

  /// DbgInfo - Tracks dbg_value information through SDISel.
  SDDbgInfo *DbgInfo;

public:
  /// DAGUpdateListener - Clients of various APIs that cause global effects on
  /// the DAG can optionally implement this interface.  This allows the clients
  /// to handle the various sorts of updates that happen.
  ///
  /// A DAGUpdateListener automatically registers itself with DAG when it is
  /// constructed, and removes itself when destroyed in RAII fashion.
  struct DAGUpdateListener {
    DAGUpdateListener *const Next;
    SelectionDAG &DAG;

    explicit DAGUpdateListener(SelectionDAG &D)
      : Next(D.UpdateListeners), DAG(D) {
      DAG.UpdateListeners = this;
    }

    virtual ~DAGUpdateListener() {
      assert(DAG.UpdateListeners == this &&
             "DAGUpdateListeners must be destroyed in LIFO order");
      DAG.UpdateListeners = Next;
    }

    /// NodeDeleted - The node N that was deleted and, if E is not null, an
    /// equivalent node E that replaced it.
    virtual void NodeDeleted(SDNode *N, SDNode *E);

    /// NodeUpdated - The node N that was updated.
    virtual void NodeUpdated(SDNode *N);
  };

  /// NewNodesMustHaveLegalTypes - When true, additional steps are taken to
  /// ensure that getConstant() and similar functions return DAG nodes that
  /// have legal types. This is important after type legalization since
  /// any illegally typed nodes generated after this point will not experience
  /// type legalization.
  bool NewNodesMustHaveLegalTypes;

private:
  /// DAGUpdateListener is a friend so it can manipulate the listener stack.
  friend struct DAGUpdateListener;

  /// UpdateListeners - Linked list of registered DAGUpdateListener instances.
  /// This stack is maintained by DAGUpdateListener RAII.
  DAGUpdateListener *UpdateListeners;

  /// setGraphColorHelper - Implementation of setSubgraphColor.
  /// Return whether we had to truncate the search.
  ///
  bool setSubgraphColorHelper(SDNode *N, const char *Color,
                              DenseSet<SDNode *> &visited,
                              int level, bool &printed);

  void operator=(const SelectionDAG&) LLVM_DELETED_FUNCTION;
  SelectionDAG(const SelectionDAG&) LLVM_DELETED_FUNCTION;

public:
  explicit SelectionDAG(const TargetMachine &TM, llvm::CodeGenOpt::Level);
  ~SelectionDAG();

  /// init - Prepare this SelectionDAG to process code in the given
  /// MachineFunction.
  ///
  void init(MachineFunction &mf, const TargetTransformInfo *TTI,
            const TargetLowering *TLI);

  /// clear - Clear state and free memory necessary to make this
  /// SelectionDAG ready to process a new block.
  ///
  void clear();

  MachineFunction &getMachineFunction() const { return *MF; }
  const TargetMachine &getTarget() const { return TM; }
  const TargetLowering &getTargetLoweringInfo() const { return *TLI; }
  const TargetSelectionDAGInfo &getSelectionDAGInfo() const { return TSI; }
  const TargetTransformInfo *getTargetTransformInfo() const { return TTI; }
  LLVMContext *getContext() const {return Context; }

  /// viewGraph - Pop up a GraphViz/gv window with the DAG rendered using 'dot'.
  ///
  void viewGraph(const std::string &Title);
  void viewGraph();

#ifndef NDEBUG
  std::map<const SDNode *, std::string> NodeGraphAttrs;
#endif

  /// clearGraphAttrs - Clear all previously defined node graph attributes.
  /// Intended to be used from a debugging tool (eg. gdb).
  void clearGraphAttrs();

  /// setGraphAttrs - Set graph attributes for a node. (eg. "color=red".)
  ///
  void setGraphAttrs(const SDNode *N, const char *Attrs);

  /// getGraphAttrs - Get graph attributes for a node. (eg. "color=red".)
  /// Used from getNodeAttributes.
  const std::string getGraphAttrs(const SDNode *N) const;

  /// setGraphColor - Convenience for setting node color attribute.
  ///
  void setGraphColor(const SDNode *N, const char *Color);

  /// setGraphColor - Convenience for setting subgraph color attribute.
  ///
  void setSubgraphColor(SDNode *N, const char *Color);

  typedef ilist<SDNode>::const_iterator allnodes_const_iterator;
  allnodes_const_iterator allnodes_begin() const { return AllNodes.begin(); }
  allnodes_const_iterator allnodes_end() const { return AllNodes.end(); }
  typedef ilist<SDNode>::iterator allnodes_iterator;
  allnodes_iterator allnodes_begin() { return AllNodes.begin(); }
  allnodes_iterator allnodes_end() { return AllNodes.end(); }
  ilist<SDNode>::size_type allnodes_size() const {
    return AllNodes.size();
  }

  /// getRoot - Return the root tag of the SelectionDAG.
  ///
  const SDValue &getRoot() const { return Root; }

  /// getEntryNode - Return the token chain corresponding to the entry of the
  /// function.
  SDValue getEntryNode() const {
    return SDValue(const_cast<SDNode *>(&EntryNode), 0);
  }

  /// setRoot - Set the current root tag of the SelectionDAG.
  ///
  const SDValue &setRoot(SDValue N) {
    assert((!N.getNode() || N.getValueType() == MVT::Other) &&
           "DAG root value is not a chain!");
    if (N.getNode())
      checkForCycles(N.getNode());
    Root = N;
    if (N.getNode())
      checkForCycles(this);
    return Root;
  }

  /// Combine - This iterates over the nodes in the SelectionDAG, folding
  /// certain types of nodes together, or eliminating superfluous nodes.  The
  /// Level argument controls whether Combine is allowed to produce nodes and
  /// types that are illegal on the target.
  void Combine(CombineLevel Level, AliasAnalysis &AA,
               CodeGenOpt::Level OptLevel);

  /// LegalizeTypes - This transforms the SelectionDAG into a SelectionDAG that
  /// only uses types natively supported by the target.  Returns "true" if it
  /// made any changes.
  ///
  /// Note that this is an involved process that may invalidate pointers into
  /// the graph.
  bool LegalizeTypes();

  /// Legalize - This transforms the SelectionDAG into a SelectionDAG that is
  /// compatible with the target instruction selector, as indicated by the
  /// TargetLowering object.
  ///
  /// Note that this is an involved process that may invalidate pointers into
  /// the graph.
  void Legalize();

  /// LegalizeVectors - This transforms the SelectionDAG into a SelectionDAG
  /// that only uses vector math operations supported by the target.  This is
  /// necessary as a separate step from Legalize because unrolling a vector
  /// operation can introduce illegal types, which requires running
  /// LegalizeTypes again.
  ///
  /// This returns true if it made any changes; in that case, LegalizeTypes
  /// is called again before Legalize.
  ///
  /// Note that this is an involved process that may invalidate pointers into
  /// the graph.
  bool LegalizeVectors();

  /// RemoveDeadNodes - This method deletes all unreachable nodes in the
  /// SelectionDAG.
  void RemoveDeadNodes();

  /// DeleteNode - Remove the specified node from the system.  This node must
  /// have no referrers.
  void DeleteNode(SDNode *N);

  /// getVTList - Return an SDVTList that represents the list of values
  /// specified.
  SDVTList getVTList(EVT VT);
  SDVTList getVTList(EVT VT1, EVT VT2);
  SDVTList getVTList(EVT VT1, EVT VT2, EVT VT3);
  SDVTList getVTList(EVT VT1, EVT VT2, EVT VT3, EVT VT4);
  SDVTList getVTList(const EVT *VTs, unsigned NumVTs);

  //===--------------------------------------------------------------------===//
  // Node creation methods.
  //
  SDValue getConstant(uint64_t Val, EVT VT, bool isTarget = false);
  SDValue getConstant(const APInt &Val, EVT VT, bool isTarget = false);
  SDValue getConstant(const ConstantInt &Val, EVT VT, bool isTarget = false);
  SDValue getIntPtrConstant(uint64_t Val, bool isTarget = false);
  SDValue getTargetConstant(uint64_t Val, EVT VT) {
    return getConstant(Val, VT, true);
  }
  SDValue getTargetConstant(const APInt &Val, EVT VT) {
    return getConstant(Val, VT, true);
  }
  SDValue getTargetConstant(const ConstantInt &Val, EVT VT) {
    return getConstant(Val, VT, true);
  }
  // The forms below that take a double should only be used for simple
  // constants that can be exactly represented in VT.  No checks are made.
  SDValue getConstantFP(double Val, EVT VT, bool isTarget = false);
  SDValue getConstantFP(const APFloat& Val, EVT VT, bool isTarget = false);
  SDValue getConstantFP(const ConstantFP &CF, EVT VT, bool isTarget = false);
  SDValue getTargetConstantFP(double Val, EVT VT) {
    return getConstantFP(Val, VT, true);
  }
  SDValue getTargetConstantFP(const APFloat& Val, EVT VT) {
    return getConstantFP(Val, VT, true);
  }
  SDValue getTargetConstantFP(const ConstantFP &Val, EVT VT) {
    return getConstantFP(Val, VT, true);
  }
  SDValue getGlobalAddress(const GlobalValue *GV, SDLoc DL, EVT VT,
                           int64_t offset = 0, bool isTargetGA = false,
                           unsigned char TargetFlags = 0);
  SDValue getTargetGlobalAddress(const GlobalValue *GV, SDLoc DL, EVT VT,
                                 int64_t offset = 0,
                                 unsigned char TargetFlags = 0) {
    return getGlobalAddress(GV, DL, VT, offset, true, TargetFlags);
  }
  SDValue getFrameIndex(int FI, EVT VT, bool isTarget = false);
  SDValue getTargetFrameIndex(int FI, EVT VT) {
    return getFrameIndex(FI, VT, true);
  }
  SDValue getJumpTable(int JTI, EVT VT, bool isTarget = false,
                       unsigned char TargetFlags = 0);
  SDValue getTargetJumpTable(int JTI, EVT VT, unsigned char TargetFlags = 0) {
    return getJumpTable(JTI, VT, true, TargetFlags);
  }
  SDValue getConstantPool(const Constant *C, EVT VT,
                          unsigned Align = 0, int Offs = 0, bool isT=false,
                          unsigned char TargetFlags = 0);
  SDValue getTargetConstantPool(const Constant *C, EVT VT,
                                unsigned Align = 0, int Offset = 0,
                                unsigned char TargetFlags = 0) {
    return getConstantPool(C, VT, Align, Offset, true, TargetFlags);
  }
  SDValue getConstantPool(MachineConstantPoolValue *C, EVT VT,
                          unsigned Align = 0, int Offs = 0, bool isT=false,
                          unsigned char TargetFlags = 0);
  SDValue getTargetConstantPool(MachineConstantPoolValue *C,
                                  EVT VT, unsigned Align = 0,
                                  int Offset = 0, unsigned char TargetFlags=0) {
    return getConstantPool(C, VT, Align, Offset, true, TargetFlags);
  }
  SDValue getTargetIndex(int Index, EVT VT, int64_t Offset = 0,
                         unsigned char TargetFlags = 0);
  // When generating a branch to a BB, we don't in general know enough
  // to provide debug info for the BB at that time, so keep this one around.
  SDValue getBasicBlock(MachineBasicBlock *MBB);
  SDValue getBasicBlock(MachineBasicBlock *MBB, SDLoc dl);
  SDValue getExternalSymbol(const char *Sym, EVT VT);
  SDValue getExternalSymbol(const char *Sym, SDLoc dl, EVT VT);
  SDValue getTargetExternalSymbol(const char *Sym, EVT VT,
                                  unsigned char TargetFlags = 0);
  SDValue getValueType(EVT);
  SDValue getRegister(unsigned Reg, EVT VT);
  SDValue getRegisterMask(const uint32_t *RegMask);
  SDValue getEHLabel(SDLoc dl, SDValue Root, MCSymbol *Label);
  SDValue getBlockAddress(const BlockAddress *BA, EVT VT,
                          int64_t Offset = 0, bool isTarget = false,
                          unsigned char TargetFlags = 0);
  SDValue getTargetBlockAddress(const BlockAddress *BA, EVT VT,
                                int64_t Offset = 0,
                                unsigned char TargetFlags = 0) {
    return getBlockAddress(BA, VT, Offset, true, TargetFlags);
  }

  SDValue getCopyToReg(SDValue Chain, SDLoc dl, unsigned Reg, SDValue N) {
    return getNode(ISD::CopyToReg, dl, MVT::Other, Chain,
                   getRegister(Reg, N.getValueType()), N);
  }

  // This version of the getCopyToReg method takes an extra operand, which
  // indicates that there is potentially an incoming glue value (if Glue is not
  // null) and that there should be a glue result.
  SDValue getCopyToReg(SDValue Chain, SDLoc dl, unsigned Reg, SDValue N,
                       SDValue Glue) {
    SDVTList VTs = getVTList(MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain, getRegister(Reg, N.getValueType()), N, Glue };
    return getNode(ISD::CopyToReg, dl, VTs, Ops, Glue.getNode() ? 4 : 3);
  }

  // Similar to last getCopyToReg() except parameter Reg is a SDValue
  SDValue getCopyToReg(SDValue Chain, SDLoc dl, SDValue Reg, SDValue N,
                         SDValue Glue) {
    SDVTList VTs = getVTList(MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain, Reg, N, Glue };
    return getNode(ISD::CopyToReg, dl, VTs, Ops, Glue.getNode() ? 4 : 3);
  }

  SDValue getCopyFromReg(SDValue Chain, SDLoc dl, unsigned Reg, EVT VT) {
    SDVTList VTs = getVTList(VT, MVT::Other);
    SDValue Ops[] = { Chain, getRegister(Reg, VT) };
    return getNode(ISD::CopyFromReg, dl, VTs, Ops, 2);
  }

  // This version of the getCopyFromReg method takes an extra operand, which
  // indicates that there is potentially an incoming glue value (if Glue is not
  // null) and that there should be a glue result.
  SDValue getCopyFromReg(SDValue Chain, SDLoc dl, unsigned Reg, EVT VT,
                           SDValue Glue) {
    SDVTList VTs = getVTList(VT, MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain, getRegister(Reg, VT), Glue };
    return getNode(ISD::CopyFromReg, dl, VTs, Ops, Glue.getNode() ? 3 : 2);
  }

  SDValue getCondCode(ISD::CondCode Cond);

  /// Returns the ConvertRndSat Note: Avoid using this node because it may
  /// disappear in the future and most targets don't support it.
  SDValue getConvertRndSat(EVT VT, SDLoc dl, SDValue Val, SDValue DTy,
                           SDValue STy,
                           SDValue Rnd, SDValue Sat, ISD::CvtCode Code);

  /// getVectorShuffle - Return an ISD::VECTOR_SHUFFLE node.  The number of
  /// elements in VT, which must be a vector type, must match the number of
  /// mask elements NumElts.  A integer mask element equal to -1 is treated as
  /// undefined.
  SDValue getVectorShuffle(EVT VT, SDLoc dl, SDValue N1, SDValue N2,
                           const int *MaskElts);

  /// getAnyExtOrTrunc - Convert Op, which must be of integer type, to the
  /// integer type VT, by either any-extending or truncating it.
  SDValue getAnyExtOrTrunc(SDValue Op, SDLoc DL, EVT VT);

  /// getSExtOrTrunc - Convert Op, which must be of integer type, to the
  /// integer type VT, by either sign-extending or truncating it.
  SDValue getSExtOrTrunc(SDValue Op, SDLoc DL, EVT VT);

  /// getZExtOrTrunc - Convert Op, which must be of integer type, to the
  /// integer type VT, by either zero-extending or truncating it.
  SDValue getZExtOrTrunc(SDValue Op, SDLoc DL, EVT VT);

  /// getZeroExtendInReg - Return the expression required to zero extend the Op
  /// value assuming it was the smaller SrcTy value.
  SDValue getZeroExtendInReg(SDValue Op, SDLoc DL, EVT SrcTy);

  /// getNOT - Create a bitwise NOT operation as (XOR Val, -1).
  SDValue getNOT(SDLoc DL, SDValue Val, EVT VT);

  /// getCALLSEQ_START - Return a new CALLSEQ_START node, which always must have
  /// a glue result (to ensure it's not CSE'd).  CALLSEQ_START does not have a
  /// useful SDLoc.
  SDValue getCALLSEQ_START(SDValue Chain, SDValue Op, SDLoc DL) {
    SDVTList VTs = getVTList(MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain,  Op };
    return getNode(ISD::CALLSEQ_START, DL, VTs, Ops, 2);
  }

  /// getCALLSEQ_END - Return a new CALLSEQ_END node, which always must have a
  /// glue result (to ensure it's not CSE'd).  CALLSEQ_END does not have
  /// a useful SDLoc.
  SDValue getCALLSEQ_END(SDValue Chain, SDValue Op1, SDValue Op2,
                           SDValue InGlue, SDLoc DL) {
    SDVTList NodeTys = getVTList(MVT::Other, MVT::Glue);
    SmallVector<SDValue, 4> Ops;
    Ops.push_back(Chain);
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(InGlue);
    return getNode(ISD::CALLSEQ_END, DL, NodeTys, &Ops[0],
                   (unsigned)Ops.size() - (InGlue.getNode() == 0 ? 1 : 0));
  }

  /// getUNDEF - Return an UNDEF node.  UNDEF does not have a useful SDLoc.
  SDValue getUNDEF(EVT VT) {
    return getNode(ISD::UNDEF, SDLoc(), VT);
  }

  /// getGLOBAL_OFFSET_TABLE - Return a GLOBAL_OFFSET_TABLE node.  This does
  /// not have a useful SDLoc.
  SDValue getGLOBAL_OFFSET_TABLE(EVT VT) {
    return getNode(ISD::GLOBAL_OFFSET_TABLE, SDLoc(), VT);
  }

  /// getNode - Gets or creates the specified node.
  ///
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT, SDValue N);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT, SDValue N1, SDValue N2);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT,
                  SDValue N1, SDValue N2, SDValue N3);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                  SDValue N5);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT,
                  const SDUse *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, SDLoc DL, EVT VT,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, SDLoc DL,
                  ArrayRef<EVT> ResultTys,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, SDLoc DL, const EVT *VTs, unsigned NumVTs,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs, SDValue N);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4);
  SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                  SDValue N5);

  /// getStackArgumentTokenFactor - Compute a TokenFactor to force all
  /// the incoming stack arguments to be loaded from the stack. This is
  /// used in tail call lowering to protect stack arguments from being
  /// clobbered.
  SDValue getStackArgumentTokenFactor(SDValue Chain);

  SDValue getMemcpy(SDValue Chain, SDLoc dl, SDValue Dst, SDValue Src,
                    SDValue Size, unsigned Align, bool isVol, bool AlwaysInline,
                    MachinePointerInfo DstPtrInfo,
                    MachinePointerInfo SrcPtrInfo);

  SDValue getMemmove(SDValue Chain, SDLoc dl, SDValue Dst, SDValue Src,
                     SDValue Size, unsigned Align, bool isVol,
                     MachinePointerInfo DstPtrInfo,
                     MachinePointerInfo SrcPtrInfo);

  SDValue getMemset(SDValue Chain, SDLoc dl, SDValue Dst, SDValue Src,
                    SDValue Size, unsigned Align, bool isVol,
                    MachinePointerInfo DstPtrInfo);

  /// getSetCC - Helper function to make it easier to build SetCC's if you just
  /// have an ISD::CondCode instead of an SDValue.
  ///
  SDValue getSetCC(SDLoc DL, EVT VT, SDValue LHS, SDValue RHS,
                   ISD::CondCode Cond) {
    assert(LHS.getValueType().isVector() == RHS.getValueType().isVector() &&
      "Cannot compare scalars to vectors");
    assert(LHS.getValueType().isVector() == VT.isVector() &&
      "Cannot compare scalars to vectors");
    assert(Cond != ISD::SETCC_INVALID &&
        "Cannot create a setCC of an invalid node.");
    return getNode(ISD::SETCC, DL, VT, LHS, RHS, getCondCode(Cond));
  }

  // getSelect - Helper function to make it easier to build Select's if you just
  // have operands and don't want to check for vector.
  SDValue getSelect(SDLoc DL, EVT VT, SDValue Cond,
                    SDValue LHS, SDValue RHS) {
    assert(LHS.getValueType() == RHS.getValueType() &&
           "Cannot use select on differing types");
    assert(VT.isVector() == LHS.getValueType().isVector() &&
           "Cannot mix vectors and scalars");
    return getNode(Cond.getValueType().isVector() ? ISD::VSELECT : ISD::SELECT, DL, VT,
                   Cond, LHS, RHS);
  }

  /// getSelectCC - Helper function to make it easier to build SelectCC's if you
  /// just have an ISD::CondCode instead of an SDValue.
  ///
  SDValue getSelectCC(SDLoc DL, SDValue LHS, SDValue RHS,
                      SDValue True, SDValue False, ISD::CondCode Cond) {
    return getNode(ISD::SELECT_CC, DL, True.getValueType(),
                   LHS, RHS, True, False, getCondCode(Cond));
  }

  /// getVAArg - VAArg produces a result and token chain, and takes a pointer
  /// and a source value as input.
  SDValue getVAArg(EVT VT, SDLoc dl, SDValue Chain, SDValue Ptr,
                   SDValue SV, unsigned Align);

  /// getAtomic - Gets a node for an atomic op, produces result and chain and
  /// takes 3 operands
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, SDValue Chain,
                    SDValue Ptr, SDValue Cmp, SDValue Swp,
                    MachinePointerInfo PtrInfo, unsigned Alignment,
                    AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, SDValue Chain,
                    SDValue Ptr, SDValue Cmp, SDValue Swp,
                    MachineMemOperand *MMO,
                    AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);

  /// getAtomic - Gets a node for an atomic op, produces result (if relevant)
  /// and chain and takes 2 operands.
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, SDValue Chain,
                    SDValue Ptr, SDValue Val, const Value* PtrVal,
                    unsigned Alignment, AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, SDValue Chain,
                    SDValue Ptr, SDValue Val, MachineMemOperand *MMO,
                    AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);

  /// getAtomic - Gets a node for an atomic op, produces result and chain and
  /// takes 1 operand.
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, EVT VT,
                    SDValue Chain, SDValue Ptr, const Value* PtrVal,
                    unsigned Alignment,
                    AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, EVT VT,
                    SDValue Chain, SDValue Ptr, MachineMemOperand *MMO,
                    AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);

  /// getAtomic - Gets a node for an atomic op, produces result and chain and
  /// takes N operands.
  SDValue getAtomic(unsigned Opcode, SDLoc dl, EVT MemVT, SDVTList VTList,
                    SDValue* Ops, unsigned NumOps, MachineMemOperand *MMO,
                    AtomicOrdering Ordering,
                    SynchronizationScope SynchScope);

  /// getMemIntrinsicNode - Creates a MemIntrinsicNode that may produce a
  /// result and takes a list of operands. Opcode may be INTRINSIC_VOID,
  /// INTRINSIC_W_CHAIN, or a target-specific opcode with a value not
  /// less than FIRST_TARGET_MEMORY_OPCODE.
  SDValue getMemIntrinsicNode(unsigned Opcode, SDLoc dl,
                              const EVT *VTs, unsigned NumVTs,
                              const SDValue *Ops, unsigned NumOps,
                              EVT MemVT, MachinePointerInfo PtrInfo,
                              unsigned Align = 0, bool Vol = false,
                              bool ReadMem = true, bool WriteMem = true);

  SDValue getMemIntrinsicNode(unsigned Opcode, SDLoc dl, SDVTList VTList,
                              const SDValue *Ops, unsigned NumOps,
                              EVT MemVT, MachinePointerInfo PtrInfo,
                              unsigned Align = 0, bool Vol = false,
                              bool ReadMem = true, bool WriteMem = true);

  SDValue getMemIntrinsicNode(unsigned Opcode, SDLoc dl, SDVTList VTList,
                              const SDValue *Ops, unsigned NumOps,
                              EVT MemVT, MachineMemOperand *MMO);

  /// getMergeValues - Create a MERGE_VALUES node from the given operands.
  SDValue getMergeValues(const SDValue *Ops, unsigned NumOps, SDLoc dl);

  /// getLoad - Loads are not normal binary operators: their result type is not
  /// determined by their operands, and they produce a value AND a token chain.
  ///
  SDValue getLoad(EVT VT, SDLoc dl, SDValue Chain, SDValue Ptr,
                  MachinePointerInfo PtrInfo, bool isVolatile,
                  bool isNonTemporal, bool isInvariant, unsigned Alignment,
                  const MDNode *TBAAInfo = 0, const MDNode *Ranges = 0);
  SDValue getLoad(EVT VT, SDLoc dl, SDValue Chain, SDValue Ptr,
                  MachineMemOperand *MMO);
  SDValue getExtLoad(ISD::LoadExtType ExtType, SDLoc dl, EVT VT,
                     SDValue Chain, SDValue Ptr, MachinePointerInfo PtrInfo,
                     EVT MemVT, bool isVolatile,
                     bool isNonTemporal, unsigned Alignment,
                     const MDNode *TBAAInfo = 0);
  SDValue getExtLoad(ISD::LoadExtType ExtType, SDLoc dl, EVT VT,
                     SDValue Chain, SDValue Ptr, EVT MemVT,
                     MachineMemOperand *MMO);
  SDValue getIndexedLoad(SDValue OrigLoad, SDLoc dl, SDValue Base,
                         SDValue Offset, ISD::MemIndexedMode AM);
  SDValue getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                  EVT VT, SDLoc dl,
                  SDValue Chain, SDValue Ptr, SDValue Offset,
                  MachinePointerInfo PtrInfo, EVT MemVT,
                  bool isVolatile, bool isNonTemporal, bool isInvariant,
                  unsigned Alignment, const MDNode *TBAAInfo = 0,
                  const MDNode *Ranges = 0);
  SDValue getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                  EVT VT, SDLoc dl,
                  SDValue Chain, SDValue Ptr, SDValue Offset,
                  EVT MemVT, MachineMemOperand *MMO);

  /// getStore - Helper function to build ISD::STORE nodes.
  ///
  SDValue getStore(SDValue Chain, SDLoc dl, SDValue Val, SDValue Ptr,
                   MachinePointerInfo PtrInfo, bool isVolatile,
                   bool isNonTemporal, unsigned Alignment,
                   const MDNode *TBAAInfo = 0);
  SDValue getStore(SDValue Chain, SDLoc dl, SDValue Val, SDValue Ptr,
                   MachineMemOperand *MMO);
  SDValue getTruncStore(SDValue Chain, SDLoc dl, SDValue Val, SDValue Ptr,
                        MachinePointerInfo PtrInfo, EVT TVT,
                        bool isNonTemporal, bool isVolatile,
                        unsigned Alignment,
                        const MDNode *TBAAInfo = 0);
  SDValue getTruncStore(SDValue Chain, SDLoc dl, SDValue Val, SDValue Ptr,
                        EVT TVT, MachineMemOperand *MMO);
  SDValue getIndexedStore(SDValue OrigStoe, SDLoc dl, SDValue Base,
                           SDValue Offset, ISD::MemIndexedMode AM);

  /// getSrcValue - Construct a node to track a Value* through the backend.
  SDValue getSrcValue(const Value *v);

  /// getMDNode - Return an MDNodeSDNode which holds an MDNode.
  SDValue getMDNode(const MDNode *MD);

  /// getAddrSpaceCast - Return an AddrSpaceCastSDNode.
  SDValue getAddrSpaceCast(SDLoc dl, EVT VT, SDValue Ptr,
                           unsigned SrcAS, unsigned DestAS);

  /// getShiftAmountOperand - Return the specified value casted to
  /// the target's desired shift amount type.
  SDValue getShiftAmountOperand(EVT LHSTy, SDValue Op);

  /// UpdateNodeOperands - *Mutate* the specified node in-place to have the
  /// specified operands.  If the resultant node already exists in the DAG,
  /// this does not modify the specified node, instead it returns the node that
  /// already exists.  If the resultant node does not exist in the DAG, the
  /// input node is returned.  As a degenerate case, if you specify the same
  /// input operands as the node already has, the input node is returned.
  SDNode *UpdateNodeOperands(SDNode *N, SDValue Op);
  SDNode *UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2);
  SDNode *UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                               SDValue Op3);
  SDNode *UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                               SDValue Op3, SDValue Op4);
  SDNode *UpdateNodeOperands(SDNode *N, SDValue Op1, SDValue Op2,
                               SDValue Op3, SDValue Op4, SDValue Op5);
  SDNode *UpdateNodeOperands(SDNode *N,
                               const SDValue *Ops, unsigned NumOps);

  /// SelectNodeTo - These are used for target selectors to *mutate* the
  /// specified node to have the specified return type, Target opcode, and
  /// operands.  Note that target opcodes are stored as
  /// ~TargetOpcode in the node opcode field.  The resultant node is returned.
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT, SDValue Op1);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT,
                       SDValue Op1, SDValue Op2);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT,
                       SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT,
                       const SDValue *Ops, unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1, EVT VT2);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1,
                       EVT VT2, const SDValue *Ops, unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1,
                       EVT VT2, EVT VT3, const SDValue *Ops, unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned MachineOpc, EVT VT1,
                       EVT VT2, EVT VT3, EVT VT4, const SDValue *Ops,
                       unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1,
                       EVT VT2, SDValue Op1);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1,
                       EVT VT2, SDValue Op1, SDValue Op2);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1,
                       EVT VT2, SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, EVT VT1,
                       EVT VT2, EVT VT3, SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, SDVTList VTs,
                       const SDValue *Ops, unsigned NumOps);

  /// MorphNodeTo - This *mutates* the specified node to have the specified
  /// return type, opcode, and operands.
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, SDVTList VTs,
                      const SDValue *Ops, unsigned NumOps);

  /// getMachineNode - These are used for target selectors to create a new node
  /// with specified return type(s), MachineInstr opcode, and operands.
  ///
  /// Note that getMachineNode returns the resultant node.  If there is already
  /// a node of the specified opcode and operands, it returns that node instead
  /// of the current one.
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                                SDValue Op1);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                                SDValue Op1, SDValue Op2);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                                SDValue Op1, SDValue Op2, SDValue Op3);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT,
                                ArrayRef<SDValue> Ops);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                SDValue Op1);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                SDValue Op1, SDValue Op2);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                SDValue Op1, SDValue Op2, SDValue Op3);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                ArrayRef<SDValue> Ops);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                EVT VT3, SDValue Op1, SDValue Op2);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                EVT VT3, SDValue Op1, SDValue Op2,
                                SDValue Op3);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                EVT VT3, ArrayRef<SDValue> Ops);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, EVT VT1, EVT VT2,
                                EVT VT3, EVT VT4, ArrayRef<SDValue> Ops);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl,
                                ArrayRef<EVT> ResultTys,
                                ArrayRef<SDValue> Ops);
  MachineSDNode *getMachineNode(unsigned Opcode, SDLoc dl, SDVTList VTs,
                                ArrayRef<SDValue> Ops);

  /// getTargetExtractSubreg - A convenience function for creating
  /// TargetInstrInfo::EXTRACT_SUBREG nodes.
  SDValue getTargetExtractSubreg(int SRIdx, SDLoc DL, EVT VT,
                                 SDValue Operand);

  /// getTargetInsertSubreg - A convenience function for creating
  /// TargetInstrInfo::INSERT_SUBREG nodes.
  SDValue getTargetInsertSubreg(int SRIdx, SDLoc DL, EVT VT,
                                SDValue Operand, SDValue Subreg);

  /// getNodeIfExists - Get the specified node if it's already available, or
  /// else return NULL.
  SDNode *getNodeIfExists(unsigned Opcode, SDVTList VTs,
                          const SDValue *Ops, unsigned NumOps);

  /// getDbgValue - Creates a SDDbgValue node.
  ///
  SDDbgValue *getDbgValue(MDNode *MDPtr, SDNode *N, unsigned R, uint64_t Off,
                          DebugLoc DL, unsigned O);
  SDDbgValue *getDbgValue(MDNode *MDPtr, const Value *C, uint64_t Off,
                          DebugLoc DL, unsigned O);
  SDDbgValue *getDbgValue(MDNode *MDPtr, unsigned FI, uint64_t Off,
                          DebugLoc DL, unsigned O);

  /// RemoveDeadNode - Remove the specified node from the system. If any of its
  /// operands then becomes dead, remove them as well. Inform UpdateListener
  /// for each node deleted.
  void RemoveDeadNode(SDNode *N);

  /// RemoveDeadNodes - This method deletes the unreachable nodes in the
  /// given list, and any nodes that become unreachable as a result.
  void RemoveDeadNodes(SmallVectorImpl<SDNode *> &DeadNodes);

  /// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
  /// This can cause recursive merging of nodes in the DAG.  Use the first
  /// version if 'From' is known to have a single result, use the second
  /// if you have two nodes with identical results (or if 'To' has a superset
  /// of the results of 'From'), use the third otherwise.
  ///
  /// These methods all take an optional UpdateListener, which (if not null) is
  /// informed about nodes that are deleted and modified due to recursive
  /// changes in the dag.
  ///
  /// These functions only replace all existing uses. It's possible that as
  /// these replacements are being performed, CSE may cause the From node
  /// to be given new uses. These new uses of From are left in place, and
  /// not automatically transferred to To.
  ///
  void ReplaceAllUsesWith(SDValue From, SDValue Op);
  void ReplaceAllUsesWith(SDNode *From, SDNode *To);
  void ReplaceAllUsesWith(SDNode *From, const SDValue *To);

  /// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
  /// uses of other values produced by From.Val alone.
  void ReplaceAllUsesOfValueWith(SDValue From, SDValue To);

  /// ReplaceAllUsesOfValuesWith - Like ReplaceAllUsesOfValueWith, but
  /// for multiple values at once. This correctly handles the case where
  /// there is an overlap between the From values and the To values.
  void ReplaceAllUsesOfValuesWith(const SDValue *From, const SDValue *To,
                                  unsigned Num);

  /// AssignTopologicalOrder - Topological-sort the AllNodes list and a
  /// assign a unique node id for each node in the DAG based on their
  /// topological order. Returns the number of nodes.
  unsigned AssignTopologicalOrder();

  /// RepositionNode - Move node N in the AllNodes list to be immediately
  /// before the given iterator Position. This may be used to update the
  /// topological ordering when the list of nodes is modified.
  void RepositionNode(allnodes_iterator Position, SDNode *N) {
    AllNodes.insert(Position, AllNodes.remove(N));
  }

  /// isCommutativeBinOp - Returns true if the opcode is a commutative binary
  /// operation.
  static bool isCommutativeBinOp(unsigned Opcode) {
    // FIXME: This should get its info from the td file, so that we can include
    // target info.
    switch (Opcode) {
    case ISD::ADD:
    case ISD::MUL:
    case ISD::MULHU:
    case ISD::MULHS:
    case ISD::SMUL_LOHI:
    case ISD::UMUL_LOHI:
    case ISD::FADD:
    case ISD::FMUL:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
    case ISD::SADDO:
    case ISD::UADDO:
    case ISD::ADDC:
    case ISD::ADDE: return true;
    default: return false;
    }
  }

  /// Returns an APFloat semantics tag appropriate for the given type. If VT is
  /// a vector type, the element semantics are returned.
  static const fltSemantics &EVTToAPFloatSemantics(EVT VT) {
    switch (VT.getScalarType().getSimpleVT().SimpleTy) {
    default: llvm_unreachable("Unknown FP format");
    case MVT::f16:     return APFloat::IEEEhalf;
    case MVT::f32:     return APFloat::IEEEsingle;
    case MVT::f64:     return APFloat::IEEEdouble;
    case MVT::f80:     return APFloat::x87DoubleExtended;
    case MVT::f128:    return APFloat::IEEEquad;
    case MVT::ppcf128: return APFloat::PPCDoubleDouble;
    }
  }

  /// AddDbgValue - Add a dbg_value SDNode. If SD is non-null that means the
  /// value is produced by SD.
  void AddDbgValue(SDDbgValue *DB, SDNode *SD, bool isParameter);

  /// GetDbgValues - Get the debug values which reference the given SDNode.
  ArrayRef<SDDbgValue*> GetDbgValues(const SDNode* SD) {
    return DbgInfo->getSDDbgValues(SD);
  }

  /// TransferDbgValues - Transfer SDDbgValues.
  void TransferDbgValues(SDValue From, SDValue To);

  /// hasDebugValues - Return true if there are any SDDbgValue nodes associated
  /// with this SelectionDAG.
  bool hasDebugValues() const { return !DbgInfo->empty(); }

  SDDbgInfo::DbgIterator DbgBegin() { return DbgInfo->DbgBegin(); }
  SDDbgInfo::DbgIterator DbgEnd()   { return DbgInfo->DbgEnd(); }
  SDDbgInfo::DbgIterator ByvalParmDbgBegin() {
    return DbgInfo->ByvalParmDbgBegin();
  }
  SDDbgInfo::DbgIterator ByvalParmDbgEnd()   {
    return DbgInfo->ByvalParmDbgEnd();
  }

  void dump() const;

  /// CreateStackTemporary - Create a stack temporary, suitable for holding the
  /// specified value type.  If minAlign is specified, the slot size will have
  /// at least that alignment.
  SDValue CreateStackTemporary(EVT VT, unsigned minAlign = 1);

  /// CreateStackTemporary - Create a stack temporary suitable for holding
  /// either of the specified value types.
  SDValue CreateStackTemporary(EVT VT1, EVT VT2);

  /// FoldConstantArithmetic -
  SDValue FoldConstantArithmetic(unsigned Opcode, EVT VT,
                                 SDNode *Cst1, SDNode *Cst2);

  /// FoldSetCC - Constant fold a setcc to true or false.
  SDValue FoldSetCC(EVT VT, SDValue N1,
                    SDValue N2, ISD::CondCode Cond, SDLoc dl);

  /// SignBitIsZero - Return true if the sign bit of Op is known to be zero.  We
  /// use this predicate to simplify operations downstream.
  bool SignBitIsZero(SDValue Op, unsigned Depth = 0) const;

  /// MaskedValueIsZero - Return true if 'Op & Mask' is known to be zero.  We
  /// use this predicate to simplify operations downstream.  Op and Mask are
  /// known to be the same type.
  bool MaskedValueIsZero(SDValue Op, const APInt &Mask, unsigned Depth = 0)
    const;

  /// ComputeMaskedBits - Determine which of the bits specified in Mask are
  /// known to be either zero or one and return them in the KnownZero/KnownOne
  /// bitsets.  This code only analyzes bits in Mask, in order to short-circuit
  /// processing.  Targets can implement the computeMaskedBitsForTargetNode
  /// method in the TargetLowering class to allow target nodes to be understood.
  void ComputeMaskedBits(SDValue Op, APInt &KnownZero, APInt &KnownOne,
                         unsigned Depth = 0) const;

  /// ComputeNumSignBits - Return the number of times the sign bit of the
  /// register is replicated into the other bits.  We know that at least 1 bit
  /// is always equal to the sign bit (itself), but other cases can give us
  /// information.  For example, immediately after an "SRA X, 2", we know that
  /// the top 3 bits are all equal to each other, so we return 3.  Targets can
  /// implement the ComputeNumSignBitsForTarget method in the TargetLowering
  /// class to allow target nodes to be understood.
  unsigned ComputeNumSignBits(SDValue Op, unsigned Depth = 0) const;

  /// isBaseWithConstantOffset - Return true if the specified operand is an
  /// ISD::ADD with a ConstantSDNode on the right-hand side, or if it is an
  /// ISD::OR with a ConstantSDNode that is guaranteed to have the same
  /// semantics as an ADD.  This handles the equivalence:
  ///     X|Cst == X+Cst iff X&Cst = 0.
  bool isBaseWithConstantOffset(SDValue Op) const;

  /// isKnownNeverNan - Test whether the given SDValue is known to never be NaN.
  bool isKnownNeverNaN(SDValue Op) const;

  /// isKnownNeverZero - Test whether the given SDValue is known to never be
  /// positive or negative Zero.
  bool isKnownNeverZero(SDValue Op) const;

  /// isEqualTo - Test whether two SDValues are known to compare equal. This
  /// is true if they are the same value, or if one is negative zero and the
  /// other positive zero.
  bool isEqualTo(SDValue A, SDValue B) const;

  /// UnrollVectorOp - Utility function used by legalize and lowering to
  /// "unroll" a vector operation by splitting out the scalars and operating
  /// on each element individually.  If the ResNE is 0, fully unroll the vector
  /// op. If ResNE is less than the width of the vector op, unroll up to ResNE.
  /// If the  ResNE is greater than the width of the vector op, unroll the
  /// vector op and fill the end of the resulting vector with UNDEFS.
  SDValue UnrollVectorOp(SDNode *N, unsigned ResNE = 0);

  /// isConsecutiveLoad - Return true if LD is loading 'Bytes' bytes from a
  /// location that is 'Dist' units away from the location that the 'Base' load
  /// is loading from.
  bool isConsecutiveLoad(LoadSDNode *LD, LoadSDNode *Base,
                         unsigned Bytes, int Dist) const;

  /// InferPtrAlignment - Infer alignment of a load / store address. Return 0 if
  /// it cannot be inferred.
  unsigned InferPtrAlignment(SDValue Ptr) const;

  /// GetSplitDestVTs - Compute the VTs needed for the low/hi parts of a type
  /// which is split (or expanded) into two not necessarily identical pieces.
  std::pair<EVT, EVT> GetSplitDestVTs(const EVT &VT) const;

  /// SplitVector - Split the vector with EXTRACT_SUBVECTOR using the provides
  /// VTs and return the low/high part.
  std::pair<SDValue, SDValue> SplitVector(const SDValue &N, const SDLoc &DL,
                                          const EVT &LoVT, const EVT &HiVT);

  /// SplitVector - Split the vector with EXTRACT_SUBVECTOR and return the
  /// low/high part.
  std::pair<SDValue, SDValue> SplitVector(const SDValue &N, const SDLoc &DL) {
    EVT LoVT, HiVT;
    llvm::tie(LoVT, HiVT) = GetSplitDestVTs(N.getValueType());
    return SplitVector(N, DL, LoVT, HiVT);
  }

  /// SplitVectorOperand - Split the node's operand with EXTRACT_SUBVECTOR and
  /// return the low/high part.
  std::pair<SDValue, SDValue> SplitVectorOperand(const SDNode *N, unsigned OpNo)
  {
    return SplitVector(N->getOperand(OpNo), SDLoc(N));
  }

private:
  bool RemoveNodeFromCSEMaps(SDNode *N);
  void AddModifiedNodeToCSEMaps(SDNode *N);
  SDNode *FindModifiedNodeSlot(SDNode *N, SDValue Op, void *&InsertPos);
  SDNode *FindModifiedNodeSlot(SDNode *N, SDValue Op1, SDValue Op2,
                               void *&InsertPos);
  SDNode *FindModifiedNodeSlot(SDNode *N, const SDValue *Ops, unsigned NumOps,
                               void *&InsertPos);
  SDNode *UpdadeSDLocOnMergedSDNode(SDNode *N, SDLoc loc);

  void DeleteNodeNotInCSEMaps(SDNode *N);
  void DeallocateNode(SDNode *N);

  unsigned getEVTAlignment(EVT MemoryVT) const;

  void allnodes_clear();

  /// VTList - List of non-single value types.
  FoldingSet<SDVTListNode> VTListMap;

  /// CondCodeNodes - Maps to auto-CSE operations.
  std::vector<CondCodeSDNode*> CondCodeNodes;

  std::vector<SDNode*> ValueTypeNodes;
  std::map<EVT, SDNode*, EVT::compareRawBits> ExtendedValueTypeNodes;
  StringMap<SDNode*> ExternalSymbols;

  std::map<std::pair<std::string, unsigned char>,SDNode*> TargetExternalSymbols;
};

template <> struct GraphTraits<SelectionDAG*> : public GraphTraits<SDNode*> {
  typedef SelectionDAG::allnodes_iterator nodes_iterator;
  static nodes_iterator nodes_begin(SelectionDAG *G) {
    return G->allnodes_begin();
  }
  static nodes_iterator nodes_end(SelectionDAG *G) {
    return G->allnodes_end();
  }
};

}  // end namespace llvm

#endif
