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

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

#include <cassert>
#include <vector>
#include <map>
#include <string>

namespace llvm {

class AliasAnalysis;
class TargetLowering;
class TargetMachine;
class MachineModuleInfo;
class DwarfWriter;
class MachineFunction;
class MachineConstantPoolValue;
class FunctionLoweringInfo;

template<> struct ilist_traits<SDNode> : public ilist_default_traits<SDNode> {
private:
  mutable SDNode Sentinel;
public:
  ilist_traits() : Sentinel(ISD::DELETED_NODE, SDVTList()) {}

  SDNode *createSentinel() const {
    return &Sentinel;
  }
  static void destroySentinel(SDNode *) {}

  static void deleteNode(SDNode *) {
    assert(0 && "ilist_traits<SDNode> shouldn't see a deleteNode call!");
  }
private:
  static void createNode(const SDNode &);
};

enum CombineLevel {
  Unrestricted,   // Combine may create illegal operations and illegal types.
  NoIllegalTypes, // Combine may create illegal operations but no illegal types.
  NoIllegalOperations // Combine may only create legal operations and types.
};

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
  TargetLowering &TLI;
  MachineFunction *MF;
  FunctionLoweringInfo &FLI;
  MachineModuleInfo *MMI;
  DwarfWriter *DW;

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
  /// CSE with existing nodes with a duplicate is requested.
  FoldingSet<SDNode> CSEMap;

  /// OperandAllocator - Pool allocation for machine-opcode SDNode operands.
  BumpPtrAllocator OperandAllocator;

  /// Allocator - Pool allocation for misc. objects that are created once per
  /// SelectionDAG.
  BumpPtrAllocator Allocator;

  /// VerifyNode - Sanity check the given node.  Aborts if it is invalid.
  void VerifyNode(SDNode *N);

  /// setGraphColorHelper - Implementation of setSubgraphColor.
  /// Return whether we had to truncate the search.
  ///
  bool setSubgraphColorHelper(SDNode *N, const char *Color, DenseSet<SDNode *> &visited,
                              int level, bool &printed);

public:
  SelectionDAG(TargetLowering &tli, FunctionLoweringInfo &fli);
  ~SelectionDAG();

  /// init - Prepare this SelectionDAG to process code in the given
  /// MachineFunction.
  ///
  void init(MachineFunction &mf, MachineModuleInfo *mmi, DwarfWriter *dw);

  /// clear - Clear state and free memory necessary to make this
  /// SelectionDAG ready to process a new block.
  ///
  void clear();

  MachineFunction &getMachineFunction() const { return *MF; }
  const TargetMachine &getTarget() const;
  TargetLowering &getTargetLoweringInfo() const { return TLI; }
  FunctionLoweringInfo &getFunctionLoweringInfo() const { return FLI; }
  MachineModuleInfo *getMachineModuleInfo() const { return MMI; }
  DwarfWriter *getDwarfWriter() const { return DW; }

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
    return Root = N;
  }

  /// Combine - This iterates over the nodes in the SelectionDAG, folding
  /// certain types of nodes together, or eliminating superfluous nodes.  The
  /// Level argument controls whether Combine is allowed to produce nodes and
  /// types that are illegal on the target.
  void Combine(CombineLevel Level, AliasAnalysis &AA, bool Fast);

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
  void Legalize(bool TypesNeedLegalizing);

  /// RemoveDeadNodes - This method deletes all unreachable nodes in the
  /// SelectionDAG.
  void RemoveDeadNodes();

  /// DeleteNode - Remove the specified node from the system.  This node must
  /// have no referrers.
  void DeleteNode(SDNode *N);

  /// getVTList - Return an SDVTList that represents the list of values
  /// specified.
  SDVTList getVTList(MVT VT);
  SDVTList getVTList(MVT VT1, MVT VT2);
  SDVTList getVTList(MVT VT1, MVT VT2, MVT VT3);
  SDVTList getVTList(MVT VT1, MVT VT2, MVT VT3, MVT VT4);
  SDVTList getVTList(const MVT *VTs, unsigned NumVTs);
  
  /// getNodeValueTypes - These are obsolete, use getVTList instead.
  const MVT *getNodeValueTypes(MVT VT) {
    return getVTList(VT).VTs;
  }
  const MVT *getNodeValueTypes(MVT VT1, MVT VT2) {
    return getVTList(VT1, VT2).VTs;
  }
  const MVT *getNodeValueTypes(MVT VT1, MVT VT2, MVT VT3) {
    return getVTList(VT1, VT2, VT3).VTs;
  }
  const MVT *getNodeValueTypes(MVT VT1, MVT VT2, MVT VT3, MVT VT4) {
    return getVTList(VT1, VT2, VT3, VT4).VTs;
  }
  const MVT *getNodeValueTypes(const std::vector<MVT> &vtList) {
    return getVTList(&vtList[0], (unsigned)vtList.size()).VTs;
  }
  
  
  //===--------------------------------------------------------------------===//
  // Node creation methods.
  //
  SDValue getConstant(uint64_t Val, MVT VT, bool isTarget = false);
  SDValue getConstant(const APInt &Val, MVT VT, bool isTarget = false);
  SDValue getConstant(const ConstantInt &Val, MVT VT, bool isTarget = false);
  SDValue getIntPtrConstant(uint64_t Val, bool isTarget = false);
  SDValue getTargetConstant(uint64_t Val, MVT VT) {
    return getConstant(Val, VT, true);
  }
  SDValue getTargetConstant(const APInt &Val, MVT VT) {
    return getConstant(Val, VT, true);
  }
  SDValue getTargetConstant(const ConstantInt &Val, MVT VT) {
    return getConstant(Val, VT, true);
  }
  SDValue getConstantFP(double Val, MVT VT, bool isTarget = false);
  SDValue getConstantFP(const APFloat& Val, MVT VT, bool isTarget = false);
  SDValue getConstantFP(const ConstantFP &CF, MVT VT, bool isTarget = false);
  SDValue getTargetConstantFP(double Val, MVT VT) {
    return getConstantFP(Val, VT, true);
  }
  SDValue getTargetConstantFP(const APFloat& Val, MVT VT) {
    return getConstantFP(Val, VT, true);
  }
  SDValue getTargetConstantFP(const ConstantFP &Val, MVT VT) {
    return getConstantFP(Val, VT, true);
  }
  SDValue getGlobalAddress(const GlobalValue *GV, MVT VT,
                           int64_t offset = 0, bool isTargetGA = false);
  SDValue getTargetGlobalAddress(const GlobalValue *GV, MVT VT,
                                 int64_t offset = 0) {
    return getGlobalAddress(GV, VT, offset, true);
  }
  SDValue getFrameIndex(int FI, MVT VT, bool isTarget = false);
  SDValue getTargetFrameIndex(int FI, MVT VT) {
    return getFrameIndex(FI, VT, true);
  }
  SDValue getJumpTable(int JTI, MVT VT, bool isTarget = false);
  SDValue getTargetJumpTable(int JTI, MVT VT) {
    return getJumpTable(JTI, VT, true);
  }
  SDValue getConstantPool(Constant *C, MVT VT,
                            unsigned Align = 0, int Offs = 0, bool isT=false);
  SDValue getTargetConstantPool(Constant *C, MVT VT,
                                  unsigned Align = 0, int Offset = 0) {
    return getConstantPool(C, VT, Align, Offset, true);
  }
  SDValue getConstantPool(MachineConstantPoolValue *C, MVT VT,
                            unsigned Align = 0, int Offs = 0, bool isT=false);
  SDValue getTargetConstantPool(MachineConstantPoolValue *C,
                                  MVT VT, unsigned Align = 0,
                                  int Offset = 0) {
    return getConstantPool(C, VT, Align, Offset, true);
  }
  SDValue getBasicBlock(MachineBasicBlock *MBB);
  SDValue getBasicBlock(MachineBasicBlock *MBB, DebugLoc dl);
  SDValue getExternalSymbol(const char *Sym, MVT VT);
  SDValue getExternalSymbol(const char *Sym, DebugLoc dl, MVT VT);
  SDValue getTargetExternalSymbol(const char *Sym, MVT VT);
  SDValue getTargetExternalSymbol(const char *Sym, DebugLoc dl, MVT VT);
  SDValue getArgFlags(ISD::ArgFlagsTy Flags);
  SDValue getValueType(MVT);
  SDValue getRegister(unsigned Reg, MVT VT);
  SDValue getDbgStopPoint(SDValue Root, unsigned Line, unsigned Col,
                          Value *CU);
  SDValue getLabel(unsigned Opcode, SDValue Root, unsigned LabelID);
  SDValue getLabel(unsigned Opcode, DebugLoc dl, SDValue Root, 
                   unsigned LabelID);

  SDValue getCopyToReg(SDValue Chain, unsigned Reg, SDValue N) {
    return getNode(ISD::CopyToReg, MVT::Other, Chain,
                   getRegister(Reg, N.getValueType()), N);
  }

  // This version of the getCopyToReg method takes an extra operand, which
  // indicates that there is potentially an incoming flag value (if Flag is not
  // null) and that there should be a flag result.
  SDValue getCopyToReg(SDValue Chain, unsigned Reg, SDValue N,
                         SDValue Flag) {
    const MVT *VTs = getNodeValueTypes(MVT::Other, MVT::Flag);
    SDValue Ops[] = { Chain, getRegister(Reg, N.getValueType()), N, Flag };
    return getNode(ISD::CopyToReg, VTs, 2, Ops, Flag.getNode() ? 4 : 3);
  }

  // Similar to last getCopyToReg() except parameter Reg is a SDValue
  SDValue getCopyToReg(SDValue Chain, SDValue Reg, SDValue N,
                         SDValue Flag) {
    const MVT *VTs = getNodeValueTypes(MVT::Other, MVT::Flag);
    SDValue Ops[] = { Chain, Reg, N, Flag };
    return getNode(ISD::CopyToReg, VTs, 2, Ops, Flag.getNode() ? 4 : 3);
  }
  
  SDValue getCopyFromReg(SDValue Chain, unsigned Reg, MVT VT) {
    const MVT *VTs = getNodeValueTypes(VT, MVT::Other);
    SDValue Ops[] = { Chain, getRegister(Reg, VT) };
    return getNode(ISD::CopyFromReg, VTs, 2, Ops, 2);
  }
  
  // This version of the getCopyFromReg method takes an extra operand, which
  // indicates that there is potentially an incoming flag value (if Flag is not
  // null) and that there should be a flag result.
  SDValue getCopyFromReg(SDValue Chain, unsigned Reg, MVT VT,
                           SDValue Flag) {
    const MVT *VTs = getNodeValueTypes(VT, MVT::Other, MVT::Flag);
    SDValue Ops[] = { Chain, getRegister(Reg, VT), Flag };
    return getNode(ISD::CopyFromReg, VTs, 3, Ops, Flag.getNode() ? 3 : 2);
  }

  SDValue getCondCode(ISD::CondCode Cond);

  /// Returns the ConvertRndSat Note: Avoid using this node because it may
  /// disappear in the future and most targets don't support it.
  SDValue getConvertRndSat(MVT VT, SDValue Val, SDValue DTy, SDValue STy,
                           SDValue Rnd, SDValue Sat, ISD::CvtCode Code);

  /// getZeroExtendInReg - Return the expression required to zero extend the Op
  /// value assuming it was the smaller SrcTy value.
  SDValue getZeroExtendInReg(SDValue Op, MVT SrcTy);
  SDValue getZeroExtendInReg(SDValue Op, DebugLoc DL, MVT SrcTy);
  
  /// getNOT - Create a bitwise NOT operation as (XOR Val, -1).
  SDValue getNOT(DebugLoc DL, SDValue Val, MVT VT);

  /// getCALLSEQ_START - Return a new CALLSEQ_START node, which always must have
  /// a flag result (to ensure it's not CSE'd).
  SDValue getCALLSEQ_START(SDValue Chain, SDValue Op) {
    const MVT *VTs = getNodeValueTypes(MVT::Other, MVT::Flag);
    SDValue Ops[] = { Chain,  Op };
    return getNode(ISD::CALLSEQ_START, VTs, 2, Ops, 2);
  }

  /// getCALLSEQ_END - Return a new CALLSEQ_END node, which always must have a
  /// flag result (to ensure it's not CSE'd).
  SDValue getCALLSEQ_END(SDValue Chain, SDValue Op1, SDValue Op2,
                           SDValue InFlag) {
    SDVTList NodeTys = getVTList(MVT::Other, MVT::Flag);
    SmallVector<SDValue, 4> Ops;
    Ops.push_back(Chain);
    Ops.push_back(Op1);
    Ops.push_back(Op2);
    Ops.push_back(InFlag);
    return getNode(ISD::CALLSEQ_END, NodeTys, &Ops[0],
                   (unsigned)Ops.size() - (InFlag.getNode() == 0 ? 1 : 0));
  }

  /// getNode - Gets or creates the specified node.
  ///
  SDValue getNode(unsigned Opcode, MVT VT);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT);
  SDValue getNode(unsigned Opcode, MVT VT, SDValue N);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT, SDValue N);
  SDValue getNode(unsigned Opcode, MVT VT, SDValue N1, SDValue N2);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT, SDValue N1, SDValue N2);
  SDValue getNode(unsigned Opcode, MVT VT,
                  SDValue N1, SDValue N2, SDValue N3);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT,
                  SDValue N1, SDValue N2, SDValue N3);
  SDValue getNode(unsigned Opcode, MVT VT,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4);
  SDValue getNode(unsigned Opcode, MVT VT,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                  SDValue N5);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                  SDValue N5);
  SDValue getNode(unsigned Opcode, MVT VT,
                  const SDUse *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT,
                  const SDUse *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, MVT VT,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, DebugLoc DL, MVT VT,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, const std::vector<MVT> &ResultTys,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, DebugLoc DL,
                  const std::vector<MVT> &ResultTys,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, const MVT *VTs, unsigned NumVTs,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, DebugLoc DL, const MVT *VTs, unsigned NumVTs,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, SDVTList VTs,
                  const SDValue *Ops, unsigned NumOps);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs,
                  const SDValue *Ops, unsigned NumOps);

  SDValue getNode(unsigned Opcode, SDVTList VTs);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs);
  SDValue getNode(unsigned Opcode, SDVTList VTs, SDValue N);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs, SDValue N);
  SDValue getNode(unsigned Opcode, SDVTList VTs, SDValue N1, SDValue N2);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2);
  SDValue getNode(unsigned Opcode, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3);
  SDValue getNode(unsigned Opcode, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4);
  SDValue getNode(unsigned Opcode, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                  SDValue N5);
  SDValue getNode(unsigned Opcode, DebugLoc DL, SDVTList VTs,
                  SDValue N1, SDValue N2, SDValue N3, SDValue N4,
                  SDValue N5);

  SDValue getMemcpy(SDValue Chain, SDValue Dst, SDValue Src,
                    SDValue Size, unsigned Align, bool AlwaysInline,
                    const Value *DstSV, uint64_t DstSVOff,
                    const Value *SrcSV, uint64_t SrcSVOff);

  SDValue getMemmove(SDValue Chain, SDValue Dst, SDValue Src,
                     SDValue Size, unsigned Align,
                     const Value *DstSV, uint64_t DstOSVff,
                     const Value *SrcSV, uint64_t SrcSVOff);

  SDValue getMemset(SDValue Chain, SDValue Dst, SDValue Src,
                    SDValue Size, unsigned Align,
                    const Value *DstSV, uint64_t DstSVOff);

  /// getSetCC - Helper function to make it easier to build SetCC's if you just
  /// have an ISD::CondCode instead of an SDValue.
  ///
  SDValue getSetCC(MVT VT, SDValue LHS, SDValue RHS,
                   ISD::CondCode Cond) {
    return getNode(ISD::SETCC, DebugLoc::getUnknownLoc(), VT,
                   LHS, RHS, getCondCode(Cond));
  }
  SDValue getSetCC(DebugLoc DL, MVT VT, SDValue LHS, SDValue RHS,
                   ISD::CondCode Cond) {
    return getNode(ISD::SETCC, DL, VT, LHS, RHS, getCondCode(Cond));
  }

  /// getVSetCC - Helper function to make it easier to build VSetCC's nodes
  /// if you just have an ISD::CondCode instead of an SDValue.
  ///
  SDValue getVSetCC(MVT VT, SDValue LHS, SDValue RHS,
                    ISD::CondCode Cond) {
    return getNode(ISD::VSETCC, DebugLoc::getUnknownLoc(), VT,
                   LHS, RHS, getCondCode(Cond));
  }
  SDValue getVSetCC(DebugLoc DL, MVT VT, SDValue LHS, SDValue RHS,
                    ISD::CondCode Cond) {
    return getNode(ISD::VSETCC, DL, VT, LHS, RHS, getCondCode(Cond));
  }

  /// getSelectCC - Helper function to make it easier to build SelectCC's if you
  /// just have an ISD::CondCode instead of an SDValue.
  ///
  SDValue getSelectCC(SDValue LHS, SDValue RHS,
                      SDValue True, SDValue False, ISD::CondCode Cond) {
    return getNode(ISD::SELECT_CC, DebugLoc::getUnknownLoc(), True.getValueType(),
                   LHS, RHS, True, False, getCondCode(Cond));
  }
  SDValue getSelectCC(DebugLoc DL, SDValue LHS, SDValue RHS,
                      SDValue True, SDValue False, ISD::CondCode Cond) {
    return getNode(ISD::SELECT_CC, DL, True.getValueType(),
                   LHS, RHS, True, False, getCondCode(Cond));
  }
  
  /// getVAArg - VAArg produces a result and token chain, and takes a pointer
  /// and a source value as input.
  SDValue getVAArg(MVT VT, SDValue Chain, SDValue Ptr,
                   SDValue SV);

  /// getAtomic - Gets a node for an atomic op, produces result and chain and 
  /// takes 3 operands
  SDValue getAtomic(unsigned Opcode, MVT MemVT, SDValue Chain, SDValue Ptr, 
                    SDValue Cmp, SDValue Swp, const Value* PtrVal,
                    unsigned Alignment=0);
  SDValue getAtomic(unsigned Opcode, DebugLoc dl, MVT MemVT, SDValue Chain, 
                    SDValue Ptr, SDValue Cmp, SDValue Swp, const Value* PtrVal,
                    unsigned Alignment=0);

  /// getAtomic - Gets a node for an atomic op, produces result and chain and
  /// takes 2 operands.
  SDValue getAtomic(unsigned Opcode, MVT MemVT, SDValue Chain, SDValue Ptr, 
                    SDValue Val, const Value* PtrVal,
                    unsigned Alignment = 0);
  SDValue getAtomic(unsigned Opcode, DebugLoc dl, MVT MemVT, SDValue Chain,
                    SDValue Ptr, SDValue Val, const Value* PtrVal,
                    unsigned Alignment = 0);

  /// getMemIntrinsicNode - Creates a MemIntrinsicNode that may produce a
  /// result and takes a list of operands.
  SDValue getMemIntrinsicNode(unsigned Opcode,
                              const MVT *VTs, unsigned NumVTs,
                              const SDValue *Ops, unsigned NumOps,
                              MVT MemVT, const Value *srcValue, int SVOff,
                              unsigned Align = 0, bool Vol = false,
                              bool ReadMem = true, bool WriteMem = true);
  SDValue getMemIntrinsicNode(unsigned Opcode, DebugLoc dl,
                              const MVT *VTs, unsigned NumVTs,
                              const SDValue *Ops, unsigned NumOps,
                              MVT MemVT, const Value *srcValue, int SVOff,
                              unsigned Align = 0, bool Vol = false,
                              bool ReadMem = true, bool WriteMem = true);

  SDValue getMemIntrinsicNode(unsigned Opcode, SDVTList VTList,
                              const SDValue *Ops, unsigned NumOps,
                              MVT MemVT, const Value *srcValue, int SVOff,
                              unsigned Align = 0, bool Vol = false,
                              bool ReadMem = true, bool WriteMem = true);
  SDValue getMemIntrinsicNode(unsigned Opcode, DebugLoc dl, SDVTList VTList,
                              const SDValue *Ops, unsigned NumOps,
                              MVT MemVT, const Value *srcValue, int SVOff,
                              unsigned Align = 0, bool Vol = false,
                              bool ReadMem = true, bool WriteMem = true);

  /// getMergeValues - Create a MERGE_VALUES node from the given operands.
  SDValue getMergeValues(const SDValue *Ops, unsigned NumOps);
  SDValue getMergeValues(const SDValue *Ops, unsigned NumOps, DebugLoc dl);

  /// getCall - Create a CALL node from the given information.
  ///
  SDValue getCall(unsigned CallingConv, bool IsVarArgs, bool IsTailCall,
                  bool isInreg, SDVTList VTs, const SDValue *Operands, 
                  unsigned NumOperands);
  SDValue getCall(unsigned CallingConv, DebugLoc dl, bool IsVarArgs,
                  bool IsTailCall, bool isInreg, SDVTList VTs,
                  const SDValue *Operands, unsigned NumOperands);

  /// getLoad - Loads are not normal binary operators: their result type is not
  /// determined by their operands, and they produce a value AND a token chain.
  ///
  SDValue getLoad(MVT VT, SDValue Chain, SDValue Ptr,
                    const Value *SV, int SVOffset, bool isVolatile=false,
                    unsigned Alignment=0);
  SDValue getLoad(MVT VT, DebugLoc dl, SDValue Chain, SDValue Ptr,
                    const Value *SV, int SVOffset, bool isVolatile=false,
                    unsigned Alignment=0);
  SDValue getExtLoad(ISD::LoadExtType ExtType, MVT VT,
                       SDValue Chain, SDValue Ptr, const Value *SV,
                       int SVOffset, MVT EVT, bool isVolatile=false,
                       unsigned Alignment=0);
  SDValue getExtLoad(ISD::LoadExtType ExtType, DebugLoc dl, MVT VT,
                       SDValue Chain, SDValue Ptr, const Value *SV,
                       int SVOffset, MVT EVT, bool isVolatile=false,
                       unsigned Alignment=0);
  SDValue getIndexedLoad(SDValue OrigLoad, SDValue Base,
                           SDValue Offset, ISD::MemIndexedMode AM);
  SDValue getIndexedLoad(SDValue OrigLoad, DebugLoc dl, SDValue Base,
                           SDValue Offset, ISD::MemIndexedMode AM);
  SDValue getLoad(ISD::MemIndexedMode AM, ISD::LoadExtType ExtType,
                    MVT VT, SDValue Chain,
                    SDValue Ptr, SDValue Offset,
                    const Value *SV, int SVOffset, MVT EVT,
                    bool isVolatile=false, unsigned Alignment=0);
  SDValue getLoad(ISD::MemIndexedMode AM, DebugLoc dl, ISD::LoadExtType ExtType,
                    MVT VT, SDValue Chain,
                    SDValue Ptr, SDValue Offset,
                    const Value *SV, int SVOffset, MVT EVT,
                    bool isVolatile=false, unsigned Alignment=0);

  /// getStore - Helper function to build ISD::STORE nodes.
  ///
  SDValue getStore(SDValue Chain, SDValue Val, SDValue Ptr,
                     const Value *SV, int SVOffset, bool isVolatile=false,
                     unsigned Alignment=0);
  SDValue getStore(SDValue Chain, DebugLoc dl, SDValue Val, SDValue Ptr,
                     const Value *SV, int SVOffset, bool isVolatile=false,
                     unsigned Alignment=0);
  SDValue getTruncStore(SDValue Chain, SDValue Val, SDValue Ptr,
                          const Value *SV, int SVOffset, MVT TVT,
                          bool isVolatile=false, unsigned Alignment=0);
  SDValue getTruncStore(SDValue Chain, DebugLoc dl, SDValue Val, SDValue Ptr,
                          const Value *SV, int SVOffset, MVT TVT,
                          bool isVolatile=false, unsigned Alignment=0);
  SDValue getIndexedStore(SDValue OrigStoe, SDValue Base,
                           SDValue Offset, ISD::MemIndexedMode AM);
  SDValue getIndexedStore(SDValue OrigStoe, DebugLoc dl, SDValue Base,
                           SDValue Offset, ISD::MemIndexedMode AM);

  /// getSrcValue - Construct a node to track a Value* through the backend.
  SDValue getSrcValue(const Value *v);

  /// getMemOperand - Construct a node to track a memory reference
  /// through the backend.
  SDValue getMemOperand(const MachineMemOperand &MO);

  /// getShiftAmountOperand - Return the specified value casted to
  /// the target's desired shift amount type.
  SDValue getShiftAmountOperand(SDValue Op);

  /// UpdateNodeOperands - *Mutate* the specified node in-place to have the
  /// specified operands.  If the resultant node already exists in the DAG,
  /// this does not modify the specified node, instead it returns the node that
  /// already exists.  If the resultant node does not exist in the DAG, the
  /// input node is returned.  As a degenerate case, if you specify the same
  /// input operands as the node already has, the input node is returned.
  SDValue UpdateNodeOperands(SDValue N, SDValue Op);
  SDValue UpdateNodeOperands(SDValue N, SDValue Op1, SDValue Op2);
  SDValue UpdateNodeOperands(SDValue N, SDValue Op1, SDValue Op2,
                               SDValue Op3);
  SDValue UpdateNodeOperands(SDValue N, SDValue Op1, SDValue Op2,
                               SDValue Op3, SDValue Op4);
  SDValue UpdateNodeOperands(SDValue N, SDValue Op1, SDValue Op2,
                               SDValue Op3, SDValue Op4, SDValue Op5);
  SDValue UpdateNodeOperands(SDValue N,
                               const SDValue *Ops, unsigned NumOps);
  
  /// SelectNodeTo - These are used for target selectors to *mutate* the
  /// specified node to have the specified return type, Target opcode, and
  /// operands.  Note that target opcodes are stored as
  /// ~TargetOpcode in the node opcode field.  The resultant node is returned.
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT, SDValue Op1);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT,
                       SDValue Op1, SDValue Op2);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT,
                       SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT,
                       const SDValue *Ops, unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1, MVT VT2);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1,
                       MVT VT2, const SDValue *Ops, unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1,
                       MVT VT2, MVT VT3, const SDValue *Ops, unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned MachineOpc, MVT VT1,
                       MVT VT2, MVT VT3, MVT VT4, const SDValue *Ops,
                       unsigned NumOps);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1,
                       MVT VT2, SDValue Op1);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1,
                       MVT VT2, SDValue Op1, SDValue Op2);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1,
                       MVT VT2, SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, MVT VT1,
                       MVT VT2, MVT VT3, SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *SelectNodeTo(SDNode *N, unsigned TargetOpc, SDVTList VTs,
                       const SDValue *Ops, unsigned NumOps);

  /// MorphNodeTo - These *mutate* the specified node to have the specified
  /// return type, opcode, and operands.
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT, SDValue Op1);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT,
                      SDValue Op1, SDValue Op2);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT,
                      SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT,
                      const SDValue *Ops, unsigned NumOps);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT1, MVT VT2);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT1,
                      MVT VT2, const SDValue *Ops, unsigned NumOps);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT1,
                      MVT VT2, MVT VT3, const SDValue *Ops, unsigned NumOps);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT1,
                      MVT VT2, SDValue Op1);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT1,
                      MVT VT2, SDValue Op1, SDValue Op2);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, MVT VT1,
                      MVT VT2, SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *MorphNodeTo(SDNode *N, unsigned Opc, SDVTList VTs,
                      const SDValue *Ops, unsigned NumOps);

  /// getTargetNode - These are used for target selectors to create a new node
  /// with specified return type(s), target opcode, and operands.
  ///
  /// Note that getTargetNode returns the resultant node.  If there is already a
  /// node of the specified opcode and operands, it returns that node instead of
  /// the current one.
  SDNode *getTargetNode(unsigned Opcode, MVT VT);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT);

  SDNode *getTargetNode(unsigned Opcode, MVT VT, SDValue Op1);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT, SDValue Op1);

  SDNode *getTargetNode(unsigned Opcode, MVT VT, SDValue Op1, SDValue Op2);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT, SDValue Op1, 
                        SDValue Op2);

  SDNode *getTargetNode(unsigned Opcode, MVT VT,
                        SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT,
                        SDValue Op1, SDValue Op2, SDValue Op3);

  SDNode *getTargetNode(unsigned Opcode, MVT VT,
                        const SDValue *Ops, unsigned NumOps);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT,
                        const SDValue *Ops, unsigned NumOps);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2, SDValue Op1);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2, 
                        SDValue Op1);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1,
                        MVT VT2, SDValue Op1, SDValue Op2);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1,
                        MVT VT2, SDValue Op1, SDValue Op2);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1,
                        MVT VT2, SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1,
                        MVT VT2, SDValue Op1, SDValue Op2, SDValue Op3);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2,
                        const SDValue *Ops, unsigned NumOps);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2,
                        const SDValue *Ops, unsigned NumOps);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2, MVT VT3,
                        SDValue Op1, SDValue Op2);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2, MVT VT3,
                        SDValue Op1, SDValue Op2);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2, MVT VT3,
                        SDValue Op1, SDValue Op2, SDValue Op3);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2, MVT VT3,
                        SDValue Op1, SDValue Op2, SDValue Op3);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2, MVT VT3,
                        const SDValue *Ops, unsigned NumOps);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2, MVT VT3,
                        const SDValue *Ops, unsigned NumOps);

  SDNode *getTargetNode(unsigned Opcode, MVT VT1, MVT VT2, MVT VT3, MVT VT4,
                        const SDValue *Ops, unsigned NumOps);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl, MVT VT1, MVT VT2, MVT VT3,
                        MVT VT4, const SDValue *Ops, unsigned NumOps);

  SDNode *getTargetNode(unsigned Opcode, const std::vector<MVT> &ResultTys,
                        const SDValue *Ops, unsigned NumOps);
  SDNode *getTargetNode(unsigned Opcode, DebugLoc dl,
                        const std::vector<MVT> &ResultTys, const SDValue *Ops,
                        unsigned NumOps);

  /// getNodeIfExists - Get the specified node if it's already available, or
  /// else return NULL.
  SDNode *getNodeIfExists(unsigned Opcode, SDVTList VTs,
                          const SDValue *Ops, unsigned NumOps);
  
  /// DAGUpdateListener - Clients of various APIs that cause global effects on
  /// the DAG can optionally implement this interface.  This allows the clients
  /// to handle the various sorts of updates that happen.
  class DAGUpdateListener {
  public:
    virtual ~DAGUpdateListener();

    /// NodeDeleted - The node N that was deleted and, if E is not null, an
    /// equivalent node E that replaced it.
    virtual void NodeDeleted(SDNode *N, SDNode *E) = 0;

    /// NodeUpdated - The node N that was updated.
    virtual void NodeUpdated(SDNode *N) = 0;
  };
  
  /// RemoveDeadNode - Remove the specified node from the system. If any of its
  /// operands then becomes dead, remove them as well. Inform UpdateListener
  /// for each node deleted.
  void RemoveDeadNode(SDNode *N, DAGUpdateListener *UpdateListener = 0);
  
  /// RemoveDeadNodes - This method deletes the unreachable nodes in the
  /// given list, and any nodes that become unreachable as a result.
  void RemoveDeadNodes(SmallVectorImpl<SDNode *> &DeadNodes,
                       DAGUpdateListener *UpdateListener = 0);

  /// ReplaceAllUsesWith - Modify anything using 'From' to use 'To' instead.
  /// This can cause recursive merging of nodes in the DAG.  Use the first
  /// version if 'From' is known to have a single result, use the second
  /// if you have two nodes with identical results, use the third otherwise.
  ///
  /// These methods all take an optional UpdateListener, which (if not null) is 
  /// informed about nodes that are deleted and modified due to recursive
  /// changes in the dag.
  ///
  /// These functions only replace all existing uses. It's possible that as
  /// these replacements are being performed, CSE may cause the From node
  /// to be given new uses. These new uses of From are left in place, and
  /// not automatically transfered to To.
  ///
  void ReplaceAllUsesWith(SDValue From, SDValue Op,
                          DAGUpdateListener *UpdateListener = 0);
  void ReplaceAllUsesWith(SDNode *From, SDNode *To,
                          DAGUpdateListener *UpdateListener = 0);
  void ReplaceAllUsesWith(SDNode *From, const SDValue *To,
                          DAGUpdateListener *UpdateListener = 0);

  /// ReplaceAllUsesOfValueWith - Replace any uses of From with To, leaving
  /// uses of other values produced by From.Val alone.
  void ReplaceAllUsesOfValueWith(SDValue From, SDValue To,
                                 DAGUpdateListener *UpdateListener = 0);

  /// ReplaceAllUsesOfValuesWith - Like ReplaceAllUsesOfValueWith, but
  /// for multiple values at once. This correctly handles the case where
  /// there is an overlap between the From values and the To values.
  void ReplaceAllUsesOfValuesWith(const SDValue *From, const SDValue *To,
                                  unsigned Num,
                                  DAGUpdateListener *UpdateListener = 0);

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
    case ISD::ADDC: 
    case ISD::ADDE: return true;
    default: return false;
    }
  }

  void dump() const;

  /// CreateStackTemporary - Create a stack temporary, suitable for holding the
  /// specified value type.  If minAlign is specified, the slot size will have
  /// at least that alignment.
  SDValue CreateStackTemporary(MVT VT, unsigned minAlign = 1);

  /// CreateStackTemporary - Create a stack temporary suitable for holding
  /// either of the specified value types.
  SDValue CreateStackTemporary(MVT VT1, MVT VT2);

  /// FoldConstantArithmetic - 
  SDValue FoldConstantArithmetic(unsigned Opcode,
                                 MVT VT,
                                 ConstantSDNode *Cst1,
                                 ConstantSDNode *Cst2);

  /// FoldSetCC - Constant fold a setcc to true or false.
  SDValue FoldSetCC(MVT VT, SDValue N1,
                    SDValue N2, ISD::CondCode Cond, DebugLoc dl);
  
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
  void ComputeMaskedBits(SDValue Op, const APInt &Mask, APInt &KnownZero,
                         APInt &KnownOne, unsigned Depth = 0) const;

  /// ComputeNumSignBits - Return the number of times the sign bit of the
  /// register is replicated into the other bits.  We know that at least 1 bit
  /// is always equal to the sign bit (itself), but other cases can give us
  /// information.  For example, immediately after an "SRA X, 2", we know that
  /// the top 3 bits are all equal to each other, so we return 3.  Targets can
  /// implement the ComputeNumSignBitsForTarget method in the TargetLowering
  /// class to allow target nodes to be understood.
  unsigned ComputeNumSignBits(SDValue Op, unsigned Depth = 0) const;

  /// isVerifiedDebugInfoDesc - Returns true if the specified SDValue has
  /// been verified as a debug information descriptor.
  bool isVerifiedDebugInfoDesc(SDValue Op) const;

  /// getShuffleScalarElt - Returns the scalar element that will make up the ith
  /// element of the result of the vector shuffle.
  SDValue getShuffleScalarElt(const SDNode *N, unsigned Idx);
  
private:
  bool RemoveNodeFromCSEMaps(SDNode *N);
  void AddModifiedNodeToCSEMaps(SDNode *N, DAGUpdateListener *UpdateListener);
  SDNode *FindModifiedNodeSlot(SDNode *N, SDValue Op, void *&InsertPos);
  SDNode *FindModifiedNodeSlot(SDNode *N, SDValue Op1, SDValue Op2,
                               void *&InsertPos);
  SDNode *FindModifiedNodeSlot(SDNode *N, const SDValue *Ops, unsigned NumOps,
                               void *&InsertPos);

  void DeleteNodeNotInCSEMaps(SDNode *N);
  void DeallocateNode(SDNode *N);

  unsigned getMVTAlignment(MVT MemoryVT) const;

  void allnodes_clear();
  
  /// VTList - List of non-single value types.
  std::vector<SDVTList> VTList;
  
  /// CondCodeNodes - Maps to auto-CSE operations.
  std::vector<CondCodeSDNode*> CondCodeNodes;

  std::vector<SDNode*> ValueTypeNodes;
  std::map<MVT, SDNode*, MVT::compareRawBits> ExtendedValueTypeNodes;
  StringMap<SDNode*> ExternalSymbols;
  StringMap<SDNode*> TargetExternalSymbols;
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
