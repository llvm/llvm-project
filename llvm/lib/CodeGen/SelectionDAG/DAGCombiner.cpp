//===-- DAGCombiner.cpp - Implement a DAG node combiner -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass combines dag nodes to form fewer, simpler DAG nodes.  It can be run
// both before and after the DAG is legalized.
//
// FIXME: Missing folds
// sdiv, udiv, srem, urem (X, const) where X is an integer can be expanded into
//  a sequence of multiplies, shifts, and adds.  This should be controlled by
//  some kind of hint from the target that int div is expensive.
// various folds of mulh[s,u] by constants such as -1, powers of 2, etc.
//
// FIXME: Should add a corresponding version of fold AND with
// ZERO_EXTEND/SIGN_EXTEND by converting them to an ANY_EXTEND node which
// we don't have yet.
//
// FIXME: select C, pow2, pow2 -> something smart
// FIXME: trunc(select X, Y, Z) -> select X, trunc(Y), trunc(Z)
// FIXME: Dead stores -> nuke
// FIXME: shr X, (and Y,31) -> shr X, Y   (TRICKY!)
// FIXME: mul (x, const) -> shifts + adds
// FIXME: undef values
// FIXME: make truncate see through SIGN_EXTEND and AND
// FIXME: (sra (sra x, c1), c2) -> (sra x, c1+c2)
// FIXME: verify that getNode can't return extends with an operand whose type
//        is >= to that of the extend.
// FIXME: divide by zero is currently left unfolded.  do we want to turn this
//        into an undef?
// FIXME: select ne (select cc, 1, 0), 0, true, false -> select cc, true, false
// FIXME: reassociate (X+C)+Y  into (X+Y)+C  if the inner expression has one use
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dagcombine"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetLowering.h"
#include <algorithm>
#include <cmath>
using namespace llvm;

namespace {
  Statistic<> NodesCombined ("dagcombiner", "Number of dag nodes combined");

  class DAGCombiner {
    SelectionDAG &DAG;
    TargetLowering &TLI;
    bool AfterLegalize;

    // Worklist of all of the nodes that need to be simplified.
    std::vector<SDNode*> WorkList;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(SDNode *N) {
      for (SDNode::use_iterator UI = N->use_begin(), UE = N->use_end();
           UI != UE; ++UI)
        WorkList.push_back(*UI);
    }

    /// removeFromWorkList - remove all instances of N from the worklist.
    void removeFromWorkList(SDNode *N) {
      WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), N),
                     WorkList.end());
    }
    
    SDOperand CombineTo(SDNode *N, const std::vector<SDOperand> &To) {
      ++NodesCombined;
      DEBUG(std::cerr << "\nReplacing "; N->dump();
            std::cerr << "\nWith: "; To[0].Val->dump();
            std::cerr << " and " << To.size()-1 << " other values\n");
      std::vector<SDNode*> NowDead;
      DAG.ReplaceAllUsesWith(N, To, &NowDead);
      
      // Push the new nodes and any users onto the worklist
      for (unsigned i = 0, e = To.size(); i != e; ++i) {
        WorkList.push_back(To[i].Val);
        AddUsersToWorkList(To[i].Val);
      }
      
      // Nodes can end up on the worklist more than once.  Make sure we do
      // not process a node that has been replaced.
      removeFromWorkList(N);
      for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
        removeFromWorkList(NowDead[i]);
      
      // Finally, since the node is now dead, remove it from the graph.
      DAG.DeleteNode(N);
      return SDOperand(N, 0);
    }

    SDOperand CombineTo(SDNode *N, SDOperand Res) {
      std::vector<SDOperand> To;
      To.push_back(Res);
      return CombineTo(N, To);
    }
    
    SDOperand CombineTo(SDNode *N, SDOperand Res0, SDOperand Res1) {
      std::vector<SDOperand> To;
      To.push_back(Res0);
      To.push_back(Res1);
      return CombineTo(N, To);
    }
    
    /// visit - call the node-specific routine that knows how to fold each
    /// particular type of node.
    SDOperand visit(SDNode *N);

    // Visitation implementation - Implement dag node combining for different
    // node types.  The semantics are as follows:
    // Return Value:
    //   SDOperand.Val == 0   - No change was made
    //   SDOperand.Val == N   - N was replaced, is dead, and is already handled.
    //   otherwise            - N should be replaced by the returned Operand.
    //
    SDOperand visitTokenFactor(SDNode *N);
    SDOperand visitADD(SDNode *N);
    SDOperand visitSUB(SDNode *N);
    SDOperand visitMUL(SDNode *N);
    SDOperand visitSDIV(SDNode *N);
    SDOperand visitUDIV(SDNode *N);
    SDOperand visitSREM(SDNode *N);
    SDOperand visitUREM(SDNode *N);
    SDOperand visitMULHU(SDNode *N);
    SDOperand visitMULHS(SDNode *N);
    SDOperand visitAND(SDNode *N);
    SDOperand visitOR(SDNode *N);
    SDOperand visitXOR(SDNode *N);
    SDOperand visitSHL(SDNode *N);
    SDOperand visitSRA(SDNode *N);
    SDOperand visitSRL(SDNode *N);
    SDOperand visitCTLZ(SDNode *N);
    SDOperand visitCTTZ(SDNode *N);
    SDOperand visitCTPOP(SDNode *N);
    SDOperand visitSELECT(SDNode *N);
    SDOperand visitSELECT_CC(SDNode *N);
    SDOperand visitSETCC(SDNode *N);
    SDOperand visitADD_PARTS(SDNode *N);
    SDOperand visitSUB_PARTS(SDNode *N);
    SDOperand visitSIGN_EXTEND(SDNode *N);
    SDOperand visitZERO_EXTEND(SDNode *N);
    SDOperand visitSIGN_EXTEND_INREG(SDNode *N);
    SDOperand visitTRUNCATE(SDNode *N);
    
    SDOperand visitFADD(SDNode *N);
    SDOperand visitFSUB(SDNode *N);
    SDOperand visitFMUL(SDNode *N);
    SDOperand visitFDIV(SDNode *N);
    SDOperand visitFREM(SDNode *N);
    SDOperand visitSINT_TO_FP(SDNode *N);
    SDOperand visitUINT_TO_FP(SDNode *N);
    SDOperand visitFP_TO_SINT(SDNode *N);
    SDOperand visitFP_TO_UINT(SDNode *N);
    SDOperand visitFP_ROUND(SDNode *N);
    SDOperand visitFP_ROUND_INREG(SDNode *N);
    SDOperand visitFP_EXTEND(SDNode *N);
    SDOperand visitFNEG(SDNode *N);
    SDOperand visitFABS(SDNode *N);
    SDOperand visitBRCOND(SDNode *N);
    SDOperand visitBRCONDTWOWAY(SDNode *N);
    SDOperand visitBR_CC(SDNode *N);
    SDOperand visitBRTWOWAY_CC(SDNode *N);

    SDOperand visitLOAD(SDNode *N);
    SDOperand visitSTORE(SDNode *N);

    bool SimplifySelectOps(SDNode *SELECT, SDOperand LHS, SDOperand RHS);
    SDOperand SimplifySelect(SDOperand N0, SDOperand N1, SDOperand N2);
    SDOperand SimplifySelectCC(SDOperand N0, SDOperand N1, SDOperand N2, 
                               SDOperand N3, ISD::CondCode CC);
    SDOperand SimplifySetCC(MVT::ValueType VT, SDOperand N0, SDOperand N1,
                            ISD::CondCode Cond, bool foldBooleans = true);
    
    SDOperand BuildSDIV(SDNode *N);
    SDOperand BuildUDIV(SDNode *N);    
public:
    DAGCombiner(SelectionDAG &D)
      : DAG(D), TLI(D.getTargetLoweringInfo()), AfterLegalize(false) {}
    
    /// Run - runs the dag combiner on all nodes in the work list
    void Run(bool RunningAfterLegalize); 
  };
}

struct ms {
  int64_t m;  // magic number
  int64_t s;  // shift amount
};

struct mu {
  uint64_t m; // magic number
  int64_t a;  // add indicator
  int64_t s;  // shift amount
};

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static ms magic32(int32_t d) {
  int32_t p;
  uint32_t ad, anc, delta, q1, r1, q2, r2, t;
  const uint32_t two31 = 0x80000000U;
  struct ms mag;
  
  ad = abs(d);
  t = two31 + ((uint32_t)d >> 31);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 31;               // initialize p
  q1 = two31/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two31 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two31/ad;        // initialize q2 = 2p/abs(d)
  r2 = two31 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  
  mag.m = (int32_t)(q2 + 1); // make sure to sign extend
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 32;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static mu magicu32(uint32_t d) {
  int32_t p;
  uint32_t nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 31;                   // initialize p
  q1 = 0x80000000/nc;       // initialize q1 = 2p/nc
  r1 = 0x80000000 - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFF/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFF - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFF) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x80000000) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 32;  // resulting shift
  return magu;
}

/// magic - calculate the magic numbers required to codegen an integer sdiv as
/// a sequence of multiply and shifts.  Requires that the divisor not be 0, 1,
/// or -1.
static ms magic64(int64_t d) {
  int64_t p;
  uint64_t ad, anc, delta, q1, r1, q2, r2, t;
  const uint64_t two63 = 9223372036854775808ULL; // 2^63
  struct ms mag;
  
  ad = d >= 0 ? d : -d;
  t = two63 + ((uint64_t)d >> 63);
  anc = t - 1 - t%ad;   // absolute value of nc
  p = 63;               // initialize p
  q1 = two63/anc;       // initialize q1 = 2p/abs(nc)
  r1 = two63 - q1*anc;  // initialize r1 = rem(2p,abs(nc))
  q2 = two63/ad;        // initialize q2 = 2p/abs(d)
  r2 = two63 - q2*ad;   // initialize r2 = rem(2p,abs(d))
  do {
    p = p + 1;
    q1 = 2*q1;        // update q1 = 2p/abs(nc)
    r1 = 2*r1;        // update r1 = rem(2p/abs(nc))
    if (r1 >= anc) {  // must be unsigned comparison
      q1 = q1 + 1;
      r1 = r1 - anc;
    }
    q2 = 2*q2;        // update q2 = 2p/abs(d)
    r2 = 2*r2;        // update r2 = rem(2p/abs(d))
    if (r2 >= ad) {   // must be unsigned comparison
      q2 = q2 + 1;
      r2 = r2 - ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  
  mag.m = q2 + 1;
  if (d < 0) mag.m = -mag.m; // resulting magic number
  mag.s = p - 64;            // resulting shift
  return mag;
}

/// magicu - calculate the magic numbers required to codegen an integer udiv as
/// a sequence of multiply, add and shifts.  Requires that the divisor not be 0.
static mu magicu64(uint64_t d)
{
  int64_t p;
  uint64_t nc, delta, q1, r1, q2, r2;
  struct mu magu;
  magu.a = 0;               // initialize "add" indicator
  nc = - 1 - (-d)%d;
  p = 63;                   // initialize p
  q1 = 0x8000000000000000ull/nc;       // initialize q1 = 2p/nc
  r1 = 0x8000000000000000ull - q1*nc;  // initialize r1 = rem(2p,nc)
  q2 = 0x7FFFFFFFFFFFFFFFull/d;        // initialize q2 = (2p-1)/d
  r2 = 0x7FFFFFFFFFFFFFFFull - q2*d;   // initialize r2 = rem((2p-1),d)
  do {
    p = p + 1;
    if (r1 >= nc - r1 ) {
      q1 = 2*q1 + 1;  // update q1
      r1 = 2*r1 - nc; // update r1
    }
    else {
      q1 = 2*q1; // update q1
      r1 = 2*r1; // update r1
    }
    if (r2 + 1 >= d - r2) {
      if (q2 >= 0x7FFFFFFFFFFFFFFFull) magu.a = 1;
      q2 = 2*q2 + 1;     // update q2
      r2 = 2*r2 + 1 - d; // update r2
    }
    else {
      if (q2 >= 0x8000000000000000ull) magu.a = 1;
      q2 = 2*q2;     // update q2
      r2 = 2*r2 + 1; // update r2
    }
    delta = d - 1 - r2;
  } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 64;  // resulting shift
  return magu;
}

/// MaskedValueIsZero - Return true if 'Op & Mask' is known to be zero.  We use
/// this predicate to simplify operations downstream.  Op and Mask are known to
/// be the same type.
static bool MaskedValueIsZero(const SDOperand &Op, uint64_t Mask,
                              const TargetLowering &TLI) {
  unsigned SrcBits;
  if (Mask == 0) return true;
  
  // If we know the result of a setcc has the top bits zero, use this info.
  switch (Op.getOpcode()) {
  case ISD::Constant:
    return (cast<ConstantSDNode>(Op)->getValue() & Mask) == 0;
  case ISD::SETCC:
    return ((Mask & 1) == 0) &&
    TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult;
  case ISD::ZEXTLOAD:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(3))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::ZERO_EXTEND:
    SrcBits = MVT::getSizeInBits(Op.getOperand(0).getValueType());
    return MaskedValueIsZero(Op.getOperand(0),Mask & ((1ULL << SrcBits)-1),TLI);
  case ISD::AssertZext:
    SrcBits = MVT::getSizeInBits(cast<VTSDNode>(Op.getOperand(1))->getVT());
    return (Mask & ((1ULL << SrcBits)-1)) == 0; // Returning only the zext bits.
  case ISD::AND:
    // If either of the operands has zero bits, the result will too.
    if (MaskedValueIsZero(Op.getOperand(1), Mask, TLI) ||
        MaskedValueIsZero(Op.getOperand(0), Mask, TLI))
      return true;
    // (X & C1) & C2 == 0   iff   C1 & C2 == 0.
    if (ConstantSDNode *AndRHS = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      return MaskedValueIsZero(Op.getOperand(0),AndRHS->getValue() & Mask, TLI);
    return false;
  case ISD::OR:
  case ISD::XOR:
    return MaskedValueIsZero(Op.getOperand(0), Mask, TLI) &&
    MaskedValueIsZero(Op.getOperand(1), Mask, TLI);
  case ISD::SELECT:
    return MaskedValueIsZero(Op.getOperand(1), Mask, TLI) &&
    MaskedValueIsZero(Op.getOperand(2), Mask, TLI);
  case ISD::SELECT_CC:
    return MaskedValueIsZero(Op.getOperand(2), Mask, TLI) &&
    MaskedValueIsZero(Op.getOperand(3), Mask, TLI);
  case ISD::SRL:
    // (ushr X, C1) & C2 == 0   iff  X & (C2 << C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask << ShAmt->getValue();
      SrcBits = MVT::getSizeInBits(Op.getValueType());
      if (SrcBits != 64) NewVal &= (1ULL << SrcBits)-1;
      return MaskedValueIsZero(Op.getOperand(0), NewVal, TLI);
    }
    return false;
  case ISD::SHL:
    // (ushl X, C1) & C2 == 0   iff  X & (C2 >> C1) == 0
    if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t NewVal = Mask >> ShAmt->getValue();
      return MaskedValueIsZero(Op.getOperand(0), NewVal, TLI);
    }
    return false;
  case ISD::ADD:
    // (add X, Y) & C == 0 iff (X&C)|(Y&C) == 0 and all bits are low bits.
    if ((Mask&(Mask+1)) == 0) {  // All low bits
      if (MaskedValueIsZero(Op.getOperand(0), Mask, TLI) &&
          MaskedValueIsZero(Op.getOperand(1), Mask, TLI))
        return true;
    }
    break;
  case ISD::SUB:
    if (ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(Op.getOperand(0))) {
      // We know that the top bits of C-X are clear if X contains less bits
      // than C (i.e. no wrap-around can happen).  For example, 20-X is
      // positive if we can prove that X is >= 0 and < 16.
      unsigned Bits = MVT::getSizeInBits(CLHS->getValueType(0));
      if ((CLHS->getValue() & (1 << (Bits-1))) == 0) {  // sign bit clear
        unsigned NLZ = CountLeadingZeros_64(CLHS->getValue()+1);
        uint64_t MaskV = (1ULL << (63-NLZ))-1;
        if (MaskedValueIsZero(Op.getOperand(1), ~MaskV, TLI)) {
          // High bits are clear this value is known to be >= C.
          unsigned NLZ2 = CountLeadingZeros_64(CLHS->getValue());
          if ((Mask & ((1ULL << (64-NLZ2))-1)) == 0)
            return true;
        }
      }
    }
    break;
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP:
    // Bit counting instructions can not set the high bits of the result
    // register.  The max number of bits sets depends on the input.
    return (Mask & (MVT::getSizeInBits(Op.getValueType())*2-1)) == 0;
  default: break;
  }
  return false;
}

// isSetCCEquivalent - Return true if this node is a setcc, or is a select_cc
// that selects between the values 1 and 0, making it equivalent to a setcc.
// Also, set the incoming LHS, RHS, and CC references to the appropriate 
// nodes based on the type of node we are checking.  This simplifies life a
// bit for the callers.
static bool isSetCCEquivalent(SDOperand N, SDOperand &LHS, SDOperand &RHS,
                              SDOperand &CC) {
  if (N.getOpcode() == ISD::SETCC) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(2);
    return true;
  }
  if (N.getOpcode() == ISD::SELECT_CC && 
      N.getOperand(2).getOpcode() == ISD::Constant &&
      N.getOperand(3).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N.getOperand(2))->getValue() == 1 &&
      cast<ConstantSDNode>(N.getOperand(3))->isNullValue()) {
    LHS = N.getOperand(0);
    RHS = N.getOperand(1);
    CC  = N.getOperand(4);
    return true;
  }
  return false;
}

// isOneUseSetCC - Return true if this is a SetCC-equivalent operation with only
// one use.  If this is true, it allows the users to invert the operation for
// free when it is profitable to do so.
static bool isOneUseSetCC(SDOperand N) {
  SDOperand N0, N1, N2;
  if (isSetCCEquivalent(N, N0, N1, N2) && N.Val->hasOneUse())
    return true;
  return false;
}

// FIXME: This should probably go in the ISD class rather than being duplicated
// in several files.
static bool isCommutativeBinOp(unsigned Opcode) {
  switch (Opcode) {
    case ISD::ADD:
    case ISD::MUL:
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR: return true;
    default: return false; // FIXME: Need commutative info for user ops!
  }
}

void DAGCombiner::Run(bool RunningAfterLegalize) {
  // set the instance variable, so that the various visit routines may use it.
  AfterLegalize = RunningAfterLegalize;

  // Add all the dag nodes to the worklist.
  WorkList.insert(WorkList.end(), DAG.allnodes_begin(), DAG.allnodes_end());
  
  // Create a dummy node (which is not added to allnodes), that adds a reference
  // to the root node, preventing it from being deleted, and tracking any
  // changes of the root.
  HandleSDNode Dummy(DAG.getRoot());
  
  // while the worklist isn't empty, inspect the node on the end of it and
  // try and combine it.
  while (!WorkList.empty()) {
    SDNode *N = WorkList.back();
    WorkList.pop_back();
    
    // If N has no uses, it is dead.  Make sure to revisit all N's operands once
    // N is deleted from the DAG, since they too may now be dead or may have a
    // reduced number of uses, allowing other xforms.
    if (N->use_empty() && N != &Dummy) {
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        WorkList.push_back(N->getOperand(i).Val);
      
      removeFromWorkList(N);
      DAG.DeleteNode(N);
      continue;
    }
    
    SDOperand RV = visit(N);
    if (RV.Val) {
      ++NodesCombined;
      // If we get back the same node we passed in, rather than a new node or
      // zero, we know that the node must have defined multiple values and
      // CombineTo was used.  Since CombineTo takes care of the worklist 
      // mechanics for us, we have no work to do in this case.
      if (RV.Val != N) {
        DEBUG(std::cerr << "\nReplacing "; N->dump();
              std::cerr << "\nWith: "; RV.Val->dump();
              std::cerr << '\n');
        std::vector<SDNode*> NowDead;
        DAG.ReplaceAllUsesWith(N, std::vector<SDOperand>(1, RV), &NowDead);
          
        // Push the new node and any users onto the worklist
        WorkList.push_back(RV.Val);
        AddUsersToWorkList(RV.Val);
          
        // Nodes can end up on the worklist more than once.  Make sure we do
        // not process a node that has been replaced.
        removeFromWorkList(N);
        for (unsigned i = 0, e = NowDead.size(); i != e; ++i)
          removeFromWorkList(NowDead[i]);
        
        // Finally, since the node is now dead, remove it from the graph.
        DAG.DeleteNode(N);
      }
    }
  }
  
  // If the root changed (e.g. it was a dead load, update the root).
  DAG.setRoot(Dummy.getValue());
}

SDOperand DAGCombiner::visit(SDNode *N) {
  switch(N->getOpcode()) {
  default: break;
  case ISD::TokenFactor:        return visitTokenFactor(N);
  case ISD::ADD:                return visitADD(N);
  case ISD::SUB:                return visitSUB(N);
  case ISD::MUL:                return visitMUL(N);
  case ISD::SDIV:               return visitSDIV(N);
  case ISD::UDIV:               return visitUDIV(N);
  case ISD::SREM:               return visitSREM(N);
  case ISD::UREM:               return visitUREM(N);
  case ISD::MULHU:              return visitMULHU(N);
  case ISD::MULHS:              return visitMULHS(N);
  case ISD::AND:                return visitAND(N);
  case ISD::OR:                 return visitOR(N);
  case ISD::XOR:                return visitXOR(N);
  case ISD::SHL:                return visitSHL(N);
  case ISD::SRA:                return visitSRA(N);
  case ISD::SRL:                return visitSRL(N);
  case ISD::CTLZ:               return visitCTLZ(N);
  case ISD::CTTZ:               return visitCTTZ(N);
  case ISD::CTPOP:              return visitCTPOP(N);
  case ISD::SELECT:             return visitSELECT(N);
  case ISD::SELECT_CC:          return visitSELECT_CC(N);
  case ISD::SETCC:              return visitSETCC(N);
  case ISD::ADD_PARTS:          return visitADD_PARTS(N);
  case ISD::SUB_PARTS:          return visitSUB_PARTS(N);
  case ISD::SIGN_EXTEND:        return visitSIGN_EXTEND(N);
  case ISD::ZERO_EXTEND:        return visitZERO_EXTEND(N);
  case ISD::SIGN_EXTEND_INREG:  return visitSIGN_EXTEND_INREG(N);
  case ISD::TRUNCATE:           return visitTRUNCATE(N);
  case ISD::FADD:               return visitFADD(N);
  case ISD::FSUB:               return visitFSUB(N);
  case ISD::FMUL:               return visitFMUL(N);
  case ISD::FDIV:               return visitFDIV(N);
  case ISD::FREM:               return visitFREM(N);
  case ISD::SINT_TO_FP:         return visitSINT_TO_FP(N);
  case ISD::UINT_TO_FP:         return visitUINT_TO_FP(N);
  case ISD::FP_TO_SINT:         return visitFP_TO_SINT(N);
  case ISD::FP_TO_UINT:         return visitFP_TO_UINT(N);
  case ISD::FP_ROUND:           return visitFP_ROUND(N);
  case ISD::FP_ROUND_INREG:     return visitFP_ROUND_INREG(N);
  case ISD::FP_EXTEND:          return visitFP_EXTEND(N);
  case ISD::FNEG:               return visitFNEG(N);
  case ISD::FABS:               return visitFABS(N);
  case ISD::BRCOND:             return visitBRCOND(N);
  case ISD::BRCONDTWOWAY:       return visitBRCONDTWOWAY(N);
  case ISD::BR_CC:              return visitBR_CC(N);
  case ISD::BRTWOWAY_CC:        return visitBRTWOWAY_CC(N);
  case ISD::LOAD:               return visitLOAD(N);
  case ISD::STORE:              return visitSTORE(N);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitTokenFactor(SDNode *N) {
  std::vector<SDOperand> Ops;
  bool Changed = false;

  // If the token factor has two operands and one is the entry token, replace
  // the token factor with the other operand.
  if (N->getNumOperands() == 2) {
    if (N->getOperand(0).getOpcode() == ISD::EntryToken)
      return N->getOperand(1);
    if (N->getOperand(1).getOpcode() == ISD::EntryToken)
      return N->getOperand(0);
  }
  
  // fold (tokenfactor (tokenfactor)) -> tokenfactor
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    SDOperand Op = N->getOperand(i);
    if (Op.getOpcode() == ISD::TokenFactor && Op.hasOneUse()) {
      Changed = true;
      for (unsigned j = 0, e = Op.getNumOperands(); j != e; ++j)
        Ops.push_back(Op.getOperand(j));
    } else {
      Ops.push_back(Op);
    }
  }
  if (Changed)
    return DAG.getNode(ISD::TokenFactor, MVT::Other, Ops);
  return SDOperand();
}

SDOperand DAGCombiner::visitADD(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (add c1, c2) -> c1+c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() + N1C->getValue(), VT);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::ADD, VT, N1, N0);
  // fold (add x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (add (add x, c1), c2) -> (add x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::ADD) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::ADD, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getValue()+N00C->getValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::ADD, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getValue()+N01C->getValue(), VT));
  }
  // fold ((0-A) + B) -> B-A
  if (N0.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N0.getOperand(0)) &&
      cast<ConstantSDNode>(N0.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, VT, N1, N0.getOperand(1));
  // fold (A + (0-B)) -> A-B
  if (N1.getOpcode() == ISD::SUB && isa<ConstantSDNode>(N1.getOperand(0)) &&
      cast<ConstantSDNode>(N1.getOperand(0))->isNullValue())
    return DAG.getNode(ISD::SUB, VT, N0, N1.getOperand(1));
  // fold (A+(B-A)) -> B
  if (N1.getOpcode() == ISD::SUB && N0 == N1.getOperand(1))
    return N1.getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitSUB(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (sub x, x) -> 0
  if (N0 == N1)
    return DAG.getConstant(0, N->getValueType(0));
  
  // fold (sub c1, c2) -> c1-c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() - N1C->getValue(),
                           N->getValueType(0));
  // fold (sub x, c) -> (add x, -c)
  if (N1C)
    return DAG.getNode(ISD::ADD, N0.getValueType(), N0,
                       DAG.getConstant(-N1C->getValue(), N0.getValueType()));

  // fold (A+B)-A -> B
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(0) == N1)
    return N0.getOperand(1);
  // fold (A+B)-B -> A
  if (N0.getOpcode() == ISD::ADD && N0.getOperand(1) == N1)
    return N0.getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitMUL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (mul c1, c2) -> c1*c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() * N1C->getValue(), VT);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::MUL, VT, N1, N0);
  // fold (mul x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mul x, -1) -> 0-x
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), N0);
  // fold (mul x, (1 << c)) -> x << c
  if (N1C && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::SHL, VT, N0,
                       DAG.getConstant(Log2_64(N1C->getValue()),
                                       TLI.getShiftAmountTy()));
  // fold (mul x, -(1 << c)) -> -(x << c) or (-x) << c
  if (N1C && isPowerOf2_64(-N1C->getSignExtended())) {
    // FIXME: If the input is something that is easily negated (e.g. a 
    // single-use add), we should put the negate there.
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT),
                       DAG.getNode(ISD::SHL, VT, N0,
                            DAG.getConstant(Log2_64(-N1C->getSignExtended()),
                                            TLI.getShiftAmountTy())));
  }
  
  
  // fold (mul (mul x, c1), c2) -> (mul x, c1*c2)
  if (N1C && N0.getOpcode() == ISD::MUL) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::MUL, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getValue()*N00C->getValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::MUL, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getValue()*N01C->getValue(), VT));
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  MVT::ValueType VT = N->getValueType(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);

  // fold (sdiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getSignExtended() / N1C->getSignExtended(),
                           N->getValueType(0));
  // fold (sdiv X, 1) -> X
  if (N1C && N1C->getSignExtended() == 1LL)
    return N0;
  // fold (sdiv X, -1) -> 0-X
  if (N1C && N1C->isAllOnesValue())
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), N0);
  // If we know the sign bits of both operands are zero, strength reduce to a
  // udiv instead.  Handles (X&15) /s 4 -> X&15 >> 2
  uint64_t SignBit = 1ULL << (MVT::getSizeInBits(VT)-1);
  if (MaskedValueIsZero(N1, SignBit, TLI) &&
      MaskedValueIsZero(N0, SignBit, TLI))
    return DAG.getNode(ISD::UDIV, N1.getValueType(), N0, N1);
  // fold (sdiv X, pow2) -> (add (sra X, log(pow2)), (srl X, sizeof(X)-1))
  if (N1C && N1C->getValue() && !TLI.isIntDivCheap() && 
      (isPowerOf2_64(N1C->getSignExtended()) || 
       isPowerOf2_64(-N1C->getSignExtended()))) {
    // If dividing by powers of two is cheap, then don't perform the following
    // fold.
    if (TLI.isPow2DivCheap())
      return SDOperand();
    int64_t pow2 = N1C->getSignExtended();
    int64_t abs2 = pow2 > 0 ? pow2 : -pow2;
    SDOperand SRL = DAG.getNode(ISD::SRL, VT, N0,
                                DAG.getConstant(MVT::getSizeInBits(VT)-1,
                                                TLI.getShiftAmountTy()));
    WorkList.push_back(SRL.Val);
    SDOperand SGN = DAG.getNode(ISD::ADD, VT, N0, SRL);
    WorkList.push_back(SGN.Val);
    SDOperand SRA = DAG.getNode(ISD::SRA, VT, SGN, 
                                DAG.getConstant(Log2_64(abs2),
                                                TLI.getShiftAmountTy()));
    // If we're dividing by a positive value, we're done.  Otherwise, we must
    // negate the result.
    if (pow2 > 0)
      return SRA;
    WorkList.push_back(SRA.Val);
    return DAG.getNode(ISD::SUB, VT, DAG.getConstant(0, VT), SRA);
  }
  // if integer divide is expensive and we satisfy the requirements, emit an
  // alternate sequence.
  if (N1C && (N1C->getSignExtended() < -1 || N1C->getSignExtended() > 1) && 
      !TLI.isIntDivCheap()) {
    SDOperand Op = BuildSDIV(N);
    if (Op.Val) return Op;
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitUDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  MVT::ValueType VT = N->getValueType(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  
  // fold (udiv c1, c2) -> c1/c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getValue() / N1C->getValue(),
                           N->getValueType(0));
  // fold (udiv x, (1 << c)) -> x >>u c
  if (N1C && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::SRL, N->getValueType(0), N0,
                       DAG.getConstant(Log2_64(N1C->getValue()),
                                       TLI.getShiftAmountTy()));
  // fold (udiv x, c) -> alternate
  if (N1C && N1C->getValue() && !TLI.isIntDivCheap()) {
    SDOperand Op = BuildUDIV(N);
    if (Op.Val) return Op;
  }
      
  return SDOperand();
}

SDOperand DAGCombiner::visitSREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  MVT::ValueType VT = N->getValueType(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (srem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getSignExtended() % N1C->getSignExtended(),
                           N->getValueType(0));
  // If we know the sign bits of both operands are zero, strength reduce to a
  // urem instead.  Handles (X & 0x0FFFFFFF) %s 16 -> X&15
  uint64_t SignBit = 1ULL << (MVT::getSizeInBits(VT)-1);
  if (MaskedValueIsZero(N1, SignBit, TLI) &&
      MaskedValueIsZero(N0, SignBit, TLI))
    return DAG.getNode(ISD::UREM, N1.getValueType(), N0, N1);
  return SDOperand();
}

SDOperand DAGCombiner::visitUREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (urem c1, c2) -> c1%c2
  if (N0C && N1C && !N1C->isNullValue())
    return DAG.getConstant(N0C->getValue() % N1C->getValue(),
                           N->getValueType(0));
  // fold (urem x, pow2) -> (and x, pow2-1)
  if (N1C && !N1C->isNullValue() && isPowerOf2_64(N1C->getValue()))
    return DAG.getNode(ISD::AND, N0.getValueType(), N0, 
                       DAG.getConstant(N1C->getValue()-1, N1.getValueType()));
  return SDOperand();
}

SDOperand DAGCombiner::visitMULHS(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (mulhs x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhs x, 1) -> (sra x, size(x)-1)
  if (N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::SRA, N0.getValueType(), N0, 
                       DAG.getConstant(MVT::getSizeInBits(N0.getValueType())-1,
                                       TLI.getShiftAmountTy()));
  return SDOperand();
}

SDOperand DAGCombiner::visitMULHU(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // fold (mulhu x, 0) -> 0
  if (N1C && N1C->isNullValue())
    return N1;
  // fold (mulhu x, 1) -> 0
  if (N1C && N1C->getValue() == 1)
    return DAG.getConstant(0, N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitAND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N1.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (and c1, c2) -> c1&c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() & N1C->getValue(), VT);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::AND, VT, N1, N0);
  // fold (and x, -1) -> x
  if (N1C && N1C->isAllOnesValue())
    return N0;
  // if (and x, c) is known to be zero, return 0
  if (N1C && MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits),TLI))
    return DAG.getConstant(0, VT);
  // fold (and x, c) -> x iff (x & ~c) == 0
  if (N1C && MaskedValueIsZero(N0,~N1C->getValue() & (~0ULL>>(64-OpSizeInBits)),
                               TLI))
    return N0;
  // fold (and (and x, c1), c2) -> (and x, c1^c2)
  if (N1C && N0.getOpcode() == ISD::AND) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::AND, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getValue()&N00C->getValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getValue()&N01C->getValue(), VT));
  }
  // fold (and (sign_extend_inreg x, i16 to i32), 1) -> (and x, 1)
  if (N1C && N0.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    unsigned ExtendBits =
    MVT::getSizeInBits(cast<VTSDNode>(N0.getOperand(1))->getVT());
    if (ExtendBits == 64 || (N1C->getValue() & (~0ULL << ExtendBits) == 0))
      return DAG.getNode(ISD::AND, VT, N0.getOperand(0), N1);
  }
  // fold (and (or x, 0xFFFF), 0xFF) -> 0xFF
  if (N1C && N0.getOpcode() == ISD::OR)
    if (ConstantSDNode *ORI = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
      if ((ORI->getValue() & N1C->getValue()) == N1C->getValue())
        return N1;
  // fold (and (setcc x), (setcc y)) -> (setcc (and x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();
    
    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        MVT::isInteger(LL.getValueType())) {
      // fold (X == 0) & (Y == 0) -> (X|Y == 0)
      if (cast<ConstantSDNode>(LR)->getValue() == 0 && Op1 == ISD::SETEQ) {
        SDOperand ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        WorkList.push_back(ORNode.Val);
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
      // fold (X == -1) & (Y == -1) -> (X&Y == -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETEQ) {
        SDOperand ANDNode = DAG.getNode(ISD::AND, LR.getValueType(), LL, RL);
        WorkList.push_back(ANDNode.Val);
        return DAG.getSetCC(VT, ANDNode, LR, Op1);
      }
      // fold (X >  -1) & (Y >  -1) -> (X|Y > -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && Op1 == ISD::SETGT) {
        SDOperand ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        WorkList.push_back(ORNode.Val);
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = MVT::isInteger(LL.getValueType());
      ISD::CondCode Result = ISD::getSetCCAndOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID)
        return DAG.getSetCC(N0.getValueType(), LL, LR, Result);
    }
  }
  // fold (and (zext x), (zext y)) -> (zext (and x, y))
  if (N0.getOpcode() == ISD::ZERO_EXTEND && 
      N1.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()) {
    SDOperand ANDNode = DAG.getNode(ISD::AND, N0.getOperand(0).getValueType(),
                                    N0.getOperand(0), N1.getOperand(0));
    WorkList.push_back(ANDNode.Val);
    return DAG.getNode(ISD::ZERO_EXTEND, VT, ANDNode);
  }
  // fold (and (shl/srl x), (shl/srl y)) -> (shl/srl (and x, y))
  if (((N0.getOpcode() == ISD::SHL && N1.getOpcode() == ISD::SHL) ||
       (N0.getOpcode() == ISD::SRL && N1.getOpcode() == ISD::SRL)) &&
      N0.getOperand(1) == N1.getOperand(1)) {
    SDOperand ANDNode = DAG.getNode(ISD::AND, N0.getOperand(0).getValueType(),
                                    N0.getOperand(0), N1.getOperand(0));
    WorkList.push_back(ANDNode.Val);
    return DAG.getNode(N0.getOpcode(), VT, ANDNode, N0.getOperand(1));
  }
  // fold (and (sra)) -> (and (srl)) when possible.
  if (N0.getOpcode() == ISD::SRA && N0.Val->hasOneUse()) {
    if (ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
      // If the RHS of the AND has zeros where the sign bits of the SRA will
      // land, turn the SRA into an SRL.
      if (MaskedValueIsZero(N1, (~0ULL << (OpSizeInBits-N01C->getValue())) &
                            (~0ULL>>(64-OpSizeInBits)), TLI)) {
        WorkList.push_back(N);
        CombineTo(N0.Val, DAG.getNode(ISD::SRL, VT, N0.getOperand(0),
                                      N0.getOperand(1)));
        return SDOperand();
      }
    }
  }
  // fold (zext_inreg (extload x)) -> (zextload x)
  if (N0.getOpcode() == ISD::EXTLOAD) {
    MVT::ValueType EVT = cast<VTSDNode>(N0.getOperand(3))->getVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    if (MaskedValueIsZero(N1, ~0ULL << MVT::getSizeInBits(EVT), TLI) &&
        (!AfterLegalize || TLI.isOperationLegal(ISD::ZEXTLOAD, EVT))) {
      SDOperand ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, N0.getOperand(0),
                                         N0.getOperand(1), N0.getOperand(2),
                                         EVT);
      WorkList.push_back(N);
      CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
      return SDOperand();
    }
  }
  // fold (zext_inreg (sextload x)) -> (zextload x) iff load has one use
  if (N0.getOpcode() == ISD::SEXTLOAD && N0.hasOneUse()) {
    MVT::ValueType EVT = cast<VTSDNode>(N0.getOperand(3))->getVT();
    // If we zero all the possible extended bits, then we can turn this into
    // a zextload if we are running before legalize or the operation is legal.
    if (MaskedValueIsZero(N1, ~0ULL << MVT::getSizeInBits(EVT), TLI) &&
        (!AfterLegalize || TLI.isOperationLegal(ISD::ZEXTLOAD, EVT))) {
      SDOperand ExtLoad = DAG.getExtLoad(ISD::ZEXTLOAD, VT, N0.getOperand(0),
                                         N0.getOperand(1), N0.getOperand(2),
                                         EVT);
      WorkList.push_back(N);
      CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
      return SDOperand();
    }
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitOR(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LL, LR, RL, RR, CC0, CC1;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N1.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (or c1, c2) -> c1|c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() | N1C->getValue(),
                           N->getValueType(0));
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::OR, VT, N1, N0);
  // fold (or x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold (or x, -1) -> -1
  if (N1C && N1C->isAllOnesValue())
    return N1;
  // fold (or x, c) -> c iff (x & ~c) == 0
  if (N1C && MaskedValueIsZero(N0,~N1C->getValue() & (~0ULL>>(64-OpSizeInBits)),
                               TLI))
    return N1;
  // fold (or (or x, c1), c2) -> (or x, c1|c2)
  if (N1C && N0.getOpcode() == ISD::OR) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::OR, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getValue()|N00C->getValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::OR, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getValue()|N01C->getValue(), VT));
  } else if (N1C && N0.getOpcode() == ISD::AND && N0.Val->hasOneUse() &&
             isa<ConstantSDNode>(N0.getOperand(1))) {
    // Canonicalize (or (and X, c1), c2) -> (and (or X, c2), c1|c2)
    ConstantSDNode *C1 = cast<ConstantSDNode>(N0.getOperand(1));
    return DAG.getNode(ISD::AND, VT, DAG.getNode(ISD::OR, VT, N0.getOperand(0),
                                                 N1),
                       DAG.getConstant(N1C->getValue() | C1->getValue(), VT));
  }
  
  
  // fold (or (setcc x), (setcc y)) -> (setcc (or x, y))
  if (isSetCCEquivalent(N0, LL, LR, CC0) && isSetCCEquivalent(N1, RL, RR, CC1)){
    ISD::CondCode Op0 = cast<CondCodeSDNode>(CC0)->get();
    ISD::CondCode Op1 = cast<CondCodeSDNode>(CC1)->get();
    
    if (LR == RR && isa<ConstantSDNode>(LR) && Op0 == Op1 &&
        MVT::isInteger(LL.getValueType())) {
      // fold (X != 0) | (Y != 0) -> (X|Y != 0)
      // fold (X <  0) | (Y <  0) -> (X|Y < 0)
      if (cast<ConstantSDNode>(LR)->getValue() == 0 && 
          (Op1 == ISD::SETNE || Op1 == ISD::SETLT)) {
        SDOperand ORNode = DAG.getNode(ISD::OR, LR.getValueType(), LL, RL);
        WorkList.push_back(ORNode.Val);
        return DAG.getSetCC(VT, ORNode, LR, Op1);
      }
      // fold (X != -1) | (Y != -1) -> (X&Y != -1)
      // fold (X >  -1) | (Y >  -1) -> (X&Y >  -1)
      if (cast<ConstantSDNode>(LR)->isAllOnesValue() && 
          (Op1 == ISD::SETNE || Op1 == ISD::SETGT)) {
        SDOperand ANDNode = DAG.getNode(ISD::AND, LR.getValueType(), LL, RL);
        WorkList.push_back(ANDNode.Val);
        return DAG.getSetCC(VT, ANDNode, LR, Op1);
      }
    }
    // canonicalize equivalent to ll == rl
    if (LL == RR && LR == RL) {
      Op1 = ISD::getSetCCSwappedOperands(Op1);
      std::swap(RL, RR);
    }
    if (LL == RL && LR == RR) {
      bool isInteger = MVT::isInteger(LL.getValueType());
      ISD::CondCode Result = ISD::getSetCCOrOperation(Op0, Op1, isInteger);
      if (Result != ISD::SETCC_INVALID)
        return DAG.getSetCC(N0.getValueType(), LL, LR, Result);
    }
  }
  // fold (or (zext x), (zext y)) -> (zext (or x, y))
  if (N0.getOpcode() == ISD::ZERO_EXTEND && 
      N1.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()) {
    SDOperand ORNode = DAG.getNode(ISD::OR, N0.getOperand(0).getValueType(),
                                   N0.getOperand(0), N1.getOperand(0));
    WorkList.push_back(ORNode.Val);
    return DAG.getNode(ISD::ZERO_EXTEND, VT, ORNode);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitXOR(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand LHS, RHS, CC;
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  
  // fold (xor c1, c2) -> c1^c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() ^ N1C->getValue(), VT);
  // canonicalize constant to RHS
  if (N0C && !N1C)
    return DAG.getNode(ISD::XOR, VT, N1, N0);
  // fold (xor x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // fold !(x cc y) -> (x !cc y)
  if (N1C && N1C->getValue() == 1 && isSetCCEquivalent(N0, LHS, RHS, CC)) {
    bool isInt = MVT::isInteger(LHS.getValueType());
    ISD::CondCode NotCC = ISD::getSetCCInverse(cast<CondCodeSDNode>(CC)->get(),
                                               isInt);
    if (N0.getOpcode() == ISD::SETCC)
      return DAG.getSetCC(VT, LHS, RHS, NotCC);
    if (N0.getOpcode() == ISD::SELECT_CC)
      return DAG.getSelectCC(LHS, RHS, N0.getOperand(2),N0.getOperand(3),NotCC);
    assert(0 && "Unhandled SetCC Equivalent!");
    abort();
  }
  // fold !(x or y) -> (!x and !y) iff x or y are setcc
  if (N1C && N1C->getValue() == 1 && 
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isOneUseSetCC(RHS) || isOneUseSetCC(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      WorkList.push_back(LHS.Val); WorkList.push_back(RHS.Val);
      return DAG.getNode(NewOpcode, VT, LHS, RHS);
    }
  }
  // fold !(x or y) -> (!x and !y) iff x or y are constants
  if (N1C && N1C->isAllOnesValue() && 
      (N0.getOpcode() == ISD::OR || N0.getOpcode() == ISD::AND)) {
    SDOperand LHS = N0.getOperand(0), RHS = N0.getOperand(1);
    if (isa<ConstantSDNode>(RHS) || isa<ConstantSDNode>(LHS)) {
      unsigned NewOpcode = N0.getOpcode() == ISD::AND ? ISD::OR : ISD::AND;
      LHS = DAG.getNode(ISD::XOR, VT, LHS, N1);  // RHS = ~LHS
      RHS = DAG.getNode(ISD::XOR, VT, RHS, N1);  // RHS = ~RHS
      WorkList.push_back(LHS.Val); WorkList.push_back(RHS.Val);
      return DAG.getNode(NewOpcode, VT, LHS, RHS);
    }
  }
  // fold (xor (xor x, c1), c2) -> (xor x, c1^c2)
  if (N1C && N0.getOpcode() == ISD::XOR) {
    ConstantSDNode *N00C = dyn_cast<ConstantSDNode>(N0.getOperand(0));
    ConstantSDNode *N01C = dyn_cast<ConstantSDNode>(N0.getOperand(1));
    if (N00C)
      return DAG.getNode(ISD::XOR, VT, N0.getOperand(1),
                         DAG.getConstant(N1C->getValue()^N00C->getValue(), VT));
    if (N01C)
      return DAG.getNode(ISD::XOR, VT, N0.getOperand(0),
                         DAG.getConstant(N1C->getValue()^N01C->getValue(), VT));
  }
  // fold (xor x, x) -> 0
  if (N0 == N1)
    return DAG.getConstant(0, VT);
  // fold (xor (zext x), (zext y)) -> (zext (xor x, y))
  if (N0.getOpcode() == ISD::ZERO_EXTEND && 
      N1.getOpcode() == ISD::ZERO_EXTEND &&
      N0.getOperand(0).getValueType() == N1.getOperand(0).getValueType()) {
    SDOperand XORNode = DAG.getNode(ISD::XOR, N0.getOperand(0).getValueType(),
                                   N0.getOperand(0), N1.getOperand(0));
    WorkList.push_back(XORNode.Val);
    return DAG.getNode(ISD::ZERO_EXTEND, VT, XORNode);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSHL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (shl c1, c2) -> c1<<c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() << N1C->getValue(), VT);
  // fold (shl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (shl x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (shl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (shl x, c) is known to be zero, return 0
  if (N1C && MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits),TLI))
    return DAG.getConstant(0, VT);
  // fold (shl (shl x, c1), c2) -> 0 or (shl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SHL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SHL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  // fold (shl (srl x, c1), c2) -> (shl (and x, -1 << c1), c2-c1) or
  //                               (srl (and x, -1 << c1), c1-c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    SDOperand Mask = DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                                 DAG.getConstant(~0ULL << c1, VT));
    if (c2 > c1)
      return DAG.getNode(ISD::SHL, VT, Mask, 
                         DAG.getConstant(c2-c1, N1.getValueType()));
    else
      return DAG.getNode(ISD::SRL, VT, Mask, 
                         DAG.getConstant(c1-c2, N1.getValueType()));
  }
  // fold (shl (sra x, c1), c1) -> (and x, -1 << c1)
  if (N1C && N0.getOpcode() == ISD::SRA && N1 == N0.getOperand(1))
    return DAG.getNode(ISD::AND, VT, N0.getOperand(0),
                       DAG.getConstant(~0ULL << N1C->getValue(), VT));
  return SDOperand();
}

SDOperand DAGCombiner::visitSRA(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (sra c1, c2) -> c1>>c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getSignExtended() >> N1C->getValue(), VT);
  // fold (sra 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (sra -1, x) -> -1
  if (N0C && N0C->isAllOnesValue())
    return N0;
  // fold (sra x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (sra x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // If the sign bit is known to be zero, switch this to a SRL.
  if (MaskedValueIsZero(N0, (1ULL << (OpSizeInBits-1)), TLI))
    return DAG.getNode(ISD::SRL, VT, N0, N1);
  return SDOperand();
}

SDOperand DAGCombiner::visitSRL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  MVT::ValueType VT = N0.getValueType();
  unsigned OpSizeInBits = MVT::getSizeInBits(VT);
  
  // fold (srl c1, c2) -> c1 >>u c2
  if (N0C && N1C)
    return DAG.getConstant(N0C->getValue() >> N1C->getValue(), VT);
  // fold (srl 0, x) -> 0
  if (N0C && N0C->isNullValue())
    return N0;
  // fold (srl x, c >= size(x)) -> undef
  if (N1C && N1C->getValue() >= OpSizeInBits)
    return DAG.getNode(ISD::UNDEF, VT);
  // fold (srl x, 0) -> x
  if (N1C && N1C->isNullValue())
    return N0;
  // if (srl x, c) is known to be zero, return 0
  if (N1C && MaskedValueIsZero(SDOperand(N, 0), ~0ULL >> (64-OpSizeInBits),TLI))
    return DAG.getConstant(0, VT);
  // fold (srl (srl x, c1), c2) -> 0 or (srl x, c1+c2)
  if (N1C && N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant) {
    uint64_t c1 = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
    uint64_t c2 = N1C->getValue();
    if (c1 + c2 > OpSizeInBits)
      return DAG.getConstant(0, VT);
    return DAG.getNode(ISD::SRL, VT, N0.getOperand(0), 
                       DAG.getConstant(c1 + c2, N1.getValueType()));
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitCTLZ(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);

  // fold (ctlz c1) -> c2
  if (N0C)
    return DAG.getConstant(CountLeadingZeros_64(N0C->getValue()),
                           N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitCTTZ(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (cttz c1) -> c2
  if (N0C)
    return DAG.getConstant(CountTrailingZeros_64(N0C->getValue()),
                           N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitCTPOP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (ctpop c1) -> c2
  if (N0C)
    return DAG.getConstant(CountPopulation_64(N0C->getValue()),
                           N0.getValueType());
  return SDOperand();
}

SDOperand DAGCombiner::visitSELECT(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  MVT::ValueType VT = N->getValueType(0);

  // fold select C, X, X -> X
  if (N1 == N2)
    return N1;
  // fold select true, X, Y -> X
  if (N0C && !N0C->isNullValue())
    return N1;
  // fold select false, X, Y -> Y
  if (N0C && N0C->isNullValue())
    return N2;
  // fold select C, 1, X -> C | X
  if (MVT::i1 == VT && N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::OR, VT, N0, N2);
  // fold select C, 0, X -> ~C & X
  // FIXME: this should check for C type == X type, not i1?
  if (MVT::i1 == VT && N1C && N1C->isNullValue()) {
    SDOperand XORNode = DAG.getNode(ISD::XOR, VT, N0, DAG.getConstant(1, VT));
    WorkList.push_back(XORNode.Val);
    return DAG.getNode(ISD::AND, VT, XORNode, N2);
  }
  // fold select C, X, 1 -> ~C | X
  if (MVT::i1 == VT && N2C && N2C->getValue() == 1) {
    SDOperand XORNode = DAG.getNode(ISD::XOR, VT, N0, DAG.getConstant(1, VT));
    WorkList.push_back(XORNode.Val);
    return DAG.getNode(ISD::OR, VT, XORNode, N1);
  }
  // fold select C, X, 0 -> C & X
  // FIXME: this should check for C type == X type, not i1?
  if (MVT::i1 == VT && N2C && N2C->isNullValue())
    return DAG.getNode(ISD::AND, VT, N0, N1);
  // fold  X ? X : Y --> X ? 1 : Y --> X | Y
  if (MVT::i1 == VT && N0 == N1)
    return DAG.getNode(ISD::OR, VT, N0, N2);
  // fold X ? Y : X --> X ? Y : 0 --> X & Y
  if (MVT::i1 == VT && N0 == N2)
    return DAG.getNode(ISD::AND, VT, N0, N1);
  
  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N1, N2))
    return SDOperand();
  
  // fold selects based on a setcc into other things, such as min/max/abs
  if (N0.getOpcode() == ISD::SETCC)
    return SimplifySelect(N0, N1, N2);
  return SDOperand();
}

SDOperand DAGCombiner::visitSELECT_CC(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  SDOperand N3 = N->getOperand(3);
  SDOperand N4 = N->getOperand(4);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
  ISD::CondCode CC = cast<CondCodeSDNode>(N4)->get();
  
  // Determine if the condition we're dealing with is constant
  SDOperand SCC = SimplifySetCC(TLI.getSetCCResultTy(), N0, N1, CC, false);
  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.Val);
  
  // fold select_cc lhs, rhs, x, x, cc -> x
  if (N2 == N3)
    return N2;
  
  // If we can fold this based on the true/false value, do so.
  if (SimplifySelectOps(N, N2, N3))
    return SDOperand();
  
  // fold select_cc into other things, such as min/max/abs
  return SimplifySelectCC(N0, N1, N2, N3, CC);
}

SDOperand DAGCombiner::visitSETCC(SDNode *N) {
  return SimplifySetCC(N->getValueType(0), N->getOperand(0), N->getOperand(1),
                       cast<CondCodeSDNode>(N->getOperand(2))->get());
}

SDOperand DAGCombiner::visitADD_PARTS(SDNode *N) {
  SDOperand LHSLo = N->getOperand(0);
  SDOperand RHSLo = N->getOperand(2);
  MVT::ValueType VT = LHSLo.getValueType();
  
  // fold (a_Hi, 0) + (b_Hi, b_Lo) -> (b_Hi + a_Hi, b_Lo)
  if (MaskedValueIsZero(LHSLo, (1ULL << MVT::getSizeInBits(VT))-1, TLI)) {
    SDOperand Hi = DAG.getNode(ISD::ADD, VT, N->getOperand(1),
                               N->getOperand(3));
    WorkList.push_back(Hi.Val);
    CombineTo(N, RHSLo, Hi);
    return SDOperand();
  }
  // fold (a_Hi, a_Lo) + (b_Hi, 0) -> (a_Hi + b_Hi, a_Lo)
  if (MaskedValueIsZero(RHSLo, (1ULL << MVT::getSizeInBits(VT))-1, TLI)) {
    SDOperand Hi = DAG.getNode(ISD::ADD, VT, N->getOperand(1),
                               N->getOperand(3));
    WorkList.push_back(Hi.Val);
    CombineTo(N, LHSLo, Hi);
    return SDOperand();
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSUB_PARTS(SDNode *N) {
  SDOperand LHSLo = N->getOperand(0);
  SDOperand RHSLo = N->getOperand(2);
  MVT::ValueType VT = LHSLo.getValueType();
  
  // fold (a_Hi, a_Lo) - (b_Hi, 0) -> (a_Hi - b_Hi, a_Lo)
  if (MaskedValueIsZero(RHSLo, (1ULL << MVT::getSizeInBits(VT))-1, TLI)) {
    SDOperand Hi = DAG.getNode(ISD::SUB, VT, N->getOperand(1),
                               N->getOperand(3));
    WorkList.push_back(Hi.Val);
    CombineTo(N, LHSLo, Hi);
    return SDOperand();
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitSIGN_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (sext c1) -> c1
  if (N0C)
    return DAG.getConstant(N0C->getSignExtended(), VT);
  // fold (sext (sext x)) -> (sext x)
  if (N0.getOpcode() == ISD::SIGN_EXTEND)
    return DAG.getNode(ISD::SIGN_EXTEND, VT, N0.getOperand(0));
  // fold (sext (sextload x)) -> (sextload x)
  if (N0.getOpcode() == ISD::SEXTLOAD && VT == N0.getValueType())
    return N0;
  // fold (sext (load x)) -> (sextload x)
  if (N0.getOpcode() == ISD::LOAD && N0.hasOneUse()) {
    SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, N0.getOperand(0),
                                       N0.getOperand(1), N0.getOperand(2),
                                       N0.getValueType());
    WorkList.push_back(N);
    CombineTo(N0.Val, DAG.getNode(ISD::TRUNCATE, N0.getValueType(), ExtLoad),
              ExtLoad.getValue(1));
    return SDOperand();
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitZERO_EXTEND(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // fold (zext c1) -> c1
  if (N0C)
    return DAG.getConstant(N0C->getValue(), VT);
  // fold (zext (zext x)) -> (zext x)
  if (N0.getOpcode() == ISD::ZERO_EXTEND)
    return DAG.getNode(ISD::ZERO_EXTEND, VT, N0.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitSIGN_EXTEND_INREG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N1)->getVT();
  unsigned EVTBits = MVT::getSizeInBits(EVT);
  
  // fold (sext_in_reg c1) -> c1
  if (N0C) {
    SDOperand Truncate = DAG.getConstant(N0C->getValue(), EVT);
    return DAG.getNode(ISD::SIGN_EXTEND, VT, Truncate);
  }
  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt1
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG && 
      cast<VTSDNode>(N0.getOperand(1))->getVT() <= EVT) {
    return N0;
  }
  // fold (sext_in_reg (sext_in_reg x, VT2), VT1) -> (sext_in_reg x, minVT) pt2
  if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
      EVT < cast<VTSDNode>(N0.getOperand(1))->getVT()) {
    return DAG.getNode(ISD::SIGN_EXTEND_INREG, VT, N0.getOperand(0), N1);
  }
  // fold (sext_in_reg (assert_sext x)) -> (assert_sext x)
  if (N0.getOpcode() == ISD::AssertSext && 
      cast<VTSDNode>(N0.getOperand(1))->getVT() <= EVT) {
    return N0;
  }
  // fold (sext_in_reg (sextload x)) -> (sextload x)
  if (N0.getOpcode() == ISD::SEXTLOAD && 
      cast<VTSDNode>(N0.getOperand(3))->getVT() <= EVT) {
    return N0;
  }
  // fold (sext_in_reg (setcc x)) -> setcc x iff (setcc x) == 0 or -1
  if (N0.getOpcode() == ISD::SETCC &&
      TLI.getSetCCResultContents() == 
        TargetLowering::ZeroOrNegativeOneSetCCResult)
    return N0;
  // fold (sext_in_reg x) -> (zext_in_reg x) if the sign bit is zero
  if (MaskedValueIsZero(N0, 1ULL << (EVTBits-1), TLI))
    return DAG.getNode(ISD::AND, N0.getValueType(), N0,
                       DAG.getConstant(~0ULL >> (64-EVTBits), VT));
  // fold (sext_in_reg (srl x)) -> sra x
  if (N0.getOpcode() == ISD::SRL && 
      N0.getOperand(1).getOpcode() == ISD::Constant &&
      cast<ConstantSDNode>(N0.getOperand(1))->getValue() == EVTBits) {
    return DAG.getNode(ISD::SRA, N0.getValueType(), N0.getOperand(0), 
                       N0.getOperand(1));
  }
  // fold (sext_inreg (extload x)) -> (sextload x)
  if (N0.getOpcode() == ISD::EXTLOAD && 
      EVT == cast<VTSDNode>(N0.getOperand(3))->getVT() &&
      (!AfterLegalize || TLI.isOperationLegal(ISD::SEXTLOAD, EVT))) {
    SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, N0.getOperand(0),
                                       N0.getOperand(1), N0.getOperand(2),
                                       EVT);
    WorkList.push_back(N);
    CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
    return SDOperand();
  }
  // fold (sext_inreg (zextload x)) -> (sextload x) iff load has one use
  if (N0.getOpcode() == ISD::ZEXTLOAD && N0.hasOneUse() &&
      EVT == cast<VTSDNode>(N0.getOperand(3))->getVT() &&
      (!AfterLegalize || TLI.isOperationLegal(ISD::SEXTLOAD, EVT))) {
    SDOperand ExtLoad = DAG.getExtLoad(ISD::SEXTLOAD, VT, N0.getOperand(0),
                                       N0.getOperand(1), N0.getOperand(2),
                                       EVT);
    WorkList.push_back(N);
    CombineTo(N0.Val, ExtLoad, ExtLoad.getValue(1));
    return SDOperand();
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitTRUNCATE(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  MVT::ValueType VT = N->getValueType(0);

  // noop truncate
  if (N0.getValueType() == N->getValueType(0))
    return N0;
  // fold (truncate c1) -> c1
  if (N0C)
    return DAG.getConstant(N0C->getValue(), VT);
  // fold (truncate (truncate x)) -> (truncate x)
  if (N0.getOpcode() == ISD::TRUNCATE)
    return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
  // fold (truncate (ext x)) -> (ext x) or (truncate x) or x
  if (N0.getOpcode() == ISD::ZERO_EXTEND || N0.getOpcode() == ISD::SIGN_EXTEND){
    if (N0.getValueType() < VT)
      // if the source is smaller than the dest, we still need an extend
      return DAG.getNode(N0.getOpcode(), VT, N0.getOperand(0));
    else if (N0.getValueType() > VT)
      // if the source is larger than the dest, than we just need the truncate
      return DAG.getNode(ISD::TRUNCATE, VT, N0.getOperand(0));
    else
      // if the source and dest are the same type, we can drop both the extend
      // and the truncate
      return N0.getOperand(0);
  }
  // fold (truncate (load x)) -> (smaller load x)
  if (N0.getOpcode() == ISD::LOAD && N0.hasOneUse()) {
    assert(MVT::getSizeInBits(N0.getValueType()) > MVT::getSizeInBits(VT) &&
           "Cannot truncate to larger type!");
    MVT::ValueType PtrType = N0.getOperand(1).getValueType();
    // For big endian targets, we need to add an offset to the pointer to load
    // the correct bytes.  For little endian systems, we merely need to read
    // fewer bytes from the same pointer.
    uint64_t PtrOff = 
      (MVT::getSizeInBits(N0.getValueType()) - MVT::getSizeInBits(VT)) / 8;
    SDOperand NewPtr = TLI.isLittleEndian() ? N0.getOperand(1) : 
      DAG.getNode(ISD::ADD, PtrType, N0.getOperand(1),
                  DAG.getConstant(PtrOff, PtrType));
    WorkList.push_back(NewPtr.Val);
    SDOperand Load = DAG.getLoad(VT, N0.getOperand(0), NewPtr,N0.getOperand(2));
    WorkList.push_back(N);
    CombineTo(N0.Val, Load, Load.getValue(1));
    return SDOperand();
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitFADD(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fadd c1, c2) -> c1+c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() + N1CFP->getValue(), VT);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FADD, VT, N1, N0);
  // fold (A + (-B)) -> A-B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FSUB, VT, N0, N1.getOperand(0));
  // fold ((-A) + B) -> B-A
  if (N0.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FSUB, VT, N1, N0.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFSUB(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);
  
  // fold (fsub c1, c2) -> c1-c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() - N1CFP->getValue(), VT);
  // fold (A-(-B)) -> A+B
  if (N1.getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FADD, N0.getValueType(), N0, N1.getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFMUL(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1);
  MVT::ValueType VT = N->getValueType(0);

  // fold (fmul c1, c2) -> c1*c2
  if (N0CFP && N1CFP)
    return DAG.getConstantFP(N0CFP->getValue() * N1CFP->getValue(), VT);
  // canonicalize constant to RHS
  if (N0CFP && !N1CFP)
    return DAG.getNode(ISD::FMUL, VT, N1, N0);
  // fold (fmul X, 2.0) -> (fadd X, X)
  if (N1CFP && N1CFP->isExactlyValue(+2.0))
    return DAG.getNode(ISD::FADD, VT, N0, N0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFDIV(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  MVT::ValueType VT = N->getValueType(0);

  if (ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0))
    if (ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1)) {
      // fold floating point (fdiv c1, c2)
      return DAG.getConstantFP(N0CFP->getValue() / N1CFP->getValue(), VT);
    }
  return SDOperand();
}

SDOperand DAGCombiner::visitFREM(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  MVT::ValueType VT = N->getValueType(0);

  if (ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0))
    if (ConstantFPSDNode *N1CFP = dyn_cast<ConstantFPSDNode>(N1)) {
      // fold floating point (frem c1, c2) -> fmod(c1, c2)
      return DAG.getConstantFP(fmod(N0CFP->getValue(),N1CFP->getValue()), VT);
    }
  return SDOperand();
}


SDOperand DAGCombiner::visitSINT_TO_FP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (sint_to_fp c1) -> c1fp
  if (N0C)
    return DAG.getConstantFP(N0C->getSignExtended(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitUINT_TO_FP(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
  
  // fold (uint_to_fp c1) -> c1fp
  if (N0C)
    return DAG.getConstantFP(N0C->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_TO_SINT(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_to_sint c1fp) -> c1
  if (N0CFP)
    return DAG.getConstant((int64_t)N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_TO_UINT(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_to_uint c1fp) -> c1
  if (N0CFP)
    return DAG.getConstant((uint64_t)N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_ROUND(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_round c1fp) -> c1fp
  if (N0CFP)
    return DAG.getConstantFP(N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_ROUND_INREG(SDNode *N) {
  SDOperand N0 = N->getOperand(0);
  MVT::ValueType VT = N->getValueType(0);
  MVT::ValueType EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N0);
  
  // fold (fp_round_inreg c1fp) -> c1fp
  if (N0CFP) {
    SDOperand Round = DAG.getConstantFP(N0CFP->getValue(), EVT);
    return DAG.getNode(ISD::FP_EXTEND, VT, Round);
  }
  return SDOperand();
}

SDOperand DAGCombiner::visitFP_EXTEND(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  
  // fold (fp_extend c1fp) -> c1fp
  if (N0CFP)
    return DAG.getConstantFP(N0CFP->getValue(), N->getValueType(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitFNEG(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  // fold (neg c1) -> -c1
  if (N0CFP)
    return DAG.getConstantFP(-N0CFP->getValue(), N->getValueType(0));
  // fold (neg (sub x, y)) -> (sub y, x)
  if (N->getOperand(0).getOpcode() == ISD::SUB)
    return DAG.getNode(ISD::SUB, N->getValueType(0), N->getOperand(1), 
                       N->getOperand(0));
  // fold (neg (neg x)) -> x
  if (N->getOperand(0).getOpcode() == ISD::FNEG)
    return N->getOperand(0).getOperand(0);
  return SDOperand();
}

SDOperand DAGCombiner::visitFABS(SDNode *N) {
  ConstantFPSDNode *N0CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  // fold (fabs c1) -> fabs(c1)
  if (N0CFP)
    return DAG.getConstantFP(fabs(N0CFP->getValue()), N->getValueType(0));
  // fold (fabs (fabs x)) -> (fabs x)
  if (N->getOperand(0).getOpcode() == ISD::FABS)
    return N->getOperand(0);
  // fold (fabs (fneg x)) -> (fabs x)
  if (N->getOperand(0).getOpcode() == ISD::FNEG)
    return DAG.getNode(ISD::FABS, N->getValueType(0), 
                       N->getOperand(0).getOperand(0));
  return SDOperand();
}

SDOperand DAGCombiner::visitBRCOND(SDNode *N) {
  SDOperand Chain = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // never taken branch, fold to chain
  if (N1C && N1C->isNullValue())
    return Chain;
  // unconditional branch
  if (N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N2);
  return SDOperand();
}

SDOperand DAGCombiner::visitBRCONDTWOWAY(SDNode *N) {
  SDOperand Chain = N->getOperand(0);
  SDOperand N1 = N->getOperand(1);
  SDOperand N2 = N->getOperand(2);
  SDOperand N3 = N->getOperand(3);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
  
  // unconditional branch to true mbb
  if (N1C && N1C->getValue() == 1)
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N2);
  // unconditional branch to false mbb
  if (N1C && N1C->isNullValue())
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N3);
  return SDOperand();
}

// Operand List for BR_CC: Chain, CondCC, CondLHS, CondRHS, DestBB.
//
SDOperand DAGCombiner::visitBR_CC(SDNode *N) {
  CondCodeSDNode *CC = cast<CondCodeSDNode>(N->getOperand(1));
  SDOperand CondLHS = N->getOperand(2), CondRHS = N->getOperand(3);
  
  // Use SimplifySetCC  to simplify SETCC's.
  SDOperand Simp = SimplifySetCC(MVT::i1, CondLHS, CondRHS, CC->get(), false);
  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(Simp.Val);

  // fold br_cc true, dest -> br dest (unconditional branch)
  if (SCCC && SCCC->getValue())
    return DAG.getNode(ISD::BR, MVT::Other, N->getOperand(0),
                       N->getOperand(4));
  // fold br_cc false, dest -> unconditional fall through
  if (SCCC && SCCC->isNullValue())
    return N->getOperand(0);
  // fold to a simpler setcc
  if (Simp.Val && Simp.getOpcode() == ISD::SETCC)
    return DAG.getNode(ISD::BR_CC, MVT::Other, N->getOperand(0), 
                       Simp.getOperand(2), Simp.getOperand(0),
                       Simp.getOperand(1), N->getOperand(4));
  return SDOperand();
}

SDOperand DAGCombiner::visitBRTWOWAY_CC(SDNode *N) {
  SDOperand Chain = N->getOperand(0);
  SDOperand CCN = N->getOperand(1);
  SDOperand LHS = N->getOperand(2);
  SDOperand RHS = N->getOperand(3);
  SDOperand N4 = N->getOperand(4);
  SDOperand N5 = N->getOperand(5);
  
  SDOperand SCC = SimplifySetCC(TLI.getSetCCResultTy(), LHS, RHS,
                                cast<CondCodeSDNode>(CCN)->get(), false);
  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.Val);
  
  // fold select_cc lhs, rhs, x, x, cc -> x
  if (N4 == N5)
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N4);
  // fold select_cc true, x, y -> x
  if (SCCC && SCCC->getValue())
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N4);
  // fold select_cc false, x, y -> y
  if (SCCC && SCCC->isNullValue())
    return DAG.getNode(ISD::BR, MVT::Other, Chain, N5);
  // fold to a simpler setcc
  if (SCC.Val && SCC.getOpcode() == ISD::SETCC)
    return DAG.getBR2Way_CC(Chain, SCC.getOperand(2), SCC.getOperand(0), 
                            SCC.getOperand(1), N4, N5);
  return SDOperand();
}

SDOperand DAGCombiner::visitLOAD(SDNode *N) {
  SDOperand Chain    = N->getOperand(0);
  SDOperand Ptr      = N->getOperand(1);
  SDOperand SrcValue = N->getOperand(2);
  
  // If this load is directly stored, replace the load value with the stored
  // value.
  // TODO: Handle store large -> read small portion.
  // TODO: Handle TRUNCSTORE/EXTLOAD
  if (Chain.getOpcode() == ISD::STORE && Chain.getOperand(2) == Ptr &&
      Chain.getOperand(1).getValueType() == N->getValueType(0))
    return CombineTo(N, Chain.getOperand(1), Chain);
  
  return SDOperand();
}

SDOperand DAGCombiner::visitSTORE(SDNode *N) {
  SDOperand Chain    = N->getOperand(0);
  SDOperand Value    = N->getOperand(1);
  SDOperand Ptr      = N->getOperand(2);
  SDOperand SrcValue = N->getOperand(3);
 
  // If this is a store that kills a previous store, remove the previous store.
  if (Chain.getOpcode() == ISD::STORE && Chain.getOperand(2) == Ptr &&
      Chain.Val->hasOneUse() /* Avoid introducing DAG cycles */ &&
      // Make sure that these stores are the same value type:
      // FIXME: we really care that the second store is >= size of the first.
      Value.getValueType() == Chain.getOperand(1).getValueType()) {
    // Create a new store of Value that replaces both stores.
    SDNode *PrevStore = Chain.Val;
    if (PrevStore->getOperand(1) == Value) // Same value multiply stored.
      return Chain;
    SDOperand NewStore = DAG.getNode(ISD::STORE, MVT::Other,
                                     PrevStore->getOperand(0), Value, Ptr,
                                     SrcValue);
    CombineTo(N, NewStore);                 // Nuke this store.
    CombineTo(PrevStore, NewStore);  // Nuke the previous store.
    return SDOperand(N, 0);
  }
  
  return SDOperand();
}

SDOperand DAGCombiner::SimplifySelect(SDOperand N0, SDOperand N1, SDOperand N2){
  assert(N0.getOpcode() ==ISD::SETCC && "First argument must be a SetCC node!");
  
  SDOperand SCC = SimplifySelectCC(N0.getOperand(0), N0.getOperand(1), N1, N2,
                                 cast<CondCodeSDNode>(N0.getOperand(2))->get());
  // If we got a simplified select_cc node back from SimplifySelectCC, then
  // break it down into a new SETCC node, and a new SELECT node, and then return
  // the SELECT node, since we were called with a SELECT node.
  if (SCC.Val) {
    // Check to see if we got a select_cc back (to turn into setcc/select).
    // Otherwise, just return whatever node we got back, like fabs.
    if (SCC.getOpcode() == ISD::SELECT_CC) {
      SDOperand SETCC = DAG.getNode(ISD::SETCC, N0.getValueType(),
                                    SCC.getOperand(0), SCC.getOperand(1), 
                                    SCC.getOperand(4));
      WorkList.push_back(SETCC.Val);
      return DAG.getNode(ISD::SELECT, SCC.getValueType(), SCC.getOperand(2),
                         SCC.getOperand(3), SETCC);
    }
    return SCC;
  }
  return SDOperand();
}

/// SimplifySelectOps - Given a SELECT or a SELECT_CC node, where LHS and RHS
/// are the two values being selected between, see if we can simplify the
/// select.
///
bool DAGCombiner::SimplifySelectOps(SDNode *TheSelect, SDOperand LHS, 
                                    SDOperand RHS) {
  
  // If this is a select from two identical things, try to pull the operation
  // through the select.
  if (LHS.getOpcode() == RHS.getOpcode() && LHS.hasOneUse() && RHS.hasOneUse()){
#if 0
    std::cerr << "SELECT: ["; LHS.Val->dump();
    std::cerr << "] ["; RHS.Val->dump();
    std::cerr << "]\n";
#endif
    
    // If this is a load and the token chain is identical, replace the select
    // of two loads with a load through a select of the address to load from.
    // This triggers in things like "select bool X, 10.0, 123.0" after the FP
    // constants have been dropped into the constant pool.
    if ((LHS.getOpcode() == ISD::LOAD ||
         LHS.getOpcode() == ISD::EXTLOAD ||
         LHS.getOpcode() == ISD::ZEXTLOAD ||
         LHS.getOpcode() == ISD::SEXTLOAD) &&
        // Token chains must be identical.
        LHS.getOperand(0) == RHS.getOperand(0) &&
        // If this is an EXTLOAD, the VT's must match.
        (LHS.getOpcode() == ISD::LOAD ||
         LHS.getOperand(3) == RHS.getOperand(3))) {
      // FIXME: this conflates two src values, discarding one.  This is not
      // the right thing to do, but nothing uses srcvalues now.  When they do,
      // turn SrcValue into a list of locations.
      SDOperand Addr;
      if (TheSelect->getOpcode() == ISD::SELECT)
        Addr = DAG.getNode(ISD::SELECT, LHS.getOperand(1).getValueType(),
                           TheSelect->getOperand(0), LHS.getOperand(1),
                           RHS.getOperand(1));
      else
        Addr = DAG.getNode(ISD::SELECT_CC, LHS.getOperand(1).getValueType(),
                           TheSelect->getOperand(0),
                           TheSelect->getOperand(1), 
                           LHS.getOperand(1), RHS.getOperand(1),
                           TheSelect->getOperand(4));
      
      SDOperand Load;
      if (LHS.getOpcode() == ISD::LOAD)
        Load = DAG.getLoad(TheSelect->getValueType(0), LHS.getOperand(0),
                           Addr, LHS.getOperand(2));
      else
        Load = DAG.getExtLoad(LHS.getOpcode(), TheSelect->getValueType(0),
                              LHS.getOperand(0), Addr, LHS.getOperand(2),
                              cast<VTSDNode>(LHS.getOperand(3))->getVT());
      // Users of the select now use the result of the load.
      CombineTo(TheSelect, Load);
      
      // Users of the old loads now use the new load's chain.  We know the
      // old-load value is dead now.
      CombineTo(LHS.Val, Load.getValue(0), Load.getValue(1));
      CombineTo(RHS.Val, Load.getValue(0), Load.getValue(1));
      return true;
    }
  }
  
  return false;
}

SDOperand DAGCombiner::SimplifySelectCC(SDOperand N0, SDOperand N1, 
                                        SDOperand N2, SDOperand N3,
                                        ISD::CondCode CC) {
  
  MVT::ValueType VT = N2.getValueType();
  ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val);
  ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val);
  ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2.Val);
  ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N3.Val);

  // Determine if the condition we're dealing with is constant
  SDOperand SCC = SimplifySetCC(TLI.getSetCCResultTy(), N0, N1, CC, false);
  ConstantSDNode *SCCC = dyn_cast_or_null<ConstantSDNode>(SCC.Val);

  // fold select_cc true, x, y -> x
  if (SCCC && SCCC->getValue())
    return N2;
  // fold select_cc false, x, y -> y
  if (SCCC && SCCC->getValue() == 0)
    return N3;
  
  // Check to see if we can simplify the select into an fabs node
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1)) {
    // Allow either -0.0 or 0.0
    if (CFP->getValue() == 0.0) {
      // select (setg[te] X, +/-0.0), X, fneg(X) -> fabs
      if ((CC == ISD::SETGE || CC == ISD::SETGT) &&
          N0 == N2 && N3.getOpcode() == ISD::FNEG &&
          N2 == N3.getOperand(0))
        return DAG.getNode(ISD::FABS, VT, N0);
      
      // select (setl[te] X, +/-0.0), fneg(X), X -> fabs
      if ((CC == ISD::SETLT || CC == ISD::SETLE) &&
          N0 == N3 && N2.getOpcode() == ISD::FNEG &&
          N2.getOperand(0) == N3)
        return DAG.getNode(ISD::FABS, VT, N3);
    }
  }
  
  // Check to see if we can perform the "gzip trick", transforming
  // select_cc setlt X, 0, A, 0 -> and (sra X, size(X)-1), A
  if (N1C && N1C->isNullValue() && N3C && N3C->isNullValue() &&
      MVT::isInteger(N0.getValueType()) && 
      MVT::isInteger(N2.getValueType()) && CC == ISD::SETLT) {
    MVT::ValueType XType = N0.getValueType();
    MVT::ValueType AType = N2.getValueType();
    if (XType >= AType) {
      // and (sra X, size(X)-1, A) -> "and (srl X, C2), A" iff A is a
      // single-bit constant.
      if (N2C && ((N2C->getValue() & (N2C->getValue()-1)) == 0)) {
        unsigned ShCtV = Log2_64(N2C->getValue());
        ShCtV = MVT::getSizeInBits(XType)-ShCtV-1;
        SDOperand ShCt = DAG.getConstant(ShCtV, TLI.getShiftAmountTy());
        SDOperand Shift = DAG.getNode(ISD::SRL, XType, N0, ShCt);
        WorkList.push_back(Shift.Val);
        if (XType > AType) {
          Shift = DAG.getNode(ISD::TRUNCATE, AType, Shift);
          WorkList.push_back(Shift.Val);
        }
        return DAG.getNode(ISD::AND, AType, Shift, N2);
      }
      SDOperand Shift = DAG.getNode(ISD::SRA, XType, N0,
                                    DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                    TLI.getShiftAmountTy()));
      WorkList.push_back(Shift.Val);
      if (XType > AType) {
        Shift = DAG.getNode(ISD::TRUNCATE, AType, Shift);
        WorkList.push_back(Shift.Val);
      }
      return DAG.getNode(ISD::AND, AType, Shift, N2);
    }
  }
  
  // fold select C, 16, 0 -> shl C, 4
  if (N2C && N3C && N3C->isNullValue() && isPowerOf2_64(N2C->getValue()) &&
      TLI.getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult) {
    // Get a SetCC of the condition
    // FIXME: Should probably make sure that setcc is legal if we ever have a
    // target where it isn't.
    SDOperand Temp, SCC = DAG.getSetCC(TLI.getSetCCResultTy(), N0, N1, CC);
    WorkList.push_back(SCC.Val);
    // cast from setcc result type to select result type
    if (AfterLegalize)
      Temp = DAG.getZeroExtendInReg(SCC, N2.getValueType());
    else
      Temp = DAG.getNode(ISD::ZERO_EXTEND, N2.getValueType(), SCC);
    WorkList.push_back(Temp.Val);
    // shl setcc result by log2 n2c
    return DAG.getNode(ISD::SHL, N2.getValueType(), Temp,
                       DAG.getConstant(Log2_64(N2C->getValue()),
                                       TLI.getShiftAmountTy()));
  }
    
  // Check to see if this is the equivalent of setcc
  // FIXME: Turn all of these into setcc if setcc if setcc is legal
  // otherwise, go ahead with the folds.
  if (0 && N3C && N3C->isNullValue() && N2C && (N2C->getValue() == 1ULL)) {
    MVT::ValueType XType = N0.getValueType();
    if (TLI.isOperationLegal(ISD::SETCC, TLI.getSetCCResultTy())) {
      SDOperand Res = DAG.getSetCC(TLI.getSetCCResultTy(), N0, N1, CC);
      if (Res.getValueType() != VT)
        Res = DAG.getNode(ISD::ZERO_EXTEND, VT, Res);
      return Res;
    }
    
    // seteq X, 0 -> srl (ctlz X, log2(size(X)))
    if (N1C && N1C->isNullValue() && CC == ISD::SETEQ && 
        TLI.isOperationLegal(ISD::CTLZ, XType)) {
      SDOperand Ctlz = DAG.getNode(ISD::CTLZ, XType, N0);
      return DAG.getNode(ISD::SRL, XType, Ctlz, 
                         DAG.getConstant(Log2_32(MVT::getSizeInBits(XType)),
                                         TLI.getShiftAmountTy()));
    }
    // setgt X, 0 -> srl (and (-X, ~X), size(X)-1)
    if (N1C && N1C->isNullValue() && CC == ISD::SETGT) { 
      SDOperand NegN0 = DAG.getNode(ISD::SUB, XType, DAG.getConstant(0, XType),
                                    N0);
      SDOperand NotN0 = DAG.getNode(ISD::XOR, XType, N0, 
                                    DAG.getConstant(~0ULL, XType));
      return DAG.getNode(ISD::SRL, XType, 
                         DAG.getNode(ISD::AND, XType, NegN0, NotN0),
                         DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                         TLI.getShiftAmountTy()));
    }
    // setgt X, -1 -> xor (srl (X, size(X)-1), 1)
    if (N1C && N1C->isAllOnesValue() && CC == ISD::SETGT) {
      SDOperand Sign = DAG.getNode(ISD::SRL, XType, N0,
                                   DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                   TLI.getShiftAmountTy()));
      return DAG.getNode(ISD::XOR, XType, Sign, DAG.getConstant(1, XType));
    }
  }
  
  // Check to see if this is an integer abs. select_cc setl[te] X, 0, -X, X ->
  // Y = sra (X, size(X)-1); xor (add (X, Y), Y)
  if (N1C && N1C->isNullValue() && (CC == ISD::SETLT || CC == ISD::SETLE) &&
      N0 == N3 && N2.getOpcode() == ISD::SUB && N0 == N2.getOperand(1)) {
    if (ConstantSDNode *SubC = dyn_cast<ConstantSDNode>(N2.getOperand(0))) {
      MVT::ValueType XType = N0.getValueType();
      if (SubC->isNullValue() && MVT::isInteger(XType)) {
        SDOperand Shift = DAG.getNode(ISD::SRA, XType, N0,
                                    DAG.getConstant(MVT::getSizeInBits(XType)-1,
                                                    TLI.getShiftAmountTy()));
        SDOperand Add = DAG.getNode(ISD::ADD, XType, N0, Shift);
        WorkList.push_back(Shift.Val);
        WorkList.push_back(Add.Val);
        return DAG.getNode(ISD::XOR, XType, Add, Shift);
      }
    }
  }

  return SDOperand();
}

SDOperand DAGCombiner::SimplifySetCC(MVT::ValueType VT, SDOperand N0,
                                     SDOperand N1, ISD::CondCode Cond,
                                     bool foldBooleans) {
  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return DAG.getConstant(0, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2:  return DAG.getConstant(1, VT);
  }

  if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.Val)) {
    uint64_t C1 = N1C->getValue();
    if (ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0.Val)) {
      uint64_t C0 = N0C->getValue();

      // Sign extend the operands if required
      if (ISD::isSignedIntSetCC(Cond)) {
        C0 = N0C->getSignExtended();
        C1 = N1C->getSignExtended();
      }

      switch (Cond) {
      default: assert(0 && "Unknown integer setcc!");
      case ISD::SETEQ:  return DAG.getConstant(C0 == C1, VT);
      case ISD::SETNE:  return DAG.getConstant(C0 != C1, VT);
      case ISD::SETULT: return DAG.getConstant(C0 <  C1, VT);
      case ISD::SETUGT: return DAG.getConstant(C0 >  C1, VT);
      case ISD::SETULE: return DAG.getConstant(C0 <= C1, VT);
      case ISD::SETUGE: return DAG.getConstant(C0 >= C1, VT);
      case ISD::SETLT:  return DAG.getConstant((int64_t)C0 <  (int64_t)C1, VT);
      case ISD::SETGT:  return DAG.getConstant((int64_t)C0 >  (int64_t)C1, VT);
      case ISD::SETLE:  return DAG.getConstant((int64_t)C0 <= (int64_t)C1, VT);
      case ISD::SETGE:  return DAG.getConstant((int64_t)C0 >= (int64_t)C1, VT);
      }
    } else {
      // If the LHS is a ZERO_EXTEND, perform the comparison on the input.
      if (N0.getOpcode() == ISD::ZERO_EXTEND) {
        unsigned InSize = MVT::getSizeInBits(N0.getOperand(0).getValueType());

        // If the comparison constant has bits in the upper part, the
        // zero-extended value could never match.
        if (C1 & (~0ULL << InSize)) {
          unsigned VSize = MVT::getSizeInBits(N0.getValueType());
          switch (Cond) {
          case ISD::SETUGT:
          case ISD::SETUGE:
          case ISD::SETEQ: return DAG.getConstant(0, VT);
          case ISD::SETULT:
          case ISD::SETULE:
          case ISD::SETNE: return DAG.getConstant(1, VT);
          case ISD::SETGT:
          case ISD::SETGE:
            // True if the sign bit of C1 is set.
            return DAG.getConstant((C1 & (1ULL << VSize)) != 0, VT);
          case ISD::SETLT:
          case ISD::SETLE:
            // True if the sign bit of C1 isn't set.
            return DAG.getConstant((C1 & (1ULL << VSize)) == 0, VT);
          default:
            break;
          }
        }

        // Otherwise, we can perform the comparison with the low bits.
        switch (Cond) {
        case ISD::SETEQ:
        case ISD::SETNE:
        case ISD::SETUGT:
        case ISD::SETUGE:
        case ISD::SETULT:
        case ISD::SETULE:
          return DAG.getSetCC(VT, N0.getOperand(0),
                          DAG.getConstant(C1, N0.getOperand(0).getValueType()),
                          Cond);
        default:
          break;   // todo, be more careful with signed comparisons
        }
      } else if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
                 (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
        MVT::ValueType ExtSrcTy = cast<VTSDNode>(N0.getOperand(1))->getVT();
        unsigned ExtSrcTyBits = MVT::getSizeInBits(ExtSrcTy);
        MVT::ValueType ExtDstTy = N0.getValueType();
        unsigned ExtDstTyBits = MVT::getSizeInBits(ExtDstTy);

        // If the extended part has any inconsistent bits, it cannot ever
        // compare equal.  In other words, they have to be all ones or all
        // zeros.
        uint64_t ExtBits =
          (~0ULL >> (64-ExtSrcTyBits)) & (~0ULL << (ExtDstTyBits-1));
        if ((C1 & ExtBits) != 0 && (C1 & ExtBits) != ExtBits)
          return DAG.getConstant(Cond == ISD::SETNE, VT);
        
        SDOperand ZextOp;
        MVT::ValueType Op0Ty = N0.getOperand(0).getValueType();
        if (Op0Ty == ExtSrcTy) {
          ZextOp = N0.getOperand(0);
        } else {
          int64_t Imm = ~0ULL >> (64-ExtSrcTyBits);
          ZextOp = DAG.getNode(ISD::AND, Op0Ty, N0.getOperand(0),
                               DAG.getConstant(Imm, Op0Ty));
        }
        WorkList.push_back(ZextOp.Val);
        // Otherwise, make this a use of a zext.
        return DAG.getSetCC(VT, ZextOp, 
                            DAG.getConstant(C1 & (~0ULL>>(64-ExtSrcTyBits)), 
                                            ExtDstTy),
                            Cond);
      }
      
      uint64_t MinVal, MaxVal;
      unsigned OperandBitSize = MVT::getSizeInBits(N1C->getValueType(0));
      if (ISD::isSignedIntSetCC(Cond)) {
        MinVal = 1ULL << (OperandBitSize-1);
        if (OperandBitSize != 1)   // Avoid X >> 64, which is undefined.
          MaxVal = ~0ULL >> (65-OperandBitSize);
        else
          MaxVal = 0;
      } else {
        MinVal = 0;
        MaxVal = ~0ULL >> (64-OperandBitSize);
      }

      // Canonicalize GE/LE comparisons to use GT/LT comparisons.
      if (Cond == ISD::SETGE || Cond == ISD::SETUGE) {
        if (C1 == MinVal) return DAG.getConstant(1, VT);   // X >= MIN --> true
        --C1;                                          // X >= C0 --> X > (C0-1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(C1, N1.getValueType()),
                        (Cond == ISD::SETGE) ? ISD::SETGT : ISD::SETUGT);
      }

      if (Cond == ISD::SETLE || Cond == ISD::SETULE) {
        if (C1 == MaxVal) return DAG.getConstant(1, VT);   // X <= MAX --> true
        ++C1;                                          // X <= C0 --> X < (C0+1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(C1, N1.getValueType()),
                        (Cond == ISD::SETLE) ? ISD::SETLT : ISD::SETULT);
      }

      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal)
        return DAG.getConstant(0, VT);      // X < MIN --> false

      // Canonicalize setgt X, Min --> setne X, Min
      if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MinVal)
        return DAG.getSetCC(VT, N0, N1, ISD::SETNE);
      // Canonicalize setlt X, Max --> setne X, Max
      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MaxVal)
        return DAG.getSetCC(VT, N0, N1, ISD::SETNE);

      // If we have setult X, 1, turn it into seteq X, 0
      if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal+1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(MinVal, N0.getValueType()),
                        ISD::SETEQ);
      // If we have setugt X, Max-1, turn it into seteq X, Max
      else if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal-1)
        return DAG.getSetCC(VT, N0, DAG.getConstant(MaxVal, N0.getValueType()),
                        ISD::SETEQ);

      // If we have "setcc X, C0", check to see if we can shrink the immediate
      // by changing cc.

      // SETUGT X, SINTMAX  -> SETLT X, 0
      if (Cond == ISD::SETUGT && OperandBitSize != 1 &&
          C1 == (~0ULL >> (65-OperandBitSize)))
        return DAG.getSetCC(VT, N0, DAG.getConstant(0, N1.getValueType()),
                            ISD::SETLT);

      // FIXME: Implement the rest of these.

      // Fold bit comparisons when we can.
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          VT == N0.getValueType() && N0.getOpcode() == ISD::AND)
        if (ConstantSDNode *AndRHS =
                    dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          if (Cond == ISD::SETNE && C1 == 0) {// (X & 8) != 0  -->  (X & 8) >> 3
            // Perform the xform if the AND RHS is a single bit.
            if ((AndRHS->getValue() & (AndRHS->getValue()-1)) == 0) {
              return DAG.getNode(ISD::SRL, VT, N0,
                             DAG.getConstant(Log2_64(AndRHS->getValue()),
                                                   TLI.getShiftAmountTy()));
            }
          } else if (Cond == ISD::SETEQ && C1 == AndRHS->getValue()) {
            // (X & 8) == 8  -->  (X & 8) >> 3
            // Perform the xform if C1 is a single bit.
            if ((C1 & (C1-1)) == 0) {
              return DAG.getNode(ISD::SRL, VT, N0,
                             DAG.getConstant(Log2_64(C1),TLI.getShiftAmountTy()));
            }
          }
        }
    }
  } else if (isa<ConstantSDNode>(N0.Val)) {
      // Ensure that the constant occurs on the RHS.
    return DAG.getSetCC(VT, N1, N0, ISD::getSetCCSwappedOperands(Cond));
  }

  if (ConstantFPSDNode *N0C = dyn_cast<ConstantFPSDNode>(N0.Val))
    if (ConstantFPSDNode *N1C = dyn_cast<ConstantFPSDNode>(N1.Val)) {
      double C0 = N0C->getValue(), C1 = N1C->getValue();

      switch (Cond) {
      default: break; // FIXME: Implement the rest of these!
      case ISD::SETEQ:  return DAG.getConstant(C0 == C1, VT);
      case ISD::SETNE:  return DAG.getConstant(C0 != C1, VT);
      case ISD::SETLT:  return DAG.getConstant(C0 < C1, VT);
      case ISD::SETGT:  return DAG.getConstant(C0 > C1, VT);
      case ISD::SETLE:  return DAG.getConstant(C0 <= C1, VT);
      case ISD::SETGE:  return DAG.getConstant(C0 >= C1, VT);
      }
    } else {
      // Ensure that the constant occurs on the RHS.
      return DAG.getSetCC(VT, N1, N0, ISD::getSetCCSwappedOperands(Cond));
    }

  if (N0 == N1) {
    // We can always fold X == Y for integer setcc's.
    if (MVT::isInteger(N0.getValueType()))
      return DAG.getConstant(ISD::isTrueWhenEqual(Cond), VT);
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return DAG.getConstant(ISD::isTrueWhenEqual(Cond), VT);
    if (UOF == unsigned(ISD::isTrueWhenEqual(Cond)))
      return DAG.getConstant(UOF, VT);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    ISD::CondCode NewCond = UOF == 0 ? ISD::SETUO : ISD::SETO;
    if (NewCond != Cond)
      return DAG.getSetCC(VT, N0, N1, NewCond);
  }

  if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
      MVT::isInteger(N0.getValueType())) {
    if (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::SUB ||
        N0.getOpcode() == ISD::XOR) {
      // Simplify (X+Y) == (X+Z) -->  Y == Z
      if (N0.getOpcode() == N1.getOpcode()) {
        if (N0.getOperand(0) == N1.getOperand(0))
          return DAG.getSetCC(VT, N0.getOperand(1), N1.getOperand(1), Cond);
        if (N0.getOperand(1) == N1.getOperand(1))
          return DAG.getSetCC(VT, N0.getOperand(0), N1.getOperand(0), Cond);
        if (isCommutativeBinOp(N0.getOpcode())) {
          // If X op Y == Y op X, try other combinations.
          if (N0.getOperand(0) == N1.getOperand(1))
            return DAG.getSetCC(VT, N0.getOperand(1), N1.getOperand(0), Cond);
          if (N0.getOperand(1) == N1.getOperand(0))
            return DAG.getSetCC(VT, N0.getOperand(0), N1.getOperand(1), Cond);
        }
      }

      // Turn (X^C1) == C2 into X == C1^C2 iff X&~C1 = 0.  Common for condcodes.
      if (N0.getOpcode() == ISD::XOR)
        if (ConstantSDNode *XORC = dyn_cast<ConstantSDNode>(N0.getOperand(1)))
          if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(N1)) {
            // If we know that all of the inverted bits are zero, don't bother
            // performing the inversion.
            if (MaskedValueIsZero(N0.getOperand(0), ~XORC->getValue(), TLI))
              return DAG.getSetCC(VT, N0.getOperand(0),
                              DAG.getConstant(XORC->getValue()^RHSC->getValue(),
                                              N0.getValueType()), Cond);
          }
      
      // Simplify (X+Z) == X -->  Z == 0
      if (N0.getOperand(0) == N1)
        return DAG.getSetCC(VT, N0.getOperand(1),
                        DAG.getConstant(0, N0.getValueType()), Cond);
      if (N0.getOperand(1) == N1) {
        if (isCommutativeBinOp(N0.getOpcode()))
          return DAG.getSetCC(VT, N0.getOperand(0),
                          DAG.getConstant(0, N0.getValueType()), Cond);
        else {
          assert(N0.getOpcode() == ISD::SUB && "Unexpected operation!");
          // (Z-X) == X  --> Z == X<<1
          SDOperand SH = DAG.getNode(ISD::SHL, N1.getValueType(),
                                     N1, 
                                     DAG.getConstant(1,TLI.getShiftAmountTy()));
          WorkList.push_back(SH.Val);
          return DAG.getSetCC(VT, N0.getOperand(0), SH, Cond);
        }
      }
    }

    if (N1.getOpcode() == ISD::ADD || N1.getOpcode() == ISD::SUB ||
        N1.getOpcode() == ISD::XOR) {
      // Simplify  X == (X+Z) -->  Z == 0
      if (N1.getOperand(0) == N0) {
        return DAG.getSetCC(VT, N1.getOperand(1),
                        DAG.getConstant(0, N1.getValueType()), Cond);
      } else if (N1.getOperand(1) == N0) {
        if (isCommutativeBinOp(N1.getOpcode())) {
          return DAG.getSetCC(VT, N1.getOperand(0),
                          DAG.getConstant(0, N1.getValueType()), Cond);
        } else {
          assert(N1.getOpcode() == ISD::SUB && "Unexpected operation!");
          // X == (Z-X)  --> X<<1 == Z
          SDOperand SH = DAG.getNode(ISD::SHL, N1.getValueType(), N0, 
                                     DAG.getConstant(1,TLI.getShiftAmountTy()));
          WorkList.push_back(SH.Val);
          return DAG.getSetCC(VT, SH, N1.getOperand(0), Cond);
        }
      }
    }
  }

  // Fold away ALL boolean setcc's.
  SDOperand Temp;
  if (N0.getValueType() == MVT::i1 && foldBooleans) {
    switch (Cond) {
    default: assert(0 && "Unknown integer setcc!");
    case ISD::SETEQ:  // X == Y  -> (X^Y)^1
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, N1);
      N0 = DAG.getNode(ISD::XOR, MVT::i1, Temp, DAG.getConstant(1, MVT::i1));
      WorkList.push_back(Temp.Val);
      break;
    case ISD::SETNE:  // X != Y   -->  (X^Y)
      N0 = DAG.getNode(ISD::XOR, MVT::i1, N0, N1);
      break;
    case ISD::SETGT:  // X >s Y   -->  X == 0 & Y == 1  -->  X^1 & Y
    case ISD::SETULT: // X <u Y   -->  X == 0 & Y == 1  -->  X^1 & Y
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::AND, MVT::i1, N1, Temp);
      WorkList.push_back(Temp.Val);
      break;
    case ISD::SETLT:  // X <s Y   --> X == 1 & Y == 0  -->  Y^1 & X
    case ISD::SETUGT: // X >u Y   --> X == 1 & Y == 0  -->  Y^1 & X
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N1, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::AND, MVT::i1, N0, Temp);
      WorkList.push_back(Temp.Val);
      break;
    case ISD::SETULE: // X <=u Y  --> X == 0 | Y == 1  -->  X^1 | Y
    case ISD::SETGE:  // X >=s Y  --> X == 0 | Y == 1  -->  X^1 | Y
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::OR, MVT::i1, N1, Temp);
      WorkList.push_back(Temp.Val);
      break;
    case ISD::SETUGE: // X >=u Y  --> X == 1 | Y == 0  -->  Y^1 | X
    case ISD::SETLE:  // X <=s Y  --> X == 1 | Y == 0  -->  Y^1 | X
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N1, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::OR, MVT::i1, N0, Temp);
      break;
    }
    if (VT != MVT::i1) {
      WorkList.push_back(N0.Val);
      // FIXME: If running after legalize, we probably can't do this.
      N0 = DAG.getNode(ISD::ZERO_EXTEND, VT, N0);
    }
    return N0;
  }

  // Could not fold it.
  return SDOperand();
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand DAGCombiner::BuildSDIV(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  
  // Check to see if we can do this.
  if (!TLI.isTypeLegal(VT) || (VT != MVT::i32 && VT != MVT::i64))
    return SDOperand();       // BuildSDIV only operates on i32 or i64
  if (!TLI.isOperationLegal(ISD::MULHS, VT))
    return SDOperand();       // Make sure the target supports MULHS.
  
  int64_t d = cast<ConstantSDNode>(N->getOperand(1))->getSignExtended();
  ms magics = (VT == MVT::i32) ? magic32(d) : magic64(d);
  
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = DAG.getNode(ISD::MULHS, VT, N->getOperand(0),
                            DAG.getConstant(magics.m, VT));
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0) { 
    Q = DAG.getNode(ISD::ADD, VT, Q, N->getOperand(0));
    WorkList.push_back(Q.Val);
  }
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0) {
    Q = DAG.getNode(ISD::SUB, VT, Q, N->getOperand(0));
    WorkList.push_back(Q.Val);
  }
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0) {
    Q = DAG.getNode(ISD::SRA, VT, Q, 
                    DAG.getConstant(magics.s, TLI.getShiftAmountTy()));
    WorkList.push_back(Q.Val);
  }
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    DAG.getNode(ISD::SRL, VT, Q, DAG.getConstant(MVT::getSizeInBits(VT)-1,
                                                 TLI.getShiftAmountTy()));
  WorkList.push_back(T.Val);
  return DAG.getNode(ISD::ADD, VT, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand DAGCombiner::BuildUDIV(SDNode *N) {
  MVT::ValueType VT = N->getValueType(0);
  
  // Check to see if we can do this.
  if (!TLI.isTypeLegal(VT) || (VT != MVT::i32 && VT != MVT::i64))
    return SDOperand();       // BuildUDIV only operates on i32 or i64
  if (!TLI.isOperationLegal(ISD::MULHU, VT))
    return SDOperand();       // Make sure the target supports MULHU.
  
  uint64_t d = cast<ConstantSDNode>(N->getOperand(1))->getValue();
  mu magics = (VT == MVT::i32) ? magicu32(d) : magicu64(d);
  
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = DAG.getNode(ISD::MULHU, VT, N->getOperand(0),
                            DAG.getConstant(magics.m, VT));
  WorkList.push_back(Q.Val);

  if (magics.a == 0) {
    return DAG.getNode(ISD::SRL, VT, Q, 
                       DAG.getConstant(magics.s, TLI.getShiftAmountTy()));
  } else {
    SDOperand NPQ = DAG.getNode(ISD::SUB, VT, N->getOperand(0), Q);
    WorkList.push_back(NPQ.Val);
    NPQ = DAG.getNode(ISD::SRL, VT, NPQ, 
                      DAG.getConstant(1, TLI.getShiftAmountTy()));
    WorkList.push_back(NPQ.Val);
    NPQ = DAG.getNode(ISD::ADD, VT, NPQ, Q);
    WorkList.push_back(NPQ.Val);
    return DAG.getNode(ISD::SRL, VT, NPQ, 
                       DAG.getConstant(magics.s-1, TLI.getShiftAmountTy()));
  }
}

// SelectionDAG::Combine - This is the entry point for the file.
//
void SelectionDAG::Combine(bool RunningAfterLegalize) {
  /// run - This is the main entry point to this class.
  ///
  DAGCombiner(*this).Run(RunningAfterLegalize);
}
