//===----- LegalizeIntegerTypes.cpp - Legalization of integer types -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements integer type expansion and promotion for LegalizeTypes.
// Promotion is the act of changing a computation in an illegal type into a
// computation in a larger type.  For example, implementing i8 arithmetic in an
// i32 register (often needed on powerpc).
// Expansion is the act of changing a computation in an illegal type into a
// computation in two identical registers of a smaller type.  For example,
// implementing i64 arithmetic in two i32 registers (often needed on 32-bit
// targets).
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Integer Result Promotion
//===----------------------------------------------------------------------===//

/// PromoteIntegerResult - This method is called when a result of a node is
/// found to be in need of promotion to a larger type.  At this point, the node
/// may also have invalid operands or may have other results that need
/// expansion, we just know that (at least) one result needs promotion.
void DAGTypeLegalizer::PromoteIntegerResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Promote integer result: "; N->dump(&DAG); cerr << "\n");
  SDValue Res = SDValue();

  // See if the target wants to custom expand this node.
  if (CustomLowerResults(N, N->getValueType(ResNo), true))
    return;

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "PromoteIntegerResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to promote this operator!");
    abort();
  case ISD::AssertSext:  Res = PromoteIntRes_AssertSext(N); break;
  case ISD::AssertZext:  Res = PromoteIntRes_AssertZext(N); break;
  case ISD::BIT_CONVERT: Res = PromoteIntRes_BIT_CONVERT(N); break;
  case ISD::BSWAP:       Res = PromoteIntRes_BSWAP(N); break;
  case ISD::BUILD_PAIR:  Res = PromoteIntRes_BUILD_PAIR(N); break;
  case ISD::Constant:    Res = PromoteIntRes_Constant(N); break;
  case ISD::CONVERT_RNDSAT:
                         Res = PromoteIntRes_CONVERT_RNDSAT(N); break;
  case ISD::CTLZ:        Res = PromoteIntRes_CTLZ(N); break;
  case ISD::CTPOP:       Res = PromoteIntRes_CTPOP(N); break;
  case ISD::CTTZ:        Res = PromoteIntRes_CTTZ(N); break;
  case ISD::EXTRACT_VECTOR_ELT:
                         Res = PromoteIntRes_EXTRACT_VECTOR_ELT(N); break;
  case ISD::LOAD:        Res = PromoteIntRes_LOAD(cast<LoadSDNode>(N));break;
  case ISD::SELECT:      Res = PromoteIntRes_SELECT(N); break;
  case ISD::SELECT_CC:   Res = PromoteIntRes_SELECT_CC(N); break;
  case ISD::SETCC:       Res = PromoteIntRes_SETCC(N); break;
  case ISD::SHL:         Res = PromoteIntRes_SHL(N); break;
  case ISD::SIGN_EXTEND_INREG:
                         Res = PromoteIntRes_SIGN_EXTEND_INREG(N); break;
  case ISD::SRA:         Res = PromoteIntRes_SRA(N); break;
  case ISD::SRL:         Res = PromoteIntRes_SRL(N); break;
  case ISD::TRUNCATE:    Res = PromoteIntRes_TRUNCATE(N); break;
  case ISD::UNDEF:       Res = PromoteIntRes_UNDEF(N); break;
  case ISD::VAARG:       Res = PromoteIntRes_VAARG(N); break;

  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:  Res = PromoteIntRes_INT_EXTEND(N); break;

  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:  Res = PromoteIntRes_FP_TO_XINT(N); break;

  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::MUL:         Res = PromoteIntRes_SimpleIntBinOp(N); break;

  case ISD::SDIV:
  case ISD::SREM:        Res = PromoteIntRes_SDIV(N); break;

  case ISD::UDIV:
  case ISD::UREM:        Res = PromoteIntRes_UDIV(N); break;

  case ISD::SADDO:
  case ISD::SSUBO:       Res = PromoteIntRes_SADDSUBO(N, ResNo); break;
  case ISD::UADDO:
  case ISD::USUBO:       Res = PromoteIntRes_UADDSUBO(N, ResNo); break;
  case ISD::SMULO:
  case ISD::UMULO:       Res = PromoteIntRes_XMULO(N, ResNo); break;

  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_SWAP:
    Res = PromoteIntRes_Atomic1(cast<AtomicSDNode>(N)); break;

  case ISD::ATOMIC_CMP_SWAP:
    Res = PromoteIntRes_Atomic2(cast<AtomicSDNode>(N)); break;
  }

  // If the result is null then the sub-method took care of registering it.
  if (Res.getNode())
    SetPromotedInteger(SDValue(N, ResNo), Res);
}

SDValue DAGTypeLegalizer::PromoteIntRes_AssertSext(SDNode *N) {
  // Sign-extend the new bits, and continue the assertion.
  SDValue Op = SExtPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::AssertSext, N->getDebugLoc(),
                     Op.getValueType(), Op, N->getOperand(1));
}

SDValue DAGTypeLegalizer::PromoteIntRes_AssertZext(SDNode *N) {
  // Zero the new bits, and continue the assertion.
  SDValue Op = ZExtPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::AssertZext, N->getDebugLoc(),
                     Op.getValueType(), Op, N->getOperand(1));
}

SDValue DAGTypeLegalizer::PromoteIntRes_Atomic1(AtomicSDNode *N) {
  SDValue Op2 = GetPromotedInteger(N->getOperand(2));
  SDValue Res = DAG.getAtomic(N->getOpcode(), N->getDebugLoc(),
                              N->getMemoryVT(),
                              N->getChain(), N->getBasePtr(),
                              Op2, N->getSrcValue(), N->getAlignment());
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Res.getValue(1));
  return Res;
}

SDValue DAGTypeLegalizer::PromoteIntRes_Atomic2(AtomicSDNode *N) {
  SDValue Op2 = GetPromotedInteger(N->getOperand(2));
  SDValue Op3 = GetPromotedInteger(N->getOperand(3));
  SDValue Res = DAG.getAtomic(N->getOpcode(), N->getMemoryVT(),
                              N->getChain(), N->getBasePtr(),
                              Op2, Op3, N->getSrcValue(), N->getAlignment());
  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Res.getValue(1));
  return Res;
}

SDValue DAGTypeLegalizer::PromoteIntRes_BIT_CONVERT(SDNode *N) {
  SDValue InOp = N->getOperand(0);
  MVT InVT = InOp.getValueType();
  MVT NInVT = TLI.getTypeToTransformTo(InVT);
  MVT OutVT = N->getValueType(0);
  MVT NOutVT = TLI.getTypeToTransformTo(OutVT);
  DebugLoc dl = N->getDebugLoc();

  switch (getTypeAction(InVT)) {
  default:
    assert(false && "Unknown type action!");
    break;
  case Legal:
    break;
  case PromoteInteger:
    if (NOutVT.bitsEq(NInVT))
      // The input promotes to the same size.  Convert the promoted value.
      return DAG.getNode(ISD::BIT_CONVERT, dl,
                         NOutVT, GetPromotedInteger(InOp));
    break;
  case SoftenFloat:
    // Promote the integer operand by hand.
    return DAG.getNode(ISD::ANY_EXTEND, dl, NOutVT, GetSoftenedFloat(InOp));
  case ExpandInteger:
  case ExpandFloat:
    break;
  case ScalarizeVector:
    // Convert the element to an integer and promote it by hand.
    return DAG.getNode(ISD::ANY_EXTEND, dl, NOutVT,
                       BitConvertToInteger(GetScalarizedVector(InOp)));
  case SplitVector: {
    // For example, i32 = BIT_CONVERT v2i16 on alpha.  Convert the split
    // pieces of the input into integers and reassemble in the final type.
    SDValue Lo, Hi;
    GetSplitVector(N->getOperand(0), Lo, Hi);
    Lo = BitConvertToInteger(Lo);
    Hi = BitConvertToInteger(Hi);

    if (TLI.isBigEndian())
      std::swap(Lo, Hi);

    InOp = DAG.getNode(ISD::ANY_EXTEND, dl,
                       MVT::getIntegerVT(NOutVT.getSizeInBits()),
                       JoinIntegers(Lo, Hi));
    return DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, InOp);
  }
  case WidenVector:
    if (OutVT.bitsEq(NInVT))
      // The input is widened to the same size.  Convert to the widened value.
      return DAG.getNode(ISD::BIT_CONVERT, dl, OutVT, GetWidenedVector(InOp));
  }

  // Otherwise, lower the bit-convert to a store/load from the stack.
  // Create the stack frame object.  Make sure it is aligned for both
  // the source and destination types.
  SDValue FIPtr = DAG.CreateStackTemporary(InVT, OutVT);
  int FI = cast<FrameIndexSDNode>(FIPtr.getNode())->getIndex();
  const Value *SV = PseudoSourceValue::getFixedStack(FI);

  // Emit a store to the stack slot.
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl, InOp, FIPtr, SV, 0);

  // Result is an extending load from the stack slot.
  return DAG.getExtLoad(ISD::EXTLOAD, dl, NOutVT, Store, FIPtr, SV, 0, OutVT);
}

SDValue DAGTypeLegalizer::PromoteIntRes_BSWAP(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  MVT OVT = N->getValueType(0);
  MVT NVT = Op.getValueType();
  DebugLoc dl = N->getDebugLoc();

  unsigned DiffBits = NVT.getSizeInBits() - OVT.getSizeInBits();
  return DAG.getNode(ISD::SRL, dl, NVT, DAG.getNode(ISD::BSWAP, dl, NVT, Op),
                     DAG.getConstant(DiffBits, TLI.getPointerTy()));
}

SDValue DAGTypeLegalizer::PromoteIntRes_BUILD_PAIR(SDNode *N) {
  // The pair element type may be legal, or may not promote to the same type as
  // the result, for example i14 = BUILD_PAIR (i7, i7).  Handle all cases.
  return DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(),
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     JoinIntegers(N->getOperand(0), N->getOperand(1)));
}

SDValue DAGTypeLegalizer::PromoteIntRes_Constant(SDNode *N) {
  MVT VT = N->getValueType(0);
  // Zero extend things like i1, sign extend everything else.  It shouldn't
  // matter in theory which one we pick, but this tends to give better code?
  unsigned Opc = VT.isByteSized() ? ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
  SDValue Result = DAG.getNode(Opc, TLI.getTypeToTransformTo(VT),
                               SDValue(N, 0));
  assert(isa<ConstantSDNode>(Result) && "Didn't constant fold ext?");
  return Result;
}

SDValue DAGTypeLegalizer::PromoteIntRes_CONVERT_RNDSAT(SDNode *N) {
  ISD::CvtCode CvtCode = cast<CvtRndSatSDNode>(N)->getCvtCode();
  assert ((CvtCode == ISD::CVT_SS || CvtCode == ISD::CVT_SU ||
           CvtCode == ISD::CVT_US || CvtCode == ISD::CVT_UU ||
           CvtCode == ISD::CVT_SF || CvtCode == ISD::CVT_UF) &&
          "can only promote integers");
  MVT OutVT = TLI.getTypeToTransformTo(N->getValueType(0));
  return DAG.getConvertRndSat(OutVT, N->getOperand(0),
                              N->getOperand(1), N->getOperand(2),
                              N->getOperand(3), N->getOperand(4), CvtCode);
}

SDValue DAGTypeLegalizer::PromoteIntRes_CTLZ(SDNode *N) {
  // Zero extend to the promoted type and do the count there.
  SDValue Op = ZExtPromotedInteger(N->getOperand(0));
  MVT OVT = N->getValueType(0);
  MVT NVT = Op.getValueType();
  Op = DAG.getNode(ISD::CTLZ, NVT, Op);
  // Subtract off the extra leading bits in the bigger type.
  return DAG.getNode(ISD::SUB, N->getDebugLoc(), NVT, Op,
                     DAG.getConstant(NVT.getSizeInBits() -
                                     OVT.getSizeInBits(), NVT));
}

SDValue DAGTypeLegalizer::PromoteIntRes_CTPOP(SDNode *N) {
  // Zero extend to the promoted type and do the count there.
  SDValue Op = ZExtPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::CTPOP, N->getDebugLoc(), Op.getValueType(), Op);
}

SDValue DAGTypeLegalizer::PromoteIntRes_CTTZ(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  MVT OVT = N->getValueType(0);
  MVT NVT = Op.getValueType();
  DebugLoc dl = N->getDebugLoc();
  // The count is the same in the promoted type except if the original
  // value was zero.  This can be handled by setting the bit just off
  // the top of the original type.
  APInt TopBit(NVT.getSizeInBits(), 0);
  TopBit.set(OVT.getSizeInBits());
  Op = DAG.getNode(ISD::OR, dl, NVT, Op, DAG.getConstant(TopBit, NVT));
  return DAG.getNode(ISD::CTTZ, dl, NVT, Op);
}

SDValue DAGTypeLegalizer::PromoteIntRes_EXTRACT_VECTOR_ELT(SDNode *N) {
  MVT OldVT = N->getValueType(0);
  SDValue OldVec = N->getOperand(0);
  if (getTypeAction(OldVec.getValueType()) == WidenVector)
    OldVec = GetWidenedVector(N->getOperand(0));
  unsigned OldElts = OldVec.getValueType().getVectorNumElements();
  DebugLoc dl = N->getDebugLoc();

  if (OldElts == 1) {
    assert(!isTypeLegal(OldVec.getValueType()) &&
           "Legal one-element vector of a type needing promotion!");
    // It is tempting to follow GetScalarizedVector by a call to
    // GetPromotedInteger, but this would be wrong because the
    // scalarized value may not yet have been processed.
    return DAG.getNode(ISD::ANY_EXTEND, dl, TLI.getTypeToTransformTo(OldVT),
                       GetScalarizedVector(OldVec));
  }

  // Convert to a vector half as long with an element type of twice the width,
  // for example <4 x i16> -> <2 x i32>.
  assert(!(OldElts & 1) && "Odd length vectors not supported!");
  MVT NewVT = MVT::getIntegerVT(2 * OldVT.getSizeInBits());
  assert(OldVT.isSimple() && NewVT.isSimple());

  SDValue NewVec = DAG.getNode(ISD::BIT_CONVERT, dl,
                                 MVT::getVectorVT(NewVT, OldElts / 2),
                                 OldVec);

  // Extract the element at OldIdx / 2 from the new vector.
  SDValue OldIdx = N->getOperand(1);
  SDValue NewIdx = DAG.getNode(ISD::SRL, dl, OldIdx.getValueType(), OldIdx,
                               DAG.getConstant(1, TLI.getPointerTy()));
  SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, NewVT, NewVec, NewIdx);

  // Select the appropriate half of the element: Lo if OldIdx was even,
  // Hi if it was odd.
  SDValue Lo = Elt;
  SDValue Hi = DAG.getNode(ISD::SRL, dl, NewVT, Elt,
                           DAG.getConstant(OldVT.getSizeInBits(),
                                           TLI.getPointerTy()));
  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  // Extend to the promoted type.
  SDValue Odd = DAG.getNode(ISD::TRUNCATE, dl, MVT::i1, OldIdx);
  SDValue Res = DAG.getNode(ISD::SELECT, dl, NewVT, Odd, Hi, Lo);
  return DAG.getNode(ISD::ANY_EXTEND, dl, TLI.getTypeToTransformTo(OldVT), Res);
}

SDValue DAGTypeLegalizer::PromoteIntRes_FP_TO_XINT(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  unsigned NewOpc = N->getOpcode();
  DebugLoc dl = N->getDebugLoc();

  // If we're promoting a UINT to a larger size, check to see if the new node
  // will be legal.  If it isn't, check to see if FP_TO_SINT is legal, since
  // we can use that instead.  This allows us to generate better code for
  // FP_TO_UINT for small destination sizes on targets where FP_TO_UINT is not
  // legal, such as PowerPC.
  if (N->getOpcode() == ISD::FP_TO_UINT &&
      !TLI.isOperationLegalOrCustom(ISD::FP_TO_UINT, NVT) &&
      TLI.isOperationLegalOrCustom(ISD::FP_TO_SINT, NVT))
    NewOpc = ISD::FP_TO_SINT;

  SDValue Res = DAG.getNode(NewOpc, dl, NVT, N->getOperand(0));

  // Assert that the converted value fits in the original type.  If it doesn't
  // (eg: because the value being converted is too big), then the result of the
  // original operation was undefined anyway, so the assert is still correct.
  return DAG.getNode(N->getOpcode() == ISD::FP_TO_UINT ?
                     ISD::AssertZext : ISD::AssertSext, dl,
                     NVT, Res, DAG.getValueType(N->getValueType(0)));
}

SDValue DAGTypeLegalizer::PromoteIntRes_INT_EXTEND(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  DebugLoc dl = N->getDebugLoc();

  if (getTypeAction(N->getOperand(0).getValueType()) == PromoteInteger) {
    SDValue Res = GetPromotedInteger(N->getOperand(0));
    assert(Res.getValueType().bitsLE(NVT) && "Extension doesn't make sense!");

    // If the result and operand types are the same after promotion, simplify
    // to an in-register extension.
    if (NVT == Res.getValueType()) {
      // The high bits are not guaranteed to be anything.  Insert an extend.
      if (N->getOpcode() == ISD::SIGN_EXTEND)
        return DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Res,
                           DAG.getValueType(N->getOperand(0).getValueType()));
      if (N->getOpcode() == ISD::ZERO_EXTEND)
        return DAG.getZeroExtendInReg(Res, N->getOperand(0).getValueType());
      assert(N->getOpcode() == ISD::ANY_EXTEND && "Unknown integer extension!");
      return Res;
    }
  }

  // Otherwise, just extend the original operand all the way to the larger type.
  return DAG.getNode(N->getOpcode(), dl, NVT, N->getOperand(0));
}

SDValue DAGTypeLegalizer::PromoteIntRes_LOAD(LoadSDNode *N) {
  assert(ISD::isUNINDEXEDLoad(N) && "Indexed load during type legalization!");
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  ISD::LoadExtType ExtType =
    ISD::isNON_EXTLoad(N) ? ISD::EXTLOAD : N->getExtensionType();
  DebugLoc dl = N->getDebugLoc();
  SDValue Res = DAG.getExtLoad(ExtType, dl, NVT, N->getChain(), N->getBasePtr(),
                               N->getSrcValue(), N->getSrcValueOffset(),
                               N->getMemoryVT(), N->isVolatile(),
                               N->getAlignment());

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Res.getValue(1));
  return Res;
}

/// Promote the overflow flag of an overflowing arithmetic node.
SDValue DAGTypeLegalizer::PromoteIntRes_Overflow(SDNode *N) {
  // Simply change the return type of the boolean result.
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(1));
  MVT ValueVTs[] = { N->getValueType(0), NVT };
  SDValue Ops[] = { N->getOperand(0), N->getOperand(1) };
  SDValue Res = DAG.getNode(N->getOpcode(), N->getDebugLoc(),
                            DAG.getVTList(ValueVTs, 2), Ops, 2);

  // Modified the sum result - switch anything that used the old sum to use
  // the new one.
  ReplaceValueWith(SDValue(N, 0), Res);

  return SDValue(Res.getNode(), 1);
}

SDValue DAGTypeLegalizer::PromoteIntRes_SADDSUBO(SDNode *N, unsigned ResNo) {
  if (ResNo == 1)
    return PromoteIntRes_Overflow(N);

  // The operation overflowed iff the result in the larger type is not the
  // sign extension of its truncation to the original type.
  SDValue LHS = SExtPromotedInteger(N->getOperand(0));
  SDValue RHS = SExtPromotedInteger(N->getOperand(1));
  MVT OVT = N->getOperand(0).getValueType();
  MVT NVT = LHS.getValueType();
  DebugLoc dl = N->getDebugLoc();

  // Do the arithmetic in the larger type.
  unsigned Opcode = N->getOpcode() == ISD::SADDO ? ISD::ADD : ISD::SUB;
  SDValue Res = DAG.getNode(Opcode, dl, NVT, LHS, RHS);

  // Calculate the overflow flag: sign extend the arithmetic result from
  // the original type.
  SDValue Ofl = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, NVT, Res,
                            DAG.getValueType(OVT));
  // Overflowed if and only if this is not equal to Res.
  Ofl = DAG.getSetCC(dl, N->getValueType(1), Ofl, Res, ISD::SETNE);

  // Use the calculated overflow everywhere.
  ReplaceValueWith(SDValue(N, 1), Ofl);

  return Res;
}

SDValue DAGTypeLegalizer::PromoteIntRes_SDIV(SDNode *N) {
  // Sign extend the input.
  SDValue LHS = SExtPromotedInteger(N->getOperand(0));
  SDValue RHS = SExtPromotedInteger(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), N->getDebugLoc(),
                     LHS.getValueType(), LHS, RHS);
}

SDValue DAGTypeLegalizer::PromoteIntRes_SELECT(SDNode *N) {
  SDValue LHS = GetPromotedInteger(N->getOperand(1));
  SDValue RHS = GetPromotedInteger(N->getOperand(2));
  return DAG.getNode(ISD::SELECT, N->getDebugLoc(),
                     LHS.getValueType(), N->getOperand(0),LHS,RHS);
}

SDValue DAGTypeLegalizer::PromoteIntRes_SELECT_CC(SDNode *N) {
  SDValue LHS = GetPromotedInteger(N->getOperand(2));
  SDValue RHS = GetPromotedInteger(N->getOperand(3));
  return DAG.getNode(ISD::SELECT_CC, N->getDebugLoc(),
                     LHS.getValueType(), N->getOperand(0),
                     N->getOperand(1), LHS, RHS, N->getOperand(4));
}

SDValue DAGTypeLegalizer::PromoteIntRes_SETCC(SDNode *N) {
  MVT SVT = TLI.getSetCCResultType(N->getOperand(0).getValueType());
  assert(isTypeLegal(SVT) && "Illegal SetCC type!");
  DebugLoc dl = N->getDebugLoc();

  // Get the SETCC result using the canonical SETCC type.
  SDValue SetCC = DAG.getNode(ISD::SETCC, dl, SVT, N->getOperand(0),
                              N->getOperand(1), N->getOperand(2));

  // Convert to the expected type.
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  assert(NVT.bitsLE(SVT) && "Integer type overpromoted?");
  return DAG.getNode(ISD::TRUNCATE, dl, NVT, SetCC);
}

SDValue DAGTypeLegalizer::PromoteIntRes_SHL(SDNode *N) {
  return DAG.getNode(ISD::SHL, N->getDebugLoc(),
                     TLI.getTypeToTransformTo(N->getValueType(0)),
                     GetPromotedInteger(N->getOperand(0)), N->getOperand(1));
}

SDValue DAGTypeLegalizer::PromoteIntRes_SIGN_EXTEND_INREG(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, N->getDebugLoc(),
                     Op.getValueType(), Op, N->getOperand(1));
}

SDValue DAGTypeLegalizer::PromoteIntRes_SimpleIntBinOp(SDNode *N) {
  // The input may have strange things in the top bits of the registers, but
  // these operations don't care.  They may have weird bits going out, but
  // that too is okay if they are integer operations.
  SDValue LHS = GetPromotedInteger(N->getOperand(0));
  SDValue RHS = GetPromotedInteger(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), N->getDebugLoc(),
                    LHS.getValueType(), LHS, RHS);
}

SDValue DAGTypeLegalizer::PromoteIntRes_SRA(SDNode *N) {
  // The input value must be properly sign extended.
  SDValue Res = SExtPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::SRA, N->getDebugLoc(),
                     Res.getValueType(), Res, N->getOperand(1));
}

SDValue DAGTypeLegalizer::PromoteIntRes_SRL(SDNode *N) {
  // The input value must be properly zero extended.
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDValue Res = ZExtPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::SRL, N->getDebugLoc(), NVT, Res, N->getOperand(1));
}

SDValue DAGTypeLegalizer::PromoteIntRes_TRUNCATE(SDNode *N) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Res;

  switch (getTypeAction(N->getOperand(0).getValueType())) {
  default: assert(0 && "Unknown type action!");
  case Legal:
  case ExpandInteger:
    Res = N->getOperand(0);
    break;
  case PromoteInteger:
    Res = GetPromotedInteger(N->getOperand(0));
    break;
  }

  // Truncate to NVT instead of VT
  return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), NVT, Res);
}

SDValue DAGTypeLegalizer::PromoteIntRes_UADDSUBO(SDNode *N, unsigned ResNo) {
  if (ResNo == 1)
    return PromoteIntRes_Overflow(N);

  // The operation overflowed iff the result in the larger type is not the
  // zero extension of its truncation to the original type.
  SDValue LHS = ZExtPromotedInteger(N->getOperand(0));
  SDValue RHS = ZExtPromotedInteger(N->getOperand(1));
  MVT OVT = N->getOperand(0).getValueType();
  MVT NVT = LHS.getValueType();
  DebugLoc dl = N->getDebugLoc();

  // Do the arithmetic in the larger type.
  unsigned Opcode = N->getOpcode() == ISD::UADDO ? ISD::ADD : ISD::SUB;
  SDValue Res = DAG.getNode(Opcode, dl, NVT, LHS, RHS);

  // Calculate the overflow flag: zero extend the arithmetic result from
  // the original type.
  SDValue Ofl = DAG.getZeroExtendInReg(Res, OVT);
  // Overflowed if and only if this is not equal to Res.
  Ofl = DAG.getSetCC(dl, N->getValueType(1), Ofl, Res, ISD::SETNE);

  // Use the calculated overflow everywhere.
  ReplaceValueWith(SDValue(N, 1), Ofl);

  return Res;
}

SDValue DAGTypeLegalizer::PromoteIntRes_UDIV(SDNode *N) {
  // Zero extend the input.
  SDValue LHS = ZExtPromotedInteger(N->getOperand(0));
  SDValue RHS = ZExtPromotedInteger(N->getOperand(1));
  return DAG.getNode(N->getOpcode(), N->getDebugLoc(),
                     LHS.getValueType(), LHS, RHS);
}

SDValue DAGTypeLegalizer::PromoteIntRes_UNDEF(SDNode *N) {
  return DAG.getNode(ISD::UNDEF, N->getDebugLoc(),
                     TLI.getTypeToTransformTo(N->getValueType(0)));
}

SDValue DAGTypeLegalizer::PromoteIntRes_VAARG(SDNode *N) {
  SDValue Chain = N->getOperand(0); // Get the chain.
  SDValue Ptr = N->getOperand(1); // Get the pointer.
  MVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  MVT RegVT = TLI.getRegisterType(VT);
  unsigned NumRegs = TLI.getNumRegisters(VT);
  // The argument is passed as NumRegs registers of type RegVT.

  SmallVector<SDValue, 8> Parts(NumRegs);
  for (unsigned i = 0; i < NumRegs; ++i) {
    Parts[i] = DAG.getVAArg(RegVT, Chain, Ptr, N->getOperand(2));
    Chain = Parts[i].getValue(1);
  }

  // Handle endianness of the load.
  if (TLI.isBigEndian())
    std::reverse(Parts.begin(), Parts.end());

  // Assemble the parts in the promoted type.
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Res = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Parts[0]);
  for (unsigned i = 1; i < NumRegs; ++i) {
    SDValue Part = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, Parts[i]);
    // Shift it to the right position and "or" it in.
    Part = DAG.getNode(ISD::SHL, dl, NVT, Part,
                       DAG.getConstant(i * RegVT.getSizeInBits(),
                                       TLI.getPointerTy()));
    Res = DAG.getNode(ISD::OR, dl, NVT, Res, Part);
  }

  // Modified the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Chain);

  return Res;
}

SDValue DAGTypeLegalizer::PromoteIntRes_XMULO(SDNode *N, unsigned ResNo) {
  assert(ResNo == 1 && "Only boolean result promotion currently supported!");
  return PromoteIntRes_Overflow(N);
}

//===----------------------------------------------------------------------===//
//  Integer Operand Promotion
//===----------------------------------------------------------------------===//

/// PromoteIntegerOperand - This method is called when the specified operand of
/// the specified node is found to need promotion.  At this point, all of the
/// result types of the node are known to be legal, but other operands of the
/// node may need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::PromoteIntegerOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Promote integer operand: "; N->dump(&DAG); cerr << "\n");
  SDValue Res = SDValue();

  if (CustomLowerResults(N, N->getOperand(OpNo).getValueType(), false))
    return false;

  switch (N->getOpcode()) {
    default:
  #ifndef NDEBUG
    cerr << "PromoteIntegerOperand Op #" << OpNo << ": ";
    N->dump(&DAG); cerr << "\n";
  #endif
    assert(0 && "Do not know how to promote this operator's operand!");
    abort();

  case ISD::ANY_EXTEND:   Res = PromoteIntOp_ANY_EXTEND(N); break;
  case ISD::BR_CC:        Res = PromoteIntOp_BR_CC(N, OpNo); break;
  case ISD::BRCOND:       Res = PromoteIntOp_BRCOND(N, OpNo); break;
  case ISD::BUILD_PAIR:   Res = PromoteIntOp_BUILD_PAIR(N); break;
  case ISD::BUILD_VECTOR: Res = PromoteIntOp_BUILD_VECTOR(N); break;
  case ISD::CONVERT_RNDSAT:
                          Res = PromoteIntOp_CONVERT_RNDSAT(N); break;
  case ISD::INSERT_VECTOR_ELT:
                          Res = PromoteIntOp_INSERT_VECTOR_ELT(N, OpNo);break;
  case ISD::MEMBARRIER:   Res = PromoteIntOp_MEMBARRIER(N); break;
  case ISD::SELECT:       Res = PromoteIntOp_SELECT(N, OpNo); break;
  case ISD::SELECT_CC:    Res = PromoteIntOp_SELECT_CC(N, OpNo); break;
  case ISD::SETCC:        Res = PromoteIntOp_SETCC(N, OpNo); break;
  case ISD::SIGN_EXTEND:  Res = PromoteIntOp_SIGN_EXTEND(N); break;
  case ISD::SINT_TO_FP:   Res = PromoteIntOp_SINT_TO_FP(N); break;
  case ISD::STORE:        Res = PromoteIntOp_STORE(cast<StoreSDNode>(N),
                                                   OpNo); break;
  case ISD::TRUNCATE:     Res = PromoteIntOp_TRUNCATE(N); break;
  case ISD::UINT_TO_FP:   Res = PromoteIntOp_UINT_TO_FP(N); break;
  case ISD::ZERO_EXTEND:  Res = PromoteIntOp_ZERO_EXTEND(N); break;

  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR: Res = PromoteIntOp_Shift(N); break;
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Tell the legalizer
  // core about this.
  if (Res.getNode() == N)
    return true;

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// PromoteSetCCOperands - Promote the operands of a comparison.  This code is
/// shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::PromoteSetCCOperands(SDValue &NewLHS,SDValue &NewRHS,
                                            ISD::CondCode CCCode) {
  // We have to insert explicit sign or zero extends.  Note that we could
  // insert sign extends for ALL conditions, but zero extend is cheaper on
  // many machines (an AND instead of two shifts), so prefer it.
  switch (CCCode) {
  default: assert(0 && "Unknown integer comparison!");
  case ISD::SETEQ:
  case ISD::SETNE:
  case ISD::SETUGE:
  case ISD::SETUGT:
  case ISD::SETULE:
  case ISD::SETULT:
    // ALL of these operations will work if we either sign or zero extend
    // the operands (including the unsigned comparisons!).  Zero extend is
    // usually a simpler/cheaper operation, so prefer it.
    NewLHS = ZExtPromotedInteger(NewLHS);
    NewRHS = ZExtPromotedInteger(NewRHS);
    break;
  case ISD::SETGE:
  case ISD::SETGT:
  case ISD::SETLT:
  case ISD::SETLE:
    NewLHS = SExtPromotedInteger(NewLHS);
    NewRHS = SExtPromotedInteger(NewRHS);
    break;
  }
}

SDValue DAGTypeLegalizer::PromoteIntOp_ANY_EXTEND(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), N->getValueType(0), Op);
}

SDValue DAGTypeLegalizer::PromoteIntOp_BR_CC(SDNode *N, unsigned OpNo) {
  assert(OpNo == 2 && "Don't know how to promote this operand!");

  SDValue LHS = N->getOperand(2);
  SDValue RHS = N->getOperand(3);
  PromoteSetCCOperands(LHS, RHS, cast<CondCodeSDNode>(N->getOperand(1))->get());

  // The chain (Op#0), CC (#1) and basic block destination (Op#4) are always
  // legal types.
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                N->getOperand(1), LHS, RHS, N->getOperand(4));
}

SDValue DAGTypeLegalizer::PromoteIntOp_BRCOND(SDNode *N, unsigned OpNo) {
  assert(OpNo == 1 && "only know how to promote condition");

  // Promote all the way up to the canonical SetCC type.
  MVT SVT = TLI.getSetCCResultType(MVT::Other);
  SDValue Cond = PromoteTargetBoolean(N->getOperand(1), SVT);

  // The chain (Op#0) and basic block destination (Op#2) are always legal types.
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0), Cond,
                                N->getOperand(2));
}

SDValue DAGTypeLegalizer::PromoteIntOp_BUILD_PAIR(SDNode *N) {
  // Since the result type is legal, the operands must promote to it.
  MVT OVT = N->getOperand(0).getValueType();
  SDValue Lo = ZExtPromotedInteger(N->getOperand(0));
  SDValue Hi = GetPromotedInteger(N->getOperand(1));
  assert(Lo.getValueType() == N->getValueType(0) && "Operand over promoted?");
  DebugLoc dl = N->getDebugLoc();

  Hi = DAG.getNode(ISD::SHL, dl, N->getValueType(0), Hi,
                   DAG.getConstant(OVT.getSizeInBits(), TLI.getPointerTy()));
  return DAG.getNode(ISD::OR, dl, N->getValueType(0), Lo, Hi);
}

SDValue DAGTypeLegalizer::PromoteIntOp_BUILD_VECTOR(SDNode *N) {
  // The vector type is legal but the element type is not.  This implies
  // that the vector is a power-of-two in length and that the element
  // type does not have a strange size (eg: it is not i1).
  MVT VecVT = N->getValueType(0);
  unsigned NumElts = VecVT.getVectorNumElements();
  assert(!(NumElts & 1) && "Legal vector of one illegal element?");
  DebugLoc dl = N->getDebugLoc();

  // Build a vector of half the length out of elements of twice the bitwidth.
  // For example <4 x i16> -> <2 x i32>.
  MVT OldVT = N->getOperand(0).getValueType();
  MVT NewVT = MVT::getIntegerVT(2 * OldVT.getSizeInBits());
  assert(OldVT.isSimple() && NewVT.isSimple());

  std::vector<SDValue> NewElts;
  NewElts.reserve(NumElts/2);

  for (unsigned i = 0; i < NumElts; i += 2) {
    // Combine two successive elements into one promoted element.
    SDValue Lo = N->getOperand(i);
    SDValue Hi = N->getOperand(i+1);
    if (TLI.isBigEndian())
      std::swap(Lo, Hi);
    NewElts.push_back(JoinIntegers(Lo, Hi));
  }

  SDValue NewVec = DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::getVectorVT(NewVT, NewElts.size()),
                                 &NewElts[0], NewElts.size());

  // Convert the new vector to the old vector type.
  return DAG.getNode(ISD::BIT_CONVERT, dl, VecVT, NewVec);
}

SDValue DAGTypeLegalizer::PromoteIntOp_CONVERT_RNDSAT(SDNode *N) {
  ISD::CvtCode CvtCode = cast<CvtRndSatSDNode>(N)->getCvtCode();
  assert ((CvtCode == ISD::CVT_SS || CvtCode == ISD::CVT_SU ||
           CvtCode == ISD::CVT_US || CvtCode == ISD::CVT_UU ||
           CvtCode == ISD::CVT_FS || CvtCode == ISD::CVT_FU) &&
           "can only promote integer arguments");
  SDValue InOp = GetPromotedInteger(N->getOperand(0));
  return DAG.getConvertRndSat(N->getValueType(0), InOp,
                              N->getOperand(1), N->getOperand(2),
                              N->getOperand(3), N->getOperand(4), CvtCode);
}

SDValue DAGTypeLegalizer::PromoteIntOp_INSERT_VECTOR_ELT(SDNode *N,
                                                         unsigned OpNo) {
  if (OpNo == 1) {
    // Promote the inserted value.  This is valid because the type does not
    // have to match the vector element type.

    // Check that any extra bits introduced will be truncated away.
    assert(N->getOperand(1).getValueType().getSizeInBits() >=
           N->getValueType(0).getVectorElementType().getSizeInBits() &&
           "Type of inserted value narrower than vector element type!");
    return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                  GetPromotedInteger(N->getOperand(1)),
                                  N->getOperand(2));
  }

  assert(OpNo == 2 && "Different operand and result vector types?");

  // Promote the index.
  SDValue Idx = ZExtPromotedInteger(N->getOperand(2));
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                N->getOperand(1), Idx);
}

SDValue DAGTypeLegalizer::PromoteIntOp_MEMBARRIER(SDNode *N) {
  SDValue NewOps[6];
  NewOps[0] = N->getOperand(0);
  for (unsigned i = 1; i < array_lengthof(NewOps); ++i) {
    SDValue Flag = GetPromotedInteger(N->getOperand(i));
    NewOps[i] = DAG.getZeroExtendInReg(Flag, MVT::i1);
  }
  return DAG.UpdateNodeOperands(SDValue (N, 0), NewOps,
                                array_lengthof(NewOps));
}

SDValue DAGTypeLegalizer::PromoteIntOp_SELECT(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Only know how to promote condition");

  // Promote all the way up to the canonical SetCC type.
  MVT SVT = TLI.getSetCCResultType(N->getOperand(1).getValueType());
  SDValue Cond = PromoteTargetBoolean(N->getOperand(0), SVT);

  return DAG.UpdateNodeOperands(SDValue(N, 0), Cond,
                                N->getOperand(1), N->getOperand(2));
}

SDValue DAGTypeLegalizer::PromoteIntOp_SELECT_CC(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Don't know how to promote this operand!");

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  PromoteSetCCOperands(LHS, RHS, cast<CondCodeSDNode>(N->getOperand(4))->get());

  // The CC (#4) and the possible return values (#2 and #3) have legal types.
  return DAG.UpdateNodeOperands(SDValue(N, 0), LHS, RHS, N->getOperand(2),
                                N->getOperand(3), N->getOperand(4));
}

SDValue DAGTypeLegalizer::PromoteIntOp_SETCC(SDNode *N, unsigned OpNo) {
  assert(OpNo == 0 && "Don't know how to promote this operand!");

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  PromoteSetCCOperands(LHS, RHS, cast<CondCodeSDNode>(N->getOperand(2))->get());

  // The CC (#2) is always legal.
  return DAG.UpdateNodeOperands(SDValue(N, 0), LHS, RHS, N->getOperand(2));
}

SDValue DAGTypeLegalizer::PromoteIntOp_Shift(SDNode *N) {
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                ZExtPromotedInteger(N->getOperand(1)));
}

SDValue DAGTypeLegalizer::PromoteIntOp_SIGN_EXTEND(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  DebugLoc dl = N->getDebugLoc();
  Op = DAG.getNode(ISD::ANY_EXTEND, dl, N->getValueType(0), Op);
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Op.getValueType(),
                     Op, DAG.getValueType(N->getOperand(0).getValueType()));
}

SDValue DAGTypeLegalizer::PromoteIntOp_SINT_TO_FP(SDNode *N) {
  return DAG.UpdateNodeOperands(SDValue(N, 0),
                                SExtPromotedInteger(N->getOperand(0)));
}

SDValue DAGTypeLegalizer::PromoteIntOp_STORE(StoreSDNode *N, unsigned OpNo){
  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  SDValue Ch = N->getChain(), Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  DebugLoc dl = N->getDebugLoc();

  SDValue Val = GetPromotedInteger(N->getValue());  // Get promoted value.

  // Truncate the value and store the result.
  return DAG.getTruncStore(Ch, dl, Val, Ptr, N->getSrcValue(),
                           SVOffset, N->getMemoryVT(),
                           isVolatile, Alignment);
}

SDValue DAGTypeLegalizer::PromoteIntOp_TRUNCATE(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), N->getValueType(0), Op);
}

SDValue DAGTypeLegalizer::PromoteIntOp_UINT_TO_FP(SDNode *N) {
  return DAG.UpdateNodeOperands(SDValue(N, 0),
                                ZExtPromotedInteger(N->getOperand(0)));
}

SDValue DAGTypeLegalizer::PromoteIntOp_ZERO_EXTEND(SDNode *N) {
  SDValue Op = GetPromotedInteger(N->getOperand(0));
  Op = DAG.getNode(ISD::ANY_EXTEND, N->getDebugLoc(), N->getValueType(0), Op);
  return DAG.getZeroExtendInReg(Op, N->getOperand(0).getValueType());
}


//===----------------------------------------------------------------------===//
//  Integer Result Expansion
//===----------------------------------------------------------------------===//

/// ExpandIntegerResult - This method is called when the specified result of the
/// specified node is found to need expansion.  At this point, the node may also
/// have invalid operands or may have other results that need promotion, we just
/// know that (at least) one result needs expansion.
void DAGTypeLegalizer::ExpandIntegerResult(SDNode *N, unsigned ResNo) {
  DEBUG(cerr << "Expand integer result: "; N->dump(&DAG); cerr << "\n");
  SDValue Lo, Hi;
  Lo = Hi = SDValue();

  // See if the target wants to custom expand this node.
  if (CustomLowerResults(N, N->getValueType(ResNo), true))
    return;

  switch (N->getOpcode()) {
  default:
#ifndef NDEBUG
    cerr << "ExpandIntegerResult #" << ResNo << ": ";
    N->dump(&DAG); cerr << "\n";
#endif
    assert(0 && "Do not know how to expand the result of this operator!");
    abort();

  case ISD::MERGE_VALUES: SplitRes_MERGE_VALUES(N, Lo, Hi); break;
  case ISD::SELECT:       SplitRes_SELECT(N, Lo, Hi); break;
  case ISD::SELECT_CC:    SplitRes_SELECT_CC(N, Lo, Hi); break;
  case ISD::UNDEF:        SplitRes_UNDEF(N, Lo, Hi); break;

  case ISD::BIT_CONVERT:        ExpandRes_BIT_CONVERT(N, Lo, Hi); break;
  case ISD::BUILD_PAIR:         ExpandRes_BUILD_PAIR(N, Lo, Hi); break;
  case ISD::EXTRACT_ELEMENT:    ExpandRes_EXTRACT_ELEMENT(N, Lo, Hi); break;
  case ISD::EXTRACT_VECTOR_ELT: ExpandRes_EXTRACT_VECTOR_ELT(N, Lo, Hi); break;
  case ISD::VAARG:              ExpandRes_VAARG(N, Lo, Hi); break;

  case ISD::ANY_EXTEND:  ExpandIntRes_ANY_EXTEND(N, Lo, Hi); break;
  case ISD::AssertSext:  ExpandIntRes_AssertSext(N, Lo, Hi); break;
  case ISD::AssertZext:  ExpandIntRes_AssertZext(N, Lo, Hi); break;
  case ISD::BSWAP:       ExpandIntRes_BSWAP(N, Lo, Hi); break;
  case ISD::Constant:    ExpandIntRes_Constant(N, Lo, Hi); break;
  case ISD::CTLZ:        ExpandIntRes_CTLZ(N, Lo, Hi); break;
  case ISD::CTPOP:       ExpandIntRes_CTPOP(N, Lo, Hi); break;
  case ISD::CTTZ:        ExpandIntRes_CTTZ(N, Lo, Hi); break;
  case ISD::FP_TO_SINT:  ExpandIntRes_FP_TO_SINT(N, Lo, Hi); break;
  case ISD::FP_TO_UINT:  ExpandIntRes_FP_TO_UINT(N, Lo, Hi); break;
  case ISD::LOAD:        ExpandIntRes_LOAD(cast<LoadSDNode>(N), Lo, Hi); break;
  case ISD::MUL:         ExpandIntRes_MUL(N, Lo, Hi); break;
  case ISD::SDIV:        ExpandIntRes_SDIV(N, Lo, Hi); break;
  case ISD::SIGN_EXTEND: ExpandIntRes_SIGN_EXTEND(N, Lo, Hi); break;
  case ISD::SIGN_EXTEND_INREG: ExpandIntRes_SIGN_EXTEND_INREG(N, Lo, Hi); break;
  case ISD::SREM:        ExpandIntRes_SREM(N, Lo, Hi); break;
  case ISD::TRUNCATE:    ExpandIntRes_TRUNCATE(N, Lo, Hi); break;
  case ISD::UDIV:        ExpandIntRes_UDIV(N, Lo, Hi); break;
  case ISD::UREM:        ExpandIntRes_UREM(N, Lo, Hi); break;
  case ISD::ZERO_EXTEND: ExpandIntRes_ZERO_EXTEND(N, Lo, Hi); break;

  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: ExpandIntRes_Logical(N, Lo, Hi); break;

  case ISD::ADD:
  case ISD::SUB: ExpandIntRes_ADDSUB(N, Lo, Hi); break;

  case ISD::ADDC:
  case ISD::SUBC: ExpandIntRes_ADDSUBC(N, Lo, Hi); break;

  case ISD::ADDE:
  case ISD::SUBE: ExpandIntRes_ADDSUBE(N, Lo, Hi); break;

  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL: ExpandIntRes_Shift(N, Lo, Hi); break;
  }

  // If Lo/Hi is null, the sub-method took care of registering results etc.
  if (Lo.getNode())
    SetExpandedInteger(SDValue(N, ResNo), Lo, Hi);
}

/// ExpandShiftByConstant - N is a shift by a value that needs to be expanded,
/// and the shift amount is a constant 'Amt'.  Expand the operation.
void DAGTypeLegalizer::ExpandShiftByConstant(SDNode *N, unsigned Amt,
                                             SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  // Expand the incoming operand to be shifted, so that we have its parts
  SDValue InL, InH;
  GetExpandedInteger(N->getOperand(0), InL, InH);

  MVT NVT = InL.getValueType();
  unsigned VTBits = N->getValueType(0).getSizeInBits();
  unsigned NVTBits = NVT.getSizeInBits();
  MVT ShTy = N->getOperand(1).getValueType();

  if (N->getOpcode() == ISD::SHL) {
    if (Amt > VTBits) {
      Lo = Hi = DAG.getConstant(0, NVT);
    } else if (Amt > NVTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = DAG.getNode(ISD::SHL, dl,
                       NVT, InL, DAG.getConstant(Amt-NVTBits,ShTy));
    } else if (Amt == NVTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = InL;
    } else if (Amt == 1 &&
               TLI.isOperationLegalOrCustom(ISD::ADDC,
                                            TLI.getTypeToExpandTo(NVT))) {
      // Emit this X << 1 as X+X.
      SDVTList VTList = DAG.getVTList(NVT, MVT::Flag);
      SDValue LoOps[2] = { InL, InL };
      Lo = DAG.getNode(ISD::ADDC, dl, VTList, LoOps, 2);
      SDValue HiOps[3] = { InH, InH, Lo.getValue(1) };
      Hi = DAG.getNode(ISD::ADDE, dl, VTList, HiOps, 3);
    } else {
      Lo = DAG.getNode(ISD::SHL, dl, NVT, InL, DAG.getConstant(Amt, ShTy));
      Hi = DAG.getNode(ISD::OR, dl, NVT,
                       DAG.getNode(ISD::SHL, dl, NVT, InH,
                                   DAG.getConstant(Amt, ShTy)),
                       DAG.getNode(ISD::SRL, dl, NVT, InL,
                                   DAG.getConstant(NVTBits-Amt, ShTy)));
    }
    return;
  }

  if (N->getOpcode() == ISD::SRL) {
    if (Amt > VTBits) {
      Lo = DAG.getConstant(0, NVT);
      Hi = DAG.getConstant(0, NVT);
    } else if (Amt > NVTBits) {
      Lo = DAG.getNode(ISD::SRL, dl,
                       NVT, InH, DAG.getConstant(Amt-NVTBits,ShTy));
      Hi = DAG.getConstant(0, NVT);
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = DAG.getConstant(0, NVT);
    } else {
      Lo = DAG.getNode(ISD::OR, dl, NVT,
                       DAG.getNode(ISD::SRL, dl, NVT, InL,
                                   DAG.getConstant(Amt, ShTy)),
                       DAG.getNode(ISD::SHL, dl, NVT, InH,
                                   DAG.getConstant(NVTBits-Amt, ShTy)));
      Hi = DAG.getNode(ISD::SRL, dl, NVT, InH, DAG.getConstant(Amt, ShTy));
    }
    return;
  }

  assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
  if (Amt > VTBits) {
    Hi = Lo = DAG.getNode(ISD::SRA, dl, NVT, InH,
                          DAG.getConstant(NVTBits-1, ShTy));
  } else if (Amt > NVTBits) {
    Lo = DAG.getNode(ISD::SRA, dl, NVT, InH,
                     DAG.getConstant(Amt-NVTBits, ShTy));
    Hi = DAG.getNode(ISD::SRA, dl, NVT, InH,
                     DAG.getConstant(NVTBits-1, ShTy));
  } else if (Amt == NVTBits) {
    Lo = InH;
    Hi = DAG.getNode(ISD::SRA, dl, NVT, InH,
                     DAG.getConstant(NVTBits-1, ShTy));
  } else {
    Lo = DAG.getNode(ISD::OR, NVT,
                     DAG.getNode(ISD::SRL, dl, NVT, InL,
                                 DAG.getConstant(Amt, ShTy)),
                     DAG.getNode(ISD::SHL, dl, NVT, InH,
                                 DAG.getConstant(NVTBits-Amt, ShTy)));
    Hi = DAG.getNode(ISD::SRA, dl, NVT, InH, DAG.getConstant(Amt, ShTy));
  }
}

/// ExpandShiftWithKnownAmountBit - Try to determine whether we can simplify
/// this shift based on knowledge of the high bit of the shift amount.  If we
/// can tell this, we know that it is >= 32 or < 32, without knowing the actual
/// shift amount.
bool DAGTypeLegalizer::
ExpandShiftWithKnownAmountBit(SDNode *N, SDValue &Lo, SDValue &Hi) {
  SDValue Amt = N->getOperand(1);
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  MVT ShTy = Amt.getValueType();
  unsigned ShBits = ShTy.getSizeInBits();
  unsigned NVTBits = NVT.getSizeInBits();
  assert(isPowerOf2_32(NVTBits) &&
         "Expanded integer type size not a power of two!");
  DebugLoc dl = N->getDebugLoc();

  APInt HighBitMask = APInt::getHighBitsSet(ShBits, ShBits - Log2_32(NVTBits));
  APInt KnownZero, KnownOne;
  DAG.ComputeMaskedBits(N->getOperand(1), HighBitMask, KnownZero, KnownOne);

  // If we don't know anything about the high bits, exit.
  if (((KnownZero|KnownOne) & HighBitMask) == 0)
    return false;

  // Get the incoming operand to be shifted.
  SDValue InL, InH;
  GetExpandedInteger(N->getOperand(0), InL, InH);

  // If we know that any of the high bits of the shift amount are one, then we
  // can do this as a couple of simple shifts.
  if (KnownOne.intersects(HighBitMask)) {
    // Mask out the high bit, which we know is set.
    Amt = DAG.getNode(ISD::AND, dl, ShTy, Amt,
                      DAG.getConstant(~HighBitMask, ShTy));

    switch (N->getOpcode()) {
    default: assert(0 && "Unknown shift");
    case ISD::SHL:
      Lo = DAG.getConstant(0, NVT);              // Low part is zero.
      Hi = DAG.getNode(ISD::SHL, dl, NVT, InL, Amt); // High part from Lo part.
      return true;
    case ISD::SRL:
      Hi = DAG.getConstant(0, NVT);              // Hi part is zero.
      Lo = DAG.getNode(ISD::SRL, dl, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    case ISD::SRA:
      Hi = DAG.getNode(ISD::SRA, dl, NVT, InH,       // Sign extend high part.
                       DAG.getConstant(NVTBits-1, ShTy));
      Lo = DAG.getNode(ISD::SRA, dl, NVT, InH, Amt); // Lo part from Hi part.
      return true;
    }
  }

#if 0
  // FIXME: This code is broken for shifts with a zero amount!
  // If we know that all of the high bits of the shift amount are zero, then we
  // can do this as a couple of simple shifts.
  if ((KnownZero & HighBitMask) == HighBitMask) {
    // Compute 32-amt.
    SDValue Amt2 = DAG.getNode(ISD::SUB, ShTy,
                                 DAG.getConstant(NVTBits, ShTy),
                                 Amt);
    unsigned Op1, Op2;
    switch (N->getOpcode()) {
    default: assert(0 && "Unknown shift");
    case ISD::SHL:  Op1 = ISD::SHL; Op2 = ISD::SRL; break;
    case ISD::SRL:
    case ISD::SRA:  Op1 = ISD::SRL; Op2 = ISD::SHL; break;
    }

    Lo = DAG.getNode(N->getOpcode(), NVT, InL, Amt);
    Hi = DAG.getNode(ISD::OR, NVT,
                     DAG.getNode(Op1, NVT, InH, Amt),
                     DAG.getNode(Op2, NVT, InL, Amt2));
    return true;
  }
#endif

  return false;
}

void DAGTypeLegalizer::ExpandIntRes_ADDSUB(SDNode *N,
                                           SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  // Expand the subcomponents.
  SDValue LHSL, LHSH, RHSL, RHSH;
  GetExpandedInteger(N->getOperand(0), LHSL, LHSH);
  GetExpandedInteger(N->getOperand(1), RHSL, RHSH);

  MVT NVT = LHSL.getValueType();
  SDValue LoOps[2] = { LHSL, RHSL };
  SDValue HiOps[3] = { LHSH, RHSH };

  // Do not generate ADDC/ADDE or SUBC/SUBE if the target does not support
  // them.  TODO: Teach operation legalization how to expand unsupported
  // ADDC/ADDE/SUBC/SUBE.  The problem is that these operations generate
  // a carry of type MVT::Flag, but there doesn't seem to be any way to
  // generate a value of this type in the expanded code sequence.
  bool hasCarry =
    TLI.isOperationLegalOrCustom(N->getOpcode() == ISD::ADD ?
                                   ISD::ADDC : ISD::SUBC,
                                 TLI.getTypeToExpandTo(NVT));

  if (hasCarry) {
    SDVTList VTList = DAG.getVTList(NVT, MVT::Flag);
    if (N->getOpcode() == ISD::ADD) {
      Lo = DAG.getNode(ISD::ADDC, dl, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::ADDE, dl, VTList, HiOps, 3);
    } else {
      Lo = DAG.getNode(ISD::SUBC, dl, VTList, LoOps, 2);
      HiOps[2] = Lo.getValue(1);
      Hi = DAG.getNode(ISD::SUBE, dl, VTList, HiOps, 3);
    }
  } else {
    if (N->getOpcode() == ISD::ADD) {
      Lo = DAG.getNode(ISD::ADD, dl, NVT, LoOps, 2);
      Hi = DAG.getNode(ISD::ADD, dl, NVT, HiOps, 2);
      SDValue Cmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT), Lo, LoOps[0],
                                  ISD::SETULT);
      SDValue Carry1 = DAG.getNode(ISD::SELECT, dl, NVT, Cmp1,
                                   DAG.getConstant(1, NVT),
                                   DAG.getConstant(0, NVT));
      SDValue Cmp2 = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT), Lo, LoOps[1],
                                  ISD::SETULT);
      SDValue Carry2 = DAG.getNode(ISD::SELECT, dl, NVT, Cmp2,
                                   DAG.getConstant(1, NVT), Carry1);
      Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, Carry2);
    } else {
      Lo = DAG.getNode(ISD::SUB, dl, NVT, LoOps, 2);
      Hi = DAG.getNode(ISD::SUB, dl, NVT, HiOps, 2);
      SDValue Cmp =
        DAG.getSetCC(TLI.getSetCCResultType(LoOps[0].getValueType()),
                     LoOps[0], LoOps[1], ISD::SETULT);
      SDValue Borrow = DAG.getNode(ISD::SELECT, dl, NVT, Cmp,
                                   DAG.getConstant(1, NVT),
                                   DAG.getConstant(0, NVT));
      Hi = DAG.getNode(ISD::SUB, dl, NVT, Hi, Borrow);
    }
  }
}

void DAGTypeLegalizer::ExpandIntRes_ADDSUBC(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  // Expand the subcomponents.
  SDValue LHSL, LHSH, RHSL, RHSH;
  DebugLoc dl = N->getDebugLoc();
  GetExpandedInteger(N->getOperand(0), LHSL, LHSH);
  GetExpandedInteger(N->getOperand(1), RHSL, RHSH);
  SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
  SDValue LoOps[2] = { LHSL, RHSL };
  SDValue HiOps[3] = { LHSH, RHSH };

  if (N->getOpcode() == ISD::ADDC) {
    Lo = DAG.getNode(ISD::ADDC, dl, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::ADDE, dl, VTList, HiOps, 3);
  } else {
    Lo = DAG.getNode(ISD::SUBC, dl, VTList, LoOps, 2);
    HiOps[2] = Lo.getValue(1);
    Hi = DAG.getNode(ISD::SUBE, dl, VTList, HiOps, 3);
  }

  // Legalized the flag result - switch anything that used the old flag to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Hi.getValue(1));
}

void DAGTypeLegalizer::ExpandIntRes_ADDSUBE(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  // Expand the subcomponents.
  SDValue LHSL, LHSH, RHSL, RHSH;
  DebugLoc dl = N->getDebugLoc();
  GetExpandedInteger(N->getOperand(0), LHSL, LHSH);
  GetExpandedInteger(N->getOperand(1), RHSL, RHSH);
  SDVTList VTList = DAG.getVTList(LHSL.getValueType(), MVT::Flag);
  SDValue LoOps[3] = { LHSL, RHSL, N->getOperand(2) };
  SDValue HiOps[3] = { LHSH, RHSH };

  Lo = DAG.getNode(N->getOpcode(), dl, VTList, LoOps, 3);
  HiOps[2] = Lo.getValue(1);
  Hi = DAG.getNode(N->getOpcode(), dl, VTList, HiOps, 3);

  // Legalized the flag result - switch anything that used the old flag to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Hi.getValue(1));
}

void DAGTypeLegalizer::ExpandIntRes_ANY_EXTEND(SDNode *N,
                                               SDValue &Lo, SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  DebugLoc dl = N->getDebugLoc();
  SDValue Op = N->getOperand(0);
  if (Op.getValueType().bitsLE(NVT)) {
    // The low part is any extension of the input (which degenerates to a copy).
    Lo = DAG.getNode(ISD::ANY_EXTEND, dl, NVT, Op);
    Hi = DAG.getNode(ISD::UNDEF, dl, NVT);   // The high part is undefined.
  } else {
    // For example, extension of an i48 to an i64.  The operand type necessarily
    // promotes to the result type, so will end up being expanded too.
    assert(getTypeAction(Op.getValueType()) == PromoteInteger &&
           "Only know how to promote this result!");
    SDValue Res = GetPromotedInteger(Op);
    assert(Res.getValueType() == N->getValueType(0) &&
           "Operand over promoted?");
    // Split the promoted operand.  This will simplify when it is expanded.
    SplitInteger(Res, Lo, Hi);
  }
}

void DAGTypeLegalizer::ExpandIntRes_AssertSext(SDNode *N,
                                               SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  GetExpandedInteger(N->getOperand(0), Lo, Hi);
  MVT NVT = Lo.getValueType();
  MVT EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  unsigned NVTBits = NVT.getSizeInBits();
  unsigned EVTBits = EVT.getSizeInBits();

  if (NVTBits < EVTBits) {
    Hi = DAG.getNode(ISD::AssertSext, dl, NVT, Hi,
                     DAG.getValueType(MVT::getIntegerVT(EVTBits - NVTBits)));
  } else {
    Lo = DAG.getNode(ISD::AssertSext, dl, NVT, Lo, DAG.getValueType(EVT));
    // The high part replicates the sign bit of Lo, make it explicit.
    Hi = DAG.getNode(ISD::SRA, dl, NVT, Lo,
                     DAG.getConstant(NVTBits-1, TLI.getPointerTy()));
  }
}

void DAGTypeLegalizer::ExpandIntRes_AssertZext(SDNode *N,
                                               SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  GetExpandedInteger(N->getOperand(0), Lo, Hi);
  MVT NVT = Lo.getValueType();
  MVT EVT = cast<VTSDNode>(N->getOperand(1))->getVT();
  unsigned NVTBits = NVT.getSizeInBits();
  unsigned EVTBits = EVT.getSizeInBits();

  if (NVTBits < EVTBits) {
    Hi = DAG.getNode(ISD::AssertZext, dl, NVT, Hi,
                     DAG.getValueType(MVT::getIntegerVT(EVTBits - NVTBits)));
  } else {
    Lo = DAG.getNode(ISD::AssertZext, dl, NVT, Lo, DAG.getValueType(EVT));
    // The high part must be zero, make it explicit.
    Hi = DAG.getConstant(0, NVT);
  }
}

void DAGTypeLegalizer::ExpandIntRes_BSWAP(SDNode *N,
                                          SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  GetExpandedInteger(N->getOperand(0), Hi, Lo);  // Note swapped operands.
  Lo = DAG.getNode(ISD::BSWAP, dl, Lo.getValueType(), Lo);
  Hi = DAG.getNode(ISD::BSWAP, dl, Hi.getValueType(), Hi);
}

void DAGTypeLegalizer::ExpandIntRes_Constant(SDNode *N,
                                             SDValue &Lo, SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  unsigned NBitWidth = NVT.getSizeInBits();
  const APInt &Cst = cast<ConstantSDNode>(N)->getAPIntValue();
  Lo = DAG.getConstant(APInt(Cst).trunc(NBitWidth), NVT);
  Hi = DAG.getConstant(Cst.lshr(NBitWidth).trunc(NBitWidth), NVT);
}

void DAGTypeLegalizer::ExpandIntRes_CTLZ(SDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  // ctlz (HiLo) -> Hi != 0 ? ctlz(Hi) : (ctlz(Lo)+32)
  GetExpandedInteger(N->getOperand(0), Lo, Hi);
  MVT NVT = Lo.getValueType();

  SDValue HiNotZero = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT), Hi,
                                   DAG.getConstant(0, NVT), ISD::SETNE);

  SDValue LoLZ = DAG.getNode(ISD::CTLZ, dl, NVT, Lo);
  SDValue HiLZ = DAG.getNode(ISD::CTLZ, dl, NVT, Hi);

  Lo = DAG.getNode(ISD::SELECT, dl, NVT, HiNotZero, HiLZ,
                   DAG.getNode(ISD::ADD, dl, NVT, LoLZ,
                               DAG.getConstant(NVT.getSizeInBits(), NVT)));
  Hi = DAG.getConstant(0, NVT);
}

void DAGTypeLegalizer::ExpandIntRes_CTPOP(SDNode *N,
                                          SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  // ctpop(HiLo) -> ctpop(Hi)+ctpop(Lo)
  GetExpandedInteger(N->getOperand(0), Lo, Hi);
  MVT NVT = Lo.getValueType();
  Lo = DAG.getNode(ISD::ADD, dl, NVT, DAG.getNode(ISD::CTPOP, NVT, Lo),
                   DAG.getNode(ISD::CTPOP, dl, NVT, Hi));
  Hi = DAG.getConstant(0, NVT);
}

void DAGTypeLegalizer::ExpandIntRes_CTTZ(SDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  // cttz (HiLo) -> Lo != 0 ? cttz(Lo) : (cttz(Hi)+32)
  GetExpandedInteger(N->getOperand(0), Lo, Hi);
  MVT NVT = Lo.getValueType();

  SDValue LoNotZero = DAG.getSetCC(dl, TLI.getSetCCResultType(NVT), Lo,
                                   DAG.getConstant(0, NVT), ISD::SETNE);

  SDValue LoLZ = DAG.getNode(ISD::CTTZ, dl, NVT, Lo);
  SDValue HiLZ = DAG.getNode(ISD::CTTZ, dl, NVT, Hi);

  Lo = DAG.getNode(ISD::SELECT, dl, NVT, LoNotZero, LoLZ,
                   DAG.getNode(ISD::ADD, dl, NVT, HiLZ,
                               DAG.getConstant(NVT.getSizeInBits(), NVT)));
  Hi = DAG.getConstant(0, NVT);
}

void DAGTypeLegalizer::ExpandIntRes_FP_TO_SINT(SDNode *N, SDValue &Lo,
                                               SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  MVT VT = N->getValueType(0);
  SDValue Op = N->getOperand(0);
  RTLIB::Libcall LC = RTLIB::getFPTOSINT(Op.getValueType(), VT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpected fp-to-sint conversion!");
  SplitInteger(MakeLibCall(LC, VT, &Op, 1, true/*irrelevant*/, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_FP_TO_UINT(SDNode *N, SDValue &Lo,
                                               SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  MVT VT = N->getValueType(0);
  SDValue Op = N->getOperand(0);
  RTLIB::Libcall LC = RTLIB::getFPTOUINT(Op.getValueType(), VT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpected fp-to-uint conversion!");
  SplitInteger(MakeLibCall(LC, VT, &Op, 1, false/*irrelevant*/, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_LOAD(LoadSDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  if (ISD::isNormalLoad(N)) {
    ExpandRes_NormalLoad(N, Lo, Hi);
    return;
  }

  assert(ISD::isUNINDEXEDLoad(N) && "Indexed load during type legalization!");

  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDValue Ch  = N->getChain();
  SDValue Ptr = N->getBasePtr();
  ISD::LoadExtType ExtType = N->getExtensionType();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  DebugLoc dl = N->getDebugLoc();

  assert(NVT.isByteSized() && "Expanded type not byte sized!");

  if (N->getMemoryVT().bitsLE(NVT)) {
    MVT EVT = N->getMemoryVT();

    Lo = DAG.getExtLoad(ExtType, dl, NVT, Ch, Ptr, N->getSrcValue(), SVOffset,
                        EVT, isVolatile, Alignment);

    // Remember the chain.
    Ch = Lo.getValue(1);

    if (ExtType == ISD::SEXTLOAD) {
      // The high part is obtained by SRA'ing all but one of the bits of the
      // lo part.
      unsigned LoSize = Lo.getValueType().getSizeInBits();
      Hi = DAG.getNode(ISD::SRA, dl, NVT, Lo,
                       DAG.getConstant(LoSize-1, TLI.getPointerTy()));
    } else if (ExtType == ISD::ZEXTLOAD) {
      // The high part is just a zero.
      Hi = DAG.getConstant(0, NVT);
    } else {
      assert(ExtType == ISD::EXTLOAD && "Unknown extload!");
      // The high part is undefined.
      Hi = DAG.getNode(ISD::UNDEF, dl, NVT);
    }
  } else if (TLI.isLittleEndian()) {
    // Little-endian - low bits are at low addresses.
    Lo = DAG.getLoad(NVT, dl, Ch, Ptr, N->getSrcValue(), SVOffset,
                     isVolatile, Alignment);

    unsigned ExcessBits =
      N->getMemoryVT().getSizeInBits() - NVT.getSizeInBits();
    MVT NEVT = MVT::getIntegerVT(ExcessBits);

    // Increment the pointer to the other half.
    unsigned IncrementSize = NVT.getSizeInBits()/8;
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getIntPtrConstant(IncrementSize));
    Hi = DAG.getExtLoad(ExtType, dl, NVT, Ch, Ptr, N->getSrcValue(),
                        SVOffset+IncrementSize, NEVT,
                        isVolatile, MinAlign(Alignment, IncrementSize));

    // Build a factor node to remember that this load is independent of the
    // other one.
    Ch = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                     Hi.getValue(1));
  } else {
    // Big-endian - high bits are at low addresses.  Favor aligned loads at
    // the cost of some bit-fiddling.
    MVT EVT = N->getMemoryVT();
    unsigned EBytes = EVT.getStoreSizeInBits()/8;
    unsigned IncrementSize = NVT.getSizeInBits()/8;
    unsigned ExcessBits = (EBytes - IncrementSize)*8;

    // Load both the high bits and maybe some of the low bits.
    Hi = DAG.getExtLoad(ExtType, dl, NVT, Ch, Ptr, N->getSrcValue(), SVOffset,
                        MVT::getIntegerVT(EVT.getSizeInBits() - ExcessBits),
                        isVolatile, Alignment);

    // Increment the pointer to the other half.
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getIntPtrConstant(IncrementSize));
    // Load the rest of the low bits.
    Lo = DAG.getExtLoad(ISD::ZEXTLOAD, dl, NVT, Ch, Ptr, N->getSrcValue(),
                        SVOffset+IncrementSize,
                        MVT::getIntegerVT(ExcessBits),
                        isVolatile, MinAlign(Alignment, IncrementSize));

    // Build a factor node to remember that this load is independent of the
    // other one.
    Ch = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                     Hi.getValue(1));

    if (ExcessBits < NVT.getSizeInBits()) {
      // Transfer low bits from the bottom of Hi to the top of Lo.
      Lo = DAG.getNode(ISD::OR, dl, NVT, Lo,
                       DAG.getNode(ISD::SHL, dl, NVT, Hi,
                                   DAG.getConstant(ExcessBits,
                                                   TLI.getPointerTy())));
      // Move high bits to the right position in Hi.
      Hi = DAG.getNode(ExtType == ISD::SEXTLOAD ? ISD::SRA : ISD::SRL, dl,
                       NVT, Hi,
                       DAG.getConstant(NVT.getSizeInBits() - ExcessBits,
                                       TLI.getPointerTy()));
    }
  }

  // Legalized the chain result - switch anything that used the old chain to
  // use the new one.
  ReplaceValueWith(SDValue(N, 1), Ch);
}

void DAGTypeLegalizer::ExpandIntRes_Logical(SDNode *N,
                                            SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  SDValue LL, LH, RL, RH;
  GetExpandedInteger(N->getOperand(0), LL, LH);
  GetExpandedInteger(N->getOperand(1), RL, RH);
  Lo = DAG.getNode(N->getOpcode(), dl, LL.getValueType(), LL, RL);
  Hi = DAG.getNode(N->getOpcode(), dl, LL.getValueType(), LH, RH);
}

void DAGTypeLegalizer::ExpandIntRes_MUL(SDNode *N,
                                        SDValue &Lo, SDValue &Hi) {
  MVT VT = N->getValueType(0);
  MVT NVT = TLI.getTypeToTransformTo(VT);
  DebugLoc dl = N->getDebugLoc();

  bool HasMULHS = TLI.isOperationLegalOrCustom(ISD::MULHS, NVT);
  bool HasMULHU = TLI.isOperationLegalOrCustom(ISD::MULHU, NVT);
  bool HasSMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::SMUL_LOHI, NVT);
  bool HasUMUL_LOHI = TLI.isOperationLegalOrCustom(ISD::UMUL_LOHI, NVT);
  if (HasMULHU || HasMULHS || HasUMUL_LOHI || HasSMUL_LOHI) {
    SDValue LL, LH, RL, RH;
    GetExpandedInteger(N->getOperand(0), LL, LH);
    GetExpandedInteger(N->getOperand(1), RL, RH);
    unsigned OuterBitSize = VT.getSizeInBits();
    unsigned InnerBitSize = NVT.getSizeInBits();
    unsigned LHSSB = DAG.ComputeNumSignBits(N->getOperand(0));
    unsigned RHSSB = DAG.ComputeNumSignBits(N->getOperand(1));

    APInt HighMask = APInt::getHighBitsSet(OuterBitSize, InnerBitSize);
    if (DAG.MaskedValueIsZero(N->getOperand(0), HighMask) &&
        DAG.MaskedValueIsZero(N->getOperand(1), HighMask)) {
      // The inputs are both zero-extended.
      if (HasUMUL_LOHI) {
        // We can emit a umul_lohi.
        Lo = DAG.getNode(ISD::UMUL_LOHI, dl, DAG.getVTList(NVT, NVT), LL, RL);
        Hi = SDValue(Lo.getNode(), 1);
        return;
      }
      if (HasMULHU) {
        // We can emit a mulhu+mul.
        Lo = DAG.getNode(ISD::MUL, dl, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHU, dl, NVT, LL, RL);
        return;
      }
    }
    if (LHSSB > InnerBitSize && RHSSB > InnerBitSize) {
      // The input values are both sign-extended.
      if (HasSMUL_LOHI) {
        // We can emit a smul_lohi.
        Lo = DAG.getNode(ISD::SMUL_LOHI, dl, DAG.getVTList(NVT, NVT), LL, RL);
        Hi = SDValue(Lo.getNode(), 1);
        return;
      }
      if (HasMULHS) {
        // We can emit a mulhs+mul.
        Lo = DAG.getNode(ISD::MUL, dl, NVT, LL, RL);
        Hi = DAG.getNode(ISD::MULHS, dl, NVT, LL, RL);
        return;
      }
    }
    if (HasUMUL_LOHI) {
      // Lo,Hi = umul LHS, RHS.
      SDValue UMulLOHI = DAG.getNode(ISD::UMUL_LOHI, dl,
                                       DAG.getVTList(NVT, NVT), LL, RL);
      Lo = UMulLOHI;
      Hi = UMulLOHI.getValue(1);
      RH = DAG.getNode(ISD::MUL, dl, NVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, dl, NVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, LH);
      return;
    }
    if (HasMULHU) {
      Lo = DAG.getNode(ISD::MUL, dl, NVT, LL, RL);
      Hi = DAG.getNode(ISD::MULHU, dl, NVT, LL, RL);
      RH = DAG.getNode(ISD::MUL, dl, NVT, LL, RH);
      LH = DAG.getNode(ISD::MUL, dl, NVT, LH, RL);
      Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, RH);
      Hi = DAG.getNode(ISD::ADD, dl, NVT, Hi, LH);
      return;
    }
  }

  // If nothing else, we can make a libcall.
  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  if (VT == MVT::i16)
    LC = RTLIB::MUL_I16;
  else if (VT == MVT::i32)
    LC = RTLIB::MUL_I32;
  else if (VT == MVT::i64)
    LC = RTLIB::MUL_I64;
  else if (VT == MVT::i128)
    LC = RTLIB::MUL_I128;
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported MUL!");

  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SplitInteger(MakeLibCall(LC, VT, Ops, 2, true/*irrelevant*/, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_SDIV(SDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  MVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  if (VT == MVT::i32)
    LC = RTLIB::SDIV_I32;
  else if (VT == MVT::i64)
    LC = RTLIB::SDIV_I64;
  else if (VT == MVT::i128)
    LC = RTLIB::SDIV_I128;
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported SDIV!");

  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SplitInteger(MakeLibCall(LC, VT, Ops, 2, true, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_Shift(SDNode *N,
                                          SDValue &Lo, SDValue &Hi) {
  MVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  // If we can emit an efficient shift operation, do so now.  Check to see if
  // the RHS is a constant.
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return ExpandShiftByConstant(N, CN->getZExtValue(), Lo, Hi);

  // If we can determine that the high bit of the shift is zero or one, even if
  // the low bits are variable, emit this shift in an optimized form.
  if (ExpandShiftWithKnownAmountBit(N, Lo, Hi))
    return;

  // If this target supports shift_PARTS, use it.  First, map to the _PARTS opc.
  unsigned PartsOpc;
  if (N->getOpcode() == ISD::SHL) {
    PartsOpc = ISD::SHL_PARTS;
  } else if (N->getOpcode() == ISD::SRL) {
    PartsOpc = ISD::SRL_PARTS;
  } else {
    assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
    PartsOpc = ISD::SRA_PARTS;
  }

  // Next check to see if the target supports this SHL_PARTS operation or if it
  // will custom expand it.
  MVT NVT = TLI.getTypeToTransformTo(VT);
  TargetLowering::LegalizeAction Action = TLI.getOperationAction(PartsOpc, NVT);
  if ((Action == TargetLowering::Legal && TLI.isTypeLegal(NVT)) ||
      Action == TargetLowering::Custom) {
    // Expand the subcomponents.
    SDValue LHSL, LHSH;
    GetExpandedInteger(N->getOperand(0), LHSL, LHSH);

    SDValue Ops[] = { LHSL, LHSH, N->getOperand(1) };
    MVT VT = LHSL.getValueType();
    Lo = DAG.getNode(PartsOpc, dl, DAG.getNodeValueTypes(VT, VT), 2, Ops, 3);
    Hi = Lo.getValue(1);
    return;
  }

  // Otherwise, emit a libcall.
  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  bool isSigned;
  if (N->getOpcode() == ISD::SHL) {
    isSigned = false; /*sign irrelevant*/
    if (VT == MVT::i16)
      LC = RTLIB::SHL_I16;
    else if (VT == MVT::i32)
      LC = RTLIB::SHL_I32;
    else if (VT == MVT::i64)
      LC = RTLIB::SHL_I64;
    else if (VT == MVT::i128)
      LC = RTLIB::SHL_I128;
  } else if (N->getOpcode() == ISD::SRL) {
    isSigned = false;
    if (VT == MVT::i16)
      LC = RTLIB::SRL_I16;
    else if (VT == MVT::i32)
      LC = RTLIB::SRL_I32;
    else if (VT == MVT::i64)
      LC = RTLIB::SRL_I64;
    else if (VT == MVT::i128)
      LC = RTLIB::SRL_I128;
  } else {
    assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
    isSigned = true;
    if (VT == MVT::i16)
      LC = RTLIB::SRA_I16;
    else if (VT == MVT::i32)
      LC = RTLIB::SRA_I32;
    else if (VT == MVT::i64)
      LC = RTLIB::SRA_I64;
    else if (VT == MVT::i128)
      LC = RTLIB::SRA_I128;
  }
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported shift!");

  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SplitInteger(MakeLibCall(LC, VT, Ops, 2, isSigned, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_SIGN_EXTEND(SDNode *N,
                                                SDValue &Lo, SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  DebugLoc dl = N->getDebugLoc();
  SDValue Op = N->getOperand(0);
  if (Op.getValueType().bitsLE(NVT)) {
    // The low part is sign extension of the input (degenerates to a copy).
    Lo = DAG.getNode(ISD::SIGN_EXTEND, dl, NVT, N->getOperand(0));
    // The high part is obtained by SRA'ing all but one of the bits of low part.
    unsigned LoSize = NVT.getSizeInBits();
    Hi = DAG.getNode(ISD::SRA, dl, NVT, Lo,
                     DAG.getConstant(LoSize-1, TLI.getPointerTy()));
  } else {
    // For example, extension of an i48 to an i64.  The operand type necessarily
    // promotes to the result type, so will end up being expanded too.
    assert(getTypeAction(Op.getValueType()) == PromoteInteger &&
           "Only know how to promote this result!");
    SDValue Res = GetPromotedInteger(Op);
    assert(Res.getValueType() == N->getValueType(0) &&
           "Operand over promoted?");
    // Split the promoted operand.  This will simplify when it is expanded.
    SplitInteger(Res, Lo, Hi);
    unsigned ExcessBits =
      Op.getValueType().getSizeInBits() - NVT.getSizeInBits();
    Hi = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Hi.getValueType(), Hi,
                     DAG.getValueType(MVT::getIntegerVT(ExcessBits)));
  }
}

void DAGTypeLegalizer::
ExpandIntRes_SIGN_EXTEND_INREG(SDNode *N, SDValue &Lo, SDValue &Hi) {
  DebugLoc dl = N->getDebugLoc();
  GetExpandedInteger(N->getOperand(0), Lo, Hi);
  MVT EVT = cast<VTSDNode>(N->getOperand(1))->getVT();

  if (EVT.bitsLE(Lo.getValueType())) {
    // sext_inreg the low part if needed.
    Lo = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Lo.getValueType(), Lo,
                     N->getOperand(1));

    // The high part gets the sign extension from the lo-part.  This handles
    // things like sextinreg V:i64 from i8.
    Hi = DAG.getNode(ISD::SRA, dl, Hi.getValueType(), Lo,
                     DAG.getConstant(Hi.getValueType().getSizeInBits()-1,
                                     TLI.getPointerTy()));
  } else {
    // For example, extension of an i48 to an i64.  Leave the low part alone,
    // sext_inreg the high part.
    unsigned ExcessBits =
      EVT.getSizeInBits() - Lo.getValueType().getSizeInBits();
    Hi = DAG.getNode(ISD::SIGN_EXTEND_INREG, dl, Hi.getValueType(), Hi,
                     DAG.getValueType(MVT::getIntegerVT(ExcessBits)));
  }
}

void DAGTypeLegalizer::ExpandIntRes_SREM(SDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  MVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  if (VT == MVT::i32)
    LC = RTLIB::SREM_I32;
  else if (VT == MVT::i64)
    LC = RTLIB::SREM_I64;
  else if (VT == MVT::i128)
    LC = RTLIB::SREM_I128;
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported SREM!");

  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SplitInteger(MakeLibCall(LC, VT, Ops, 2, true, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_TRUNCATE(SDNode *N,
                                             SDValue &Lo, SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  DebugLoc dl = N->getDebugLoc();
  Lo = DAG.getNode(ISD::TRUNCATE, dl, NVT, N->getOperand(0));
  Hi = DAG.getNode(ISD::SRL, dl,
                   N->getOperand(0).getValueType(), N->getOperand(0),
                   DAG.getConstant(NVT.getSizeInBits(), TLI.getPointerTy()));
  Hi = DAG.getNode(ISD::TRUNCATE, dl, NVT, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_UDIV(SDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  MVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  if (VT == MVT::i32)
    LC = RTLIB::UDIV_I32;
  else if (VT == MVT::i64)
    LC = RTLIB::UDIV_I64;
  else if (VT == MVT::i128)
    LC = RTLIB::UDIV_I128;
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported UDIV!");

  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SplitInteger(MakeLibCall(LC, VT, Ops, 2, false, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_UREM(SDNode *N,
                                         SDValue &Lo, SDValue &Hi) {
  MVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  RTLIB::Libcall LC = RTLIB::UNKNOWN_LIBCALL;
  if (VT == MVT::i32)
    LC = RTLIB::UREM_I32;
  else if (VT == MVT::i64)
    LC = RTLIB::UREM_I64;
  else if (VT == MVT::i128)
    LC = RTLIB::UREM_I128;
  assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unsupported UREM!");

  SDValue Ops[2] = { N->getOperand(0), N->getOperand(1) };
  SplitInteger(MakeLibCall(LC, VT, Ops, 2, false, dl), Lo, Hi);
}

void DAGTypeLegalizer::ExpandIntRes_ZERO_EXTEND(SDNode *N,
                                                SDValue &Lo, SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  DebugLoc dl = N->getDebugLoc();
  SDValue Op = N->getOperand(0);
  if (Op.getValueType().bitsLE(NVT)) {
    // The low part is zero extension of the input (degenerates to a copy).
    Lo = DAG.getNode(ISD::ZERO_EXTEND, dl, NVT, N->getOperand(0));
    Hi = DAG.getConstant(0, NVT);   // The high part is just a zero.
  } else {
    // For example, extension of an i48 to an i64.  The operand type necessarily
    // promotes to the result type, so will end up being expanded too.
    assert(getTypeAction(Op.getValueType()) == PromoteInteger &&
           "Only know how to promote this result!");
    SDValue Res = GetPromotedInteger(Op);
    assert(Res.getValueType() == N->getValueType(0) &&
           "Operand over promoted?");
    // Split the promoted operand.  This will simplify when it is expanded.
    SplitInteger(Res, Lo, Hi);
    unsigned ExcessBits =
      Op.getValueType().getSizeInBits() - NVT.getSizeInBits();
    Hi = DAG.getZeroExtendInReg(Hi, MVT::getIntegerVT(ExcessBits));
  }
}


//===----------------------------------------------------------------------===//
//  Integer Operand Expansion
//===----------------------------------------------------------------------===//

/// ExpandIntegerOperand - This method is called when the specified operand of
/// the specified node is found to need expansion.  At this point, all of the
/// result types of the node are known to be legal, but other operands of the
/// node may need promotion or expansion as well as the specified one.
bool DAGTypeLegalizer::ExpandIntegerOperand(SDNode *N, unsigned OpNo) {
  DEBUG(cerr << "Expand integer operand: "; N->dump(&DAG); cerr << "\n");
  SDValue Res = SDValue();

  if (CustomLowerResults(N, N->getOperand(OpNo).getValueType(), false))
    return false;

  switch (N->getOpcode()) {
  default:
  #ifndef NDEBUG
    cerr << "ExpandIntegerOperand Op #" << OpNo << ": ";
    N->dump(&DAG); cerr << "\n";
  #endif
    assert(0 && "Do not know how to expand this operator's operand!");
    abort();

  case ISD::BIT_CONVERT:       Res = ExpandOp_BIT_CONVERT(N); break;
  case ISD::BR_CC:             Res = ExpandIntOp_BR_CC(N); break;
  case ISD::BUILD_VECTOR:      Res = ExpandOp_BUILD_VECTOR(N); break;
  case ISD::EXTRACT_ELEMENT:   Res = ExpandOp_EXTRACT_ELEMENT(N); break;
  case ISD::INSERT_VECTOR_ELT: Res = ExpandOp_INSERT_VECTOR_ELT(N); break;
  case ISD::SCALAR_TO_VECTOR:  Res = ExpandOp_SCALAR_TO_VECTOR(N); break;
  case ISD::SELECT_CC:         Res = ExpandIntOp_SELECT_CC(N); break;
  case ISD::SETCC:             Res = ExpandIntOp_SETCC(N); break;
  case ISD::SINT_TO_FP:        Res = ExpandIntOp_SINT_TO_FP(N); break;
  case ISD::STORE:   Res = ExpandIntOp_STORE(cast<StoreSDNode>(N), OpNo); break;
  case ISD::TRUNCATE:          Res = ExpandIntOp_TRUNCATE(N); break;
  case ISD::UINT_TO_FP:        Res = ExpandIntOp_UINT_TO_FP(N); break;

  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
  case ISD::ROTL:
  case ISD::ROTR: Res = ExpandIntOp_Shift(N); break;
  }

  // If the result is null, the sub-method took care of registering results etc.
  if (!Res.getNode()) return false;

  // If the result is N, the sub-method updated N in place.  Tell the legalizer
  // core about this.
  if (Res.getNode() == N)
    return true;

  assert(Res.getValueType() == N->getValueType(0) && N->getNumValues() == 1 &&
         "Invalid operand expansion");

  ReplaceValueWith(SDValue(N, 0), Res);
  return false;
}

/// IntegerExpandSetCCOperands - Expand the operands of a comparison.  This code
/// is shared among BR_CC, SELECT_CC, and SETCC handlers.
void DAGTypeLegalizer::IntegerExpandSetCCOperands(SDValue &NewLHS,
                                                  SDValue &NewRHS,
                                                  ISD::CondCode &CCCode,
                                                  DebugLoc dl) {
  SDValue LHSLo, LHSHi, RHSLo, RHSHi;
  GetExpandedInteger(NewLHS, LHSLo, LHSHi);
  GetExpandedInteger(NewRHS, RHSLo, RHSHi);

  MVT VT = NewLHS.getValueType();

  if (CCCode == ISD::SETEQ || CCCode == ISD::SETNE) {
    if (RHSLo == RHSHi) {
      if (ConstantSDNode *RHSCST = dyn_cast<ConstantSDNode>(RHSLo)) {
        if (RHSCST->isAllOnesValue()) {
          // Equality comparison to -1.
          NewLHS = DAG.getNode(ISD::AND, dl,
                               LHSLo.getValueType(), LHSLo, LHSHi);
          NewRHS = RHSLo;
          return;
        }
      }
    }

    NewLHS = DAG.getNode(ISD::XOR, dl, LHSLo.getValueType(), LHSLo, RHSLo);
    NewRHS = DAG.getNode(ISD::XOR, dl, LHSLo.getValueType(), LHSHi, RHSHi);
    NewLHS = DAG.getNode(ISD::OR, dl, NewLHS.getValueType(), NewLHS, NewRHS);
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    return;
  }

  // If this is a comparison of the sign bit, just look at the top part.
  // X > -1,  x < 0
  if (ConstantSDNode *CST = dyn_cast<ConstantSDNode>(NewRHS))
    if ((CCCode == ISD::SETLT && CST->isNullValue()) ||     // X < 0
        (CCCode == ISD::SETGT && CST->isAllOnesValue())) {  // X > -1
      NewLHS = LHSHi;
      NewRHS = RHSHi;
      return;
    }

  // FIXME: This generated code sucks.
  ISD::CondCode LowCC;
  switch (CCCode) {
  default: assert(0 && "Unknown integer setcc!");
  case ISD::SETLT:
  case ISD::SETULT: LowCC = ISD::SETULT; break;
  case ISD::SETGT:
  case ISD::SETUGT: LowCC = ISD::SETUGT; break;
  case ISD::SETLE:
  case ISD::SETULE: LowCC = ISD::SETULE; break;
  case ISD::SETGE:
  case ISD::SETUGE: LowCC = ISD::SETUGE; break;
  }

  // Tmp1 = lo(op1) < lo(op2)   // Always unsigned comparison
  // Tmp2 = hi(op1) < hi(op2)   // Signedness depends on operands
  // dest = hi(op1) == hi(op2) ? Tmp1 : Tmp2;

  // NOTE: on targets without efficient SELECT of bools, we can always use
  // this identity: (B1 ? B2 : B3) --> (B1 & B2)|(!B1&B3)
  TargetLowering::DAGCombinerInfo DagCombineInfo(DAG, false, true, NULL);
  SDValue Tmp1, Tmp2;
  Tmp1 = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSLo.getValueType()),
                           LHSLo, RHSLo, LowCC, false, DagCombineInfo, dl);
  if (!Tmp1.getNode())
    Tmp1 = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSLo.getValueType()),
                        LHSLo, RHSLo, LowCC);
  Tmp2 = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSHi.getValueType()),
                           LHSHi, RHSHi, CCCode, false, DagCombineInfo, dl);
  if (!Tmp2.getNode())
    Tmp2 = DAG.getNode(ISD::SETCC, dl,
                       TLI.getSetCCResultType(LHSHi.getValueType()),
                       LHSHi, RHSHi, DAG.getCondCode(CCCode));

  ConstantSDNode *Tmp1C = dyn_cast<ConstantSDNode>(Tmp1.getNode());
  ConstantSDNode *Tmp2C = dyn_cast<ConstantSDNode>(Tmp2.getNode());
  if ((Tmp1C && Tmp1C->isNullValue()) ||
      (Tmp2C && Tmp2C->isNullValue() &&
       (CCCode == ISD::SETLE || CCCode == ISD::SETGE ||
        CCCode == ISD::SETUGE || CCCode == ISD::SETULE)) ||
      (Tmp2C && Tmp2C->getAPIntValue() == 1 &&
       (CCCode == ISD::SETLT || CCCode == ISD::SETGT ||
        CCCode == ISD::SETUGT || CCCode == ISD::SETULT))) {
    // low part is known false, returns high part.
    // For LE / GE, if high part is known false, ignore the low part.
    // For LT / GT, if high part is known true, ignore the low part.
    NewLHS = Tmp2;
    NewRHS = SDValue();
    return;
  }

  NewLHS = TLI.SimplifySetCC(TLI.getSetCCResultType(LHSHi.getValueType()),
                             LHSHi, RHSHi, ISD::SETEQ, false, 
                             DagCombineInfo, dl);
  if (!NewLHS.getNode())
    NewLHS = DAG.getSetCC(dl, TLI.getSetCCResultType(LHSHi.getValueType()),
                          LHSHi, RHSHi, ISD::SETEQ);
  NewLHS = DAG.getNode(ISD::SELECT, dl, Tmp1.getValueType(),
                       NewLHS, Tmp1, Tmp2);
  NewRHS = SDValue();
}

SDValue DAGTypeLegalizer::ExpandIntOp_BR_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(2), NewRHS = N->getOperand(3);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(1))->get();
  IntegerExpandSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0),
                                DAG.getCondCode(CCCode), NewLHS, NewRHS,
                                N->getOperand(4));
}

SDValue DAGTypeLegalizer::ExpandIntOp_SELECT_CC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(4))->get();
  IntegerExpandSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If ExpandSetCCOperands returned a scalar, we need to compare the result
  // against zero to select between true and false values.
  if (NewRHS.getNode() == 0) {
    NewRHS = DAG.getConstant(0, NewLHS.getValueType());
    CCCode = ISD::SETNE;
  }

  // Update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), NewLHS, NewRHS,
                                N->getOperand(2), N->getOperand(3),
                                DAG.getCondCode(CCCode));
}

SDValue DAGTypeLegalizer::ExpandIntOp_SETCC(SDNode *N) {
  SDValue NewLHS = N->getOperand(0), NewRHS = N->getOperand(1);
  ISD::CondCode CCCode = cast<CondCodeSDNode>(N->getOperand(2))->get();
  IntegerExpandSetCCOperands(NewLHS, NewRHS, CCCode, N->getDebugLoc());

  // If ExpandSetCCOperands returned a scalar, use it.
  if (NewRHS.getNode() == 0) {
    assert(NewLHS.getValueType() == N->getValueType(0) &&
           "Unexpected setcc expansion!");
    return NewLHS;
  }

  // Otherwise, update N to have the operands specified.
  return DAG.UpdateNodeOperands(SDValue(N, 0), NewLHS, NewRHS,
                                DAG.getCondCode(CCCode));
}

SDValue DAGTypeLegalizer::ExpandIntOp_Shift(SDNode *N) {
  // The value being shifted is legal, but the shift amount is too big.
  // It follows that either the result of the shift is undefined, or the
  // upper half of the shift amount is zero.  Just use the lower half.
  SDValue Lo, Hi;
  GetExpandedInteger(N->getOperand(1), Lo, Hi);
  return DAG.UpdateNodeOperands(SDValue(N, 0), N->getOperand(0), Lo);
}

SDValue DAGTypeLegalizer::ExpandIntOp_SINT_TO_FP(SDNode *N) {
  SDValue Op = N->getOperand(0);
  MVT DstVT = N->getValueType(0);
  RTLIB::Libcall LC = RTLIB::getSINTTOFP(Op.getValueType(), DstVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL &&
         "Don't know how to expand this SINT_TO_FP!");
  return MakeLibCall(LC, DstVT, &Op, 1, true, N->getDebugLoc());
}

SDValue DAGTypeLegalizer::ExpandIntOp_STORE(StoreSDNode *N, unsigned OpNo) {
  if (ISD::isNormalStore(N))
    return ExpandOp_NormalStore(N, OpNo);

  assert(ISD::isUNINDEXEDStore(N) && "Indexed store during type legalization!");
  assert(OpNo == 1 && "Can only expand the stored value so far");

  MVT VT = N->getOperand(1).getValueType();
  MVT NVT = TLI.getTypeToTransformTo(VT);
  SDValue Ch  = N->getChain();
  SDValue Ptr = N->getBasePtr();
  int SVOffset = N->getSrcValueOffset();
  unsigned Alignment = N->getAlignment();
  bool isVolatile = N->isVolatile();
  DebugLoc dl = N->getDebugLoc();
  SDValue Lo, Hi;

  assert(NVT.isByteSized() && "Expanded type not byte sized!");

  if (N->getMemoryVT().bitsLE(NVT)) {
    GetExpandedInteger(N->getValue(), Lo, Hi);
    return DAG.getTruncStore(Ch, dl, Lo, Ptr, N->getSrcValue(), SVOffset,
                             N->getMemoryVT(), isVolatile, Alignment);
  } else if (TLI.isLittleEndian()) {
    // Little-endian - low bits are at low addresses.
    GetExpandedInteger(N->getValue(), Lo, Hi);

    Lo = DAG.getStore(Ch, dl, Lo, Ptr, N->getSrcValue(), SVOffset,
                      isVolatile, Alignment);

    unsigned ExcessBits =
      N->getMemoryVT().getSizeInBits() - NVT.getSizeInBits();
    MVT NEVT = MVT::getIntegerVT(ExcessBits);

    // Increment the pointer to the other half.
    unsigned IncrementSize = NVT.getSizeInBits()/8;
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getIntPtrConstant(IncrementSize));
    Hi = DAG.getTruncStore(Ch, dl, Hi, Ptr, N->getSrcValue(),
                           SVOffset+IncrementSize, NEVT,
                           isVolatile, MinAlign(Alignment, IncrementSize));
    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
  } else {
    // Big-endian - high bits are at low addresses.  Favor aligned stores at
    // the cost of some bit-fiddling.
    GetExpandedInteger(N->getValue(), Lo, Hi);

    MVT EVT = N->getMemoryVT();
    unsigned EBytes = EVT.getStoreSizeInBits()/8;
    unsigned IncrementSize = NVT.getSizeInBits()/8;
    unsigned ExcessBits = (EBytes - IncrementSize)*8;
    MVT HiVT = MVT::getIntegerVT(EVT.getSizeInBits() - ExcessBits);

    if (ExcessBits < NVT.getSizeInBits()) {
      // Transfer high bits from the top of Lo to the bottom of Hi.
      Hi = DAG.getNode(ISD::SHL, dl, NVT, Hi,
                       DAG.getConstant(NVT.getSizeInBits() - ExcessBits,
                                       TLI.getPointerTy()));
      Hi = DAG.getNode(ISD::OR, dl, NVT, Hi,
                       DAG.getNode(ISD::SRL, NVT, Lo,
                                   DAG.getConstant(ExcessBits,
                                                   TLI.getPointerTy())));
    }

    // Store both the high bits and maybe some of the low bits.
    Hi = DAG.getTruncStore(Ch, dl, Hi, Ptr, N->getSrcValue(),
                           SVOffset, HiVT, isVolatile, Alignment);

    // Increment the pointer to the other half.
    Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                      DAG.getIntPtrConstant(IncrementSize));
    // Store the lowest ExcessBits bits in the second half.
    Lo = DAG.getTruncStore(Ch, dl, Lo, Ptr, N->getSrcValue(),
                           SVOffset+IncrementSize,
                           MVT::getIntegerVT(ExcessBits),
                           isVolatile, MinAlign(Alignment, IncrementSize));
    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
  }
}

SDValue DAGTypeLegalizer::ExpandIntOp_TRUNCATE(SDNode *N) {
  SDValue InL, InH;
  GetExpandedInteger(N->getOperand(0), InL, InH);
  // Just truncate the low part of the source.
  return DAG.getNode(ISD::TRUNCATE, N->getDebugLoc(), N->getValueType(0), InL);
}

SDValue DAGTypeLegalizer::ExpandIntOp_UINT_TO_FP(SDNode *N) {
  SDValue Op = N->getOperand(0);
  MVT SrcVT = Op.getValueType();
  MVT DstVT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  if (TLI.getOperationAction(ISD::SINT_TO_FP, SrcVT) == TargetLowering::Custom){
    // Do a signed conversion then adjust the result.
    SDValue SignedConv = DAG.getNode(ISD::SINT_TO_FP, dl, DstVT, Op);
    SignedConv = TLI.LowerOperation(SignedConv, DAG);

    // The result of the signed conversion needs adjusting if the 'sign bit' of
    // the incoming integer was set.  To handle this, we dynamically test to see
    // if it is set, and, if so, add a fudge factor.

    const uint64_t F32TwoE32  = 0x4F800000ULL;
    const uint64_t F32TwoE64  = 0x5F800000ULL;
    const uint64_t F32TwoE128 = 0x7F800000ULL;

    APInt FF(32, 0);
    if (SrcVT == MVT::i32)
      FF = APInt(32, F32TwoE32);
    else if (SrcVT == MVT::i64)
      FF = APInt(32, F32TwoE64);
    else if (SrcVT == MVT::i128)
      FF = APInt(32, F32TwoE128);
    else
      assert(false && "Unsupported UINT_TO_FP!");

    // Check whether the sign bit is set.
    SDValue Lo, Hi;
    GetExpandedInteger(Op, Lo, Hi);
    SDValue SignSet = DAG.getSetCC(dl,
                                   TLI.getSetCCResultType(Hi.getValueType()),
                                   Hi, DAG.getConstant(0, Hi.getValueType()),
                                   ISD::SETLT);

    // Build a 64 bit pair (0, FF) in the constant pool, with FF in the lo bits.
    SDValue FudgePtr = DAG.getConstantPool(ConstantInt::get(FF.zext(64)),
                                           TLI.getPointerTy());

    // Get a pointer to FF if the sign bit was set, or to 0 otherwise.
    SDValue Zero = DAG.getIntPtrConstant(0);
    SDValue Four = DAG.getIntPtrConstant(4);
    if (TLI.isBigEndian()) std::swap(Zero, Four);
    SDValue Offset = DAG.getNode(ISD::SELECT, dl, Zero.getValueType(), SignSet,
                                 Zero, Four);
    unsigned Alignment =
      1 << cast<ConstantPoolSDNode>(FudgePtr)->getAlignment();
    FudgePtr = DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(), FudgePtr, Offset);
    Alignment = std::min(Alignment, 4u);

    // Load the value out, extending it from f32 to the destination float type.
    // FIXME: Avoid the extend by constructing the right constant pool?
    SDValue Fudge = DAG.getExtLoad(ISD::EXTLOAD, dl, DstVT, DAG.getEntryNode(),
                                   FudgePtr, NULL, 0, MVT::f32,
                                   false, Alignment);
    return DAG.getNode(ISD::FADD, dl, DstVT, SignedConv, Fudge);
  }

  // Otherwise, use a libcall.
  RTLIB::Libcall LC = RTLIB::getUINTTOFP(SrcVT, DstVT);
  assert(LC != RTLIB::UNKNOWN_LIBCALL &&
         "Don't know how to expand this UINT_TO_FP!");
  return MakeLibCall(LC, DstVT, &Op, 1, true, dl);
}
