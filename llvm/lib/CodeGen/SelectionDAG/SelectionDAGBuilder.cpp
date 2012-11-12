//===-- SelectionDAGBuilder.cpp - Selection-DAG building ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating from LLVM IR into SelectionDAG IR.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "SDNodeDbgValue.h"
#include "SelectionDAGBuilder.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Constants.h"
#include "llvm/CallingConv.h"
#include "llvm/DebugInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IntegersSubsetMapping.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

/// LimitFloatPrecision - Generate low-precision inline sequences for
/// some float libcalls (6, 8 or 12 bits).
static unsigned LimitFloatPrecision;

static cl::opt<unsigned, true>
LimitFPPrecision("limit-float-precision",
                 cl::desc("Generate low-precision inline sequences "
                          "for some float libcalls"),
                 cl::location(LimitFloatPrecision),
                 cl::init(0));

// Limit the width of DAG chains. This is important in general to prevent
// prevent DAG-based analysis from blowing up. For example, alias analysis and
// load clustering may not complete in reasonable time. It is difficult to
// recognize and avoid this situation within each individual analysis, and
// future analyses are likely to have the same behavior. Limiting DAG width is
// the safe approach, and will be especially important with global DAGs.
//
// MaxParallelChains default is arbitrarily high to avoid affecting
// optimization, but could be lowered to improve compile time. Any ld-ld-st-st
// sequence over this should have been converted to llvm.memcpy by the
// frontend. It easy to induce this behavior with .ll code such as:
// %buffer = alloca [4096 x i8]
// %data = load [4096 x i8]* %argPtr
// store [4096 x i8] %data, [4096 x i8]* %buffer
static const unsigned MaxParallelChains = 64;

static SDValue getCopyFromPartsVector(SelectionDAG &DAG, DebugLoc DL,
                                      const SDValue *Parts, unsigned NumParts,
                                      EVT PartVT, EVT ValueVT, const Value *V);

/// getCopyFromParts - Create a value that contains the specified legal parts
/// combined into the value they represent.  If the parts combine to a type
/// larger then ValueVT then AssertOp can be used to specify whether the extra
/// bits are known to be zero (ISD::AssertZext) or sign extended from ValueVT
/// (ISD::AssertSext).
static SDValue getCopyFromParts(SelectionDAG &DAG, DebugLoc DL,
                                const SDValue *Parts,
                                unsigned NumParts, EVT PartVT, EVT ValueVT,
                                const Value *V,
                                ISD::NodeType AssertOp = ISD::DELETED_NODE) {
  if (ValueVT.isVector())
    return getCopyFromPartsVector(DAG, DL, Parts, NumParts,
                                  PartVT, ValueVT, V);

  assert(NumParts > 0 && "No parts to assemble!");
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Val = Parts[0];

  if (NumParts > 1) {
    // Assemble the value from multiple parts.
    if (ValueVT.isInteger()) {
      unsigned PartBits = PartVT.getSizeInBits();
      unsigned ValueBits = ValueVT.getSizeInBits();

      // Assemble the power of 2 part.
      unsigned RoundParts = NumParts & (NumParts - 1) ?
        1 << Log2_32(NumParts) : NumParts;
      unsigned RoundBits = PartBits * RoundParts;
      EVT RoundVT = RoundBits == ValueBits ?
        ValueVT : EVT::getIntegerVT(*DAG.getContext(), RoundBits);
      SDValue Lo, Hi;

      EVT HalfVT = EVT::getIntegerVT(*DAG.getContext(), RoundBits/2);

      if (RoundParts > 2) {
        Lo = getCopyFromParts(DAG, DL, Parts, RoundParts / 2,
                              PartVT, HalfVT, V);
        Hi = getCopyFromParts(DAG, DL, Parts + RoundParts / 2,
                              RoundParts / 2, PartVT, HalfVT, V);
      } else {
        Lo = DAG.getNode(ISD::BITCAST, DL, HalfVT, Parts[0]);
        Hi = DAG.getNode(ISD::BITCAST, DL, HalfVT, Parts[1]);
      }

      if (TLI.isBigEndian())
        std::swap(Lo, Hi);

      Val = DAG.getNode(ISD::BUILD_PAIR, DL, RoundVT, Lo, Hi);

      if (RoundParts < NumParts) {
        // Assemble the trailing non-power-of-2 part.
        unsigned OddParts = NumParts - RoundParts;
        EVT OddVT = EVT::getIntegerVT(*DAG.getContext(), OddParts * PartBits);
        Hi = getCopyFromParts(DAG, DL,
                              Parts + RoundParts, OddParts, PartVT, OddVT, V);

        // Combine the round and odd parts.
        Lo = Val;
        if (TLI.isBigEndian())
          std::swap(Lo, Hi);
        EVT TotalVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
        Hi = DAG.getNode(ISD::ANY_EXTEND, DL, TotalVT, Hi);
        Hi = DAG.getNode(ISD::SHL, DL, TotalVT, Hi,
                         DAG.getConstant(Lo.getValueType().getSizeInBits(),
                                         TLI.getPointerTy()));
        Lo = DAG.getNode(ISD::ZERO_EXTEND, DL, TotalVT, Lo);
        Val = DAG.getNode(ISD::OR, DL, TotalVT, Lo, Hi);
      }
    } else if (PartVT.isFloatingPoint()) {
      // FP split into multiple FP parts (for ppcf128)
      assert(ValueVT == EVT(MVT::ppcf128) && PartVT == EVT(MVT::f64) &&
             "Unexpected split");
      SDValue Lo, Hi;
      Lo = DAG.getNode(ISD::BITCAST, DL, EVT(MVT::f64), Parts[0]);
      Hi = DAG.getNode(ISD::BITCAST, DL, EVT(MVT::f64), Parts[1]);
      if (TLI.isBigEndian())
        std::swap(Lo, Hi);
      Val = DAG.getNode(ISD::BUILD_PAIR, DL, ValueVT, Lo, Hi);
    } else {
      // FP split into integer parts (soft fp)
      assert(ValueVT.isFloatingPoint() && PartVT.isInteger() &&
             !PartVT.isVector() && "Unexpected split");
      EVT IntVT = EVT::getIntegerVT(*DAG.getContext(), ValueVT.getSizeInBits());
      Val = getCopyFromParts(DAG, DL, Parts, NumParts, PartVT, IntVT, V);
    }
  }

  // There is now one part, held in Val.  Correct it to match ValueVT.
  PartVT = Val.getValueType();

  if (PartVT == ValueVT)
    return Val;

  if (PartVT.isInteger() && ValueVT.isInteger()) {
    if (ValueVT.bitsLT(PartVT)) {
      // For a truncate, see if we have any information to
      // indicate whether the truncated bits will always be
      // zero or sign-extension.
      if (AssertOp != ISD::DELETED_NODE)
        Val = DAG.getNode(AssertOp, DL, PartVT, Val,
                          DAG.getValueType(ValueVT));
      return DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Val);
    }
    return DAG.getNode(ISD::ANY_EXTEND, DL, ValueVT, Val);
  }

  if (PartVT.isFloatingPoint() && ValueVT.isFloatingPoint()) {
    // FP_ROUND's are always exact here.
    if (ValueVT.bitsLT(Val.getValueType()))
      return DAG.getNode(ISD::FP_ROUND, DL, ValueVT, Val,
                         DAG.getTargetConstant(1, TLI.getPointerTy()));

    return DAG.getNode(ISD::FP_EXTEND, DL, ValueVT, Val);
  }

  if (PartVT.getSizeInBits() == ValueVT.getSizeInBits())
    return DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);

  llvm_unreachable("Unknown mismatch!");
}

/// getCopyFromPartsVector - Create a value that contains the specified legal
/// parts combined into the value they represent.  If the parts combine to a
/// type larger then ValueVT then AssertOp can be used to specify whether the
/// extra bits are known to be zero (ISD::AssertZext) or sign extended from
/// ValueVT (ISD::AssertSext).
static SDValue getCopyFromPartsVector(SelectionDAG &DAG, DebugLoc DL,
                                      const SDValue *Parts, unsigned NumParts,
                                      EVT PartVT, EVT ValueVT, const Value *V) {
  assert(ValueVT.isVector() && "Not a vector value");
  assert(NumParts > 0 && "No parts to assemble!");
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  SDValue Val = Parts[0];

  // Handle a multi-element vector.
  if (NumParts > 1) {
    EVT IntermediateVT, RegisterVT;
    unsigned NumIntermediates;
    unsigned NumRegs =
    TLI.getVectorTypeBreakdown(*DAG.getContext(), ValueVT, IntermediateVT,
                               NumIntermediates, RegisterVT);
    assert(NumRegs == NumParts && "Part count doesn't match vector breakdown!");
    NumParts = NumRegs; // Silence a compiler warning.
    assert(RegisterVT == PartVT && "Part type doesn't match vector breakdown!");
    assert(RegisterVT == Parts[0].getValueType() &&
           "Part type doesn't match part!");

    // Assemble the parts into intermediate operands.
    SmallVector<SDValue, 8> Ops(NumIntermediates);
    if (NumIntermediates == NumParts) {
      // If the register was not expanded, truncate or copy the value,
      // as appropriate.
      for (unsigned i = 0; i != NumParts; ++i)
        Ops[i] = getCopyFromParts(DAG, DL, &Parts[i], 1,
                                  PartVT, IntermediateVT, V);
    } else if (NumParts > 0) {
      // If the intermediate type was expanded, build the intermediate
      // operands from the parts.
      assert(NumParts % NumIntermediates == 0 &&
             "Must expand into a divisible number of parts!");
      unsigned Factor = NumParts / NumIntermediates;
      for (unsigned i = 0; i != NumIntermediates; ++i)
        Ops[i] = getCopyFromParts(DAG, DL, &Parts[i * Factor], Factor,
                                  PartVT, IntermediateVT, V);
    }

    // Build a vector with BUILD_VECTOR or CONCAT_VECTORS from the
    // intermediate operands.
    Val = DAG.getNode(IntermediateVT.isVector() ?
                      ISD::CONCAT_VECTORS : ISD::BUILD_VECTOR, DL,
                      ValueVT, &Ops[0], NumIntermediates);
  }

  // There is now one part, held in Val.  Correct it to match ValueVT.
  PartVT = Val.getValueType();

  if (PartVT == ValueVT)
    return Val;

  if (PartVT.isVector()) {
    // If the element type of the source/dest vectors are the same, but the
    // parts vector has more elements than the value vector, then we have a
    // vector widening case (e.g. <2 x float> -> <4 x float>).  Extract the
    // elements we want.
    if (PartVT.getVectorElementType() == ValueVT.getVectorElementType()) {
      assert(PartVT.getVectorNumElements() > ValueVT.getVectorNumElements() &&
             "Cannot narrow, it would be a lossy transformation");
      return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, ValueVT, Val,
                         DAG.getIntPtrConstant(0));
    }

    // Vector/Vector bitcast.
    if (ValueVT.getSizeInBits() == PartVT.getSizeInBits())
      return DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);

    assert(PartVT.getVectorNumElements() == ValueVT.getVectorNumElements() &&
      "Cannot handle this kind of promotion");
    // Promoted vector extract
    bool Smaller = ValueVT.bitsLE(PartVT);
    return DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                       DL, ValueVT, Val);

  }

  // Trivial bitcast if the types are the same size and the destination
  // vector type is legal.
  if (PartVT.getSizeInBits() == ValueVT.getSizeInBits() &&
      TLI.isTypeLegal(ValueVT))
    return DAG.getNode(ISD::BITCAST, DL, ValueVT, Val);

  // Handle cases such as i8 -> <1 x i1>
  if (ValueVT.getVectorNumElements() != 1) {
    LLVMContext &Ctx = *DAG.getContext();
    Twine ErrMsg("non-trivial scalar-to-vector conversion");
    if (const Instruction *I = dyn_cast_or_null<Instruction>(V)) {
      if (const CallInst *CI = dyn_cast<CallInst>(I))
        if (isa<InlineAsm>(CI->getCalledValue()))
          ErrMsg = ErrMsg + ", possible invalid constraint for vector type";
      Ctx.emitError(I, ErrMsg);
    } else {
      Ctx.emitError(ErrMsg);
    }
    report_fatal_error("Cannot handle scalar-to-vector conversion!");
  }

  if (ValueVT.getVectorNumElements() == 1 &&
      ValueVT.getVectorElementType() != PartVT) {
    bool Smaller = ValueVT.bitsLE(PartVT);
    Val = DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                       DL, ValueVT.getScalarType(), Val);
  }

  return DAG.getNode(ISD::BUILD_VECTOR, DL, ValueVT, Val);
}

static void getCopyToPartsVector(SelectionDAG &DAG, DebugLoc dl,
                                 SDValue Val, SDValue *Parts, unsigned NumParts,
                                 EVT PartVT, const Value *V);

/// getCopyToParts - Create a series of nodes that contain the specified value
/// split into legal parts.  If the parts contain more bits than Val, then, for
/// integers, ExtendKind can be used to specify how to generate the extra bits.
static void getCopyToParts(SelectionDAG &DAG, DebugLoc DL,
                           SDValue Val, SDValue *Parts, unsigned NumParts,
                           EVT PartVT, const Value *V,
                           ISD::NodeType ExtendKind = ISD::ANY_EXTEND) {
  EVT ValueVT = Val.getValueType();

  // Handle the vector case separately.
  if (ValueVT.isVector())
    return getCopyToPartsVector(DAG, DL, Val, Parts, NumParts, PartVT, V);

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  unsigned PartBits = PartVT.getSizeInBits();
  unsigned OrigNumParts = NumParts;
  assert(TLI.isTypeLegal(PartVT) && "Copying to an illegal type!");

  if (NumParts == 0)
    return;

  assert(!ValueVT.isVector() && "Vector case handled elsewhere");
  if (PartVT == ValueVT) {
    assert(NumParts == 1 && "No-op copy with multiple parts!");
    Parts[0] = Val;
    return;
  }

  if (NumParts * PartBits > ValueVT.getSizeInBits()) {
    // If the parts cover more bits than the value has, promote the value.
    if (PartVT.isFloatingPoint() && ValueVT.isFloatingPoint()) {
      assert(NumParts == 1 && "Do not know what to promote to!");
      Val = DAG.getNode(ISD::FP_EXTEND, DL, PartVT, Val);
    } else {
      assert((PartVT.isInteger() || PartVT == MVT::x86mmx) &&
             ValueVT.isInteger() &&
             "Unknown mismatch!");
      ValueVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
      Val = DAG.getNode(ExtendKind, DL, ValueVT, Val);
      if (PartVT == MVT::x86mmx)
        Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
    }
  } else if (PartBits == ValueVT.getSizeInBits()) {
    // Different types of the same size.
    assert(NumParts == 1 && PartVT != ValueVT);
    Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
  } else if (NumParts * PartBits < ValueVT.getSizeInBits()) {
    // If the parts cover less bits than value has, truncate the value.
    assert((PartVT.isInteger() || PartVT == MVT::x86mmx) &&
           ValueVT.isInteger() &&
           "Unknown mismatch!");
    ValueVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Val);
    if (PartVT == MVT::x86mmx)
      Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
  }

  // The value may have changed - recompute ValueVT.
  ValueVT = Val.getValueType();
  assert(NumParts * PartBits == ValueVT.getSizeInBits() &&
         "Failed to tile the value with PartVT!");

  if (NumParts == 1) {
    if (PartVT != ValueVT) {
      LLVMContext &Ctx = *DAG.getContext();
      Twine ErrMsg("scalar-to-vector conversion failed");
      if (const Instruction *I = dyn_cast_or_null<Instruction>(V)) {
        if (const CallInst *CI = dyn_cast<CallInst>(I))
          if (isa<InlineAsm>(CI->getCalledValue()))
            ErrMsg = ErrMsg + ", possible invalid constraint for vector type";
        Ctx.emitError(I, ErrMsg);
      } else {
        Ctx.emitError(ErrMsg);
      }
    }

    Parts[0] = Val;
    return;
  }

  // Expand the value into multiple parts.
  if (NumParts & (NumParts - 1)) {
    // The number of parts is not a power of 2.  Split off and copy the tail.
    assert(PartVT.isInteger() && ValueVT.isInteger() &&
           "Do not know what to expand to!");
    unsigned RoundParts = 1 << Log2_32(NumParts);
    unsigned RoundBits = RoundParts * PartBits;
    unsigned OddParts = NumParts - RoundParts;
    SDValue OddVal = DAG.getNode(ISD::SRL, DL, ValueVT, Val,
                                 DAG.getIntPtrConstant(RoundBits));
    getCopyToParts(DAG, DL, OddVal, Parts + RoundParts, OddParts, PartVT, V);

    if (TLI.isBigEndian())
      // The odd parts were reversed by getCopyToParts - unreverse them.
      std::reverse(Parts + RoundParts, Parts + NumParts);

    NumParts = RoundParts;
    ValueVT = EVT::getIntegerVT(*DAG.getContext(), NumParts * PartBits);
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValueVT, Val);
  }

  // The number of parts is a power of 2.  Repeatedly bisect the value using
  // EXTRACT_ELEMENT.
  Parts[0] = DAG.getNode(ISD::BITCAST, DL,
                         EVT::getIntegerVT(*DAG.getContext(),
                                           ValueVT.getSizeInBits()),
                         Val);

  for (unsigned StepSize = NumParts; StepSize > 1; StepSize /= 2) {
    for (unsigned i = 0; i < NumParts; i += StepSize) {
      unsigned ThisBits = StepSize * PartBits / 2;
      EVT ThisVT = EVT::getIntegerVT(*DAG.getContext(), ThisBits);
      SDValue &Part0 = Parts[i];
      SDValue &Part1 = Parts[i+StepSize/2];

      Part1 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL,
                          ThisVT, Part0, DAG.getIntPtrConstant(1));
      Part0 = DAG.getNode(ISD::EXTRACT_ELEMENT, DL,
                          ThisVT, Part0, DAG.getIntPtrConstant(0));

      if (ThisBits == PartBits && ThisVT != PartVT) {
        Part0 = DAG.getNode(ISD::BITCAST, DL, PartVT, Part0);
        Part1 = DAG.getNode(ISD::BITCAST, DL, PartVT, Part1);
      }
    }
  }

  if (TLI.isBigEndian())
    std::reverse(Parts, Parts + OrigNumParts);
}


/// getCopyToPartsVector - Create a series of nodes that contain the specified
/// value split into legal parts.
static void getCopyToPartsVector(SelectionDAG &DAG, DebugLoc DL,
                                 SDValue Val, SDValue *Parts, unsigned NumParts,
                                 EVT PartVT, const Value *V) {
  EVT ValueVT = Val.getValueType();
  assert(ValueVT.isVector() && "Not a vector");
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  if (NumParts == 1) {
    if (PartVT == ValueVT) {
      // Nothing to do.
    } else if (PartVT.getSizeInBits() == ValueVT.getSizeInBits()) {
      // Bitconvert vector->vector case.
      Val = DAG.getNode(ISD::BITCAST, DL, PartVT, Val);
    } else if (PartVT.isVector() &&
               PartVT.getVectorElementType() == ValueVT.getVectorElementType() &&
               PartVT.getVectorNumElements() > ValueVT.getVectorNumElements()) {
      EVT ElementVT = PartVT.getVectorElementType();
      // Vector widening case, e.g. <2 x float> -> <4 x float>.  Shuffle in
      // undef elements.
      SmallVector<SDValue, 16> Ops;
      for (unsigned i = 0, e = ValueVT.getVectorNumElements(); i != e; ++i)
        Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                                  ElementVT, Val, DAG.getIntPtrConstant(i)));

      for (unsigned i = ValueVT.getVectorNumElements(),
           e = PartVT.getVectorNumElements(); i != e; ++i)
        Ops.push_back(DAG.getUNDEF(ElementVT));

      Val = DAG.getNode(ISD::BUILD_VECTOR, DL, PartVT, &Ops[0], Ops.size());

      // FIXME: Use CONCAT for 2x -> 4x.

      //SDValue UndefElts = DAG.getUNDEF(VectorTy);
      //Val = DAG.getNode(ISD::CONCAT_VECTORS, DL, PartVT, Val, UndefElts);
    } else if (PartVT.isVector() &&
               PartVT.getVectorElementType().bitsGE(
                 ValueVT.getVectorElementType()) &&
               PartVT.getVectorNumElements() == ValueVT.getVectorNumElements()) {

      // Promoted vector extract
      bool Smaller = PartVT.bitsLE(ValueVT);
      Val = DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                        DL, PartVT, Val);
    } else{
      // Vector -> scalar conversion.
      assert(ValueVT.getVectorNumElements() == 1 &&
             "Only trivial vector-to-scalar conversions should get here!");
      Val = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                        PartVT, Val, DAG.getIntPtrConstant(0));

      bool Smaller = ValueVT.bitsLE(PartVT);
      Val = DAG.getNode((Smaller ? ISD::TRUNCATE : ISD::ANY_EXTEND),
                         DL, PartVT, Val);
    }

    Parts[0] = Val;
    return;
  }

  // Handle a multi-element vector.
  EVT IntermediateVT, RegisterVT;
  unsigned NumIntermediates;
  unsigned NumRegs = TLI.getVectorTypeBreakdown(*DAG.getContext(), ValueVT,
                                                IntermediateVT,
                                                NumIntermediates, RegisterVT);
  unsigned NumElements = ValueVT.getVectorNumElements();

  assert(NumRegs == NumParts && "Part count doesn't match vector breakdown!");
  NumParts = NumRegs; // Silence a compiler warning.
  assert(RegisterVT == PartVT && "Part type doesn't match vector breakdown!");

  // Split the vector into intermediate operands.
  SmallVector<SDValue, 8> Ops(NumIntermediates);
  for (unsigned i = 0; i != NumIntermediates; ++i) {
    if (IntermediateVT.isVector())
      Ops[i] = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL,
                           IntermediateVT, Val,
                   DAG.getIntPtrConstant(i * (NumElements / NumIntermediates)));
    else
      Ops[i] = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                           IntermediateVT, Val, DAG.getIntPtrConstant(i));
  }

  // Split the intermediate operands into legal parts.
  if (NumParts == NumIntermediates) {
    // If the register was not expanded, promote or copy the value,
    // as appropriate.
    for (unsigned i = 0; i != NumParts; ++i)
      getCopyToParts(DAG, DL, Ops[i], &Parts[i], 1, PartVT, V);
  } else if (NumParts > 0) {
    // If the intermediate type was expanded, split each the value into
    // legal parts.
    assert(NumParts % NumIntermediates == 0 &&
           "Must expand into a divisible number of parts!");
    unsigned Factor = NumParts / NumIntermediates;
    for (unsigned i = 0; i != NumIntermediates; ++i)
      getCopyToParts(DAG, DL, Ops[i], &Parts[i*Factor], Factor, PartVT, V);
  }
}

namespace {
  /// RegsForValue - This struct represents the registers (physical or virtual)
  /// that a particular set of values is assigned, and the type information
  /// about the value. The most common situation is to represent one value at a
  /// time, but struct or array values are handled element-wise as multiple
  /// values.  The splitting of aggregates is performed recursively, so that we
  /// never have aggregate-typed registers. The values at this point do not
  /// necessarily have legal types, so each value may require one or more
  /// registers of some legal type.
  ///
  struct RegsForValue {
    /// ValueVTs - The value types of the values, which may not be legal, and
    /// may need be promoted or synthesized from one or more registers.
    ///
    SmallVector<EVT, 4> ValueVTs;

    /// RegVTs - The value types of the registers. This is the same size as
    /// ValueVTs and it records, for each value, what the type of the assigned
    /// register or registers are. (Individual values are never synthesized
    /// from more than one type of register.)
    ///
    /// With virtual registers, the contents of RegVTs is redundant with TLI's
    /// getRegisterType member function, however when with physical registers
    /// it is necessary to have a separate record of the types.
    ///
    SmallVector<EVT, 4> RegVTs;

    /// Regs - This list holds the registers assigned to the values.
    /// Each legal or promoted value requires one register, and each
    /// expanded value requires multiple registers.
    ///
    SmallVector<unsigned, 4> Regs;

    RegsForValue() {}

    RegsForValue(const SmallVector<unsigned, 4> &regs,
                 EVT regvt, EVT valuevt)
      : ValueVTs(1, valuevt), RegVTs(1, regvt), Regs(regs) {}

    RegsForValue(LLVMContext &Context, const TargetLowering &tli,
                 unsigned Reg, Type *Ty) {
      ComputeValueVTs(tli, Ty, ValueVTs);

      for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
        EVT ValueVT = ValueVTs[Value];
        unsigned NumRegs = tli.getNumRegisters(Context, ValueVT);
        EVT RegisterVT = tli.getRegisterType(Context, ValueVT);
        for (unsigned i = 0; i != NumRegs; ++i)
          Regs.push_back(Reg + i);
        RegVTs.push_back(RegisterVT);
        Reg += NumRegs;
      }
    }

    /// areValueTypesLegal - Return true if types of all the values are legal.
    bool areValueTypesLegal(const TargetLowering &TLI) {
      for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
        EVT RegisterVT = RegVTs[Value];
        if (!TLI.isTypeLegal(RegisterVT))
          return false;
      }
      return true;
    }

    /// append - Add the specified values to this one.
    void append(const RegsForValue &RHS) {
      ValueVTs.append(RHS.ValueVTs.begin(), RHS.ValueVTs.end());
      RegVTs.append(RHS.RegVTs.begin(), RHS.RegVTs.end());
      Regs.append(RHS.Regs.begin(), RHS.Regs.end());
    }

    /// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
    /// this value and returns the result as a ValueVTs value.  This uses
    /// Chain/Flag as the input and updates them for the output Chain/Flag.
    /// If the Flag pointer is NULL, no flag is used.
    SDValue getCopyFromRegs(SelectionDAG &DAG, FunctionLoweringInfo &FuncInfo,
                            DebugLoc dl,
                            SDValue &Chain, SDValue *Flag,
                            const Value *V = 0) const;

    /// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
    /// specified value into the registers specified by this object.  This uses
    /// Chain/Flag as the input and updates them for the output Chain/Flag.
    /// If the Flag pointer is NULL, no flag is used.
    void getCopyToRegs(SDValue Val, SelectionDAG &DAG, DebugLoc dl,
                       SDValue &Chain, SDValue *Flag, const Value *V) const;

    /// AddInlineAsmOperands - Add this value to the specified inlineasm node
    /// operand list.  This adds the code marker, matching input operand index
    /// (if applicable), and includes the number of values added into it.
    void AddInlineAsmOperands(unsigned Kind,
                              bool HasMatching, unsigned MatchingIdx,
                              SelectionDAG &DAG,
                              std::vector<SDValue> &Ops) const;
  };
}

/// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
/// this value and returns the result as a ValueVT value.  This uses
/// Chain/Flag as the input and updates them for the output Chain/Flag.
/// If the Flag pointer is NULL, no flag is used.
SDValue RegsForValue::getCopyFromRegs(SelectionDAG &DAG,
                                      FunctionLoweringInfo &FuncInfo,
                                      DebugLoc dl,
                                      SDValue &Chain, SDValue *Flag,
                                      const Value *V) const {
  // A Value with type {} or [0 x %t] needs no registers.
  if (ValueVTs.empty())
    return SDValue();

  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Assemble the legal parts into the final values.
  SmallVector<SDValue, 4> Values(ValueVTs.size());
  SmallVector<SDValue, 8> Parts;
  for (unsigned Value = 0, Part = 0, e = ValueVTs.size(); Value != e; ++Value) {
    // Copy the legal parts from the registers.
    EVT ValueVT = ValueVTs[Value];
    unsigned NumRegs = TLI.getNumRegisters(*DAG.getContext(), ValueVT);
    EVT RegisterVT = RegVTs[Value];

    Parts.resize(NumRegs);
    for (unsigned i = 0; i != NumRegs; ++i) {
      SDValue P;
      if (Flag == 0) {
        P = DAG.getCopyFromReg(Chain, dl, Regs[Part+i], RegisterVT);
      } else {
        P = DAG.getCopyFromReg(Chain, dl, Regs[Part+i], RegisterVT, *Flag);
        *Flag = P.getValue(2);
      }

      Chain = P.getValue(1);
      Parts[i] = P;

      // If the source register was virtual and if we know something about it,
      // add an assert node.
      if (!TargetRegisterInfo::isVirtualRegister(Regs[Part+i]) ||
          !RegisterVT.isInteger() || RegisterVT.isVector())
        continue;

      const FunctionLoweringInfo::LiveOutInfo *LOI =
        FuncInfo.GetLiveOutRegInfo(Regs[Part+i]);
      if (!LOI)
        continue;

      unsigned RegSize = RegisterVT.getSizeInBits();
      unsigned NumSignBits = LOI->NumSignBits;
      unsigned NumZeroBits = LOI->KnownZero.countLeadingOnes();

      // FIXME: We capture more information than the dag can represent.  For
      // now, just use the tightest assertzext/assertsext possible.
      bool isSExt = true;
      EVT FromVT(MVT::Other);
      if (NumSignBits == RegSize)
        isSExt = true, FromVT = MVT::i1;   // ASSERT SEXT 1
      else if (NumZeroBits >= RegSize-1)
        isSExt = false, FromVT = MVT::i1;  // ASSERT ZEXT 1
      else if (NumSignBits > RegSize-8)
        isSExt = true, FromVT = MVT::i8;   // ASSERT SEXT 8
      else if (NumZeroBits >= RegSize-8)
        isSExt = false, FromVT = MVT::i8;  // ASSERT ZEXT 8
      else if (NumSignBits > RegSize-16)
        isSExt = true, FromVT = MVT::i16;  // ASSERT SEXT 16
      else if (NumZeroBits >= RegSize-16)
        isSExt = false, FromVT = MVT::i16; // ASSERT ZEXT 16
      else if (NumSignBits > RegSize-32)
        isSExt = true, FromVT = MVT::i32;  // ASSERT SEXT 32
      else if (NumZeroBits >= RegSize-32)
        isSExt = false, FromVT = MVT::i32; // ASSERT ZEXT 32
      else
        continue;

      // Add an assertion node.
      assert(FromVT != MVT::Other);
      Parts[i] = DAG.getNode(isSExt ? ISD::AssertSext : ISD::AssertZext, dl,
                             RegisterVT, P, DAG.getValueType(FromVT));
    }

    Values[Value] = getCopyFromParts(DAG, dl, Parts.begin(),
                                     NumRegs, RegisterVT, ValueVT, V);
    Part += NumRegs;
    Parts.clear();
  }

  return DAG.getNode(ISD::MERGE_VALUES, dl,
                     DAG.getVTList(&ValueVTs[0], ValueVTs.size()),
                     &Values[0], ValueVTs.size());
}

/// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
/// specified value into the registers specified by this object.  This uses
/// Chain/Flag as the input and updates them for the output Chain/Flag.
/// If the Flag pointer is NULL, no flag is used.
void RegsForValue::getCopyToRegs(SDValue Val, SelectionDAG &DAG, DebugLoc dl,
                                 SDValue &Chain, SDValue *Flag,
                                 const Value *V) const {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  // Get the list of the values's legal parts.
  unsigned NumRegs = Regs.size();
  SmallVector<SDValue, 8> Parts(NumRegs);
  for (unsigned Value = 0, Part = 0, e = ValueVTs.size(); Value != e; ++Value) {
    EVT ValueVT = ValueVTs[Value];
    unsigned NumParts = TLI.getNumRegisters(*DAG.getContext(), ValueVT);
    EVT RegisterVT = RegVTs[Value];

    getCopyToParts(DAG, dl, Val.getValue(Val.getResNo() + Value),
                   &Parts[Part], NumParts, RegisterVT, V);
    Part += NumParts;
  }

  // Copy the parts into the registers.
  SmallVector<SDValue, 8> Chains(NumRegs);
  for (unsigned i = 0; i != NumRegs; ++i) {
    SDValue Part;
    if (Flag == 0) {
      Part = DAG.getCopyToReg(Chain, dl, Regs[i], Parts[i]);
    } else {
      Part = DAG.getCopyToReg(Chain, dl, Regs[i], Parts[i], *Flag);
      *Flag = Part.getValue(1);
    }

    Chains[i] = Part.getValue(0);
  }

  if (NumRegs == 1 || Flag)
    // If NumRegs > 1 && Flag is used then the use of the last CopyToReg is
    // flagged to it. That is the CopyToReg nodes and the user are considered
    // a single scheduling unit. If we create a TokenFactor and return it as
    // chain, then the TokenFactor is both a predecessor (operand) of the
    // user as well as a successor (the TF operands are flagged to the user).
    // c1, f1 = CopyToReg
    // c2, f2 = CopyToReg
    // c3     = TokenFactor c1, c2
    // ...
    //        = op c3, ..., f2
    Chain = Chains[NumRegs-1];
  else
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &Chains[0], NumRegs);
}

/// AddInlineAsmOperands - Add this value to the specified inlineasm node
/// operand list.  This adds the code marker and includes the number of
/// values added into it.
void RegsForValue::AddInlineAsmOperands(unsigned Code, bool HasMatching,
                                        unsigned MatchingIdx,
                                        SelectionDAG &DAG,
                                        std::vector<SDValue> &Ops) const {
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();

  unsigned Flag = InlineAsm::getFlagWord(Code, Regs.size());
  if (HasMatching)
    Flag = InlineAsm::getFlagWordForMatchingOp(Flag, MatchingIdx);
  else if (!Regs.empty() &&
           TargetRegisterInfo::isVirtualRegister(Regs.front())) {
    // Put the register class of the virtual registers in the flag word.  That
    // way, later passes can recompute register class constraints for inline
    // assembly as well as normal instructions.
    // Don't do this for tied operands that can use the regclass information
    // from the def.
    const MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
    const TargetRegisterClass *RC = MRI.getRegClass(Regs.front());
    Flag = InlineAsm::getFlagWordForRegClass(Flag, RC->getID());
  }

  SDValue Res = DAG.getTargetConstant(Flag, MVT::i32);
  Ops.push_back(Res);

  for (unsigned Value = 0, Reg = 0, e = ValueVTs.size(); Value != e; ++Value) {
    unsigned NumRegs = TLI.getNumRegisters(*DAG.getContext(), ValueVTs[Value]);
    EVT RegisterVT = RegVTs[Value];
    for (unsigned i = 0; i != NumRegs; ++i) {
      assert(Reg < Regs.size() && "Mismatch in # registers expected");
      Ops.push_back(DAG.getRegister(Regs[Reg++], RegisterVT));
    }
  }
}

void SelectionDAGBuilder::init(GCFunctionInfo *gfi, AliasAnalysis &aa,
                               const TargetLibraryInfo *li) {
  AA = &aa;
  GFI = gfi;
  LibInfo = li;
  TD = DAG.getTarget().getDataLayout();
  Context = DAG.getContext();
  LPadToCallSiteMap.clear();
}

/// clear - Clear out the current SelectionDAG and the associated
/// state and prepare this SelectionDAGBuilder object to be used
/// for a new block. This doesn't clear out information about
/// additional blocks that are needed to complete switch lowering
/// or PHI node updating; that information is cleared out as it is
/// consumed.
void SelectionDAGBuilder::clear() {
  NodeMap.clear();
  UnusedArgNodeMap.clear();
  PendingLoads.clear();
  PendingExports.clear();
  CurDebugLoc = DebugLoc();
  HasTailCall = false;
}

/// clearDanglingDebugInfo - Clear the dangling debug information
/// map. This function is separated from the clear so that debug
/// information that is dangling in a basic block can be properly
/// resolved in a different basic block. This allows the
/// SelectionDAG to resolve dangling debug information attached
/// to PHI nodes.
void SelectionDAGBuilder::clearDanglingDebugInfo() {
  DanglingDebugInfoMap.clear();
}

/// getRoot - Return the current virtual root of the Selection DAG,
/// flushing any PendingLoad items. This must be done before emitting
/// a store or any other node that may need to be ordered after any
/// prior load instructions.
///
SDValue SelectionDAGBuilder::getRoot() {
  if (PendingLoads.empty())
    return DAG.getRoot();

  if (PendingLoads.size() == 1) {
    SDValue Root = PendingLoads[0];
    DAG.setRoot(Root);
    PendingLoads.clear();
    return Root;
  }

  // Otherwise, we have to make a token factor node.
  SDValue Root = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(), MVT::Other,
                               &PendingLoads[0], PendingLoads.size());
  PendingLoads.clear();
  DAG.setRoot(Root);
  return Root;
}

/// getControlRoot - Similar to getRoot, but instead of flushing all the
/// PendingLoad items, flush all the PendingExports items. It is necessary
/// to do this before emitting a terminator instruction.
///
SDValue SelectionDAGBuilder::getControlRoot() {
  SDValue Root = DAG.getRoot();

  if (PendingExports.empty())
    return Root;

  // Turn all of the CopyToReg chains into one factored node.
  if (Root.getOpcode() != ISD::EntryToken) {
    unsigned i = 0, e = PendingExports.size();
    for (; i != e; ++i) {
      assert(PendingExports[i].getNode()->getNumOperands() > 1);
      if (PendingExports[i].getNode()->getOperand(0) == Root)
        break;  // Don't add the root if we already indirectly depend on it.
    }

    if (i == e)
      PendingExports.push_back(Root);
  }

  Root = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(), MVT::Other,
                     &PendingExports[0],
                     PendingExports.size());
  PendingExports.clear();
  DAG.setRoot(Root);
  return Root;
}

void SelectionDAGBuilder::AssignOrderingToNode(const SDNode *Node) {
  if (DAG.GetOrdering(Node) != 0) return; // Already has ordering.
  DAG.AssignOrdering(Node, SDNodeOrder);

  for (unsigned I = 0, E = Node->getNumOperands(); I != E; ++I)
    AssignOrderingToNode(Node->getOperand(I).getNode());
}

void SelectionDAGBuilder::visit(const Instruction &I) {
  // Set up outgoing PHI node register values before emitting the terminator.
  if (isa<TerminatorInst>(&I))
    HandlePHINodesInSuccessorBlocks(I.getParent());

  CurDebugLoc = I.getDebugLoc();

  visit(I.getOpcode(), I);

  if (!isa<TerminatorInst>(&I) && !HasTailCall)
    CopyToExportRegsIfNeeded(&I);

  CurDebugLoc = DebugLoc();
}

void SelectionDAGBuilder::visitPHI(const PHINode &) {
  llvm_unreachable("SelectionDAGBuilder shouldn't visit PHI nodes!");
}

void SelectionDAGBuilder::visit(unsigned Opcode, const User &I) {
  // Note: this doesn't use InstVisitor, because it has to work with
  // ConstantExpr's in addition to instructions.
  switch (Opcode) {
  default: llvm_unreachable("Unknown instruction type encountered!");
    // Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: visit##OPCODE((const CLASS&)I); break;
#include "llvm/Instruction.def"
  }

  // Assign the ordering to the freshly created DAG nodes.
  if (NodeMap.count(&I)) {
    ++SDNodeOrder;
    AssignOrderingToNode(getValue(&I).getNode());
  }
}

// resolveDanglingDebugInfo - if we saw an earlier dbg_value referring to V,
// generate the debug data structures now that we've seen its definition.
void SelectionDAGBuilder::resolveDanglingDebugInfo(const Value *V,
                                                   SDValue Val) {
  DanglingDebugInfo &DDI = DanglingDebugInfoMap[V];
  if (DDI.getDI()) {
    const DbgValueInst *DI = DDI.getDI();
    DebugLoc dl = DDI.getdl();
    unsigned DbgSDNodeOrder = DDI.getSDNodeOrder();
    MDNode *Variable = DI->getVariable();
    uint64_t Offset = DI->getOffset();
    SDDbgValue *SDV;
    if (Val.getNode()) {
      if (!EmitFuncArgumentDbgValue(V, Variable, Offset, Val)) {
        SDV = DAG.getDbgValue(Variable, Val.getNode(),
                              Val.getResNo(), Offset, dl, DbgSDNodeOrder);
        DAG.AddDbgValue(SDV, Val.getNode(), false);
      }
    } else
      DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
    DanglingDebugInfoMap[V] = DanglingDebugInfo();
  }
}

/// getValue - Return an SDValue for the given Value.
SDValue SelectionDAGBuilder::getValue(const Value *V) {
  // If we already have an SDValue for this value, use it. It's important
  // to do this first, so that we don't create a CopyFromReg if we already
  // have a regular SDValue.
  SDValue &N = NodeMap[V];
  if (N.getNode()) return N;

  // If there's a virtual register allocated and initialized for this
  // value, use it.
  DenseMap<const Value *, unsigned>::iterator It = FuncInfo.ValueMap.find(V);
  if (It != FuncInfo.ValueMap.end()) {
    unsigned InReg = It->second;
    RegsForValue RFV(*DAG.getContext(), TLI, InReg, V->getType());
    SDValue Chain = DAG.getEntryNode();
    N = RFV.getCopyFromRegs(DAG, FuncInfo, getCurDebugLoc(), Chain, NULL, V);
    resolveDanglingDebugInfo(V, N);
    return N;
  }

  // Otherwise create a new SDValue and remember it.
  SDValue Val = getValueImpl(V);
  NodeMap[V] = Val;
  resolveDanglingDebugInfo(V, Val);
  return Val;
}

/// getNonRegisterValue - Return an SDValue for the given Value, but
/// don't look in FuncInfo.ValueMap for a virtual register.
SDValue SelectionDAGBuilder::getNonRegisterValue(const Value *V) {
  // If we already have an SDValue for this value, use it.
  SDValue &N = NodeMap[V];
  if (N.getNode()) return N;

  // Otherwise create a new SDValue and remember it.
  SDValue Val = getValueImpl(V);
  NodeMap[V] = Val;
  resolveDanglingDebugInfo(V, Val);
  return Val;
}

/// getValueImpl - Helper function for getValue and getNonRegisterValue.
/// Create an SDValue for the given value.
SDValue SelectionDAGBuilder::getValueImpl(const Value *V) {
  if (const Constant *C = dyn_cast<Constant>(V)) {
    EVT VT = TLI.getValueType(V->getType(), true);

    if (const ConstantInt *CI = dyn_cast<ConstantInt>(C))
      return DAG.getConstant(*CI, VT);

    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
      return DAG.getGlobalAddress(GV, getCurDebugLoc(), VT);

    if (isa<ConstantPointerNull>(C))
      return DAG.getConstant(0, TLI.getPointerTy());

    if (const ConstantFP *CFP = dyn_cast<ConstantFP>(C))
      return DAG.getConstantFP(*CFP, VT);

    if (isa<UndefValue>(C) && !V->getType()->isAggregateType())
      return DAG.getUNDEF(VT);

    if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      visit(CE->getOpcode(), *CE);
      SDValue N1 = NodeMap[V];
      assert(N1.getNode() && "visit didn't populate the NodeMap!");
      return N1;
    }

    if (isa<ConstantStruct>(C) || isa<ConstantArray>(C)) {
      SmallVector<SDValue, 4> Constants;
      for (User::const_op_iterator OI = C->op_begin(), OE = C->op_end();
           OI != OE; ++OI) {
        SDNode *Val = getValue(*OI).getNode();
        // If the operand is an empty aggregate, there are no values.
        if (!Val) continue;
        // Add each leaf value from the operand to the Constants list
        // to form a flattened list of all the values.
        for (unsigned i = 0, e = Val->getNumValues(); i != e; ++i)
          Constants.push_back(SDValue(Val, i));
      }

      return DAG.getMergeValues(&Constants[0], Constants.size(),
                                getCurDebugLoc());
    }
    
    if (const ConstantDataSequential *CDS =
          dyn_cast<ConstantDataSequential>(C)) {
      SmallVector<SDValue, 4> Ops;
      for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
        SDNode *Val = getValue(CDS->getElementAsConstant(i)).getNode();
        // Add each leaf value from the operand to the Constants list
        // to form a flattened list of all the values.
        for (unsigned i = 0, e = Val->getNumValues(); i != e; ++i)
          Ops.push_back(SDValue(Val, i));
      }

      if (isa<ArrayType>(CDS->getType()))
        return DAG.getMergeValues(&Ops[0], Ops.size(), getCurDebugLoc());
      return NodeMap[V] = DAG.getNode(ISD::BUILD_VECTOR, getCurDebugLoc(),
                                      VT, &Ops[0], Ops.size());
    }

    if (C->getType()->isStructTy() || C->getType()->isArrayTy()) {
      assert((isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) &&
             "Unknown struct or array constant!");

      SmallVector<EVT, 4> ValueVTs;
      ComputeValueVTs(TLI, C->getType(), ValueVTs);
      unsigned NumElts = ValueVTs.size();
      if (NumElts == 0)
        return SDValue(); // empty struct
      SmallVector<SDValue, 4> Constants(NumElts);
      for (unsigned i = 0; i != NumElts; ++i) {
        EVT EltVT = ValueVTs[i];
        if (isa<UndefValue>(C))
          Constants[i] = DAG.getUNDEF(EltVT);
        else if (EltVT.isFloatingPoint())
          Constants[i] = DAG.getConstantFP(0, EltVT);
        else
          Constants[i] = DAG.getConstant(0, EltVT);
      }

      return DAG.getMergeValues(&Constants[0], NumElts,
                                getCurDebugLoc());
    }

    if (const BlockAddress *BA = dyn_cast<BlockAddress>(C))
      return DAG.getBlockAddress(BA, VT);

    VectorType *VecTy = cast<VectorType>(V->getType());
    unsigned NumElements = VecTy->getNumElements();

    // Now that we know the number and type of the elements, get that number of
    // elements into the Ops array based on what kind of constant it is.
    SmallVector<SDValue, 16> Ops;
    if (const ConstantVector *CV = dyn_cast<ConstantVector>(C)) {
      for (unsigned i = 0; i != NumElements; ++i)
        Ops.push_back(getValue(CV->getOperand(i)));
    } else {
      assert(isa<ConstantAggregateZero>(C) && "Unknown vector constant!");
      EVT EltVT = TLI.getValueType(VecTy->getElementType());

      SDValue Op;
      if (EltVT.isFloatingPoint())
        Op = DAG.getConstantFP(0, EltVT);
      else
        Op = DAG.getConstant(0, EltVT);
      Ops.assign(NumElements, Op);
    }

    // Create a BUILD_VECTOR node.
    return NodeMap[V] = DAG.getNode(ISD::BUILD_VECTOR, getCurDebugLoc(),
                                    VT, &Ops[0], Ops.size());
  }

  // If this is a static alloca, generate it as the frameindex instead of
  // computation.
  if (const AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    DenseMap<const AllocaInst*, int>::iterator SI =
      FuncInfo.StaticAllocaMap.find(AI);
    if (SI != FuncInfo.StaticAllocaMap.end())
      return DAG.getFrameIndex(SI->second, TLI.getPointerTy());
  }

  // If this is an instruction which fast-isel has deferred, select it now.
  if (const Instruction *Inst = dyn_cast<Instruction>(V)) {
    unsigned InReg = FuncInfo.InitializeRegForValue(Inst);
    RegsForValue RFV(*DAG.getContext(), TLI, InReg, Inst->getType());
    SDValue Chain = DAG.getEntryNode();
    return RFV.getCopyFromRegs(DAG, FuncInfo, getCurDebugLoc(), Chain, NULL, V);
  }

  llvm_unreachable("Can't get register for value!");
}

void SelectionDAGBuilder::visitRet(const ReturnInst &I) {
  SDValue Chain = getControlRoot();
  SmallVector<ISD::OutputArg, 8> Outs;
  SmallVector<SDValue, 8> OutVals;

  if (!FuncInfo.CanLowerReturn) {
    unsigned DemoteReg = FuncInfo.DemoteRegister;
    const Function *F = I.getParent()->getParent();

    // Emit a store of the return value through the virtual register.
    // Leave Outs empty so that LowerReturn won't try to load return
    // registers the usual way.
    SmallVector<EVT, 1> PtrValueVTs;
    ComputeValueVTs(TLI, PointerType::getUnqual(F->getReturnType()),
                    PtrValueVTs);

    SDValue RetPtr = DAG.getRegister(DemoteReg, PtrValueVTs[0]);
    SDValue RetOp = getValue(I.getOperand(0));

    SmallVector<EVT, 4> ValueVTs;
    SmallVector<uint64_t, 4> Offsets;
    ComputeValueVTs(TLI, I.getOperand(0)->getType(), ValueVTs, &Offsets);
    unsigned NumValues = ValueVTs.size();

    SmallVector<SDValue, 4> Chains(NumValues);
    for (unsigned i = 0; i != NumValues; ++i) {
      SDValue Add = DAG.getNode(ISD::ADD, getCurDebugLoc(),
                                RetPtr.getValueType(), RetPtr,
                                DAG.getIntPtrConstant(Offsets[i]));
      Chains[i] =
        DAG.getStore(Chain, getCurDebugLoc(),
                     SDValue(RetOp.getNode(), RetOp.getResNo() + i),
                     // FIXME: better loc info would be nice.
                     Add, MachinePointerInfo(), false, false, 0);
    }

    Chain = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(),
                        MVT::Other, &Chains[0], NumValues);
  } else if (I.getNumOperands() != 0) {
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(TLI, I.getOperand(0)->getType(), ValueVTs);
    unsigned NumValues = ValueVTs.size();
    if (NumValues) {
      SDValue RetOp = getValue(I.getOperand(0));
      for (unsigned j = 0, f = NumValues; j != f; ++j) {
        EVT VT = ValueVTs[j];

        ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

        const Function *F = I.getParent()->getParent();
        if (F->getRetAttributes().hasAttribute(Attributes::SExt))
          ExtendKind = ISD::SIGN_EXTEND;
        else if (F->getRetAttributes().hasAttribute(Attributes::ZExt))
          ExtendKind = ISD::ZERO_EXTEND;

        if (ExtendKind != ISD::ANY_EXTEND && VT.isInteger())
          VT = TLI.getTypeForExtArgOrReturn(*DAG.getContext(), VT, ExtendKind);

        unsigned NumParts = TLI.getNumRegisters(*DAG.getContext(), VT);
        EVT PartVT = TLI.getRegisterType(*DAG.getContext(), VT);
        SmallVector<SDValue, 4> Parts(NumParts);
        getCopyToParts(DAG, getCurDebugLoc(),
                       SDValue(RetOp.getNode(), RetOp.getResNo() + j),
                       &Parts[0], NumParts, PartVT, &I, ExtendKind);

        // 'inreg' on function refers to return value
        ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();
        if (F->getRetAttributes().hasAttribute(Attributes::InReg))
          Flags.setInReg();

        // Propagate extension type if any
        if (ExtendKind == ISD::SIGN_EXTEND)
          Flags.setSExt();
        else if (ExtendKind == ISD::ZERO_EXTEND)
          Flags.setZExt();

        for (unsigned i = 0; i < NumParts; ++i) {
          Outs.push_back(ISD::OutputArg(Flags, Parts[i].getValueType(),
                                        /*isfixed=*/true, 0, 0));
          OutVals.push_back(Parts[i]);
        }
      }
    }
  }

  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  CallingConv::ID CallConv =
    DAG.getMachineFunction().getFunction()->getCallingConv();
  Chain = TLI.LowerReturn(Chain, CallConv, isVarArg,
                          Outs, OutVals, getCurDebugLoc(), DAG);

  // Verify that the target's LowerReturn behaved as expected.
  assert(Chain.getNode() && Chain.getValueType() == MVT::Other &&
         "LowerReturn didn't return a valid chain!");

  // Update the DAG with the new chain value resulting from return lowering.
  DAG.setRoot(Chain);
}

/// CopyToExportRegsIfNeeded - If the given value has virtual registers
/// created for it, emit nodes to copy the value into the virtual
/// registers.
void SelectionDAGBuilder::CopyToExportRegsIfNeeded(const Value *V) {
  // Skip empty types
  if (V->getType()->isEmptyTy())
    return;

  DenseMap<const Value *, unsigned>::iterator VMI = FuncInfo.ValueMap.find(V);
  if (VMI != FuncInfo.ValueMap.end()) {
    assert(!V->use_empty() && "Unused value assigned virtual registers!");
    CopyValueToVirtualRegister(V, VMI->second);
  }
}

/// ExportFromCurrentBlock - If this condition isn't known to be exported from
/// the current basic block, add it to ValueMap now so that we'll get a
/// CopyTo/FromReg.
void SelectionDAGBuilder::ExportFromCurrentBlock(const Value *V) {
  // No need to export constants.
  if (!isa<Instruction>(V) && !isa<Argument>(V)) return;

  // Already exported?
  if (FuncInfo.isExportedInst(V)) return;

  unsigned Reg = FuncInfo.InitializeRegForValue(V);
  CopyValueToVirtualRegister(V, Reg);
}

bool SelectionDAGBuilder::isExportableFromCurrentBlock(const Value *V,
                                                     const BasicBlock *FromBB) {
  // The operands of the setcc have to be in this block.  We don't know
  // how to export them from some other block.
  if (const Instruction *VI = dyn_cast<Instruction>(V)) {
    // Can export from current BB.
    if (VI->getParent() == FromBB)
      return true;

    // Is already exported, noop.
    return FuncInfo.isExportedInst(V);
  }

  // If this is an argument, we can export it if the BB is the entry block or
  // if it is already exported.
  if (isa<Argument>(V)) {
    if (FromBB == &FromBB->getParent()->getEntryBlock())
      return true;

    // Otherwise, can only export this if it is already exported.
    return FuncInfo.isExportedInst(V);
  }

  // Otherwise, constants can always be exported.
  return true;
}

/// Return branch probability calculated by BranchProbabilityInfo for IR blocks.
uint32_t SelectionDAGBuilder::getEdgeWeight(const MachineBasicBlock *Src,
                                            const MachineBasicBlock *Dst) const {
  BranchProbabilityInfo *BPI = FuncInfo.BPI;
  if (!BPI)
    return 0;
  const BasicBlock *SrcBB = Src->getBasicBlock();
  const BasicBlock *DstBB = Dst->getBasicBlock();
  return BPI->getEdgeWeight(SrcBB, DstBB);
}

void SelectionDAGBuilder::
addSuccessorWithWeight(MachineBasicBlock *Src, MachineBasicBlock *Dst,
                       uint32_t Weight /* = 0 */) {
  if (!Weight)
    Weight = getEdgeWeight(Src, Dst);
  Src->addSuccessor(Dst, Weight);
}


static bool InBlock(const Value *V, const BasicBlock *BB) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() == BB;
  return true;
}

/// EmitBranchForMergedCondition - Helper method for FindMergedConditions.
/// This function emits a branch and is used at the leaves of an OR or an
/// AND operator tree.
///
void
SelectionDAGBuilder::EmitBranchForMergedCondition(const Value *Cond,
                                                  MachineBasicBlock *TBB,
                                                  MachineBasicBlock *FBB,
                                                  MachineBasicBlock *CurBB,
                                                  MachineBasicBlock *SwitchBB) {
  const BasicBlock *BB = CurBB->getBasicBlock();

  // If the leaf of the tree is a comparison, merge the condition into
  // the caseblock.
  if (const CmpInst *BOp = dyn_cast<CmpInst>(Cond)) {
    // The operands of the cmp have to be in this block.  We don't know
    // how to export them from some other block.  If this is the first block
    // of the sequence, no exporting is needed.
    if (CurBB == SwitchBB ||
        (isExportableFromCurrentBlock(BOp->getOperand(0), BB) &&
         isExportableFromCurrentBlock(BOp->getOperand(1), BB))) {
      ISD::CondCode Condition;
      if (const ICmpInst *IC = dyn_cast<ICmpInst>(Cond)) {
        Condition = getICmpCondCode(IC->getPredicate());
      } else if (const FCmpInst *FC = dyn_cast<FCmpInst>(Cond)) {
        Condition = getFCmpCondCode(FC->getPredicate());
        if (TM.Options.NoNaNsFPMath)
          Condition = getFCmpCodeWithoutNaN(Condition);
      } else {
        Condition = ISD::SETEQ; // silence warning.
        llvm_unreachable("Unknown compare instruction");
      }

      CaseBlock CB(Condition, BOp->getOperand(0),
                   BOp->getOperand(1), NULL, TBB, FBB, CurBB);
      SwitchCases.push_back(CB);
      return;
    }
  }

  // Create a CaseBlock record representing this branch.
  CaseBlock CB(ISD::SETEQ, Cond, ConstantInt::getTrue(*DAG.getContext()),
               NULL, TBB, FBB, CurBB);
  SwitchCases.push_back(CB);
}

/// FindMergedConditions - If Cond is an expression like
void SelectionDAGBuilder::FindMergedConditions(const Value *Cond,
                                               MachineBasicBlock *TBB,
                                               MachineBasicBlock *FBB,
                                               MachineBasicBlock *CurBB,
                                               MachineBasicBlock *SwitchBB,
                                               unsigned Opc) {
  // If this node is not part of the or/and tree, emit it as a branch.
  const Instruction *BOp = dyn_cast<Instruction>(Cond);
  if (!BOp || !(isa<BinaryOperator>(BOp) || isa<CmpInst>(BOp)) ||
      (unsigned)BOp->getOpcode() != Opc || !BOp->hasOneUse() ||
      BOp->getParent() != CurBB->getBasicBlock() ||
      !InBlock(BOp->getOperand(0), CurBB->getBasicBlock()) ||
      !InBlock(BOp->getOperand(1), CurBB->getBasicBlock())) {
    EmitBranchForMergedCondition(Cond, TBB, FBB, CurBB, SwitchBB);
    return;
  }

  //  Create TmpBB after CurBB.
  MachineFunction::iterator BBI = CurBB;
  MachineFunction &MF = DAG.getMachineFunction();
  MachineBasicBlock *TmpBB = MF.CreateMachineBasicBlock(CurBB->getBasicBlock());
  CurBB->getParent()->insert(++BBI, TmpBB);

  if (Opc == Instruction::Or) {
    // Codegen X | Y as:
    //   jmp_if_X TBB
    //   jmp TmpBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //

    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TBB, TmpBB, CurBB, SwitchBB, Opc);

    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, SwitchBB, Opc);
  } else {
    assert(Opc == Instruction::And && "Unknown merge op!");
    // Codegen X & Y as:
    //   jmp_if_X TmpBB
    //   jmp FBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //
    //  This requires creation of TmpBB after CurBB.

    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TmpBB, FBB, CurBB, SwitchBB, Opc);

    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, SwitchBB, Opc);
  }
}

/// If the set of cases should be emitted as a series of branches, return true.
/// If we should emit this as a bunch of and/or'd together conditions, return
/// false.
bool
SelectionDAGBuilder::ShouldEmitAsBranches(const std::vector<CaseBlock> &Cases){
  if (Cases.size() != 2) return true;

  // If this is two comparisons of the same values or'd or and'd together, they
  // will get folded into a single comparison, so don't emit two blocks.
  if ((Cases[0].CmpLHS == Cases[1].CmpLHS &&
       Cases[0].CmpRHS == Cases[1].CmpRHS) ||
      (Cases[0].CmpRHS == Cases[1].CmpLHS &&
       Cases[0].CmpLHS == Cases[1].CmpRHS)) {
    return false;
  }

  // Handle: (X != null) | (Y != null) --> (X|Y) != 0
  // Handle: (X == null) & (Y == null) --> (X|Y) == 0
  if (Cases[0].CmpRHS == Cases[1].CmpRHS &&
      Cases[0].CC == Cases[1].CC &&
      isa<Constant>(Cases[0].CmpRHS) &&
      cast<Constant>(Cases[0].CmpRHS)->isNullValue()) {
    if (Cases[0].CC == ISD::SETEQ && Cases[0].TrueBB == Cases[1].ThisBB)
      return false;
    if (Cases[0].CC == ISD::SETNE && Cases[0].FalseBB == Cases[1].ThisBB)
      return false;
  }

  return true;
}

void SelectionDAGBuilder::visitBr(const BranchInst &I) {
  MachineBasicBlock *BrMBB = FuncInfo.MBB;

  // Update machine-CFG edges.
  MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[I.getSuccessor(0)];

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = BrMBB;
  if (++BBI != FuncInfo.MF->end())
    NextBlock = BBI;

  if (I.isUnconditional()) {
    // Update machine-CFG edges.
    BrMBB->addSuccessor(Succ0MBB);

    // If this is not a fall-through branch, emit the branch.
    if (Succ0MBB != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, getCurDebugLoc(),
                              MVT::Other, getControlRoot(),
                              DAG.getBasicBlock(Succ0MBB)));

    return;
  }

  // If this condition is one of the special cases we handle, do special stuff
  // now.
  const Value *CondVal = I.getCondition();
  MachineBasicBlock *Succ1MBB = FuncInfo.MBBMap[I.getSuccessor(1)];

  // If this is a series of conditions that are or'd or and'd together, emit
  // this as a sequence of branches instead of setcc's with and/or operations.
  // As long as jumps are not expensive, this should improve performance.
  // For example, instead of something like:
  //     cmp A, B
  //     C = seteq
  //     cmp D, E
  //     F = setle
  //     or C, F
  //     jnz foo
  // Emit:
  //     cmp A, B
  //     je foo
  //     cmp D, E
  //     jle foo
  //
  if (const BinaryOperator *BOp = dyn_cast<BinaryOperator>(CondVal)) {
    if (!TLI.isJumpExpensive() &&
        BOp->hasOneUse() &&
        (BOp->getOpcode() == Instruction::And ||
         BOp->getOpcode() == Instruction::Or)) {
      FindMergedConditions(BOp, Succ0MBB, Succ1MBB, BrMBB, BrMBB,
                           BOp->getOpcode());
      // If the compares in later blocks need to use values not currently
      // exported from this block, export them now.  This block should always
      // be the first entry.
      assert(SwitchCases[0].ThisBB == BrMBB && "Unexpected lowering!");

      // Allow some cases to be rejected.
      if (ShouldEmitAsBranches(SwitchCases)) {
        for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i) {
          ExportFromCurrentBlock(SwitchCases[i].CmpLHS);
          ExportFromCurrentBlock(SwitchCases[i].CmpRHS);
        }

        // Emit the branch for this block.
        visitSwitchCase(SwitchCases[0], BrMBB);
        SwitchCases.erase(SwitchCases.begin());
        return;
      }

      // Okay, we decided not to do this, remove any inserted MBB's and clear
      // SwitchCases.
      for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i)
        FuncInfo.MF->erase(SwitchCases[i].ThisBB);

      SwitchCases.clear();
    }
  }

  // Create a CaseBlock record representing this branch.
  CaseBlock CB(ISD::SETEQ, CondVal, ConstantInt::getTrue(*DAG.getContext()),
               NULL, Succ0MBB, Succ1MBB, BrMBB);

  // Use visitSwitchCase to actually insert the fast branch sequence for this
  // cond branch.
  visitSwitchCase(CB, BrMBB);
}

/// visitSwitchCase - Emits the necessary code to represent a single node in
/// the binary search tree resulting from lowering a switch instruction.
void SelectionDAGBuilder::visitSwitchCase(CaseBlock &CB,
                                          MachineBasicBlock *SwitchBB) {
  SDValue Cond;
  SDValue CondLHS = getValue(CB.CmpLHS);
  DebugLoc dl = getCurDebugLoc();

  // Build the setcc now.
  if (CB.CmpMHS == NULL) {
    // Fold "(X == true)" to X and "(X == false)" to !X to
    // handle common cases produced by branch lowering.
    if (CB.CmpRHS == ConstantInt::getTrue(*DAG.getContext()) &&
        CB.CC == ISD::SETEQ)
      Cond = CondLHS;
    else if (CB.CmpRHS == ConstantInt::getFalse(*DAG.getContext()) &&
             CB.CC == ISD::SETEQ) {
      SDValue True = DAG.getConstant(1, CondLHS.getValueType());
      Cond = DAG.getNode(ISD::XOR, dl, CondLHS.getValueType(), CondLHS, True);
    } else
      Cond = DAG.getSetCC(dl, MVT::i1, CondLHS, getValue(CB.CmpRHS), CB.CC);
  } else {
    assert(CB.CC == ISD::SETCC_INVALID &&
           "Condition is undefined for to-the-range belonging check.");

    const APInt& Low = cast<ConstantInt>(CB.CmpLHS)->getValue();
    const APInt& High  = cast<ConstantInt>(CB.CmpRHS)->getValue();

    SDValue CmpOp = getValue(CB.CmpMHS);
    EVT VT = CmpOp.getValueType();
    
    if (cast<ConstantInt>(CB.CmpLHS)->isMinValue(false)) {
      Cond = DAG.getSetCC(dl, MVT::i1, CmpOp, DAG.getConstant(High, VT),
                          ISD::SETULE);
    } else {
      SDValue SUB = DAG.getNode(ISD::SUB, dl,
                                VT, CmpOp, DAG.getConstant(Low, VT));
      Cond = DAG.getSetCC(dl, MVT::i1, SUB,
                          DAG.getConstant(High-Low, VT), ISD::SETULE);
    }
  }

  // Update successor info
  addSuccessorWithWeight(SwitchBB, CB.TrueBB, CB.TrueWeight);
  // TrueBB and FalseBB are always different unless the incoming IR is
  // degenerate. This only happens when running llc on weird IR.
  if (CB.TrueBB != CB.FalseBB)
    addSuccessorWithWeight(SwitchBB, CB.FalseBB, CB.FalseWeight);

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = SwitchBB;
  if (++BBI != FuncInfo.MF->end())
    NextBlock = BBI;

  // If the lhs block is the next block, invert the condition so that we can
  // fall through to the lhs instead of the rhs block.
  if (CB.TrueBB == NextBlock) {
    std::swap(CB.TrueBB, CB.FalseBB);
    SDValue True = DAG.getConstant(1, Cond.getValueType());
    Cond = DAG.getNode(ISD::XOR, dl, Cond.getValueType(), Cond, True);
  }

  SDValue BrCond = DAG.getNode(ISD::BRCOND, dl,
                               MVT::Other, getControlRoot(), Cond,
                               DAG.getBasicBlock(CB.TrueBB));

  // Insert the false branch. Do this even if it's a fall through branch,
  // this makes it easier to do DAG optimizations which require inverting
  // the branch condition.
  BrCond = DAG.getNode(ISD::BR, dl, MVT::Other, BrCond,
                       DAG.getBasicBlock(CB.FalseBB));

  DAG.setRoot(BrCond);
}

/// visitJumpTable - Emit JumpTable node in the current MBB
void SelectionDAGBuilder::visitJumpTable(JumpTable &JT) {
  // Emit the code for the jump table
  assert(JT.Reg != -1U && "Should lower JT Header first!");
  EVT PTy = TLI.getPointerTy();
  SDValue Index = DAG.getCopyFromReg(getControlRoot(), getCurDebugLoc(),
                                     JT.Reg, PTy);
  SDValue Table = DAG.getJumpTable(JT.JTI, PTy);
  SDValue BrJumpTable = DAG.getNode(ISD::BR_JT, getCurDebugLoc(),
                                    MVT::Other, Index.getValue(1),
                                    Table, Index);
  DAG.setRoot(BrJumpTable);
}

/// visitJumpTableHeader - This function emits necessary code to produce index
/// in the JumpTable from switch case.
void SelectionDAGBuilder::visitJumpTableHeader(JumpTable &JT,
                                               JumpTableHeader &JTH,
                                               MachineBasicBlock *SwitchBB) {
  // Subtract the lowest switch case value from the value being switched on and
  // conditional branch to default mbb if the result is greater than the
  // difference between smallest and largest cases.
  SDValue SwitchOp = getValue(JTH.SValue);
  EVT VT = SwitchOp.getValueType();
  SDValue Sub = DAG.getNode(ISD::SUB, getCurDebugLoc(), VT, SwitchOp,
                            DAG.getConstant(JTH.First, VT));

  // The SDNode we just created, which holds the value being switched on minus
  // the smallest case value, needs to be copied to a virtual register so it
  // can be used as an index into the jump table in a subsequent basic block.
  // This value may be smaller or larger than the target's pointer type, and
  // therefore require extension or truncating.
  SwitchOp = DAG.getZExtOrTrunc(Sub, getCurDebugLoc(), TLI.getPointerTy());

  unsigned JumpTableReg = FuncInfo.CreateReg(TLI.getPointerTy());
  SDValue CopyTo = DAG.getCopyToReg(getControlRoot(), getCurDebugLoc(),
                                    JumpTableReg, SwitchOp);
  JT.Reg = JumpTableReg;

  // Emit the range check for the jump table, and branch to the default block
  // for the switch statement if the value being switched on exceeds the largest
  // case in the switch.
  SDValue CMP = DAG.getSetCC(getCurDebugLoc(),
                             TLI.getSetCCResultType(Sub.getValueType()), Sub,
                             DAG.getConstant(JTH.Last-JTH.First,VT),
                             ISD::SETUGT);

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = SwitchBB;

  if (++BBI != FuncInfo.MF->end())
    NextBlock = BBI;

  SDValue BrCond = DAG.getNode(ISD::BRCOND, getCurDebugLoc(),
                               MVT::Other, CopyTo, CMP,
                               DAG.getBasicBlock(JT.Default));

  if (JT.MBB != NextBlock)
    BrCond = DAG.getNode(ISD::BR, getCurDebugLoc(), MVT::Other, BrCond,
                         DAG.getBasicBlock(JT.MBB));

  DAG.setRoot(BrCond);
}

/// visitBitTestHeader - This function emits necessary code to produce value
/// suitable for "bit tests"
void SelectionDAGBuilder::visitBitTestHeader(BitTestBlock &B,
                                             MachineBasicBlock *SwitchBB) {
  // Subtract the minimum value
  SDValue SwitchOp = getValue(B.SValue);
  EVT VT = SwitchOp.getValueType();
  SDValue Sub = DAG.getNode(ISD::SUB, getCurDebugLoc(), VT, SwitchOp,
                            DAG.getConstant(B.First, VT));

  // Check range
  SDValue RangeCmp = DAG.getSetCC(getCurDebugLoc(),
                                  TLI.getSetCCResultType(Sub.getValueType()),
                                  Sub, DAG.getConstant(B.Range, VT),
                                  ISD::SETUGT);

  // Determine the type of the test operands.
  bool UsePtrType = false;
  if (!TLI.isTypeLegal(VT))
    UsePtrType = true;
  else {
    for (unsigned i = 0, e = B.Cases.size(); i != e; ++i)
      if (!isUIntN(VT.getSizeInBits(), B.Cases[i].Mask)) {
        // Switch table case range are encoded into series of masks.
        // Just use pointer type, it's guaranteed to fit.
        UsePtrType = true;
        break;
      }
  }
  if (UsePtrType) {
    VT = TLI.getPointerTy();
    Sub = DAG.getZExtOrTrunc(Sub, getCurDebugLoc(), VT);
  }

  B.RegVT = VT;
  B.Reg = FuncInfo.CreateReg(VT);
  SDValue CopyTo = DAG.getCopyToReg(getControlRoot(), getCurDebugLoc(),
                                    B.Reg, Sub);

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = SwitchBB;
  if (++BBI != FuncInfo.MF->end())
    NextBlock = BBI;

  MachineBasicBlock* MBB = B.Cases[0].ThisBB;

  addSuccessorWithWeight(SwitchBB, B.Default);
  addSuccessorWithWeight(SwitchBB, MBB);

  SDValue BrRange = DAG.getNode(ISD::BRCOND, getCurDebugLoc(),
                                MVT::Other, CopyTo, RangeCmp,
                                DAG.getBasicBlock(B.Default));

  if (MBB != NextBlock)
    BrRange = DAG.getNode(ISD::BR, getCurDebugLoc(), MVT::Other, CopyTo,
                          DAG.getBasicBlock(MBB));

  DAG.setRoot(BrRange);
}

/// visitBitTestCase - this function produces one "bit test"
void SelectionDAGBuilder::visitBitTestCase(BitTestBlock &BB,
                                           MachineBasicBlock* NextMBB,
                                           uint32_t BranchWeightToNext,
                                           unsigned Reg,
                                           BitTestCase &B,
                                           MachineBasicBlock *SwitchBB) {
  EVT VT = BB.RegVT;
  SDValue ShiftOp = DAG.getCopyFromReg(getControlRoot(), getCurDebugLoc(),
                                       Reg, VT);
  SDValue Cmp;
  unsigned PopCount = CountPopulation_64(B.Mask);
  if (PopCount == 1) {
    // Testing for a single bit; just compare the shift count with what it
    // would need to be to shift a 1 bit in that position.
    Cmp = DAG.getSetCC(getCurDebugLoc(),
                       TLI.getSetCCResultType(VT),
                       ShiftOp,
                       DAG.getConstant(CountTrailingZeros_64(B.Mask), VT),
                       ISD::SETEQ);
  } else if (PopCount == BB.Range) {
    // There is only one zero bit in the range, test for it directly.
    Cmp = DAG.getSetCC(getCurDebugLoc(),
                       TLI.getSetCCResultType(VT),
                       ShiftOp,
                       DAG.getConstant(CountTrailingOnes_64(B.Mask), VT),
                       ISD::SETNE);
  } else {
    // Make desired shift
    SDValue SwitchVal = DAG.getNode(ISD::SHL, getCurDebugLoc(), VT,
                                    DAG.getConstant(1, VT), ShiftOp);

    // Emit bit tests and jumps
    SDValue AndOp = DAG.getNode(ISD::AND, getCurDebugLoc(),
                                VT, SwitchVal, DAG.getConstant(B.Mask, VT));
    Cmp = DAG.getSetCC(getCurDebugLoc(),
                       TLI.getSetCCResultType(VT),
                       AndOp, DAG.getConstant(0, VT),
                       ISD::SETNE);
  }

  // The branch weight from SwitchBB to B.TargetBB is B.ExtraWeight.
  addSuccessorWithWeight(SwitchBB, B.TargetBB, B.ExtraWeight);
  // The branch weight from SwitchBB to NextMBB is BranchWeightToNext.
  addSuccessorWithWeight(SwitchBB, NextMBB, BranchWeightToNext);

  SDValue BrAnd = DAG.getNode(ISD::BRCOND, getCurDebugLoc(),
                              MVT::Other, getControlRoot(),
                              Cmp, DAG.getBasicBlock(B.TargetBB));

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = SwitchBB;
  if (++BBI != FuncInfo.MF->end())
    NextBlock = BBI;

  if (NextMBB != NextBlock)
    BrAnd = DAG.getNode(ISD::BR, getCurDebugLoc(), MVT::Other, BrAnd,
                        DAG.getBasicBlock(NextMBB));

  DAG.setRoot(BrAnd);
}

void SelectionDAGBuilder::visitInvoke(const InvokeInst &I) {
  MachineBasicBlock *InvokeMBB = FuncInfo.MBB;

  // Retrieve successors.
  MachineBasicBlock *Return = FuncInfo.MBBMap[I.getSuccessor(0)];
  MachineBasicBlock *LandingPad = FuncInfo.MBBMap[I.getSuccessor(1)];

  const Value *Callee(I.getCalledValue());
  const Function *Fn = dyn_cast<Function>(Callee);
  if (isa<InlineAsm>(Callee))
    visitInlineAsm(&I);
  else if (Fn && Fn->isIntrinsic()) {
    assert(Fn->getIntrinsicID() == Intrinsic::donothing);
    // Ignore invokes to @llvm.donothing: jump directly to the next BB.
  } else
    LowerCallTo(&I, getValue(Callee), false, LandingPad);

  // If the value of the invoke is used outside of its defining block, make it
  // available as a virtual register.
  CopyToExportRegsIfNeeded(&I);

  // Update successor info
  addSuccessorWithWeight(InvokeMBB, Return);
  addSuccessorWithWeight(InvokeMBB, LandingPad);

  // Drop into normal successor.
  DAG.setRoot(DAG.getNode(ISD::BR, getCurDebugLoc(),
                          MVT::Other, getControlRoot(),
                          DAG.getBasicBlock(Return)));
}

void SelectionDAGBuilder::visitResume(const ResumeInst &RI) {
  llvm_unreachable("SelectionDAGBuilder shouldn't visit resume instructions!");
}

void SelectionDAGBuilder::visitLandingPad(const LandingPadInst &LP) {
  assert(FuncInfo.MBB->isLandingPad() &&
         "Call to landingpad not in landing pad!");

  MachineBasicBlock *MBB = FuncInfo.MBB;
  MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
  AddLandingPadInfo(LP, MMI, MBB);

  // If there aren't registers to copy the values into (e.g., during SjLj
  // exceptions), then don't bother to create these DAG nodes.
  if (TLI.getExceptionPointerRegister() == 0 &&
      TLI.getExceptionSelectorRegister() == 0)
    return;

  SmallVector<EVT, 2> ValueVTs;
  ComputeValueVTs(TLI, LP.getType(), ValueVTs);

  // Insert the EXCEPTIONADDR instruction.
  assert(FuncInfo.MBB->isLandingPad() &&
         "Call to eh.exception not in landing pad!");
  SDVTList VTs = DAG.getVTList(TLI.getPointerTy(), MVT::Other);
  SDValue Ops[2];
  Ops[0] = DAG.getRoot();
  SDValue Op1 = DAG.getNode(ISD::EXCEPTIONADDR, getCurDebugLoc(), VTs, Ops, 1);
  SDValue Chain = Op1.getValue(1);

  // Insert the EHSELECTION instruction.
  VTs = DAG.getVTList(TLI.getPointerTy(), MVT::Other);
  Ops[0] = Op1;
  Ops[1] = Chain;
  SDValue Op2 = DAG.getNode(ISD::EHSELECTION, getCurDebugLoc(), VTs, Ops, 2);
  Chain = Op2.getValue(1);
  Op2 = DAG.getSExtOrTrunc(Op2, getCurDebugLoc(), MVT::i32);

  Ops[0] = Op1;
  Ops[1] = Op2;
  SDValue Res = DAG.getNode(ISD::MERGE_VALUES, getCurDebugLoc(),
                            DAG.getVTList(&ValueVTs[0], ValueVTs.size()),
                            &Ops[0], 2);

  std::pair<SDValue, SDValue> RetPair = std::make_pair(Res, Chain);
  setValue(&LP, RetPair.first);
  DAG.setRoot(RetPair.second);
}

/// handleSmallSwitchCaseRange - Emit a series of specific tests (suitable for
/// small case ranges).
bool SelectionDAGBuilder::handleSmallSwitchRange(CaseRec& CR,
                                                 CaseRecVector& WorkList,
                                                 const Value* SV,
                                                 MachineBasicBlock *Default,
                                                 MachineBasicBlock *SwitchBB) {
  // Size is the number of Cases represented by this range.
  size_t Size = CR.Range.second - CR.Range.first;
  if (Size > 3)
    return false;

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = FuncInfo.MF;

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != FuncInfo.MF->end())
    NextBlock = BBI;

  BranchProbabilityInfo *BPI = FuncInfo.BPI;
  // If any two of the cases has the same destination, and if one value
  // is the same as the other, but has one bit unset that the other has set,
  // use bit manipulation to do two compares at once.  For example:
  // "if (X == 6 || X == 4)" -> "if ((X|2) == 6)"
  // TODO: This could be extended to merge any 2 cases in switches with 3 cases.
  // TODO: Handle cases where CR.CaseBB != SwitchBB.
  if (Size == 2 && CR.CaseBB == SwitchBB) {
    Case &Small = *CR.Range.first;
    Case &Big = *(CR.Range.second-1);

    if (Small.Low == Small.High && Big.Low == Big.High && Small.BB == Big.BB) {
      const APInt& SmallValue = cast<ConstantInt>(Small.Low)->getValue();
      const APInt& BigValue = cast<ConstantInt>(Big.Low)->getValue();

      // Check that there is only one bit different.
      if (BigValue.countPopulation() == SmallValue.countPopulation() + 1 &&
          (SmallValue | BigValue) == BigValue) {
        // Isolate the common bit.
        APInt CommonBit = BigValue & ~SmallValue;
        assert((SmallValue | CommonBit) == BigValue &&
               CommonBit.countPopulation() == 1 && "Not a common bit?");

        SDValue CondLHS = getValue(SV);
        EVT VT = CondLHS.getValueType();
        DebugLoc DL = getCurDebugLoc();

        SDValue Or = DAG.getNode(ISD::OR, DL, VT, CondLHS,
                                 DAG.getConstant(CommonBit, VT));
        SDValue Cond = DAG.getSetCC(DL, MVT::i1,
                                    Or, DAG.getConstant(BigValue, VT),
                                    ISD::SETEQ);

        // Update successor info.
        // Both Small and Big will jump to Small.BB, so we sum up the weights.
        addSuccessorWithWeight(SwitchBB, Small.BB,
                               Small.ExtraWeight + Big.ExtraWeight);
        addSuccessorWithWeight(SwitchBB, Default,
          // The default destination is the first successor in IR.
          BPI ? BPI->getEdgeWeight(SwitchBB->getBasicBlock(), (unsigned)0) : 0);

        // Insert the true branch.
        SDValue BrCond = DAG.getNode(ISD::BRCOND, DL, MVT::Other,
                                     getControlRoot(), Cond,
                                     DAG.getBasicBlock(Small.BB));

        // Insert the false branch.
        BrCond = DAG.getNode(ISD::BR, DL, MVT::Other, BrCond,
                             DAG.getBasicBlock(Default));

        DAG.setRoot(BrCond);
        return true;
      }
    }
  }

  // Order cases by weight so the most likely case will be checked first.
  uint32_t UnhandledWeights = 0;
  if (BPI) {
    for (CaseItr I = CR.Range.first, IE = CR.Range.second; I != IE; ++I) {
      uint32_t IWeight = I->ExtraWeight;
      UnhandledWeights += IWeight;
      for (CaseItr J = CR.Range.first; J < I; ++J) {
        uint32_t JWeight = J->ExtraWeight;
        if (IWeight > JWeight)
          std::swap(*I, *J);
      }
    }
  }
  // Rearrange the case blocks so that the last one falls through if possible.
  Case &BackCase = *(CR.Range.second-1);
  if (Size > 1 &&
      NextBlock && Default != NextBlock && BackCase.BB != NextBlock) {
    // The last case block won't fall through into 'NextBlock' if we emit the
    // branches in this order.  See if rearranging a case value would help.
    // We start at the bottom as it's the case with the least weight.
    for (Case *I = &*(CR.Range.second-2), *E = &*CR.Range.first-1; I != E; --I){
      if (I->BB == NextBlock) {
        std::swap(*I, BackCase);
        break;
      }
    }
  }

  // Create a CaseBlock record representing a conditional branch to
  // the Case's target mbb if the value being switched on SV is equal
  // to C.
  MachineBasicBlock *CurBlock = CR.CaseBB;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++I) {
    MachineBasicBlock *FallThrough;
    if (I != E-1) {
      FallThrough = CurMF->CreateMachineBasicBlock(CurBlock->getBasicBlock());
      CurMF->insert(BBI, FallThrough);

      // Put SV in a virtual register to make it available from the new blocks.
      ExportFromCurrentBlock(SV);
    } else {
      // If the last case doesn't match, go to the default block.
      FallThrough = Default;
    }

    const Value *RHS, *LHS, *MHS;
    ISD::CondCode CC;
    if (I->High == I->Low) {
      // This is just small small case range :) containing exactly 1 case
      CC = ISD::SETEQ;
      LHS = SV; RHS = I->High; MHS = NULL;
    } else {
      CC = ISD::SETCC_INVALID; 
      LHS = I->Low; MHS = SV; RHS = I->High;
    }

    // The false weight should be sum of all un-handled cases.
    UnhandledWeights -= I->ExtraWeight;
    CaseBlock CB(CC, LHS, RHS, MHS, /* truebb */ I->BB, /* falsebb */ FallThrough,
                 /* me */ CurBlock,
                 /* trueweight */ I->ExtraWeight,
                 /* falseweight */ UnhandledWeights);

    // If emitting the first comparison, just call visitSwitchCase to emit the
    // code into the current block.  Otherwise, push the CaseBlock onto the
    // vector to be later processed by SDISel, and insert the node's MBB
    // before the next MBB.
    if (CurBlock == SwitchBB)
      visitSwitchCase(CB, SwitchBB);
    else
      SwitchCases.push_back(CB);

    CurBlock = FallThrough;
  }

  return true;
}

static inline bool areJTsAllowed(const TargetLowering &TLI) {
  return TLI.supportJumpTables() &&
          (TLI.isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
           TLI.isOperationLegalOrCustom(ISD::BRIND, MVT::Other));
}

static APInt ComputeRange(const APInt &First, const APInt &Last) {
  uint32_t BitWidth = std::max(Last.getBitWidth(), First.getBitWidth()) + 1;
  APInt LastExt = Last.zext(BitWidth), FirstExt = First.zext(BitWidth);
  return (LastExt - FirstExt + 1ULL);
}

/// handleJTSwitchCase - Emit jumptable for current switch case range
bool SelectionDAGBuilder::handleJTSwitchCase(CaseRec &CR,
                                             CaseRecVector &WorkList,
                                             const Value *SV,
                                             MachineBasicBlock *Default,
                                             MachineBasicBlock *SwitchBB) {
  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);

  const APInt &First = cast<ConstantInt>(FrontCase.Low)->getValue();
  const APInt &Last  = cast<ConstantInt>(BackCase.High)->getValue();

  APInt TSize(First.getBitWidth(), 0);
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++I)
    TSize += I->size();

  if (!areJTsAllowed(TLI) || TSize.ult(TLI.getMinimumJumpTableEntries()))
    return false;

  APInt Range = ComputeRange(First, Last);
  // The density is TSize / Range. Require at least 40%.
  // It should not be possible for IntTSize to saturate for sane code, but make
  // sure we handle Range saturation correctly.
  uint64_t IntRange = Range.getLimitedValue(UINT64_MAX/10);
  uint64_t IntTSize = TSize.getLimitedValue(UINT64_MAX/10);
  if (IntTSize * 10 < IntRange * 4)
    return false;

  DEBUG(dbgs() << "Lowering jump table\n"
               << "First entry: " << First << ". Last entry: " << Last << '\n'
               << "Range: " << Range << ". Size: " << TSize << ".\n\n");

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = FuncInfo.MF;

  // Figure out which block is immediately after the current one.
  MachineFunction::iterator BBI = CR.CaseBB;
  ++BBI;

  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  // Create a new basic block to hold the code for loading the address
  // of the jump table, and jumping to it.  Update successor information;
  // we will either branch to the default case for the switch, or the jump
  // table.
  MachineBasicBlock *JumpTableBB = CurMF->CreateMachineBasicBlock(LLVMBB);
  CurMF->insert(BBI, JumpTableBB);

  addSuccessorWithWeight(CR.CaseBB, Default);
  addSuccessorWithWeight(CR.CaseBB, JumpTableBB);

  // Build a vector of destination BBs, corresponding to each target
  // of the jump table. If the value of the jump table slot corresponds to
  // a case statement, push the case's BB onto the vector, otherwise, push
  // the default BB.
  std::vector<MachineBasicBlock*> DestBBs;
  APInt TEI = First;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++TEI) {
    const APInt &Low = cast<ConstantInt>(I->Low)->getValue();
    const APInt &High = cast<ConstantInt>(I->High)->getValue();

    if (Low.ule(TEI) && TEI.ule(High)) {
      DestBBs.push_back(I->BB);
      if (TEI==High)
        ++I;
    } else {
      DestBBs.push_back(Default);
    }
  }

  // Calculate weight for each unique destination in CR.
  DenseMap<MachineBasicBlock*, uint32_t> DestWeights;
  if (FuncInfo.BPI)
    for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++I) {
      DenseMap<MachineBasicBlock*, uint32_t>::iterator Itr =
          DestWeights.find(I->BB);
      if (Itr != DestWeights.end()) 
        Itr->second += I->ExtraWeight;
      else
        DestWeights[I->BB] = I->ExtraWeight;
    }

  // Update successor info. Add one edge to each unique successor.
  BitVector SuccsHandled(CR.CaseBB->getParent()->getNumBlockIDs());
  for (std::vector<MachineBasicBlock*>::iterator I = DestBBs.begin(),
         E = DestBBs.end(); I != E; ++I) {
    if (!SuccsHandled[(*I)->getNumber()]) {
      SuccsHandled[(*I)->getNumber()] = true;
      DenseMap<MachineBasicBlock*, uint32_t>::iterator Itr =
          DestWeights.find(*I);
      addSuccessorWithWeight(JumpTableBB, *I,
                             Itr != DestWeights.end() ? Itr->second : 0);
    }
  }

  // Create a jump table index for this jump table.
  unsigned JTEncoding = TLI.getJumpTableEncoding();
  unsigned JTI = CurMF->getOrCreateJumpTableInfo(JTEncoding)
                       ->createJumpTableIndex(DestBBs);

  // Set the jump table information so that we can codegen it as a second
  // MachineBasicBlock
  JumpTable JT(-1U, JTI, JumpTableBB, Default);
  JumpTableHeader JTH(First, Last, SV, CR.CaseBB, (CR.CaseBB == SwitchBB));
  if (CR.CaseBB == SwitchBB)
    visitJumpTableHeader(JT, JTH, SwitchBB);

  JTCases.push_back(JumpTableBlock(JTH, JT));
  return true;
}

/// handleBTSplitSwitchCase - emit comparison and split binary search tree into
/// 2 subtrees.
bool SelectionDAGBuilder::handleBTSplitSwitchCase(CaseRec& CR,
                                                  CaseRecVector& WorkList,
                                                  const Value* SV,
                                                  MachineBasicBlock *Default,
                                                  MachineBasicBlock *SwitchBB) {
  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = FuncInfo.MF;

  // Figure out which block is immediately after the current one.
  MachineFunction::iterator BBI = CR.CaseBB;
  ++BBI;

  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);
  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  // Size is the number of Cases represented by this range.
  unsigned Size = CR.Range.second - CR.Range.first;

  const APInt &First = cast<ConstantInt>(FrontCase.Low)->getValue();
  const APInt &Last  = cast<ConstantInt>(BackCase.High)->getValue();
  double FMetric = 0;
  CaseItr Pivot = CR.Range.first + Size/2;

  // Select optimal pivot, maximizing sum density of LHS and RHS. This will
  // (heuristically) allow us to emit JumpTable's later.
  APInt TSize(First.getBitWidth(), 0);
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I)
    TSize += I->size();

  APInt LSize = FrontCase.size();
  APInt RSize = TSize-LSize;
  DEBUG(dbgs() << "Selecting best pivot: \n"
               << "First: " << First << ", Last: " << Last <<'\n'
               << "LSize: " << LSize << ", RSize: " << RSize << '\n');
  for (CaseItr I = CR.Range.first, J=I+1, E = CR.Range.second;
       J!=E; ++I, ++J) {
    const APInt &LEnd = cast<ConstantInt>(I->High)->getValue();
    const APInt &RBegin = cast<ConstantInt>(J->Low)->getValue();
    APInt Range = ComputeRange(LEnd, RBegin);
    assert((Range - 2ULL).isNonNegative() &&
           "Invalid case distance");
    // Use volatile double here to avoid excess precision issues on some hosts,
    // e.g. that use 80-bit X87 registers.
    volatile double LDensity =
       (double)LSize.roundToDouble() /
                           (LEnd - First + 1ULL).roundToDouble();
    volatile double RDensity =
      (double)RSize.roundToDouble() /
                           (Last - RBegin + 1ULL).roundToDouble();
    double Metric = Range.logBase2()*(LDensity+RDensity);
    // Should always split in some non-trivial place
    DEBUG(dbgs() <<"=>Step\n"
                 << "LEnd: " << LEnd << ", RBegin: " << RBegin << '\n'
                 << "LDensity: " << LDensity
                 << ", RDensity: " << RDensity << '\n'
                 << "Metric: " << Metric << '\n');
    if (FMetric < Metric) {
      Pivot = J;
      FMetric = Metric;
      DEBUG(dbgs() << "Current metric set to: " << FMetric << '\n');
    }

    LSize += J->size();
    RSize -= J->size();
  }
  if (areJTsAllowed(TLI)) {
    // If our case is dense we *really* should handle it earlier!
    assert((FMetric > 0) && "Should handle dense range earlier!");
  } else {
    Pivot = CR.Range.first + Size/2;
  }

  CaseRange LHSR(CR.Range.first, Pivot);
  CaseRange RHSR(Pivot, CR.Range.second);
  const Constant *C = Pivot->Low;
  MachineBasicBlock *FalseBB = 0, *TrueBB = 0;

  // We know that we branch to the LHS if the Value being switched on is
  // less than the Pivot value, C.  We use this to optimize our binary
  // tree a bit, by recognizing that if SV is greater than or equal to the
  // LHS's Case Value, and that Case Value is exactly one less than the
  // Pivot's Value, then we can branch directly to the LHS's Target,
  // rather than creating a leaf node for it.
  if ((LHSR.second - LHSR.first) == 1 &&
      LHSR.first->High == CR.GE &&
      cast<ConstantInt>(C)->getValue() ==
      (cast<ConstantInt>(CR.GE)->getValue() + 1LL)) {
    TrueBB = LHSR.first->BB;
  } else {
    TrueBB = CurMF->CreateMachineBasicBlock(LLVMBB);
    CurMF->insert(BBI, TrueBB);
    WorkList.push_back(CaseRec(TrueBB, C, CR.GE, LHSR));

    // Put SV in a virtual register to make it available from the new blocks.
    ExportFromCurrentBlock(SV);
  }

  // Similar to the optimization above, if the Value being switched on is
  // known to be less than the Constant CR.LT, and the current Case Value
  // is CR.LT - 1, then we can branch directly to the target block for
  // the current Case Value, rather than emitting a RHS leaf node for it.
  if ((RHSR.second - RHSR.first) == 1 && CR.LT &&
      cast<ConstantInt>(RHSR.first->Low)->getValue() ==
      (cast<ConstantInt>(CR.LT)->getValue() - 1LL)) {
    FalseBB = RHSR.first->BB;
  } else {
    FalseBB = CurMF->CreateMachineBasicBlock(LLVMBB);
    CurMF->insert(BBI, FalseBB);
    WorkList.push_back(CaseRec(FalseBB,CR.LT,C,RHSR));

    // Put SV in a virtual register to make it available from the new blocks.
    ExportFromCurrentBlock(SV);
  }

  // Create a CaseBlock record representing a conditional branch to
  // the LHS node if the value being switched on SV is less than C.
  // Otherwise, branch to LHS.
  CaseBlock CB(ISD::SETULT, SV, C, NULL, TrueBB, FalseBB, CR.CaseBB);

  if (CR.CaseBB == SwitchBB)
    visitSwitchCase(CB, SwitchBB);
  else
    SwitchCases.push_back(CB);

  return true;
}

/// handleBitTestsSwitchCase - if current case range has few destination and
/// range span less, than machine word bitwidth, encode case range into series
/// of masks and emit bit tests with these masks.
bool SelectionDAGBuilder::handleBitTestsSwitchCase(CaseRec& CR,
                                                   CaseRecVector& WorkList,
                                                   const Value* SV,
                                                   MachineBasicBlock* Default,
                                                   MachineBasicBlock *SwitchBB){
  EVT PTy = TLI.getPointerTy();
  unsigned IntPtrBits = PTy.getSizeInBits();

  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = FuncInfo.MF;

  // If target does not have legal shift left, do not emit bit tests at all.
  if (!TLI.isOperationLegal(ISD::SHL, TLI.getPointerTy()))
    return false;

  size_t numCmps = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I) {
    // Single case counts one, case range - two.
    numCmps += (I->Low == I->High ? 1 : 2);
  }

  // Count unique destinations
  SmallSet<MachineBasicBlock*, 4> Dests;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I!=E; ++I) {
    Dests.insert(I->BB);
    if (Dests.size() > 3)
      // Don't bother the code below, if there are too much unique destinations
      return false;
  }
  DEBUG(dbgs() << "Total number of unique destinations: "
        << Dests.size() << '\n'
        << "Total number of comparisons: " << numCmps << '\n');

  // Compute span of values.
  const APInt& minValue = cast<ConstantInt>(FrontCase.Low)->getValue();
  const APInt& maxValue = cast<ConstantInt>(BackCase.High)->getValue();
  APInt cmpRange = maxValue - minValue;

  DEBUG(dbgs() << "Compare range: " << cmpRange << '\n'
               << "Low bound: " << minValue << '\n'
               << "High bound: " << maxValue << '\n');

  if (cmpRange.uge(IntPtrBits) ||
      (!(Dests.size() == 1 && numCmps >= 3) &&
       !(Dests.size() == 2 && numCmps >= 5) &&
       !(Dests.size() >= 3 && numCmps >= 6)))
    return false;

  DEBUG(dbgs() << "Emitting bit tests\n");
  APInt lowBound = APInt::getNullValue(cmpRange.getBitWidth());

  // Optimize the case where all the case values fit in a
  // word without having to subtract minValue. In this case,
  // we can optimize away the subtraction.
  if (maxValue.ult(IntPtrBits)) {
    cmpRange = maxValue;
  } else {
    lowBound = minValue;
  }

  CaseBitsVector CasesBits;
  unsigned i, count = 0;

  for (CaseItr I = CR.Range.first, E = CR.Range.second; I!=E; ++I) {
    MachineBasicBlock* Dest = I->BB;
    for (i = 0; i < count; ++i)
      if (Dest == CasesBits[i].BB)
        break;

    if (i == count) {
      assert((count < 3) && "Too much destinations to test!");
      CasesBits.push_back(CaseBits(0, Dest, 0, 0/*Weight*/));
      count++;
    }

    const APInt& lowValue = cast<ConstantInt>(I->Low)->getValue();
    const APInt& highValue = cast<ConstantInt>(I->High)->getValue();

    uint64_t lo = (lowValue - lowBound).getZExtValue();
    uint64_t hi = (highValue - lowBound).getZExtValue();
    CasesBits[i].ExtraWeight += I->ExtraWeight;

    for (uint64_t j = lo; j <= hi; j++) {
      CasesBits[i].Mask |=  1ULL << j;
      CasesBits[i].Bits++;
    }

  }
  std::sort(CasesBits.begin(), CasesBits.end(), CaseBitsCmp());

  BitTestInfo BTC;

  // Figure out which block is immediately after the current one.
  MachineFunction::iterator BBI = CR.CaseBB;
  ++BBI;

  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  DEBUG(dbgs() << "Cases:\n");
  for (unsigned i = 0, e = CasesBits.size(); i!=e; ++i) {
    DEBUG(dbgs() << "Mask: " << CasesBits[i].Mask
                 << ", Bits: " << CasesBits[i].Bits
                 << ", BB: " << CasesBits[i].BB << '\n');

    MachineBasicBlock *CaseBB = CurMF->CreateMachineBasicBlock(LLVMBB);
    CurMF->insert(BBI, CaseBB);
    BTC.push_back(BitTestCase(CasesBits[i].Mask,
                              CaseBB,
                              CasesBits[i].BB, CasesBits[i].ExtraWeight));

    // Put SV in a virtual register to make it available from the new blocks.
    ExportFromCurrentBlock(SV);
  }

  BitTestBlock BTB(lowBound, cmpRange, SV,
                   -1U, MVT::Other, (CR.CaseBB == SwitchBB),
                   CR.CaseBB, Default, BTC);

  if (CR.CaseBB == SwitchBB)
    visitBitTestHeader(BTB, SwitchBB);

  BitTestCases.push_back(BTB);

  return true;
}

/// Clusterify - Transform simple list of Cases into list of CaseRange's
size_t SelectionDAGBuilder::Clusterify(CaseVector& Cases,
                                       const SwitchInst& SI) {
  
  /// Use a shorter form of declaration, and also
  /// show the we want to use CRSBuilder as Clusterifier.
  typedef IntegersSubsetMapping<MachineBasicBlock> Clusterifier;
  
  Clusterifier TheClusterifier;

  BranchProbabilityInfo *BPI = FuncInfo.BPI;
  // Start with "simple" cases
  for (SwitchInst::ConstCaseIt i = SI.case_begin(), e = SI.case_end();
       i != e; ++i) {
    const BasicBlock *SuccBB = i.getCaseSuccessor();
    MachineBasicBlock *SMBB = FuncInfo.MBBMap[SuccBB];

    TheClusterifier.add(i.getCaseValueEx(), SMBB, 
        BPI ? BPI->getEdgeWeight(SI.getParent(), i.getSuccessorIndex()) : 0);
  }
  
  TheClusterifier.optimize();
  
  size_t numCmps = 0;
  for (Clusterifier::RangeIterator i = TheClusterifier.begin(),
       e = TheClusterifier.end(); i != e; ++i, ++numCmps) {
    Clusterifier::Cluster &C = *i;
    // Update edge weight for the cluster.
    unsigned W = C.first.Weight;

    // FIXME: Currently work with ConstantInt based numbers.
    // Changing it to APInt based is a pretty heavy for this commit.
    Cases.push_back(Case(C.first.getLow().toConstantInt(),
                         C.first.getHigh().toConstantInt(), C.second, W));
    
    if (C.first.getLow() != C.first.getHigh())
    // A range counts double, since it requires two compares.
    ++numCmps;
  }

  return numCmps;
}

void SelectionDAGBuilder::UpdateSplitBlock(MachineBasicBlock *First,
                                           MachineBasicBlock *Last) {
  // Update JTCases.
  for (unsigned i = 0, e = JTCases.size(); i != e; ++i)
    if (JTCases[i].first.HeaderBB == First)
      JTCases[i].first.HeaderBB = Last;

  // Update BitTestCases.
  for (unsigned i = 0, e = BitTestCases.size(); i != e; ++i)
    if (BitTestCases[i].Parent == First)
      BitTestCases[i].Parent = Last;
}

void SelectionDAGBuilder::visitSwitch(const SwitchInst &SI) {
  MachineBasicBlock *SwitchMBB = FuncInfo.MBB;

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineBasicBlock *Default = FuncInfo.MBBMap[SI.getDefaultDest()];

  // If there is only the default destination, branch to it if it is not the
  // next basic block.  Otherwise, just fall through.
  if (!SI.getNumCases()) {
    // Update machine-CFG edges.

    // If this is not a fall-through branch, emit the branch.
    SwitchMBB->addSuccessor(Default);
    if (Default != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, getCurDebugLoc(),
                              MVT::Other, getControlRoot(),
                              DAG.getBasicBlock(Default)));

    return;
  }

  // If there are any non-default case statements, create a vector of Cases
  // representing each one, and sort the vector so that we can efficiently
  // create a binary search tree from them.
  CaseVector Cases;
  size_t numCmps = Clusterify(Cases, SI);
  DEBUG(dbgs() << "Clusterify finished. Total clusters: " << Cases.size()
               << ". Total compares: " << numCmps << '\n');
  (void)numCmps;

  // Get the Value to be switched on and default basic blocks, which will be
  // inserted into CaseBlock records, representing basic blocks in the binary
  // search tree.
  const Value *SV = SI.getCondition();

  // Push the initial CaseRec onto the worklist
  CaseRecVector WorkList;
  WorkList.push_back(CaseRec(SwitchMBB,0,0,
                             CaseRange(Cases.begin(),Cases.end())));

  while (!WorkList.empty()) {
    // Grab a record representing a case range to process off the worklist
    CaseRec CR = WorkList.back();
    WorkList.pop_back();

    if (handleBitTestsSwitchCase(CR, WorkList, SV, Default, SwitchMBB))
      continue;

    // If the range has few cases (two or less) emit a series of specific
    // tests.
    if (handleSmallSwitchRange(CR, WorkList, SV, Default, SwitchMBB))
      continue;

    // If the switch has more than N blocks, and is at least 40% dense, and the
    // target supports indirect branches, then emit a jump table rather than
    // lowering the switch to a binary tree of conditional branches.
    // N defaults to 4 and is controlled via TLS.getMinimumJumpTableEntries().
    if (handleJTSwitchCase(CR, WorkList, SV, Default, SwitchMBB))
      continue;

    // Emit binary tree. We need to pick a pivot, and push left and right ranges
    // onto the worklist. Leafs are handled via handleSmallSwitchRange() call.
    handleBTSplitSwitchCase(CR, WorkList, SV, Default, SwitchMBB);
  }
}

void SelectionDAGBuilder::visitIndirectBr(const IndirectBrInst &I) {
  MachineBasicBlock *IndirectBrMBB = FuncInfo.MBB;

  // Update machine-CFG edges with unique successors.
  SmallSet<BasicBlock*, 32> Done;
  for (unsigned i = 0, e = I.getNumSuccessors(); i != e; ++i) {
    BasicBlock *BB = I.getSuccessor(i);
    bool Inserted = Done.insert(BB);
    if (!Inserted)
        continue;

    MachineBasicBlock *Succ = FuncInfo.MBBMap[BB];
    addSuccessorWithWeight(IndirectBrMBB, Succ);
  }

  DAG.setRoot(DAG.getNode(ISD::BRIND, getCurDebugLoc(),
                          MVT::Other, getControlRoot(),
                          getValue(I.getAddress())));
}

void SelectionDAGBuilder::visitFSub(const User &I) {
  // -0.0 - X --> fneg
  Type *Ty = I.getType();
  if (isa<Constant>(I.getOperand(0)) &&
      I.getOperand(0) == ConstantFP::getZeroValueForNegation(Ty)) {
    SDValue Op2 = getValue(I.getOperand(1));
    setValue(&I, DAG.getNode(ISD::FNEG, getCurDebugLoc(),
                             Op2.getValueType(), Op2));
    return;
  }

  visitBinary(I, ISD::FSUB);
}

void SelectionDAGBuilder::visitBinary(const User &I, unsigned OpCode) {
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));
  setValue(&I, DAG.getNode(OpCode, getCurDebugLoc(),
                           Op1.getValueType(), Op1, Op2));
}

void SelectionDAGBuilder::visitShift(const User &I, unsigned Opcode) {
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));

  MVT ShiftTy = TLI.getShiftAmountTy(Op2.getValueType());

  // Coerce the shift amount to the right type if we can.
  if (!I.getType()->isVectorTy() && Op2.getValueType() != ShiftTy) {
    unsigned ShiftSize = ShiftTy.getSizeInBits();
    unsigned Op2Size = Op2.getValueType().getSizeInBits();
    DebugLoc DL = getCurDebugLoc();

    // If the operand is smaller than the shift count type, promote it.
    if (ShiftSize > Op2Size)
      Op2 = DAG.getNode(ISD::ZERO_EXTEND, DL, ShiftTy, Op2);

    // If the operand is larger than the shift count type but the shift
    // count type has enough bits to represent any shift value, truncate
    // it now. This is a common case and it exposes the truncate to
    // optimization early.
    else if (ShiftSize >= Log2_32_Ceil(Op2.getValueType().getSizeInBits()))
      Op2 = DAG.getNode(ISD::TRUNCATE, DL, ShiftTy, Op2);
    // Otherwise we'll need to temporarily settle for some other convenient
    // type.  Type legalization will make adjustments once the shiftee is split.
    else
      Op2 = DAG.getZExtOrTrunc(Op2, DL, MVT::i32);
  }

  setValue(&I, DAG.getNode(Opcode, getCurDebugLoc(),
                           Op1.getValueType(), Op1, Op2));
}

void SelectionDAGBuilder::visitSDiv(const User &I) {
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));

  // Turn exact SDivs into multiplications.
  // FIXME: This should be in DAGCombiner, but it doesn't have access to the
  // exact bit.
  if (isa<BinaryOperator>(&I) && cast<BinaryOperator>(&I)->isExact() &&
      !isa<ConstantSDNode>(Op1) &&
      isa<ConstantSDNode>(Op2) && !cast<ConstantSDNode>(Op2)->isNullValue())
    setValue(&I, TLI.BuildExactSDIV(Op1, Op2, getCurDebugLoc(), DAG));
  else
    setValue(&I, DAG.getNode(ISD::SDIV, getCurDebugLoc(), Op1.getValueType(),
                             Op1, Op2));
}

void SelectionDAGBuilder::visitICmp(const User &I) {
  ICmpInst::Predicate predicate = ICmpInst::BAD_ICMP_PREDICATE;
  if (const ICmpInst *IC = dyn_cast<ICmpInst>(&I))
    predicate = IC->getPredicate();
  else if (const ConstantExpr *IC = dyn_cast<ConstantExpr>(&I))
    predicate = ICmpInst::Predicate(IC->getPredicate());
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode = getICmpCondCode(predicate);

  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getSetCC(getCurDebugLoc(), DestVT, Op1, Op2, Opcode));
}

void SelectionDAGBuilder::visitFCmp(const User &I) {
  FCmpInst::Predicate predicate = FCmpInst::BAD_FCMP_PREDICATE;
  if (const FCmpInst *FC = dyn_cast<FCmpInst>(&I))
    predicate = FC->getPredicate();
  else if (const ConstantExpr *FC = dyn_cast<ConstantExpr>(&I))
    predicate = FCmpInst::Predicate(FC->getPredicate());
  SDValue Op1 = getValue(I.getOperand(0));
  SDValue Op2 = getValue(I.getOperand(1));
  ISD::CondCode Condition = getFCmpCondCode(predicate);
  if (TM.Options.NoNaNsFPMath)
    Condition = getFCmpCodeWithoutNaN(Condition);
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getSetCC(getCurDebugLoc(), DestVT, Op1, Op2, Condition));
}

void SelectionDAGBuilder::visitSelect(const User &I) {
  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, I.getType(), ValueVTs);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0) return;

  SmallVector<SDValue, 4> Values(NumValues);
  SDValue Cond     = getValue(I.getOperand(0));
  SDValue TrueVal  = getValue(I.getOperand(1));
  SDValue FalseVal = getValue(I.getOperand(2));
  ISD::NodeType OpCode = Cond.getValueType().isVector() ?
    ISD::VSELECT : ISD::SELECT;

  for (unsigned i = 0; i != NumValues; ++i)
    Values[i] = DAG.getNode(OpCode, getCurDebugLoc(),
                            TrueVal.getNode()->getValueType(TrueVal.getResNo()+i),
                            Cond,
                            SDValue(TrueVal.getNode(),
                                    TrueVal.getResNo() + i),
                            SDValue(FalseVal.getNode(),
                                    FalseVal.getResNo() + i));

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurDebugLoc(),
                           DAG.getVTList(&ValueVTs[0], NumValues),
                           &Values[0], NumValues));
}

void SelectionDAGBuilder::visitTrunc(const User &I) {
  // TruncInst cannot be a no-op cast because sizeof(src) > sizeof(dest).
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::TRUNCATE, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitZExt(const User &I) {
  // ZExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // ZExt also can't be a cast to bool for same reason. So, nothing much to do
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitSExt(const User &I) {
  // SExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // SExt also can't be a cast to bool for same reason. So, nothing much to do
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::SIGN_EXTEND, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitFPTrunc(const User &I) {
  // FPTrunc is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_ROUND, getCurDebugLoc(),
                           DestVT, N,
                           DAG.getTargetConstant(0, TLI.getPointerTy())));
}

void SelectionDAGBuilder::visitFPExt(const User &I){
  // FPExt is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_EXTEND, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitFPToUI(const User &I) {
  // FPToUI is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_UINT, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitFPToSI(const User &I) {
  // FPToSI is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_SINT, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitUIToFP(const User &I) {
  // UIToFP is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::UINT_TO_FP, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitSIToFP(const User &I){
  // SIToFP is never a no-op cast, no need to check
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::SINT_TO_FP, getCurDebugLoc(), DestVT, N));
}

void SelectionDAGBuilder::visitPtrToInt(const User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getZExtOrTrunc(N, getCurDebugLoc(), DestVT));
}

void SelectionDAGBuilder::visitIntToPtr(const User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getZExtOrTrunc(N, getCurDebugLoc(), DestVT));
}

void SelectionDAGBuilder::visitBitCast(const User &I) {
  SDValue N = getValue(I.getOperand(0));
  EVT DestVT = TLI.getValueType(I.getType());

  // BitCast assures us that source and destination are the same size so this is
  // either a BITCAST or a no-op.
  if (DestVT != N.getValueType())
    setValue(&I, DAG.getNode(ISD::BITCAST, getCurDebugLoc(),
                             DestVT, N)); // convert types.
  else
    setValue(&I, N);            // noop cast.
}

void SelectionDAGBuilder::visitInsertElement(const User &I) {
  SDValue InVec = getValue(I.getOperand(0));
  SDValue InVal = getValue(I.getOperand(1));
  SDValue InIdx = DAG.getNode(ISD::ZERO_EXTEND, getCurDebugLoc(),
                              TLI.getPointerTy(),
                              getValue(I.getOperand(2)));
  setValue(&I, DAG.getNode(ISD::INSERT_VECTOR_ELT, getCurDebugLoc(),
                           TLI.getValueType(I.getType()),
                           InVec, InVal, InIdx));
}

void SelectionDAGBuilder::visitExtractElement(const User &I) {
  SDValue InVec = getValue(I.getOperand(0));
  SDValue InIdx = DAG.getNode(ISD::ZERO_EXTEND, getCurDebugLoc(),
                              TLI.getPointerTy(),
                              getValue(I.getOperand(1)));
  setValue(&I, DAG.getNode(ISD::EXTRACT_VECTOR_ELT, getCurDebugLoc(),
                           TLI.getValueType(I.getType()), InVec, InIdx));
}

// Utility for visitShuffleVector - Return true if every element in Mask,
// beginning from position Pos and ending in Pos+Size, falls within the
// specified sequential range [L, L+Pos). or is undef.
static bool isSequentialInRange(const SmallVectorImpl<int> &Mask,
                                unsigned Pos, unsigned Size, int Low) {
  for (unsigned i = Pos, e = Pos+Size; i != e; ++i, ++Low)
    if (Mask[i] >= 0 && Mask[i] != Low)
      return false;
  return true;
}

void SelectionDAGBuilder::visitShuffleVector(const User &I) {
  SDValue Src1 = getValue(I.getOperand(0));
  SDValue Src2 = getValue(I.getOperand(1));

  SmallVector<int, 8> Mask;
  ShuffleVectorInst::getShuffleMask(cast<Constant>(I.getOperand(2)), Mask);
  unsigned MaskNumElts = Mask.size();
  
  EVT VT = TLI.getValueType(I.getType());
  EVT SrcVT = Src1.getValueType();
  unsigned SrcNumElts = SrcVT.getVectorNumElements();

  if (SrcNumElts == MaskNumElts) {
    setValue(&I, DAG.getVectorShuffle(VT, getCurDebugLoc(), Src1, Src2,
                                      &Mask[0]));
    return;
  }

  // Normalize the shuffle vector since mask and vector length don't match.
  if (SrcNumElts < MaskNumElts && MaskNumElts % SrcNumElts == 0) {
    // Mask is longer than the source vectors and is a multiple of the source
    // vectors.  We can use concatenate vector to make the mask and vectors
    // lengths match.
    if (SrcNumElts*2 == MaskNumElts) {
      // First check for Src1 in low and Src2 in high
      if (isSequentialInRange(Mask, 0, SrcNumElts, 0) &&
          isSequentialInRange(Mask, SrcNumElts, SrcNumElts, SrcNumElts)) {
        // The shuffle is concatenating two vectors together.
        setValue(&I, DAG.getNode(ISD::CONCAT_VECTORS, getCurDebugLoc(),
                                 VT, Src1, Src2));
        return;
      }
      // Then check for Src2 in low and Src1 in high
      if (isSequentialInRange(Mask, 0, SrcNumElts, SrcNumElts) &&
          isSequentialInRange(Mask, SrcNumElts, SrcNumElts, 0)) {
        // The shuffle is concatenating two vectors together.
        setValue(&I, DAG.getNode(ISD::CONCAT_VECTORS, getCurDebugLoc(),
                                 VT, Src2, Src1));
        return;
      }
    }

    // Pad both vectors with undefs to make them the same length as the mask.
    unsigned NumConcat = MaskNumElts / SrcNumElts;
    bool Src1U = Src1.getOpcode() == ISD::UNDEF;
    bool Src2U = Src2.getOpcode() == ISD::UNDEF;
    SDValue UndefVal = DAG.getUNDEF(SrcVT);

    SmallVector<SDValue, 8> MOps1(NumConcat, UndefVal);
    SmallVector<SDValue, 8> MOps2(NumConcat, UndefVal);
    MOps1[0] = Src1;
    MOps2[0] = Src2;

    Src1 = Src1U ? DAG.getUNDEF(VT) : DAG.getNode(ISD::CONCAT_VECTORS,
                                                  getCurDebugLoc(), VT,
                                                  &MOps1[0], NumConcat);
    Src2 = Src2U ? DAG.getUNDEF(VT) : DAG.getNode(ISD::CONCAT_VECTORS,
                                                  getCurDebugLoc(), VT,
                                                  &MOps2[0], NumConcat);

    // Readjust mask for new input vector length.
    SmallVector<int, 8> MappedOps;
    for (unsigned i = 0; i != MaskNumElts; ++i) {
      int Idx = Mask[i];
      if (Idx >= (int)SrcNumElts)
        Idx -= SrcNumElts - MaskNumElts;
      MappedOps.push_back(Idx);
    }

    setValue(&I, DAG.getVectorShuffle(VT, getCurDebugLoc(), Src1, Src2,
                                      &MappedOps[0]));
    return;
  }

  if (SrcNumElts > MaskNumElts) {
    // Analyze the access pattern of the vector to see if we can extract
    // two subvectors and do the shuffle. The analysis is done by calculating
    // the range of elements the mask access on both vectors.
    int MinRange[2] = { static_cast<int>(SrcNumElts),
                        static_cast<int>(SrcNumElts)};
    int MaxRange[2] = {-1, -1};

    for (unsigned i = 0; i != MaskNumElts; ++i) {
      int Idx = Mask[i];
      unsigned Input = 0;
      if (Idx < 0)
        continue;

      if (Idx >= (int)SrcNumElts) {
        Input = 1;
        Idx -= SrcNumElts;
      }
      if (Idx > MaxRange[Input])
        MaxRange[Input] = Idx;
      if (Idx < MinRange[Input])
        MinRange[Input] = Idx;
    }

    // Check if the access is smaller than the vector size and can we find
    // a reasonable extract index.
    int RangeUse[2] = { -1, -1 };  // 0 = Unused, 1 = Extract, -1 = Can not
                                   // Extract.
    int StartIdx[2];  // StartIdx to extract from
    for (unsigned Input = 0; Input < 2; ++Input) {
      if (MinRange[Input] >= (int)SrcNumElts && MaxRange[Input] < 0) {
        RangeUse[Input] = 0; // Unused
        StartIdx[Input] = 0;
        continue;
      }

      // Find a good start index that is a multiple of the mask length. Then
      // see if the rest of the elements are in range.
      StartIdx[Input] = (MinRange[Input]/MaskNumElts)*MaskNumElts;
      if (MaxRange[Input] - StartIdx[Input] < (int)MaskNumElts &&
          StartIdx[Input] + MaskNumElts <= SrcNumElts)
        RangeUse[Input] = 1; // Extract from a multiple of the mask length.
    }

    if (RangeUse[0] == 0 && RangeUse[1] == 0) {
      setValue(&I, DAG.getUNDEF(VT)); // Vectors are not used.
      return;
    }
    if (RangeUse[0] >= 0 && RangeUse[1] >= 0) {
      // Extract appropriate subvector and generate a vector shuffle
      for (unsigned Input = 0; Input < 2; ++Input) {
        SDValue &Src = Input == 0 ? Src1 : Src2;
        if (RangeUse[Input] == 0)
          Src = DAG.getUNDEF(VT);
        else
          Src = DAG.getNode(ISD::EXTRACT_SUBVECTOR, getCurDebugLoc(), VT,
                            Src, DAG.getIntPtrConstant(StartIdx[Input]));
      }

      // Calculate new mask.
      SmallVector<int, 8> MappedOps;
      for (unsigned i = 0; i != MaskNumElts; ++i) {
        int Idx = Mask[i];
        if (Idx >= 0) {
          if (Idx < (int)SrcNumElts)
            Idx -= StartIdx[0];
          else
            Idx -= SrcNumElts + StartIdx[1] - MaskNumElts;
        }
        MappedOps.push_back(Idx);
      }

      setValue(&I, DAG.getVectorShuffle(VT, getCurDebugLoc(), Src1, Src2,
                                        &MappedOps[0]));
      return;
    }
  }

  // We can't use either concat vectors or extract subvectors so fall back to
  // replacing the shuffle with extract and build vector.
  // to insert and build vector.
  EVT EltVT = VT.getVectorElementType();
  EVT PtrVT = TLI.getPointerTy();
  SmallVector<SDValue,8> Ops;
  for (unsigned i = 0; i != MaskNumElts; ++i) {
    int Idx = Mask[i];
    SDValue Res;

    if (Idx < 0) {
      Res = DAG.getUNDEF(EltVT);
    } else {
      SDValue &Src = Idx < (int)SrcNumElts ? Src1 : Src2;
      if (Idx >= (int)SrcNumElts) Idx -= SrcNumElts;

      Res = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, getCurDebugLoc(),
                        EltVT, Src, DAG.getConstant(Idx, PtrVT));
    }

    Ops.push_back(Res);
  }

  setValue(&I, DAG.getNode(ISD::BUILD_VECTOR, getCurDebugLoc(),
                           VT, &Ops[0], Ops.size()));
}

void SelectionDAGBuilder::visitInsertValue(const InsertValueInst &I) {
  const Value *Op0 = I.getOperand(0);
  const Value *Op1 = I.getOperand(1);
  Type *AggTy = I.getType();
  Type *ValTy = Op1->getType();
  bool IntoUndef = isa<UndefValue>(Op0);
  bool FromUndef = isa<UndefValue>(Op1);

  unsigned LinearIndex = ComputeLinearIndex(AggTy, I.getIndices());

  SmallVector<EVT, 4> AggValueVTs;
  ComputeValueVTs(TLI, AggTy, AggValueVTs);
  SmallVector<EVT, 4> ValValueVTs;
  ComputeValueVTs(TLI, ValTy, ValValueVTs);

  unsigned NumAggValues = AggValueVTs.size();
  unsigned NumValValues = ValValueVTs.size();
  SmallVector<SDValue, 4> Values(NumAggValues);

  SDValue Agg = getValue(Op0);
  unsigned i = 0;
  // Copy the beginning value(s) from the original aggregate.
  for (; i != LinearIndex; ++i)
    Values[i] = IntoUndef ? DAG.getUNDEF(AggValueVTs[i]) :
                SDValue(Agg.getNode(), Agg.getResNo() + i);
  // Copy values from the inserted value(s).
  if (NumValValues) {
    SDValue Val = getValue(Op1);
    for (; i != LinearIndex + NumValValues; ++i)
      Values[i] = FromUndef ? DAG.getUNDEF(AggValueVTs[i]) :
                  SDValue(Val.getNode(), Val.getResNo() + i - LinearIndex);
  }
  // Copy remaining value(s) from the original aggregate.
  for (; i != NumAggValues; ++i)
    Values[i] = IntoUndef ? DAG.getUNDEF(AggValueVTs[i]) :
                SDValue(Agg.getNode(), Agg.getResNo() + i);

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurDebugLoc(),
                           DAG.getVTList(&AggValueVTs[0], NumAggValues),
                           &Values[0], NumAggValues));
}

void SelectionDAGBuilder::visitExtractValue(const ExtractValueInst &I) {
  const Value *Op0 = I.getOperand(0);
  Type *AggTy = Op0->getType();
  Type *ValTy = I.getType();
  bool OutOfUndef = isa<UndefValue>(Op0);

  unsigned LinearIndex = ComputeLinearIndex(AggTy, I.getIndices());

  SmallVector<EVT, 4> ValValueVTs;
  ComputeValueVTs(TLI, ValTy, ValValueVTs);

  unsigned NumValValues = ValValueVTs.size();

  // Ignore a extractvalue that produces an empty object
  if (!NumValValues) {
    setValue(&I, DAG.getUNDEF(MVT(MVT::Other)));
    return;
  }

  SmallVector<SDValue, 4> Values(NumValValues);

  SDValue Agg = getValue(Op0);
  // Copy out the selected value(s).
  for (unsigned i = LinearIndex; i != LinearIndex + NumValValues; ++i)
    Values[i - LinearIndex] =
      OutOfUndef ?
        DAG.getUNDEF(Agg.getNode()->getValueType(Agg.getResNo() + i)) :
        SDValue(Agg.getNode(), Agg.getResNo() + i);

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurDebugLoc(),
                           DAG.getVTList(&ValValueVTs[0], NumValValues),
                           &Values[0], NumValValues));
}

void SelectionDAGBuilder::visitGetElementPtr(const User &I) {
  SDValue N = getValue(I.getOperand(0));
  // Note that the pointer operand may be a vector of pointers. Take the scalar
  // element which holds a pointer.
  Type *Ty = I.getOperand(0)->getType()->getScalarType();

  for (GetElementPtrInst::const_op_iterator OI = I.op_begin()+1, E = I.op_end();
       OI != E; ++OI) {
    const Value *Idx = *OI;
    if (StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offset = TD->getStructLayout(StTy)->getElementOffset(Field);
        N = DAG.getNode(ISD::ADD, getCurDebugLoc(), N.getValueType(), N,
                        DAG.getIntPtrConstant(Offset));
      }

      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // If this is a constant subscript, handle it quickly.
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->isZero()) continue;
        uint64_t Offs =
            TD->getTypeAllocSize(Ty)*cast<ConstantInt>(CI)->getSExtValue();
        SDValue OffsVal;
        EVT PTy = TLI.getPointerTy();
        unsigned PtrBits = PTy.getSizeInBits();
        if (PtrBits < 64)
          OffsVal = DAG.getNode(ISD::TRUNCATE, getCurDebugLoc(),
                                TLI.getPointerTy(),
                                DAG.getConstant(Offs, MVT::i64));
        else
          OffsVal = DAG.getIntPtrConstant(Offs);

        N = DAG.getNode(ISD::ADD, getCurDebugLoc(), N.getValueType(), N,
                        OffsVal);
        continue;
      }

      // N = N + Idx * ElementSize;
      APInt ElementSize = APInt(TLI.getPointerTy().getSizeInBits(),
                                TD->getTypeAllocSize(Ty));
      SDValue IdxN = getValue(Idx);

      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      IdxN = DAG.getSExtOrTrunc(IdxN, getCurDebugLoc(), N.getValueType());

      // If this is a multiply by a power of two, turn it into a shl
      // immediately.  This is a very common case.
      if (ElementSize != 1) {
        if (ElementSize.isPowerOf2()) {
          unsigned Amt = ElementSize.logBase2();
          IdxN = DAG.getNode(ISD::SHL, getCurDebugLoc(),
                             N.getValueType(), IdxN,
                             DAG.getConstant(Amt, IdxN.getValueType()));
        } else {
          SDValue Scale = DAG.getConstant(ElementSize, TLI.getPointerTy());
          IdxN = DAG.getNode(ISD::MUL, getCurDebugLoc(),
                             N.getValueType(), IdxN, Scale);
        }
      }

      N = DAG.getNode(ISD::ADD, getCurDebugLoc(),
                      N.getValueType(), N, IdxN);
    }
  }

  setValue(&I, N);
}

void SelectionDAGBuilder::visitAlloca(const AllocaInst &I) {
  // If this is a fixed sized alloca in the entry block of the function,
  // allocate it statically on the stack.
  if (FuncInfo.StaticAllocaMap.count(&I))
    return;   // getValue will auto-populate this.

  Type *Ty = I.getAllocatedType();
  uint64_t TySize = TLI.getDataLayout()->getTypeAllocSize(Ty);
  unsigned Align =
    std::max((unsigned)TLI.getDataLayout()->getPrefTypeAlignment(Ty),
             I.getAlignment());

  SDValue AllocSize = getValue(I.getArraySize());

  EVT IntPtr = TLI.getPointerTy();
  if (AllocSize.getValueType() != IntPtr)
    AllocSize = DAG.getZExtOrTrunc(AllocSize, getCurDebugLoc(), IntPtr);

  AllocSize = DAG.getNode(ISD::MUL, getCurDebugLoc(), IntPtr,
                          AllocSize,
                          DAG.getConstant(TySize, IntPtr));

  // Handle alignment.  If the requested alignment is less than or equal to
  // the stack alignment, ignore it.  If the size is greater than or equal to
  // the stack alignment, we note this in the DYNAMIC_STACKALLOC node.
  unsigned StackAlign = TM.getFrameLowering()->getStackAlignment();
  if (Align <= StackAlign)
    Align = 0;

  // Round the size of the allocation up to the stack alignment size
  // by add SA-1 to the size.
  AllocSize = DAG.getNode(ISD::ADD, getCurDebugLoc(),
                          AllocSize.getValueType(), AllocSize,
                          DAG.getIntPtrConstant(StackAlign-1));

  // Mask out the low bits for alignment purposes.
  AllocSize = DAG.getNode(ISD::AND, getCurDebugLoc(),
                          AllocSize.getValueType(), AllocSize,
                          DAG.getIntPtrConstant(~(uint64_t)(StackAlign-1)));

  SDValue Ops[] = { getRoot(), AllocSize, DAG.getIntPtrConstant(Align) };
  SDVTList VTs = DAG.getVTList(AllocSize.getValueType(), MVT::Other);
  SDValue DSA = DAG.getNode(ISD::DYNAMIC_STACKALLOC, getCurDebugLoc(),
                            VTs, Ops, 3);
  setValue(&I, DSA);
  DAG.setRoot(DSA.getValue(1));

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  FuncInfo.MF->getFrameInfo()->CreateVariableSizedObject(Align ? Align : 1);
}

void SelectionDAGBuilder::visitLoad(const LoadInst &I) {
  if (I.isAtomic())
    return visitAtomicLoad(I);

  const Value *SV = I.getOperand(0);
  SDValue Ptr = getValue(SV);

  Type *Ty = I.getType();

  bool isVolatile = I.isVolatile();
  bool isNonTemporal = I.getMetadata("nontemporal") != 0;
  bool isInvariant = I.getMetadata("invariant.load") != 0;
  unsigned Alignment = I.getAlignment();
  const MDNode *TBAAInfo = I.getMetadata(LLVMContext::MD_tbaa);
  const MDNode *Ranges = I.getMetadata(LLVMContext::MD_range);

  SmallVector<EVT, 4> ValueVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, Ty, ValueVTs, &Offsets);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0)
    return;

  SDValue Root;
  bool ConstantMemory = false;
  if (I.isVolatile() || NumValues > MaxParallelChains)
    // Serialize volatile loads with other side effects.
    Root = getRoot();
  else if (AA->pointsToConstantMemory(
             AliasAnalysis::Location(SV, AA->getTypeStoreSize(Ty), TBAAInfo))) {
    // Do not serialize (non-volatile) loads of constant memory with anything.
    Root = DAG.getEntryNode();
    ConstantMemory = true;
  } else {
    // Do not serialize non-volatile loads against each other.
    Root = DAG.getRoot();
  }

  SmallVector<SDValue, 4> Values(NumValues);
  SmallVector<SDValue, 4> Chains(std::min(unsigned(MaxParallelChains),
                                          NumValues));
  EVT PtrVT = Ptr.getValueType();
  unsigned ChainI = 0;
  for (unsigned i = 0; i != NumValues; ++i, ++ChainI) {
    // Serializing loads here may result in excessive register pressure, and
    // TokenFactor places arbitrary choke points on the scheduler. SD scheduling
    // could recover a bit by hoisting nodes upward in the chain by recognizing
    // they are side-effect free or do not alias. The optimizer should really
    // avoid this case by converting large object/array copies to llvm.memcpy
    // (MaxParallelChains should always remain as failsafe).
    if (ChainI == MaxParallelChains) {
      assert(PendingLoads.empty() && "PendingLoads must be serialized first");
      SDValue Chain = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(),
                                  MVT::Other, &Chains[0], ChainI);
      Root = Chain;
      ChainI = 0;
    }
    SDValue A = DAG.getNode(ISD::ADD, getCurDebugLoc(),
                            PtrVT, Ptr,
                            DAG.getConstant(Offsets[i], PtrVT));
    SDValue L = DAG.getLoad(ValueVTs[i], getCurDebugLoc(), Root,
                            A, MachinePointerInfo(SV, Offsets[i]), isVolatile,
                            isNonTemporal, isInvariant, Alignment, TBAAInfo,
                            Ranges);

    Values[i] = L;
    Chains[ChainI] = L.getValue(1);
  }

  if (!ConstantMemory) {
    SDValue Chain = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(),
                                MVT::Other, &Chains[0], ChainI);
    if (isVolatile)
      DAG.setRoot(Chain);
    else
      PendingLoads.push_back(Chain);
  }

  setValue(&I, DAG.getNode(ISD::MERGE_VALUES, getCurDebugLoc(),
                           DAG.getVTList(&ValueVTs[0], NumValues),
                           &Values[0], NumValues));
}

void SelectionDAGBuilder::visitStore(const StoreInst &I) {
  if (I.isAtomic())
    return visitAtomicStore(I);

  const Value *SrcV = I.getOperand(0);
  const Value *PtrV = I.getOperand(1);

  SmallVector<EVT, 4> ValueVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, SrcV->getType(), ValueVTs, &Offsets);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0)
    return;

  // Get the lowered operands. Note that we do this after
  // checking if NumResults is zero, because with zero results
  // the operands won't have values in the map.
  SDValue Src = getValue(SrcV);
  SDValue Ptr = getValue(PtrV);

  SDValue Root = getRoot();
  SmallVector<SDValue, 4> Chains(std::min(unsigned(MaxParallelChains),
                                          NumValues));
  EVT PtrVT = Ptr.getValueType();
  bool isVolatile = I.isVolatile();
  bool isNonTemporal = I.getMetadata("nontemporal") != 0;
  unsigned Alignment = I.getAlignment();
  const MDNode *TBAAInfo = I.getMetadata(LLVMContext::MD_tbaa);

  unsigned ChainI = 0;
  for (unsigned i = 0; i != NumValues; ++i, ++ChainI) {
    // See visitLoad comments.
    if (ChainI == MaxParallelChains) {
      SDValue Chain = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(),
                                  MVT::Other, &Chains[0], ChainI);
      Root = Chain;
      ChainI = 0;
    }
    SDValue Add = DAG.getNode(ISD::ADD, getCurDebugLoc(), PtrVT, Ptr,
                              DAG.getConstant(Offsets[i], PtrVT));
    SDValue St = DAG.getStore(Root, getCurDebugLoc(),
                              SDValue(Src.getNode(), Src.getResNo() + i),
                              Add, MachinePointerInfo(PtrV, Offsets[i]),
                              isVolatile, isNonTemporal, Alignment, TBAAInfo);
    Chains[ChainI] = St;
  }

  SDValue StoreNode = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(),
                                  MVT::Other, &Chains[0], ChainI);
  ++SDNodeOrder;
  AssignOrderingToNode(StoreNode.getNode());
  DAG.setRoot(StoreNode);
}

static SDValue InsertFenceForAtomic(SDValue Chain, AtomicOrdering Order,
                                    SynchronizationScope Scope,
                                    bool Before, DebugLoc dl,
                                    SelectionDAG &DAG,
                                    const TargetLowering &TLI) {
  // Fence, if necessary
  if (Before) {
    if (Order == AcquireRelease || Order == SequentiallyConsistent)
      Order = Release;
    else if (Order == Acquire || Order == Monotonic)
      return Chain;
  } else {
    if (Order == AcquireRelease)
      Order = Acquire;
    else if (Order == Release || Order == Monotonic)
      return Chain;
  }
  SDValue Ops[3];
  Ops[0] = Chain;
  Ops[1] = DAG.getConstant(Order, TLI.getPointerTy());
  Ops[2] = DAG.getConstant(Scope, TLI.getPointerTy());
  return DAG.getNode(ISD::ATOMIC_FENCE, dl, MVT::Other, Ops, 3);
}

void SelectionDAGBuilder::visitAtomicCmpXchg(const AtomicCmpXchgInst &I) {
  DebugLoc dl = getCurDebugLoc();
  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  if (TLI.getInsertFencesForAtomic())
    InChain = InsertFenceForAtomic(InChain, Order, Scope, true, dl,
                                   DAG, TLI);

  SDValue L =
    DAG.getAtomic(ISD::ATOMIC_CMP_SWAP, dl,
                  getValue(I.getCompareOperand()).getValueType().getSimpleVT(),
                  InChain,
                  getValue(I.getPointerOperand()),
                  getValue(I.getCompareOperand()),
                  getValue(I.getNewValOperand()),
                  MachinePointerInfo(I.getPointerOperand()), 0 /* Alignment */,
                  TLI.getInsertFencesForAtomic() ? Monotonic : Order,
                  Scope);

  SDValue OutChain = L.getValue(1);

  if (TLI.getInsertFencesForAtomic())
    OutChain = InsertFenceForAtomic(OutChain, Order, Scope, false, dl,
                                    DAG, TLI);

  setValue(&I, L);
  DAG.setRoot(OutChain);
}

void SelectionDAGBuilder::visitAtomicRMW(const AtomicRMWInst &I) {
  DebugLoc dl = getCurDebugLoc();
  ISD::NodeType NT;
  switch (I.getOperation()) {
  default: llvm_unreachable("Unknown atomicrmw operation");
  case AtomicRMWInst::Xchg: NT = ISD::ATOMIC_SWAP; break;
  case AtomicRMWInst::Add:  NT = ISD::ATOMIC_LOAD_ADD; break;
  case AtomicRMWInst::Sub:  NT = ISD::ATOMIC_LOAD_SUB; break;
  case AtomicRMWInst::And:  NT = ISD::ATOMIC_LOAD_AND; break;
  case AtomicRMWInst::Nand: NT = ISD::ATOMIC_LOAD_NAND; break;
  case AtomicRMWInst::Or:   NT = ISD::ATOMIC_LOAD_OR; break;
  case AtomicRMWInst::Xor:  NT = ISD::ATOMIC_LOAD_XOR; break;
  case AtomicRMWInst::Max:  NT = ISD::ATOMIC_LOAD_MAX; break;
  case AtomicRMWInst::Min:  NT = ISD::ATOMIC_LOAD_MIN; break;
  case AtomicRMWInst::UMax: NT = ISD::ATOMIC_LOAD_UMAX; break;
  case AtomicRMWInst::UMin: NT = ISD::ATOMIC_LOAD_UMIN; break;
  }
  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  if (TLI.getInsertFencesForAtomic())
    InChain = InsertFenceForAtomic(InChain, Order, Scope, true, dl,
                                   DAG, TLI);

  SDValue L =
    DAG.getAtomic(NT, dl,
                  getValue(I.getValOperand()).getValueType().getSimpleVT(),
                  InChain,
                  getValue(I.getPointerOperand()),
                  getValue(I.getValOperand()),
                  I.getPointerOperand(), 0 /* Alignment */,
                  TLI.getInsertFencesForAtomic() ? Monotonic : Order,
                  Scope);

  SDValue OutChain = L.getValue(1);

  if (TLI.getInsertFencesForAtomic())
    OutChain = InsertFenceForAtomic(OutChain, Order, Scope, false, dl,
                                    DAG, TLI);

  setValue(&I, L);
  DAG.setRoot(OutChain);
}

void SelectionDAGBuilder::visitFence(const FenceInst &I) {
  DebugLoc dl = getCurDebugLoc();
  SDValue Ops[3];
  Ops[0] = getRoot();
  Ops[1] = DAG.getConstant(I.getOrdering(), TLI.getPointerTy());
  Ops[2] = DAG.getConstant(I.getSynchScope(), TLI.getPointerTy());
  DAG.setRoot(DAG.getNode(ISD::ATOMIC_FENCE, dl, MVT::Other, Ops, 3));
}

void SelectionDAGBuilder::visitAtomicLoad(const LoadInst &I) {
  DebugLoc dl = getCurDebugLoc();
  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  EVT VT = TLI.getValueType(I.getType());

  if (I.getAlignment() * 8 < VT.getSizeInBits())
    report_fatal_error("Cannot generate unaligned atomic load");

  SDValue L =
    DAG.getAtomic(ISD::ATOMIC_LOAD, dl, VT, VT, InChain,
                  getValue(I.getPointerOperand()),
                  I.getPointerOperand(), I.getAlignment(),
                  TLI.getInsertFencesForAtomic() ? Monotonic : Order,
                  Scope);

  SDValue OutChain = L.getValue(1);

  if (TLI.getInsertFencesForAtomic())
    OutChain = InsertFenceForAtomic(OutChain, Order, Scope, false, dl,
                                    DAG, TLI);

  setValue(&I, L);
  DAG.setRoot(OutChain);
}

void SelectionDAGBuilder::visitAtomicStore(const StoreInst &I) {
  DebugLoc dl = getCurDebugLoc();

  AtomicOrdering Order = I.getOrdering();
  SynchronizationScope Scope = I.getSynchScope();

  SDValue InChain = getRoot();

  EVT VT = TLI.getValueType(I.getValueOperand()->getType());

  if (I.getAlignment() * 8 < VT.getSizeInBits())
    report_fatal_error("Cannot generate unaligned atomic store");

  if (TLI.getInsertFencesForAtomic())
    InChain = InsertFenceForAtomic(InChain, Order, Scope, true, dl,
                                   DAG, TLI);

  SDValue OutChain =
    DAG.getAtomic(ISD::ATOMIC_STORE, dl, VT,
                  InChain,
                  getValue(I.getPointerOperand()),
                  getValue(I.getValueOperand()),
                  I.getPointerOperand(), I.getAlignment(),
                  TLI.getInsertFencesForAtomic() ? Monotonic : Order,
                  Scope);

  if (TLI.getInsertFencesForAtomic())
    OutChain = InsertFenceForAtomic(OutChain, Order, Scope, false, dl,
                                    DAG, TLI);

  DAG.setRoot(OutChain);
}

/// visitTargetIntrinsic - Lower a call of a target intrinsic to an INTRINSIC
/// node.
void SelectionDAGBuilder::visitTargetIntrinsic(const CallInst &I,
                                               unsigned Intrinsic) {
  bool HasChain = !I.doesNotAccessMemory();
  bool OnlyLoad = HasChain && I.onlyReadsMemory();

  // Build the operand list.
  SmallVector<SDValue, 8> Ops;
  if (HasChain) {  // If this intrinsic has side-effects, chainify it.
    if (OnlyLoad) {
      // We don't need to serialize loads against other loads.
      Ops.push_back(DAG.getRoot());
    } else {
      Ops.push_back(getRoot());
    }
  }

  // Info is set by getTgtMemInstrinsic
  TargetLowering::IntrinsicInfo Info;
  bool IsTgtIntrinsic = TLI.getTgtMemIntrinsic(Info, I, Intrinsic);

  // Add the intrinsic ID as an integer operand if it's not a target intrinsic.
  if (!IsTgtIntrinsic || Info.opc == ISD::INTRINSIC_VOID ||
      Info.opc == ISD::INTRINSIC_W_CHAIN)
    Ops.push_back(DAG.getTargetConstant(Intrinsic, TLI.getPointerTy()));

  // Add all operands of the call to the operand list.
  for (unsigned i = 0, e = I.getNumArgOperands(); i != e; ++i) {
    SDValue Op = getValue(I.getArgOperand(i));
    Ops.push_back(Op);
  }

  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, I.getType(), ValueVTs);

  if (HasChain)
    ValueVTs.push_back(MVT::Other);

  SDVTList VTs = DAG.getVTList(ValueVTs.data(), ValueVTs.size());

  // Create the node.
  SDValue Result;
  if (IsTgtIntrinsic) {
    // This is target intrinsic that touches memory
    Result = DAG.getMemIntrinsicNode(Info.opc, getCurDebugLoc(),
                                     VTs, &Ops[0], Ops.size(),
                                     Info.memVT,
                                   MachinePointerInfo(Info.ptrVal, Info.offset),
                                     Info.align, Info.vol,
                                     Info.readMem, Info.writeMem);
  } else if (!HasChain) {
    Result = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, getCurDebugLoc(),
                         VTs, &Ops[0], Ops.size());
  } else if (!I.getType()->isVoidTy()) {
    Result = DAG.getNode(ISD::INTRINSIC_W_CHAIN, getCurDebugLoc(),
                         VTs, &Ops[0], Ops.size());
  } else {
    Result = DAG.getNode(ISD::INTRINSIC_VOID, getCurDebugLoc(),
                         VTs, &Ops[0], Ops.size());
  }

  if (HasChain) {
    SDValue Chain = Result.getValue(Result.getNode()->getNumValues()-1);
    if (OnlyLoad)
      PendingLoads.push_back(Chain);
    else
      DAG.setRoot(Chain);
  }

  if (!I.getType()->isVoidTy()) {
    if (VectorType *PTy = dyn_cast<VectorType>(I.getType())) {
      EVT VT = TLI.getValueType(PTy);
      Result = DAG.getNode(ISD::BITCAST, getCurDebugLoc(), VT, Result);
    }

    setValue(&I, Result);
  } else {
    // Assign order to result here. If the intrinsic does not produce a result,
    // it won't be mapped to a SDNode and visit() will not assign it an order
    // number.
    ++SDNodeOrder;
    AssignOrderingToNode(Result.getNode());
  }
}

/// GetSignificand - Get the significand and build it into a floating-point
/// number with exponent of 1:
///
///   Op = (Op & 0x007fffff) | 0x3f800000;
///
/// where Op is the hexidecimal representation of floating point value.
static SDValue
GetSignificand(SelectionDAG &DAG, SDValue Op, DebugLoc dl) {
  SDValue t1 = DAG.getNode(ISD::AND, dl, MVT::i32, Op,
                           DAG.getConstant(0x007fffff, MVT::i32));
  SDValue t2 = DAG.getNode(ISD::OR, dl, MVT::i32, t1,
                           DAG.getConstant(0x3f800000, MVT::i32));
  return DAG.getNode(ISD::BITCAST, dl, MVT::f32, t2);
}

/// GetExponent - Get the exponent:
///
///   (float)(int)(((Op & 0x7f800000) >> 23) - 127);
///
/// where Op is the hexidecimal representation of floating point value.
static SDValue
GetExponent(SelectionDAG &DAG, SDValue Op, const TargetLowering &TLI,
            DebugLoc dl) {
  SDValue t0 = DAG.getNode(ISD::AND, dl, MVT::i32, Op,
                           DAG.getConstant(0x7f800000, MVT::i32));
  SDValue t1 = DAG.getNode(ISD::SRL, dl, MVT::i32, t0,
                           DAG.getConstant(23, TLI.getPointerTy()));
  SDValue t2 = DAG.getNode(ISD::SUB, dl, MVT::i32, t1,
                           DAG.getConstant(127, MVT::i32));
  return DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f32, t2);
}

/// getF32Constant - Get 32-bit floating point constant.
static SDValue
getF32Constant(SelectionDAG &DAG, unsigned Flt) {
  return DAG.getConstantFP(APFloat(APInt(32, Flt)), MVT::f32);
}

/// visitExp - Lower an exp intrinsic. Handles the special sequences for
/// limited-precision mode.
void
SelectionDAGBuilder::visitExp(const CallInst &I) {
  SDValue result;
  DebugLoc dl = getCurDebugLoc();

  if (getValue(I.getArgOperand(0)).getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op = getValue(I.getArgOperand(0));

    // Put the exponent in the right bit position for later addition to the
    // final result:
    //
    //   #define LOG2OFe 1.4426950f
    //   IntegerPartOfX = ((int32_t)(X * LOG2OFe));
    SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, Op,
                             getF32Constant(DAG, 0x3fb8aa3b));
    SDValue IntegerPartOfX = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::i32, t0);

    //   FractionalPartOfX = (X * LOG2OFe) - (float)IntegerPartOfX;
    SDValue t1 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f32, IntegerPartOfX);
    SDValue X = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0, t1);

    //   IntegerPartOfX <<= 23;
    IntegerPartOfX = DAG.getNode(ISD::SHL, dl, MVT::i32, IntegerPartOfX,
                                 DAG.getConstant(23, TLI.getPointerTy()));

    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   TwoToFractionalPartOfX =
      //     0.997535578f +
      //       (0.735607626f + 0.252464424f * x) * x;
      //
      // error 0.0144103317, which is 6 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3e814304));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f3c50c8));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f7f5e7e));
      SDValue TwoToFracPartOfX = DAG.getNode(ISD::BITCAST, dl,MVT::i32, t5);

      // Add the exponent into the result in integer domain.
      SDValue t6 = DAG.getNode(ISD::ADD, dl, MVT::i32,
                               TwoToFracPartOfX, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl, MVT::f32, t6);
    } else if (LimitFloatPrecision > 6 && LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   TwoToFractionalPartOfX =
      //     0.999892986f +
      //       (0.696457318f +
      //         (0.224338339f + 0.792043434e-1f * x) * x) * x;
      //
      // 0.000107046256 error, which is 13 to 14 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3da235e3));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3e65b8f3));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f324b07));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3f7ff8fd));
      SDValue TwoToFracPartOfX = DAG.getNode(ISD::BITCAST, dl,MVT::i32, t7);

      // Add the exponent into the result in integer domain.
      SDValue t8 = DAG.getNode(ISD::ADD, dl, MVT::i32,
                               TwoToFracPartOfX, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl, MVT::f32, t8);
    } else { // LimitFloatPrecision > 12 && LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   TwoToFractionalPartOfX =
      //     0.999999982f +
      //       (0.693148872f +
      //         (0.240227044f +
      //           (0.554906021e-1f +
      //             (0.961591928e-2f +
      //               (0.136028312e-2f + 0.157059148e-3f *x)*x)*x)*x)*x)*x;
      //
      // error 2.47208000*10^(-7), which is better than 18 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3924b03e));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3ab24b87));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3c1d8c17));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3d634a1d));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x3e75fe14));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      SDValue t11 = DAG.getNode(ISD::FADD, dl, MVT::f32, t10,
                                getF32Constant(DAG, 0x3f317234));
      SDValue t12 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t11, X);
      SDValue t13 = DAG.getNode(ISD::FADD, dl, MVT::f32, t12,
                                getF32Constant(DAG, 0x3f800000));
      SDValue TwoToFracPartOfX = DAG.getNode(ISD::BITCAST, dl,
                                             MVT::i32, t13);

      // Add the exponent into the result in integer domain.
      SDValue t14 = DAG.getNode(ISD::ADD, dl, MVT::i32,
                                TwoToFracPartOfX, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl, MVT::f32, t14);
    }
  } else {
    // No special expansion.
    result = DAG.getNode(ISD::FEXP, dl,
                         getValue(I.getArgOperand(0)).getValueType(),
                         getValue(I.getArgOperand(0)));
  }

  setValue(&I, result);
}

/// visitLog - Lower a log intrinsic. Handles the special sequences for
/// limited-precision mode.
void
SelectionDAGBuilder::visitLog(const CallInst &I) {
  SDValue result;
  DebugLoc dl = getCurDebugLoc();

  if (getValue(I.getArgOperand(0)).getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op = getValue(I.getArgOperand(0));
    SDValue Op1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);

    // Scale the exponent by log(2) [0.69314718f].
    SDValue Exp = GetExponent(DAG, Op1, TLI, dl);
    SDValue LogOfExponent = DAG.getNode(ISD::FMUL, dl, MVT::f32, Exp,
                                        getF32Constant(DAG, 0x3f317218));

    // Get the significand and build it into a floating-point number with
    // exponent of 1.
    SDValue X = GetSignificand(DAG, Op1, dl);

    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   LogofMantissa =
      //     -1.1609546f +
      //       (1.4034025f - 0.23903021f * x) * x;
      //
      // error 0.0034276066, which is better than 8 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbe74c456));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3fb3a2b1));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue LogOfMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                                          getF32Constant(DAG, 0x3f949a29));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, LogOfMantissa);
    } else if (LimitFloatPrecision > 6 && LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   LogOfMantissa =
      //     -1.7417939f +
      //       (2.8212026f +
      //         (-1.4699568f +
      //           (0.44717955f - 0.56570851e-1f * x) * x) * x) * x;
      //
      // error 0.000061011436, which is 14 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbd67b6d6));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3ee4f4b8));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3fbc278b));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x40348e95));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue LogOfMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                                          getF32Constant(DAG, 0x3fdef31a));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, LogOfMantissa);
    } else { // LimitFloatPrecision > 12 && LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   LogOfMantissa =
      //     -2.1072184f +
      //       (4.2372794f +
      //         (-3.7029485f +
      //           (2.2781945f +
      //             (-0.87823314f +
      //               (0.19073739f - 0.17809712e-1f * x) * x) * x) * x) * x)*x;
      //
      // error 0.0000023660568, which is better than 18 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbc91e5ac));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3e4350aa));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f60d3e3));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x4011cdf0));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x406cfd1c));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x408797cb));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      SDValue LogOfMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t10,
                                          getF32Constant(DAG, 0x4006dcab));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, LogOfMantissa);
    }
  } else {
    // No special expansion.
    result = DAG.getNode(ISD::FLOG, dl,
                         getValue(I.getArgOperand(0)).getValueType(),
                         getValue(I.getArgOperand(0)));
  }

  setValue(&I, result);
}

/// visitLog2 - Lower a log2 intrinsic. Handles the special sequences for
/// limited-precision mode.
void
SelectionDAGBuilder::visitLog2(const CallInst &I) {
  SDValue result;
  DebugLoc dl = getCurDebugLoc();

  if (getValue(I.getArgOperand(0)).getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op = getValue(I.getArgOperand(0));
    SDValue Op1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);

    // Get the exponent.
    SDValue LogOfExponent = GetExponent(DAG, Op1, TLI, dl);

    // Get the significand and build it into a floating-point number with
    // exponent of 1.
    SDValue X = GetSignificand(DAG, Op1, dl);

    // Different possible minimax approximations of significand in
    // floating-point for various degrees of accuracy over [1,2].
    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   Log2ofMantissa = -1.6749035f + (2.0246817f - .34484768f * x) * x;
      //
      // error 0.0049451742, which is more than 7 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbeb08fe0));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x40019463));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue Log2ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                                           getF32Constant(DAG, 0x3fd6633d));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, Log2ofMantissa);
    } else if (LimitFloatPrecision > 6 && LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   Log2ofMantissa =
      //     -2.51285454f +
      //       (4.07009056f +
      //         (-2.12067489f +
      //           (.645142248f - 0.816157886e-1f * x) * x) * x) * x;
      //
      // error 0.0000876136000, which is better than 13 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbda7262e));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3f25280b));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x4007b923));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x40823e2f));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue Log2ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                                           getF32Constant(DAG, 0x4020d29c));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, Log2ofMantissa);
    } else { // LimitFloatPrecision > 12 && LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   Log2ofMantissa =
      //     -3.0400495f +
      //       (6.1129976f +
      //         (-5.3420409f +
      //           (3.2865683f +
      //             (-1.2669343f +
      //               (0.27515199f -
      //                 0.25691327e-1f * x) * x) * x) * x) * x) * x;
      //
      // error 0.0000018516, which is better than 18 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbcd2769e));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3e8ce0b9));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3fa22ae7));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x40525723));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x40aaf200));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x40c39dad));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      SDValue Log2ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t10,
                                           getF32Constant(DAG, 0x4042902c));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, Log2ofMantissa);
    }
  } else {
    // No special expansion.
    result = DAG.getNode(ISD::FLOG2, dl,
                         getValue(I.getArgOperand(0)).getValueType(),
                         getValue(I.getArgOperand(0)));
  }

  setValue(&I, result);
}

/// visitLog10 - Lower a log10 intrinsic. Handles the special sequences for
/// limited-precision mode.
void
SelectionDAGBuilder::visitLog10(const CallInst &I) {
  SDValue result;
  DebugLoc dl = getCurDebugLoc();

  if (getValue(I.getArgOperand(0)).getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op = getValue(I.getArgOperand(0));
    SDValue Op1 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);

    // Scale the exponent by log10(2) [0.30102999f].
    SDValue Exp = GetExponent(DAG, Op1, TLI, dl);
    SDValue LogOfExponent = DAG.getNode(ISD::FMUL, dl, MVT::f32, Exp,
                                        getF32Constant(DAG, 0x3e9a209a));

    // Get the significand and build it into a floating-point number with
    // exponent of 1.
    SDValue X = GetSignificand(DAG, Op1, dl);

    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   Log10ofMantissa =
      //     -0.50419619f +
      //       (0.60948995f - 0.10380950f * x) * x;
      //
      // error 0.0014886165, which is 6 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0xbdd49a13));
      SDValue t1 = DAG.getNode(ISD::FADD, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3f1c0789));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue Log10ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t2,
                                            getF32Constant(DAG, 0x3f011300));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, Log10ofMantissa);
    } else if (LimitFloatPrecision > 6 && LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   Log10ofMantissa =
      //     -0.64831180f +
      //       (0.91751397f +
      //         (-0.31664806f + 0.47637168e-1f * x) * x) * x;
      //
      // error 0.00019228036, which is better than 12 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3d431f31));
      SDValue t1 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3ea21fb2));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f6ae232));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue Log10ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t4,
                                            getF32Constant(DAG, 0x3f25f7c3));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, Log10ofMantissa);
    } else { // LimitFloatPrecision > 12 && LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   Log10ofMantissa =
      //     -0.84299375f +
      //       (1.5327582f +
      //         (-1.0688956f +
      //           (0.49102474f +
      //             (-0.12539807f + 0.13508273e-1f * x) * x) * x) * x) * x;
      //
      // error 0.0000037995730, which is better than 18 bits
      SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3c5d51ce));
      SDValue t1 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0,
                               getF32Constant(DAG, 0x3e00685a));
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t1, X);
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3efb6798));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FSUB, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f88d192));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3fc4316c));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue Log10ofMantissa = DAG.getNode(ISD::FSUB, dl, MVT::f32, t8,
                                            getF32Constant(DAG, 0x3f57ce70));

      result = DAG.getNode(ISD::FADD, dl,
                           MVT::f32, LogOfExponent, Log10ofMantissa);
    }
  } else {
    // No special expansion.
    result = DAG.getNode(ISD::FLOG10, dl,
                         getValue(I.getArgOperand(0)).getValueType(),
                         getValue(I.getArgOperand(0)));
  }

  setValue(&I, result);
}

/// visitExp2 - Lower an exp2 intrinsic. Handles the special sequences for
/// limited-precision mode.
void
SelectionDAGBuilder::visitExp2(const CallInst &I) {
  SDValue result;
  DebugLoc dl = getCurDebugLoc();

  if (getValue(I.getArgOperand(0)).getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op = getValue(I.getArgOperand(0));

    SDValue IntegerPartOfX = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::i32, Op);

    //   FractionalPartOfX = x - (float)IntegerPartOfX;
    SDValue t1 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f32, IntegerPartOfX);
    SDValue X = DAG.getNode(ISD::FSUB, dl, MVT::f32, Op, t1);

    //   IntegerPartOfX <<= 23;
    IntegerPartOfX = DAG.getNode(ISD::SHL, dl, MVT::i32, IntegerPartOfX,
                                 DAG.getConstant(23, TLI.getPointerTy()));

    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   TwoToFractionalPartOfX =
      //     0.997535578f +
      //       (0.735607626f + 0.252464424f * x) * x;
      //
      // error 0.0144103317, which is 6 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3e814304));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f3c50c8));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f7f5e7e));
      SDValue t6 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, t5);
      SDValue TwoToFractionalPartOfX =
        DAG.getNode(ISD::ADD, dl, MVT::i32, t6, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl,
                           MVT::f32, TwoToFractionalPartOfX);
    } else if (LimitFloatPrecision > 6 && LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   TwoToFractionalPartOfX =
      //     0.999892986f +
      //       (0.696457318f +
      //         (0.224338339f + 0.792043434e-1f * x) * x) * x;
      //
      // error 0.000107046256, which is 13 to 14 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3da235e3));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3e65b8f3));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f324b07));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3f7ff8fd));
      SDValue t8 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, t7);
      SDValue TwoToFractionalPartOfX =
        DAG.getNode(ISD::ADD, dl, MVT::i32, t8, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl,
                           MVT::f32, TwoToFractionalPartOfX);
    } else { // LimitFloatPrecision > 12 && LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   TwoToFractionalPartOfX =
      //     0.999999982f +
      //       (0.693148872f +
      //         (0.240227044f +
      //           (0.554906021e-1f +
      //             (0.961591928e-2f +
      //               (0.136028312e-2f + 0.157059148e-3f *x)*x)*x)*x)*x)*x;
      // error 2.47208000*10^(-7), which is better than 18 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3924b03e));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3ab24b87));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3c1d8c17));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3d634a1d));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x3e75fe14));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      SDValue t11 = DAG.getNode(ISD::FADD, dl, MVT::f32, t10,
                                getF32Constant(DAG, 0x3f317234));
      SDValue t12 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t11, X);
      SDValue t13 = DAG.getNode(ISD::FADD, dl, MVT::f32, t12,
                                getF32Constant(DAG, 0x3f800000));
      SDValue t14 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, t13);
      SDValue TwoToFractionalPartOfX =
        DAG.getNode(ISD::ADD, dl, MVT::i32, t14, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl,
                           MVT::f32, TwoToFractionalPartOfX);
    }
  } else {
    // No special expansion.
    result = DAG.getNode(ISD::FEXP2, dl,
                         getValue(I.getArgOperand(0)).getValueType(),
                         getValue(I.getArgOperand(0)));
  }

  setValue(&I, result);
}

/// visitPow - Lower a pow intrinsic. Handles the special sequences for
/// limited-precision mode with x == 10.0f.
void
SelectionDAGBuilder::visitPow(const CallInst &I) {
  SDValue result;
  const Value *Val = I.getArgOperand(0);
  DebugLoc dl = getCurDebugLoc();
  bool IsExp10 = false;

  if (getValue(Val).getValueType() == MVT::f32 &&
      getValue(I.getArgOperand(1)).getValueType() == MVT::f32 &&
      LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(Val))) {
      if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
        APFloat Ten(10.0f);
        IsExp10 = CFP->getValueAPF().bitwiseIsEqual(Ten);
      }
    }
  }

  if (IsExp10 && LimitFloatPrecision > 0 && LimitFloatPrecision <= 18) {
    SDValue Op = getValue(I.getArgOperand(1));

    // Put the exponent in the right bit position for later addition to the
    // final result:
    //
    //   #define LOG2OF10 3.3219281f
    //   IntegerPartOfX = (int32_t)(x * LOG2OF10);
    SDValue t0 = DAG.getNode(ISD::FMUL, dl, MVT::f32, Op,
                             getF32Constant(DAG, 0x40549a78));
    SDValue IntegerPartOfX = DAG.getNode(ISD::FP_TO_SINT, dl, MVT::i32, t0);

    //   FractionalPartOfX = x - (float)IntegerPartOfX;
    SDValue t1 = DAG.getNode(ISD::SINT_TO_FP, dl, MVT::f32, IntegerPartOfX);
    SDValue X = DAG.getNode(ISD::FSUB, dl, MVT::f32, t0, t1);

    //   IntegerPartOfX <<= 23;
    IntegerPartOfX = DAG.getNode(ISD::SHL, dl, MVT::i32, IntegerPartOfX,
                                 DAG.getConstant(23, TLI.getPointerTy()));

    if (LimitFloatPrecision <= 6) {
      // For floating-point precision of 6:
      //
      //   twoToFractionalPartOfX =
      //     0.997535578f +
      //       (0.735607626f + 0.252464424f * x) * x;
      //
      // error 0.0144103317, which is 6 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3e814304));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3f3c50c8));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f7f5e7e));
      SDValue t6 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, t5);
      SDValue TwoToFractionalPartOfX =
        DAG.getNode(ISD::ADD, dl, MVT::i32, t6, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl,
                           MVT::f32, TwoToFractionalPartOfX);
    } else if (LimitFloatPrecision > 6 && LimitFloatPrecision <= 12) {
      // For floating-point precision of 12:
      //
      //   TwoToFractionalPartOfX =
      //     0.999892986f +
      //       (0.696457318f +
      //         (0.224338339f + 0.792043434e-1f * x) * x) * x;
      //
      // error 0.000107046256, which is 13 to 14 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3da235e3));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3e65b8f3));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3f324b07));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3f7ff8fd));
      SDValue t8 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, t7);
      SDValue TwoToFractionalPartOfX =
        DAG.getNode(ISD::ADD, dl, MVT::i32, t8, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl,
                           MVT::f32, TwoToFractionalPartOfX);
    } else { // LimitFloatPrecision > 12 && LimitFloatPrecision <= 18
      // For floating-point precision of 18:
      //
      //   TwoToFractionalPartOfX =
      //     0.999999982f +
      //       (0.693148872f +
      //         (0.240227044f +
      //           (0.554906021e-1f +
      //             (0.961591928e-2f +
      //               (0.136028312e-2f + 0.157059148e-3f *x)*x)*x)*x)*x)*x;
      // error 2.47208000*10^(-7), which is better than 18 bits
      SDValue t2 = DAG.getNode(ISD::FMUL, dl, MVT::f32, X,
                               getF32Constant(DAG, 0x3924b03e));
      SDValue t3 = DAG.getNode(ISD::FADD, dl, MVT::f32, t2,
                               getF32Constant(DAG, 0x3ab24b87));
      SDValue t4 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t3, X);
      SDValue t5 = DAG.getNode(ISD::FADD, dl, MVT::f32, t4,
                               getF32Constant(DAG, 0x3c1d8c17));
      SDValue t6 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t5, X);
      SDValue t7 = DAG.getNode(ISD::FADD, dl, MVT::f32, t6,
                               getF32Constant(DAG, 0x3d634a1d));
      SDValue t8 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t7, X);
      SDValue t9 = DAG.getNode(ISD::FADD, dl, MVT::f32, t8,
                               getF32Constant(DAG, 0x3e75fe14));
      SDValue t10 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t9, X);
      SDValue t11 = DAG.getNode(ISD::FADD, dl, MVT::f32, t10,
                                getF32Constant(DAG, 0x3f317234));
      SDValue t12 = DAG.getNode(ISD::FMUL, dl, MVT::f32, t11, X);
      SDValue t13 = DAG.getNode(ISD::FADD, dl, MVT::f32, t12,
                                getF32Constant(DAG, 0x3f800000));
      SDValue t14 = DAG.getNode(ISD::BITCAST, dl, MVT::i32, t13);
      SDValue TwoToFractionalPartOfX =
        DAG.getNode(ISD::ADD, dl, MVT::i32, t14, IntegerPartOfX);

      result = DAG.getNode(ISD::BITCAST, dl,
                           MVT::f32, TwoToFractionalPartOfX);
    }
  } else {
    // No special expansion.
    result = DAG.getNode(ISD::FPOW, dl,
                         getValue(I.getArgOperand(0)).getValueType(),
                         getValue(I.getArgOperand(0)),
                         getValue(I.getArgOperand(1)));
  }

  setValue(&I, result);
}


/// ExpandPowI - Expand a llvm.powi intrinsic.
static SDValue ExpandPowI(DebugLoc DL, SDValue LHS, SDValue RHS,
                          SelectionDAG &DAG) {
  // If RHS is a constant, we can expand this out to a multiplication tree,
  // otherwise we end up lowering to a call to __powidf2 (for example).  When
  // optimizing for size, we only want to do this if the expansion would produce
  // a small number of multiplies, otherwise we do the full expansion.
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS)) {
    // Get the exponent as a positive value.
    unsigned Val = RHSC->getSExtValue();
    if ((int)Val < 0) Val = -Val;

    // powi(x, 0) -> 1.0
    if (Val == 0)
      return DAG.getConstantFP(1.0, LHS.getValueType());

    const Function *F = DAG.getMachineFunction().getFunction();
    if (!F->getFnAttributes().hasAttribute(Attributes::OptimizeForSize) ||
        // If optimizing for size, don't insert too many multiplies.  This
        // inserts up to 5 multiplies.
        CountPopulation_32(Val)+Log2_32(Val) < 7) {
      // We use the simple binary decomposition method to generate the multiply
      // sequence.  There are more optimal ways to do this (for example,
      // powi(x,15) generates one more multiply than it should), but this has
      // the benefit of being both really simple and much better than a libcall.
      SDValue Res;  // Logically starts equal to 1.0
      SDValue CurSquare = LHS;
      while (Val) {
        if (Val & 1) {
          if (Res.getNode())
            Res = DAG.getNode(ISD::FMUL, DL,Res.getValueType(), Res, CurSquare);
          else
            Res = CurSquare;  // 1.0*CurSquare.
        }

        CurSquare = DAG.getNode(ISD::FMUL, DL, CurSquare.getValueType(),
                                CurSquare, CurSquare);
        Val >>= 1;
      }

      // If the original was negative, invert the result, producing 1/(x*x*x).
      if (RHSC->getSExtValue() < 0)
        Res = DAG.getNode(ISD::FDIV, DL, LHS.getValueType(),
                          DAG.getConstantFP(1.0, LHS.getValueType()), Res);
      return Res;
    }
  }

  // Otherwise, expand to a libcall.
  return DAG.getNode(ISD::FPOWI, DL, LHS.getValueType(), LHS, RHS);
}

// getTruncatedArgReg - Find underlying register used for an truncated
// argument.
static unsigned getTruncatedArgReg(const SDValue &N) {
  if (N.getOpcode() != ISD::TRUNCATE)
    return 0;

  const SDValue &Ext = N.getOperand(0);
  if (Ext.getOpcode() == ISD::AssertZext || Ext.getOpcode() == ISD::AssertSext){
    const SDValue &CFR = Ext.getOperand(0);
    if (CFR.getOpcode() == ISD::CopyFromReg)
      return cast<RegisterSDNode>(CFR.getOperand(1))->getReg();
    if (CFR.getOpcode() == ISD::TRUNCATE)
      return getTruncatedArgReg(CFR);
  }
  return 0;
}

/// EmitFuncArgumentDbgValue - If the DbgValueInst is a dbg_value of a function
/// argument, create the corresponding DBG_VALUE machine instruction for it now.
/// At the end of instruction selection, they will be inserted to the entry BB.
bool
SelectionDAGBuilder::EmitFuncArgumentDbgValue(const Value *V, MDNode *Variable,
                                              int64_t Offset,
                                              const SDValue &N) {
  const Argument *Arg = dyn_cast<Argument>(V);
  if (!Arg)
    return false;

  MachineFunction &MF = DAG.getMachineFunction();
  const TargetInstrInfo *TII = DAG.getTarget().getInstrInfo();
  const TargetRegisterInfo *TRI = DAG.getTarget().getRegisterInfo();

  // Ignore inlined function arguments here.
  DIVariable DV(Variable);
  if (DV.isInlinedFnArgument(MF.getFunction()))
    return false;

  unsigned Reg = 0;
  // Some arguments' frame index is recorded during argument lowering.
  Offset = FuncInfo.getArgumentFrameIndex(Arg);
  if (Offset)
    Reg = TRI->getFrameRegister(MF);

  if (!Reg && N.getNode()) {
    if (N.getOpcode() == ISD::CopyFromReg)
      Reg = cast<RegisterSDNode>(N.getOperand(1))->getReg();
    else
      Reg = getTruncatedArgReg(N);
    if (Reg && TargetRegisterInfo::isVirtualRegister(Reg)) {
      MachineRegisterInfo &RegInfo = MF.getRegInfo();
      unsigned PR = RegInfo.getLiveInPhysReg(Reg);
      if (PR)
        Reg = PR;
    }
  }

  if (!Reg) {
    // Check if ValueMap has reg number.
    DenseMap<const Value *, unsigned>::iterator VMI = FuncInfo.ValueMap.find(V);
    if (VMI != FuncInfo.ValueMap.end())
      Reg = VMI->second;
  }

  if (!Reg && N.getNode()) {
    // Check if frame index is available.
    if (LoadSDNode *LNode = dyn_cast<LoadSDNode>(N.getNode()))
      if (FrameIndexSDNode *FINode =
          dyn_cast<FrameIndexSDNode>(LNode->getBasePtr().getNode())) {
        Reg = TRI->getFrameRegister(MF);
        Offset = FINode->getIndex();
      }
  }

  if (!Reg)
    return false;

  MachineInstrBuilder MIB = BuildMI(MF, getCurDebugLoc(),
                                    TII->get(TargetOpcode::DBG_VALUE))
    .addReg(Reg, RegState::Debug).addImm(Offset).addMetadata(Variable);
  FuncInfo.ArgDbgValues.push_back(&*MIB);
  return true;
}

// VisualStudio defines setjmp as _setjmp
#if defined(_MSC_VER) && defined(setjmp) && \
                         !defined(setjmp_undefined_for_msvc)
#  pragma push_macro("setjmp")
#  undef setjmp
#  define setjmp_undefined_for_msvc
#endif

/// visitIntrinsicCall - Lower the call to the specified intrinsic function.  If
/// we want to emit this as a call to a named external function, return the name
/// otherwise lower it and return null.
const char *
SelectionDAGBuilder::visitIntrinsicCall(const CallInst &I, unsigned Intrinsic) {
  DebugLoc dl = getCurDebugLoc();
  SDValue Res;

  switch (Intrinsic) {
  default:
    // By default, turn this into a target intrinsic node.
    visitTargetIntrinsic(I, Intrinsic);
    return 0;
  case Intrinsic::vastart:  visitVAStart(I); return 0;
  case Intrinsic::vaend:    visitVAEnd(I); return 0;
  case Intrinsic::vacopy:   visitVACopy(I); return 0;
  case Intrinsic::returnaddress:
    setValue(&I, DAG.getNode(ISD::RETURNADDR, dl, TLI.getPointerTy(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::frameaddress:
    setValue(&I, DAG.getNode(ISD::FRAMEADDR, dl, TLI.getPointerTy(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::setjmp:
    return &"_setjmp"[!TLI.usesUnderscoreSetJmp()];
  case Intrinsic::longjmp:
    return &"_longjmp"[!TLI.usesUnderscoreLongJmp()];
  case Intrinsic::memcpy: {
    // Assert for address < 256 since we support only user defined address
    // spaces.
    assert(cast<PointerType>(I.getArgOperand(0)->getType())->getAddressSpace()
           < 256 &&
           cast<PointerType>(I.getArgOperand(1)->getType())->getAddressSpace()
           < 256 &&
           "Unknown address space");
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));
    SDValue Op3 = getValue(I.getArgOperand(2));
    unsigned Align = cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();
    bool isVol = cast<ConstantInt>(I.getArgOperand(4))->getZExtValue();
    DAG.setRoot(DAG.getMemcpy(getRoot(), dl, Op1, Op2, Op3, Align, isVol, false,
                              MachinePointerInfo(I.getArgOperand(0)),
                              MachinePointerInfo(I.getArgOperand(1))));
    return 0;
  }
  case Intrinsic::memset: {
    // Assert for address < 256 since we support only user defined address
    // spaces.
    assert(cast<PointerType>(I.getArgOperand(0)->getType())->getAddressSpace()
           < 256 &&
           "Unknown address space");
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));
    SDValue Op3 = getValue(I.getArgOperand(2));
    unsigned Align = cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();
    bool isVol = cast<ConstantInt>(I.getArgOperand(4))->getZExtValue();
    DAG.setRoot(DAG.getMemset(getRoot(), dl, Op1, Op2, Op3, Align, isVol,
                              MachinePointerInfo(I.getArgOperand(0))));
    return 0;
  }
  case Intrinsic::memmove: {
    // Assert for address < 256 since we support only user defined address
    // spaces.
    assert(cast<PointerType>(I.getArgOperand(0)->getType())->getAddressSpace()
           < 256 &&
           cast<PointerType>(I.getArgOperand(1)->getType())->getAddressSpace()
           < 256 &&
           "Unknown address space");
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));
    SDValue Op3 = getValue(I.getArgOperand(2));
    unsigned Align = cast<ConstantInt>(I.getArgOperand(3))->getZExtValue();
    bool isVol = cast<ConstantInt>(I.getArgOperand(4))->getZExtValue();
    DAG.setRoot(DAG.getMemmove(getRoot(), dl, Op1, Op2, Op3, Align, isVol,
                               MachinePointerInfo(I.getArgOperand(0)),
                               MachinePointerInfo(I.getArgOperand(1))));
    return 0;
  }
  case Intrinsic::dbg_declare: {
    const DbgDeclareInst &DI = cast<DbgDeclareInst>(I);
    MDNode *Variable = DI.getVariable();
    const Value *Address = DI.getAddress();
    if (!Address || !DIVariable(Variable).Verify()) {
      DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      return 0;
    }

    // Build an entry in DbgOrdering.  Debug info input nodes get an SDNodeOrder
    // but do not always have a corresponding SDNode built.  The SDNodeOrder
    // absolute, but not relative, values are different depending on whether
    // debug info exists.
    ++SDNodeOrder;

    // Check if address has undef value.
    if (isa<UndefValue>(Address) ||
        (Address->use_empty() && !isa<Argument>(Address))) {
      DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      return 0;
    }

    SDValue &N = NodeMap[Address];
    if (!N.getNode() && isa<Argument>(Address))
      // Check unused arguments map.
      N = UnusedArgNodeMap[Address];
    SDDbgValue *SDV;
    if (N.getNode()) {
      if (const BitCastInst *BCI = dyn_cast<BitCastInst>(Address))
        Address = BCI->getOperand(0);
      // Parameters are handled specially.
      bool isParameter =
        (DIVariable(Variable).getTag() == dwarf::DW_TAG_arg_variable ||
         isa<Argument>(Address));

      const AllocaInst *AI = dyn_cast<AllocaInst>(Address);

      if (isParameter && !AI) {
        FrameIndexSDNode *FINode = dyn_cast<FrameIndexSDNode>(N.getNode());
        if (FINode)
          // Byval parameter.  We have a frame index at this point.
          SDV = DAG.getDbgValue(Variable, FINode->getIndex(),
                                0, dl, SDNodeOrder);
        else {
          // Address is an argument, so try to emit its dbg value using
          // virtual register info from the FuncInfo.ValueMap.
          EmitFuncArgumentDbgValue(Address, Variable, 0, N);
          return 0;
        }
      } else if (AI)
        SDV = DAG.getDbgValue(Variable, N.getNode(), N.getResNo(),
                              0, dl, SDNodeOrder);
      else {
        // Can't do anything with other non-AI cases yet.
        DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
        DEBUG(dbgs() << "non-AllocaInst issue for Address: \n\t");
        DEBUG(Address->dump());
        return 0;
      }
      DAG.AddDbgValue(SDV, N.getNode(), isParameter);
    } else {
      // If Address is an argument then try to emit its dbg value using
      // virtual register info from the FuncInfo.ValueMap.
      if (!EmitFuncArgumentDbgValue(Address, Variable, 0, N)) {
        // If variable is pinned by a alloca in dominating bb then
        // use StaticAllocaMap.
        if (const AllocaInst *AI = dyn_cast<AllocaInst>(Address)) {
          if (AI->getParent() != DI.getParent()) {
            DenseMap<const AllocaInst*, int>::iterator SI =
              FuncInfo.StaticAllocaMap.find(AI);
            if (SI != FuncInfo.StaticAllocaMap.end()) {
              SDV = DAG.getDbgValue(Variable, SI->second,
                                    0, dl, SDNodeOrder);
              DAG.AddDbgValue(SDV, 0, false);
              return 0;
            }
          }
        }
        DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      }
    }
    return 0;
  }
  case Intrinsic::dbg_value: {
    const DbgValueInst &DI = cast<DbgValueInst>(I);
    if (!DIVariable(DI.getVariable()).Verify())
      return 0;

    MDNode *Variable = DI.getVariable();
    uint64_t Offset = DI.getOffset();
    const Value *V = DI.getValue();
    if (!V)
      return 0;

    // Build an entry in DbgOrdering.  Debug info input nodes get an SDNodeOrder
    // but do not always have a corresponding SDNode built.  The SDNodeOrder
    // absolute, but not relative, values are different depending on whether
    // debug info exists.
    ++SDNodeOrder;
    SDDbgValue *SDV;
    if (isa<ConstantInt>(V) || isa<ConstantFP>(V) || isa<UndefValue>(V)) {
      SDV = DAG.getDbgValue(Variable, V, Offset, dl, SDNodeOrder);
      DAG.AddDbgValue(SDV, 0, false);
    } else {
      // Do not use getValue() in here; we don't want to generate code at
      // this point if it hasn't been done yet.
      SDValue N = NodeMap[V];
      if (!N.getNode() && isa<Argument>(V))
        // Check unused arguments map.
        N = UnusedArgNodeMap[V];
      if (N.getNode()) {
        if (!EmitFuncArgumentDbgValue(V, Variable, Offset, N)) {
          SDV = DAG.getDbgValue(Variable, N.getNode(),
                                N.getResNo(), Offset, dl, SDNodeOrder);
          DAG.AddDbgValue(SDV, N.getNode(), false);
        }
      } else if (!V->use_empty() ) {
        // Do not call getValue(V) yet, as we don't want to generate code.
        // Remember it for later.
        DanglingDebugInfo DDI(&DI, dl, SDNodeOrder);
        DanglingDebugInfoMap[V] = DDI;
      } else {
        // We may expand this to cover more cases.  One case where we have no
        // data available is an unreferenced parameter.
        DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      }
    }

    // Build a debug info table entry.
    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(V))
      V = BCI->getOperand(0);
    const AllocaInst *AI = dyn_cast<AllocaInst>(V);
    // Don't handle byval struct arguments or VLAs, for example.
    if (!AI) {
      DEBUG(dbgs() << "Dropping debug location info for:\n  " << DI << "\n");
      DEBUG(dbgs() << "  Last seen at:\n    " << *V << "\n");
      return 0;
    }
    DenseMap<const AllocaInst*, int>::iterator SI =
      FuncInfo.StaticAllocaMap.find(AI);
    if (SI == FuncInfo.StaticAllocaMap.end())
      return 0; // VLAs.
    int FI = SI->second;

    MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
    if (!DI.getDebugLoc().isUnknown() && MMI.hasDebugInfo())
      MMI.setVariableDbgInfo(Variable, FI, DI.getDebugLoc());
    return 0;
  }

  case Intrinsic::eh_typeid_for: {
    // Find the type id for the given typeinfo.
    GlobalVariable *GV = ExtractTypeInfo(I.getArgOperand(0));
    unsigned TypeID = DAG.getMachineFunction().getMMI().getTypeIDFor(GV);
    Res = DAG.getConstant(TypeID, MVT::i32);
    setValue(&I, Res);
    return 0;
  }

  case Intrinsic::eh_return_i32:
  case Intrinsic::eh_return_i64:
    DAG.getMachineFunction().getMMI().setCallsEHReturn(true);
    DAG.setRoot(DAG.getNode(ISD::EH_RETURN, dl,
                            MVT::Other,
                            getControlRoot(),
                            getValue(I.getArgOperand(0)),
                            getValue(I.getArgOperand(1))));
    return 0;
  case Intrinsic::eh_unwind_init:
    DAG.getMachineFunction().getMMI().setCallsUnwindInit(true);
    return 0;
  case Intrinsic::eh_dwarf_cfa: {
    SDValue CfaArg = DAG.getSExtOrTrunc(getValue(I.getArgOperand(0)), dl,
                                        TLI.getPointerTy());
    SDValue Offset = DAG.getNode(ISD::ADD, dl,
                                 TLI.getPointerTy(),
                                 DAG.getNode(ISD::FRAME_TO_ARGS_OFFSET, dl,
                                             TLI.getPointerTy()),
                                 CfaArg);
    SDValue FA = DAG.getNode(ISD::FRAMEADDR, dl,
                             TLI.getPointerTy(),
                             DAG.getConstant(0, TLI.getPointerTy()));
    setValue(&I, DAG.getNode(ISD::ADD, dl, TLI.getPointerTy(),
                             FA, Offset));
    return 0;
  }
  case Intrinsic::eh_sjlj_callsite: {
    MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
    ConstantInt *CI = dyn_cast<ConstantInt>(I.getArgOperand(0));
    assert(CI && "Non-constant call site value in eh.sjlj.callsite!");
    assert(MMI.getCurrentCallSite() == 0 && "Overlapping call sites!");

    MMI.setCurrentCallSite(CI->getZExtValue());
    return 0;
  }
  case Intrinsic::eh_sjlj_functioncontext: {
    // Get and store the index of the function context.
    MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
    AllocaInst *FnCtx =
      cast<AllocaInst>(I.getArgOperand(0)->stripPointerCasts());
    int FI = FuncInfo.StaticAllocaMap[FnCtx];
    MFI->setFunctionContextIndex(FI);
    return 0;
  }
  case Intrinsic::eh_sjlj_setjmp: {
    SDValue Ops[2];
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getArgOperand(0));
    SDValue Op = DAG.getNode(ISD::EH_SJLJ_SETJMP, dl,
                             DAG.getVTList(MVT::i32, MVT::Other),
                             Ops, 2);
    setValue(&I, Op.getValue(0));
    DAG.setRoot(Op.getValue(1));
    return 0;
  }
  case Intrinsic::eh_sjlj_longjmp: {
    DAG.setRoot(DAG.getNode(ISD::EH_SJLJ_LONGJMP, dl, MVT::Other,
                            getRoot(), getValue(I.getArgOperand(0))));
    return 0;
  }

  case Intrinsic::x86_mmx_pslli_w:
  case Intrinsic::x86_mmx_pslli_d:
  case Intrinsic::x86_mmx_pslli_q:
  case Intrinsic::x86_mmx_psrli_w:
  case Intrinsic::x86_mmx_psrli_d:
  case Intrinsic::x86_mmx_psrli_q:
  case Intrinsic::x86_mmx_psrai_w:
  case Intrinsic::x86_mmx_psrai_d: {
    SDValue ShAmt = getValue(I.getArgOperand(1));
    if (isa<ConstantSDNode>(ShAmt)) {
      visitTargetIntrinsic(I, Intrinsic);
      return 0;
    }
    unsigned NewIntrinsic = 0;
    EVT ShAmtVT = MVT::v2i32;
    switch (Intrinsic) {
    case Intrinsic::x86_mmx_pslli_w:
      NewIntrinsic = Intrinsic::x86_mmx_psll_w;
      break;
    case Intrinsic::x86_mmx_pslli_d:
      NewIntrinsic = Intrinsic::x86_mmx_psll_d;
      break;
    case Intrinsic::x86_mmx_pslli_q:
      NewIntrinsic = Intrinsic::x86_mmx_psll_q;
      break;
    case Intrinsic::x86_mmx_psrli_w:
      NewIntrinsic = Intrinsic::x86_mmx_psrl_w;
      break;
    case Intrinsic::x86_mmx_psrli_d:
      NewIntrinsic = Intrinsic::x86_mmx_psrl_d;
      break;
    case Intrinsic::x86_mmx_psrli_q:
      NewIntrinsic = Intrinsic::x86_mmx_psrl_q;
      break;
    case Intrinsic::x86_mmx_psrai_w:
      NewIntrinsic = Intrinsic::x86_mmx_psra_w;
      break;
    case Intrinsic::x86_mmx_psrai_d:
      NewIntrinsic = Intrinsic::x86_mmx_psra_d;
      break;
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    }

    // The vector shift intrinsics with scalars uses 32b shift amounts but
    // the sse2/mmx shift instructions reads 64 bits. Set the upper 32 bits
    // to be zero.
    // We must do this early because v2i32 is not a legal type.
    DebugLoc dl = getCurDebugLoc();
    SDValue ShOps[2];
    ShOps[0] = ShAmt;
    ShOps[1] = DAG.getConstant(0, MVT::i32);
    ShAmt =  DAG.getNode(ISD::BUILD_VECTOR, dl, ShAmtVT, &ShOps[0], 2);
    EVT DestVT = TLI.getValueType(I.getType());
    ShAmt = DAG.getNode(ISD::BITCAST, dl, DestVT, ShAmt);
    Res = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, DestVT,
                       DAG.getConstant(NewIntrinsic, MVT::i32),
                       getValue(I.getArgOperand(0)), ShAmt);
    setValue(&I, Res);
    return 0;
  }
  case Intrinsic::x86_avx_vinsertf128_pd_256:
  case Intrinsic::x86_avx_vinsertf128_ps_256:
  case Intrinsic::x86_avx_vinsertf128_si_256:
  case Intrinsic::x86_avx2_vinserti128: {
    DebugLoc dl = getCurDebugLoc();
    EVT DestVT = TLI.getValueType(I.getType());
    EVT ElVT = TLI.getValueType(I.getArgOperand(1)->getType());
    uint64_t Idx = (cast<ConstantInt>(I.getArgOperand(2))->getZExtValue() & 1) *
                   ElVT.getVectorNumElements();
    Res = DAG.getNode(ISD::INSERT_SUBVECTOR, dl, DestVT,
                      getValue(I.getArgOperand(0)),
                      getValue(I.getArgOperand(1)),
                      DAG.getIntPtrConstant(Idx));
    setValue(&I, Res);
    return 0;
  }
  case Intrinsic::x86_avx_vextractf128_pd_256:
  case Intrinsic::x86_avx_vextractf128_ps_256:
  case Intrinsic::x86_avx_vextractf128_si_256:
  case Intrinsic::x86_avx2_vextracti128: {
    DebugLoc dl = getCurDebugLoc();
    EVT DestVT = TLI.getValueType(I.getType());
    uint64_t Idx = (cast<ConstantInt>(I.getArgOperand(1))->getZExtValue() & 1) *
                   DestVT.getVectorNumElements();
    Res = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, DestVT,
                      getValue(I.getArgOperand(0)),
                      DAG.getIntPtrConstant(Idx));
    setValue(&I, Res);
    return 0;
  }
  case Intrinsic::convertff:
  case Intrinsic::convertfsi:
  case Intrinsic::convertfui:
  case Intrinsic::convertsif:
  case Intrinsic::convertuif:
  case Intrinsic::convertss:
  case Intrinsic::convertsu:
  case Intrinsic::convertus:
  case Intrinsic::convertuu: {
    ISD::CvtCode Code = ISD::CVT_INVALID;
    switch (Intrinsic) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::convertff:  Code = ISD::CVT_FF; break;
    case Intrinsic::convertfsi: Code = ISD::CVT_FS; break;
    case Intrinsic::convertfui: Code = ISD::CVT_FU; break;
    case Intrinsic::convertsif: Code = ISD::CVT_SF; break;
    case Intrinsic::convertuif: Code = ISD::CVT_UF; break;
    case Intrinsic::convertss:  Code = ISD::CVT_SS; break;
    case Intrinsic::convertsu:  Code = ISD::CVT_SU; break;
    case Intrinsic::convertus:  Code = ISD::CVT_US; break;
    case Intrinsic::convertuu:  Code = ISD::CVT_UU; break;
    }
    EVT DestVT = TLI.getValueType(I.getType());
    const Value *Op1 = I.getArgOperand(0);
    Res = DAG.getConvertRndSat(DestVT, getCurDebugLoc(), getValue(Op1),
                               DAG.getValueType(DestVT),
                               DAG.getValueType(getValue(Op1).getValueType()),
                               getValue(I.getArgOperand(1)),
                               getValue(I.getArgOperand(2)),
                               Code);
    setValue(&I, Res);
    return 0;
  }
  case Intrinsic::sqrt:
    setValue(&I, DAG.getNode(ISD::FSQRT, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::powi:
    setValue(&I, ExpandPowI(dl, getValue(I.getArgOperand(0)),
                            getValue(I.getArgOperand(1)), DAG));
    return 0;
  case Intrinsic::sin:
    setValue(&I, DAG.getNode(ISD::FSIN, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::cos:
    setValue(&I, DAG.getNode(ISD::FCOS, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::log:
    visitLog(I);
    return 0;
  case Intrinsic::log2:
    visitLog2(I);
    return 0;
  case Intrinsic::log10:
    visitLog10(I);
    return 0;
  case Intrinsic::exp:
    visitExp(I);
    return 0;
  case Intrinsic::exp2:
    visitExp2(I);
    return 0;
  case Intrinsic::pow:
    visitPow(I);
    return 0;
  case Intrinsic::fabs:
    setValue(&I, DAG.getNode(ISD::FABS, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::floor:
    setValue(&I, DAG.getNode(ISD::FFLOOR, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::fma:
    setValue(&I, DAG.getNode(ISD::FMA, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0)),
                             getValue(I.getArgOperand(1)),
                             getValue(I.getArgOperand(2))));
    return 0;
  case Intrinsic::fmuladd: {
    EVT VT = TLI.getValueType(I.getType());
    if (TM.Options.AllowFPOpFusion != FPOpFusion::Strict &&
        TLI.isOperationLegal(ISD::FMA, VT) &&
        TLI.isFMAFasterThanMulAndAdd(VT)){
      setValue(&I, DAG.getNode(ISD::FMA, dl,
                               getValue(I.getArgOperand(0)).getValueType(),
                               getValue(I.getArgOperand(0)),
                               getValue(I.getArgOperand(1)),
                               getValue(I.getArgOperand(2))));
    } else {
      SDValue Mul = DAG.getNode(ISD::FMUL, dl,
                                getValue(I.getArgOperand(0)).getValueType(),
                                getValue(I.getArgOperand(0)),
                                getValue(I.getArgOperand(1)));
      SDValue Add = DAG.getNode(ISD::FADD, dl,
                                getValue(I.getArgOperand(0)).getValueType(),
                                Mul,
                                getValue(I.getArgOperand(2)));
      setValue(&I, Add);
    }
    return 0;
  }
  case Intrinsic::convert_to_fp16:
    setValue(&I, DAG.getNode(ISD::FP32_TO_FP16, dl,
                             MVT::i16, getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::convert_from_fp16:
    setValue(&I, DAG.getNode(ISD::FP16_TO_FP32, dl,
                             MVT::f32, getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::pcmarker: {
    SDValue Tmp = getValue(I.getArgOperand(0));
    DAG.setRoot(DAG.getNode(ISD::PCMARKER, dl, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::readcyclecounter: {
    SDValue Op = getRoot();
    Res = DAG.getNode(ISD::READCYCLECOUNTER, dl,
                      DAG.getVTList(MVT::i64, MVT::Other),
                      &Op, 1);
    setValue(&I, Res);
    DAG.setRoot(Res.getValue(1));
    return 0;
  }
  case Intrinsic::bswap:
    setValue(&I, DAG.getNode(ISD::BSWAP, dl,
                             getValue(I.getArgOperand(0)).getValueType(),
                             getValue(I.getArgOperand(0))));
    return 0;
  case Intrinsic::cttz: {
    SDValue Arg = getValue(I.getArgOperand(0));
    ConstantInt *CI = cast<ConstantInt>(I.getArgOperand(1));
    EVT Ty = Arg.getValueType();
    setValue(&I, DAG.getNode(CI->isZero() ? ISD::CTTZ : ISD::CTTZ_ZERO_UNDEF,
                             dl, Ty, Arg));
    return 0;
  }
  case Intrinsic::ctlz: {
    SDValue Arg = getValue(I.getArgOperand(0));
    ConstantInt *CI = cast<ConstantInt>(I.getArgOperand(1));
    EVT Ty = Arg.getValueType();
    setValue(&I, DAG.getNode(CI->isZero() ? ISD::CTLZ : ISD::CTLZ_ZERO_UNDEF,
                             dl, Ty, Arg));
    return 0;
  }
  case Intrinsic::ctpop: {
    SDValue Arg = getValue(I.getArgOperand(0));
    EVT Ty = Arg.getValueType();
    setValue(&I, DAG.getNode(ISD::CTPOP, dl, Ty, Arg));
    return 0;
  }
  case Intrinsic::stacksave: {
    SDValue Op = getRoot();
    Res = DAG.getNode(ISD::STACKSAVE, dl,
                      DAG.getVTList(TLI.getPointerTy(), MVT::Other), &Op, 1);
    setValue(&I, Res);
    DAG.setRoot(Res.getValue(1));
    return 0;
  }
  case Intrinsic::stackrestore: {
    Res = getValue(I.getArgOperand(0));
    DAG.setRoot(DAG.getNode(ISD::STACKRESTORE, dl, MVT::Other, getRoot(), Res));
    return 0;
  }
  case Intrinsic::stackprotector: {
    // Emit code into the DAG to store the stack guard onto the stack.
    MachineFunction &MF = DAG.getMachineFunction();
    MachineFrameInfo *MFI = MF.getFrameInfo();
    EVT PtrTy = TLI.getPointerTy();

    SDValue Src = getValue(I.getArgOperand(0));   // The guard's value.
    AllocaInst *Slot = cast<AllocaInst>(I.getArgOperand(1));

    int FI = FuncInfo.StaticAllocaMap[Slot];
    MFI->setStackProtectorIndex(FI);

    SDValue FIN = DAG.getFrameIndex(FI, PtrTy);

    // Store the stack protector onto the stack.
    Res = DAG.getStore(getRoot(), getCurDebugLoc(), Src, FIN,
                       MachinePointerInfo::getFixedStack(FI),
                       true, false, 0);
    setValue(&I, Res);
    DAG.setRoot(Res);
    return 0;
  }
  case Intrinsic::objectsize: {
    // If we don't know by now, we're never going to know.
    ConstantInt *CI = dyn_cast<ConstantInt>(I.getArgOperand(1));

    assert(CI && "Non-constant type in __builtin_object_size?");

    SDValue Arg = getValue(I.getCalledValue());
    EVT Ty = Arg.getValueType();

    if (CI->isZero())
      Res = DAG.getConstant(-1ULL, Ty);
    else
      Res = DAG.getConstant(0, Ty);

    setValue(&I, Res);
    return 0;
  }
  case Intrinsic::var_annotation:
    // Discard annotate attributes
    return 0;

  case Intrinsic::init_trampoline: {
    const Function *F = cast<Function>(I.getArgOperand(1)->stripPointerCasts());

    SDValue Ops[6];
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getArgOperand(0));
    Ops[2] = getValue(I.getArgOperand(1));
    Ops[3] = getValue(I.getArgOperand(2));
    Ops[4] = DAG.getSrcValue(I.getArgOperand(0));
    Ops[5] = DAG.getSrcValue(F);

    Res = DAG.getNode(ISD::INIT_TRAMPOLINE, dl, MVT::Other, Ops, 6);

    DAG.setRoot(Res);
    return 0;
  }
  case Intrinsic::adjust_trampoline: {
    setValue(&I, DAG.getNode(ISD::ADJUST_TRAMPOLINE, dl,
                             TLI.getPointerTy(),
                             getValue(I.getArgOperand(0))));
    return 0;
  }
  case Intrinsic::gcroot:
    if (GFI) {
      const Value *Alloca = I.getArgOperand(0)->stripPointerCasts();
      const Constant *TypeMap = cast<Constant>(I.getArgOperand(1));

      FrameIndexSDNode *FI = cast<FrameIndexSDNode>(getValue(Alloca).getNode());
      GFI->addStackRoot(FI->getIndex(), TypeMap);
    }
    return 0;
  case Intrinsic::gcread:
  case Intrinsic::gcwrite:
    llvm_unreachable("GC failed to lower gcread/gcwrite intrinsics!");
  case Intrinsic::flt_rounds:
    setValue(&I, DAG.getNode(ISD::FLT_ROUNDS_, dl, MVT::i32));
    return 0;

  case Intrinsic::expect: {
    // Just replace __builtin_expect(exp, c) with EXP.
    setValue(&I, getValue(I.getArgOperand(0)));
    return 0;
  }

  case Intrinsic::debugtrap:
  case Intrinsic::trap: {
    StringRef TrapFuncName = TM.Options.getTrapFunctionName();
    if (TrapFuncName.empty()) {
      ISD::NodeType Op = (Intrinsic == Intrinsic::trap) ? 
        ISD::TRAP : ISD::DEBUGTRAP;
      DAG.setRoot(DAG.getNode(Op, dl,MVT::Other, getRoot()));
      return 0;
    }
    TargetLowering::ArgListTy Args;
    TargetLowering::
    CallLoweringInfo CLI(getRoot(), I.getType(),
                 false, false, false, false, 0, CallingConv::C,
                 /*isTailCall=*/false,
                 /*doesNotRet=*/false, /*isReturnValueUsed=*/true,
                 DAG.getExternalSymbol(TrapFuncName.data(), TLI.getPointerTy()),
                 Args, DAG, getCurDebugLoc());
    std::pair<SDValue, SDValue> Result = TLI.LowerCallTo(CLI);
    DAG.setRoot(Result.second);
    return 0;
  }

  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow: {
    ISD::NodeType Op;
    switch (Intrinsic) {
    default: llvm_unreachable("Impossible intrinsic");  // Can't reach here.
    case Intrinsic::uadd_with_overflow: Op = ISD::UADDO; break;
    case Intrinsic::sadd_with_overflow: Op = ISD::SADDO; break;
    case Intrinsic::usub_with_overflow: Op = ISD::USUBO; break;
    case Intrinsic::ssub_with_overflow: Op = ISD::SSUBO; break;
    case Intrinsic::umul_with_overflow: Op = ISD::UMULO; break;
    case Intrinsic::smul_with_overflow: Op = ISD::SMULO; break;
    }
    SDValue Op1 = getValue(I.getArgOperand(0));
    SDValue Op2 = getValue(I.getArgOperand(1));

    SDVTList VTs = DAG.getVTList(Op1.getValueType(), MVT::i1);
    setValue(&I, DAG.getNode(Op, getCurDebugLoc(), VTs, Op1, Op2));
    return 0;
  }
  case Intrinsic::prefetch: {
    SDValue Ops[5];
    unsigned rw = cast<ConstantInt>(I.getArgOperand(1))->getZExtValue();
    Ops[0] = getRoot();
    Ops[1] = getValue(I.getArgOperand(0));
    Ops[2] = getValue(I.getArgOperand(1));
    Ops[3] = getValue(I.getArgOperand(2));
    Ops[4] = getValue(I.getArgOperand(3));
    DAG.setRoot(DAG.getMemIntrinsicNode(ISD::PREFETCH, dl,
                                        DAG.getVTList(MVT::Other),
                                        &Ops[0], 5,
                                        EVT::getIntegerVT(*Context, 8),
                                        MachinePointerInfo(I.getArgOperand(0)),
                                        0, /* align */
                                        false, /* volatile */
                                        rw==0, /* read */
                                        rw==1)); /* write */
    return 0;
  }
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end: {
    bool IsStart = (Intrinsic == Intrinsic::lifetime_start);
    // Stack coloring is not enabled in O0, discard region information.
    if (TM.getOptLevel() == CodeGenOpt::None)
      return 0;

    SmallVector<Value *, 4> Allocas;
    GetUnderlyingObjects(I.getArgOperand(1), Allocas, TD);

    for (SmallVector<Value*, 4>::iterator Object = Allocas.begin(),
         E = Allocas.end(); Object != E; ++Object) {
      AllocaInst *LifetimeObject = dyn_cast_or_null<AllocaInst>(*Object);

      // Could not find an Alloca.
      if (!LifetimeObject)
        continue;

      int FI = FuncInfo.StaticAllocaMap[LifetimeObject];

      SDValue Ops[2];
      Ops[0] = getRoot();
      Ops[1] = DAG.getFrameIndex(FI, TLI.getPointerTy(), true);
      unsigned Opcode = (IsStart ? ISD::LIFETIME_START : ISD::LIFETIME_END);

      Res = DAG.getNode(Opcode, dl, MVT::Other, Ops, 2);
      DAG.setRoot(Res);
    }
  }
  case Intrinsic::invariant_start:
    // Discard region information.
    setValue(&I, DAG.getUNDEF(TLI.getPointerTy()));
    return 0;
  case Intrinsic::invariant_end:
    // Discard region information.
    return 0;
  case Intrinsic::donothing:
    // ignore
    return 0;
  }
}

void SelectionDAGBuilder::LowerCallTo(ImmutableCallSite CS, SDValue Callee,
                                      bool isTailCall,
                                      MachineBasicBlock *LandingPad) {
  PointerType *PT = cast<PointerType>(CS.getCalledValue()->getType());
  FunctionType *FTy = cast<FunctionType>(PT->getElementType());
  Type *RetTy = FTy->getReturnType();
  MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
  MCSymbol *BeginLabel = 0;

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Args.reserve(CS.arg_size());

  // Check whether the function can return without sret-demotion.
  SmallVector<ISD::OutputArg, 4> Outs;
  GetReturnInfo(RetTy, CS.getAttributes().getRetAttributes(),
                Outs, TLI);

  bool CanLowerReturn = TLI.CanLowerReturn(CS.getCallingConv(),
                                           DAG.getMachineFunction(),
                                           FTy->isVarArg(), Outs,
                                           FTy->getContext());

  SDValue DemoteStackSlot;
  int DemoteStackIdx = -100;

  if (!CanLowerReturn) {
    uint64_t TySize = TLI.getDataLayout()->getTypeAllocSize(
                      FTy->getReturnType());
    unsigned Align  = TLI.getDataLayout()->getPrefTypeAlignment(
                      FTy->getReturnType());
    MachineFunction &MF = DAG.getMachineFunction();
    DemoteStackIdx = MF.getFrameInfo()->CreateStackObject(TySize, Align, false);
    Type *StackSlotPtrType = PointerType::getUnqual(FTy->getReturnType());

    DemoteStackSlot = DAG.getFrameIndex(DemoteStackIdx, TLI.getPointerTy());
    Entry.Node = DemoteStackSlot;
    Entry.Ty = StackSlotPtrType;
    Entry.isSExt = false;
    Entry.isZExt = false;
    Entry.isInReg = false;
    Entry.isSRet = true;
    Entry.isNest = false;
    Entry.isByVal = false;
    Entry.Alignment = Align;
    Args.push_back(Entry);
    RetTy = Type::getVoidTy(FTy->getContext());
  }

  for (ImmutableCallSite::arg_iterator i = CS.arg_begin(), e = CS.arg_end();
       i != e; ++i) {
    const Value *V = *i;

    // Skip empty types
    if (V->getType()->isEmptyTy())
      continue;

    SDValue ArgNode = getValue(V);
    Entry.Node = ArgNode; Entry.Ty = V->getType();

    unsigned attrInd = i - CS.arg_begin() + 1;
    Entry.isSExt  = CS.paramHasAttr(attrInd, Attributes::SExt);
    Entry.isZExt  = CS.paramHasAttr(attrInd, Attributes::ZExt);
    Entry.isInReg = CS.paramHasAttr(attrInd, Attributes::InReg);
    Entry.isSRet  = CS.paramHasAttr(attrInd, Attributes::StructRet);
    Entry.isNest  = CS.paramHasAttr(attrInd, Attributes::Nest);
    Entry.isByVal = CS.paramHasAttr(attrInd, Attributes::ByVal);
    Entry.Alignment = CS.getParamAlignment(attrInd);
    Args.push_back(Entry);
  }

  if (LandingPad) {
    // Insert a label before the invoke call to mark the try range.  This can be
    // used to detect deletion of the invoke via the MachineModuleInfo.
    BeginLabel = MMI.getContext().CreateTempSymbol();

    // For SjLj, keep track of which landing pads go with which invokes
    // so as to maintain the ordering of pads in the LSDA.
    unsigned CallSiteIndex = MMI.getCurrentCallSite();
    if (CallSiteIndex) {
      MMI.setCallSiteBeginLabel(BeginLabel, CallSiteIndex);
      LPadToCallSiteMap[LandingPad].push_back(CallSiteIndex);

      // Now that the call site is handled, stop tracking it.
      MMI.setCurrentCallSite(0);
    }

    // Both PendingLoads and PendingExports must be flushed here;
    // this call might not return.
    (void)getRoot();
    DAG.setRoot(DAG.getEHLabel(getCurDebugLoc(), getControlRoot(), BeginLabel));
  }

  // Check if target-independent constraints permit a tail call here.
  // Target-dependent constraints are checked within TLI.LowerCallTo.
  if (isTailCall &&
      !isInTailCallPosition(CS, CS.getAttributes().getRetAttributes(), TLI))
    isTailCall = false;

  // If there's a possibility that fast-isel has already selected some amount
  // of the current basic block, don't emit a tail call.
  if (isTailCall && TM.Options.EnableFastISel)
    isTailCall = false;

  TargetLowering::
  CallLoweringInfo CLI(getRoot(), RetTy, FTy, isTailCall, Callee, Args, DAG,
                       getCurDebugLoc(), CS);
  std::pair<SDValue,SDValue> Result = TLI.LowerCallTo(CLI);
  assert((isTailCall || Result.second.getNode()) &&
         "Non-null chain expected with non-tail call!");
  assert((Result.second.getNode() || !Result.first.getNode()) &&
         "Null value expected with tail call!");
  if (Result.first.getNode()) {
    setValue(CS.getInstruction(), Result.first);
  } else if (!CanLowerReturn && Result.second.getNode()) {
    // The instruction result is the result of loading from the
    // hidden sret parameter.
    SmallVector<EVT, 1> PVTs;
    Type *PtrRetTy = PointerType::getUnqual(FTy->getReturnType());

    ComputeValueVTs(TLI, PtrRetTy, PVTs);
    assert(PVTs.size() == 1 && "Pointers should fit in one register");
    EVT PtrVT = PVTs[0];

    SmallVector<EVT, 4> RetTys;
    SmallVector<uint64_t, 4> Offsets;
    RetTy = FTy->getReturnType();
    ComputeValueVTs(TLI, RetTy, RetTys, &Offsets);

    unsigned NumValues = RetTys.size();
    SmallVector<SDValue, 4> Values(NumValues);
    SmallVector<SDValue, 4> Chains(NumValues);

    for (unsigned i = 0; i < NumValues; ++i) {
      SDValue Add = DAG.getNode(ISD::ADD, getCurDebugLoc(), PtrVT,
                                DemoteStackSlot,
                                DAG.getConstant(Offsets[i], PtrVT));
      SDValue L = DAG.getLoad(RetTys[i], getCurDebugLoc(), Result.second, Add,
                  MachinePointerInfo::getFixedStack(DemoteStackIdx, Offsets[i]),
                              false, false, false, 1);
      Values[i] = L;
      Chains[i] = L.getValue(1);
    }

    SDValue Chain = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(),
                                MVT::Other, &Chains[0], NumValues);
    PendingLoads.push_back(Chain);

    setValue(CS.getInstruction(),
             DAG.getNode(ISD::MERGE_VALUES, getCurDebugLoc(),
                         DAG.getVTList(&RetTys[0], RetTys.size()),
                         &Values[0], Values.size()));
  }

  // Assign order to nodes here. If the call does not produce a result, it won't
  // be mapped to a SDNode and visit() will not assign it an order number.
  if (!Result.second.getNode()) {
    // As a special case, a null chain means that a tail call has been emitted and
    // the DAG root is already updated.
    HasTailCall = true;
    ++SDNodeOrder;
    AssignOrderingToNode(DAG.getRoot().getNode());
  } else {
    DAG.setRoot(Result.second);
    ++SDNodeOrder;
    AssignOrderingToNode(Result.second.getNode());
  }

  if (LandingPad) {
    // Insert a label at the end of the invoke call to mark the try range.  This
    // can be used to detect deletion of the invoke via the MachineModuleInfo.
    MCSymbol *EndLabel = MMI.getContext().CreateTempSymbol();
    DAG.setRoot(DAG.getEHLabel(getCurDebugLoc(), getRoot(), EndLabel));

    // Inform MachineModuleInfo of range.
    MMI.addInvoke(LandingPad, BeginLabel, EndLabel);
  }
}

/// IsOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero.
static bool IsOnlyUsedInZeroEqualityComparison(const Value *V) {
  for (Value::const_use_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    if (const ICmpInst *IC = dyn_cast<ICmpInst>(*UI))
      if (IC->isEquality())
        if (const Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

static SDValue getMemCmpLoad(const Value *PtrVal, MVT LoadVT,
                             Type *LoadTy,
                             SelectionDAGBuilder &Builder) {

  // Check to see if this load can be trivially constant folded, e.g. if the
  // input is from a string literal.
  if (const Constant *LoadInput = dyn_cast<Constant>(PtrVal)) {
    // Cast pointer to the type we really want to load.
    LoadInput = ConstantExpr::getBitCast(const_cast<Constant *>(LoadInput),
                                         PointerType::getUnqual(LoadTy));

    if (const Constant *LoadCst =
          ConstantFoldLoadFromConstPtr(const_cast<Constant *>(LoadInput),
                                       Builder.TD))
      return Builder.getValue(LoadCst);
  }

  // Otherwise, we have to emit the load.  If the pointer is to unfoldable but
  // still constant memory, the input chain can be the entry node.
  SDValue Root;
  bool ConstantMemory = false;

  // Do not serialize (non-volatile) loads of constant memory with anything.
  if (Builder.AA->pointsToConstantMemory(PtrVal)) {
    Root = Builder.DAG.getEntryNode();
    ConstantMemory = true;
  } else {
    // Do not serialize non-volatile loads against each other.
    Root = Builder.DAG.getRoot();
  }

  SDValue Ptr = Builder.getValue(PtrVal);
  SDValue LoadVal = Builder.DAG.getLoad(LoadVT, Builder.getCurDebugLoc(), Root,
                                        Ptr, MachinePointerInfo(PtrVal),
                                        false /*volatile*/,
                                        false /*nontemporal*/, 
                                        false /*isinvariant*/, 1 /* align=1 */);

  if (!ConstantMemory)
    Builder.PendingLoads.push_back(LoadVal.getValue(1));
  return LoadVal;
}


/// visitMemCmpCall - See if we can lower a call to memcmp in an optimized form.
/// If so, return true and lower it, otherwise return false and it will be
/// lowered like a normal call.
bool SelectionDAGBuilder::visitMemCmpCall(const CallInst &I) {
  // Verify that the prototype makes sense.  int memcmp(void*,void*,size_t)
  if (I.getNumArgOperands() != 3)
    return false;

  const Value *LHS = I.getArgOperand(0), *RHS = I.getArgOperand(1);
  if (!LHS->getType()->isPointerTy() || !RHS->getType()->isPointerTy() ||
      !I.getArgOperand(2)->getType()->isIntegerTy() ||
      !I.getType()->isIntegerTy())
    return false;

  const ConstantInt *Size = dyn_cast<ConstantInt>(I.getArgOperand(2));

  // memcmp(S1,S2,2) != 0 -> (*(short*)LHS != *(short*)RHS)  != 0
  // memcmp(S1,S2,4) != 0 -> (*(int*)LHS != *(int*)RHS)  != 0
  if (Size && IsOnlyUsedInZeroEqualityComparison(&I)) {
    bool ActuallyDoIt = true;
    MVT LoadVT;
    Type *LoadTy;
    switch (Size->getZExtValue()) {
    default:
      LoadVT = MVT::Other;
      LoadTy = 0;
      ActuallyDoIt = false;
      break;
    case 2:
      LoadVT = MVT::i16;
      LoadTy = Type::getInt16Ty(Size->getContext());
      break;
    case 4:
      LoadVT = MVT::i32;
      LoadTy = Type::getInt32Ty(Size->getContext());
      break;
    case 8:
      LoadVT = MVT::i64;
      LoadTy = Type::getInt64Ty(Size->getContext());
      break;
        /*
    case 16:
      LoadVT = MVT::v4i32;
      LoadTy = Type::getInt32Ty(Size->getContext());
      LoadTy = VectorType::get(LoadTy, 4);
      break;
         */
    }

    // This turns into unaligned loads.  We only do this if the target natively
    // supports the MVT we'll be loading or if it is small enough (<= 4) that
    // we'll only produce a small number of byte loads.

    // Require that we can find a legal MVT, and only do this if the target
    // supports unaligned loads of that type.  Expanding into byte loads would
    // bloat the code.
    if (ActuallyDoIt && Size->getZExtValue() > 4) {
      // TODO: Handle 5 byte compare as 4-byte + 1 byte.
      // TODO: Handle 8 byte compare on x86-32 as two 32-bit loads.
      if (!TLI.isTypeLegal(LoadVT) ||!TLI.allowsUnalignedMemoryAccesses(LoadVT))
        ActuallyDoIt = false;
    }

    if (ActuallyDoIt) {
      SDValue LHSVal = getMemCmpLoad(LHS, LoadVT, LoadTy, *this);
      SDValue RHSVal = getMemCmpLoad(RHS, LoadVT, LoadTy, *this);

      SDValue Res = DAG.getSetCC(getCurDebugLoc(), MVT::i1, LHSVal, RHSVal,
                                 ISD::SETNE);
      EVT CallVT = TLI.getValueType(I.getType(), true);
      setValue(&I, DAG.getZExtOrTrunc(Res, getCurDebugLoc(), CallVT));
      return true;
    }
  }


  return false;
}

/// visitUnaryFloatCall - If a call instruction is a unary floating-point
/// operation (as expected), translate it to an SDNode with the specified opcode
/// and return true.
bool SelectionDAGBuilder::visitUnaryFloatCall(const CallInst &I,
                                              unsigned Opcode) {
  // Sanity check that it really is a unary floating-point call.
  if (I.getNumArgOperands() != 1 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      !I.onlyReadsMemory())
    return false;

  SDValue Tmp = getValue(I.getArgOperand(0));
  setValue(&I, DAG.getNode(Opcode, getCurDebugLoc(), Tmp.getValueType(), Tmp));
  return true;
}

void SelectionDAGBuilder::visitCall(const CallInst &I) {
  // Handle inline assembly differently.
  if (isa<InlineAsm>(I.getCalledValue())) {
    visitInlineAsm(&I);
    return;
  }

  MachineModuleInfo &MMI = DAG.getMachineFunction().getMMI();
  ComputeUsesVAFloatArgument(I, &MMI);

  const char *RenameFn = 0;
  if (Function *F = I.getCalledFunction()) {
    if (F->isDeclaration()) {
      if (const TargetIntrinsicInfo *II = TM.getIntrinsicInfo()) {
        if (unsigned IID = II->getIntrinsicID(F)) {
          RenameFn = visitIntrinsicCall(I, IID);
          if (!RenameFn)
            return;
        }
      }
      if (unsigned IID = F->getIntrinsicID()) {
        RenameFn = visitIntrinsicCall(I, IID);
        if (!RenameFn)
          return;
      }
    }

    // Check for well-known libc/libm calls.  If the function is internal, it
    // can't be a library call.
    LibFunc::Func Func;
    if (!F->hasLocalLinkage() && F->hasName() &&
        LibInfo->getLibFunc(F->getName(), Func) &&
        LibInfo->hasOptimizedCodeGen(Func)) {
      switch (Func) {
      default: break;
      case LibFunc::copysign:
      case LibFunc::copysignf:
      case LibFunc::copysignl:
        if (I.getNumArgOperands() == 2 &&   // Basic sanity checks.
            I.getArgOperand(0)->getType()->isFloatingPointTy() &&
            I.getType() == I.getArgOperand(0)->getType() &&
            I.getType() == I.getArgOperand(1)->getType() &&
            I.onlyReadsMemory()) {
          SDValue LHS = getValue(I.getArgOperand(0));
          SDValue RHS = getValue(I.getArgOperand(1));
          setValue(&I, DAG.getNode(ISD::FCOPYSIGN, getCurDebugLoc(),
                                   LHS.getValueType(), LHS, RHS));
          return;
        }
        break;
      case LibFunc::fabs:
      case LibFunc::fabsf:
      case LibFunc::fabsl:
        if (visitUnaryFloatCall(I, ISD::FABS))
          return;
        break;
      case LibFunc::sin:
      case LibFunc::sinf:
      case LibFunc::sinl:
        if (visitUnaryFloatCall(I, ISD::FSIN))
          return;
        break;
      case LibFunc::cos:
      case LibFunc::cosf:
      case LibFunc::cosl:
        if (visitUnaryFloatCall(I, ISD::FCOS))
          return;
        break;
      case LibFunc::sqrt:
      case LibFunc::sqrtf:
      case LibFunc::sqrtl:
        if (visitUnaryFloatCall(I, ISD::FSQRT))
          return;
        break;
      case LibFunc::floor:
      case LibFunc::floorf:
      case LibFunc::floorl:
        if (visitUnaryFloatCall(I, ISD::FFLOOR))
          return;
        break;
      case LibFunc::nearbyint:
      case LibFunc::nearbyintf:
      case LibFunc::nearbyintl:
        if (visitUnaryFloatCall(I, ISD::FNEARBYINT))
          return;
        break;
      case LibFunc::ceil:
      case LibFunc::ceilf:
      case LibFunc::ceill:
        if (visitUnaryFloatCall(I, ISD::FCEIL))
          return;
        break;
      case LibFunc::rint:
      case LibFunc::rintf:
      case LibFunc::rintl:
        if (visitUnaryFloatCall(I, ISD::FRINT))
          return;
        break;
      case LibFunc::trunc:
      case LibFunc::truncf:
      case LibFunc::truncl:
        if (visitUnaryFloatCall(I, ISD::FTRUNC))
          return;
        break;
      case LibFunc::log2:
      case LibFunc::log2f:
      case LibFunc::log2l:
        if (visitUnaryFloatCall(I, ISD::FLOG2))
          return;
        break;
      case LibFunc::exp2:
      case LibFunc::exp2f:
      case LibFunc::exp2l:
        if (visitUnaryFloatCall(I, ISD::FEXP2))
          return;
        break;
      case LibFunc::memcmp:
        if (visitMemCmpCall(I))
          return;
        break;
      }
    }
  }

  SDValue Callee;
  if (!RenameFn)
    Callee = getValue(I.getCalledValue());
  else
    Callee = DAG.getExternalSymbol(RenameFn, TLI.getPointerTy());

  // Check if we can potentially perform a tail call. More detailed checking is
  // be done within LowerCallTo, after more information about the call is known.
  LowerCallTo(&I, Callee, I.isTailCall());
}

namespace {

/// AsmOperandInfo - This contains information for each constraint that we are
/// lowering.
class SDISelAsmOperandInfo : public TargetLowering::AsmOperandInfo {
public:
  /// CallOperand - If this is the result output operand or a clobber
  /// this is null, otherwise it is the incoming operand to the CallInst.
  /// This gets modified as the asm is processed.
  SDValue CallOperand;

  /// AssignedRegs - If this is a register or register class operand, this
  /// contains the set of register corresponding to the operand.
  RegsForValue AssignedRegs;

  explicit SDISelAsmOperandInfo(const TargetLowering::AsmOperandInfo &info)
    : TargetLowering::AsmOperandInfo(info), CallOperand(0,0) {
  }

  /// getCallOperandValEVT - Return the EVT of the Value* that this operand
  /// corresponds to.  If there is no Value* for this operand, it returns
  /// MVT::Other.
  EVT getCallOperandValEVT(LLVMContext &Context,
                           const TargetLowering &TLI,
                           const DataLayout *TD) const {
    if (CallOperandVal == 0) return MVT::Other;

    if (isa<BasicBlock>(CallOperandVal))
      return TLI.getPointerTy();

    llvm::Type *OpTy = CallOperandVal->getType();

    // FIXME: code duplicated from TargetLowering::ParseConstraints().
    // If this is an indirect operand, the operand is a pointer to the
    // accessed type.
    if (isIndirect) {
      llvm::PointerType *PtrTy = dyn_cast<PointerType>(OpTy);
      if (!PtrTy)
        report_fatal_error("Indirect operand for inline asm not a pointer!");
      OpTy = PtrTy->getElementType();
    }

    // Look for vector wrapped in a struct. e.g. { <16 x i8> }.
    if (StructType *STy = dyn_cast<StructType>(OpTy))
      if (STy->getNumElements() == 1)
        OpTy = STy->getElementType(0);

    // If OpTy is not a single value, it may be a struct/union that we
    // can tile with integers.
    if (!OpTy->isSingleValueType() && OpTy->isSized()) {
      unsigned BitSize = TD->getTypeSizeInBits(OpTy);
      switch (BitSize) {
      default: break;
      case 1:
      case 8:
      case 16:
      case 32:
      case 64:
      case 128:
        OpTy = IntegerType::get(Context, BitSize);
        break;
      }
    }

    return TLI.getValueType(OpTy, true);
  }
};

typedef SmallVector<SDISelAsmOperandInfo,16> SDISelAsmOperandInfoVector;

} // end anonymous namespace

/// GetRegistersForValue - Assign registers (virtual or physical) for the
/// specified operand.  We prefer to assign virtual registers, to allow the
/// register allocator to handle the assignment process.  However, if the asm
/// uses features that we can't model on machineinstrs, we have SDISel do the
/// allocation.  This produces generally horrible, but correct, code.
///
///   OpInfo describes the operand.
///
static void GetRegistersForValue(SelectionDAG &DAG,
                                 const TargetLowering &TLI,
                                 DebugLoc DL,
                                 SDISelAsmOperandInfo &OpInfo) {
  LLVMContext &Context = *DAG.getContext();

  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<unsigned, 4> Regs;

  // If this is a constraint for a single physreg, or a constraint for a
  // register class, find it.
  std::pair<unsigned, const TargetRegisterClass*> PhysReg =
    TLI.getRegForInlineAsmConstraint(OpInfo.ConstraintCode,
                                     OpInfo.ConstraintVT);

  unsigned NumRegs = 1;
  if (OpInfo.ConstraintVT != MVT::Other) {
    // If this is a FP input in an integer register (or visa versa) insert a bit
    // cast of the input value.  More generally, handle any case where the input
    // value disagrees with the register class we plan to stick this in.
    if (OpInfo.Type == InlineAsm::isInput &&
        PhysReg.second && !PhysReg.second->hasType(OpInfo.ConstraintVT)) {
      // Try to convert to the first EVT that the reg class contains.  If the
      // types are identical size, use a bitcast to convert (e.g. two differing
      // vector types).
      EVT RegVT = *PhysReg.second->vt_begin();
      if (RegVT.getSizeInBits() == OpInfo.ConstraintVT.getSizeInBits()) {
        OpInfo.CallOperand = DAG.getNode(ISD::BITCAST, DL,
                                         RegVT, OpInfo.CallOperand);
        OpInfo.ConstraintVT = RegVT;
      } else if (RegVT.isInteger() && OpInfo.ConstraintVT.isFloatingPoint()) {
        // If the input is a FP value and we want it in FP registers, do a
        // bitcast to the corresponding integer type.  This turns an f64 value
        // into i64, which can be passed with two i32 values on a 32-bit
        // machine.
        RegVT = EVT::getIntegerVT(Context,
                                  OpInfo.ConstraintVT.getSizeInBits());
        OpInfo.CallOperand = DAG.getNode(ISD::BITCAST, DL,
                                         RegVT, OpInfo.CallOperand);
        OpInfo.ConstraintVT = RegVT;
      }
    }

    NumRegs = TLI.getNumRegisters(Context, OpInfo.ConstraintVT);
  }

  EVT RegVT;
  EVT ValueVT = OpInfo.ConstraintVT;

  // If this is a constraint for a specific physical register, like {r17},
  // assign it now.
  if (unsigned AssignedReg = PhysReg.first) {
    const TargetRegisterClass *RC = PhysReg.second;
    if (OpInfo.ConstraintVT == MVT::Other)
      ValueVT = *RC->vt_begin();

    // Get the actual register value type.  This is important, because the user
    // may have asked for (e.g.) the AX register in i32 type.  We need to
    // remember that AX is actually i16 to get the right extension.
    RegVT = *RC->vt_begin();

    // This is a explicit reference to a physical register.
    Regs.push_back(AssignedReg);

    // If this is an expanded reference, add the rest of the regs to Regs.
    if (NumRegs != 1) {
      TargetRegisterClass::iterator I = RC->begin();
      for (; *I != AssignedReg; ++I)
        assert(I != RC->end() && "Didn't find reg!");

      // Already added the first reg.
      --NumRegs; ++I;
      for (; NumRegs; --NumRegs, ++I) {
        assert(I != RC->end() && "Ran out of registers to allocate!");
        Regs.push_back(*I);
      }
    }

    OpInfo.AssignedRegs = RegsForValue(Regs, RegVT, ValueVT);
    return;
  }

  // Otherwise, if this was a reference to an LLVM register class, create vregs
  // for this reference.
  if (const TargetRegisterClass *RC = PhysReg.second) {
    RegVT = *RC->vt_begin();
    if (OpInfo.ConstraintVT == MVT::Other)
      ValueVT = RegVT;

    // Create the appropriate number of virtual registers.
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    for (; NumRegs; --NumRegs)
      Regs.push_back(RegInfo.createVirtualRegister(RC));

    OpInfo.AssignedRegs = RegsForValue(Regs, RegVT, ValueVT);
    return;
  }

  // Otherwise, we couldn't allocate enough registers for this.
}

/// visitInlineAsm - Handle a call to an InlineAsm object.
///
void SelectionDAGBuilder::visitInlineAsm(ImmutableCallSite CS) {
  const InlineAsm *IA = cast<InlineAsm>(CS.getCalledValue());

  /// ConstraintOperands - Information about all of the constraints.
  SDISelAsmOperandInfoVector ConstraintOperands;

  TargetLowering::AsmOperandInfoVector
    TargetConstraints = TLI.ParseConstraints(CS);

  bool hasMemory = false;

  unsigned ArgNo = 0;   // ArgNo - The argument of the CallInst.
  unsigned ResNo = 0;   // ResNo - The result number of the next output.
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    ConstraintOperands.push_back(SDISelAsmOperandInfo(TargetConstraints[i]));
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands.back();

    EVT OpVT = MVT::Other;

    // Compute the value type for each operand.
    switch (OpInfo.Type) {
    case InlineAsm::isOutput:
      // Indirect outputs just consume an argument.
      if (OpInfo.isIndirect) {
        OpInfo.CallOperandVal = const_cast<Value *>(CS.getArgument(ArgNo++));
        break;
      }

      // The return value of the call is this value.  As such, there is no
      // corresponding argument.
      assert(!CS.getType()->isVoidTy() && "Bad inline asm!");
      if (StructType *STy = dyn_cast<StructType>(CS.getType())) {
        OpVT = TLI.getValueType(STy->getElementType(ResNo));
      } else {
        assert(ResNo == 0 && "Asm only has one result!");
        OpVT = TLI.getValueType(CS.getType());
      }
      ++ResNo;
      break;
    case InlineAsm::isInput:
      OpInfo.CallOperandVal = const_cast<Value *>(CS.getArgument(ArgNo++));
      break;
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }

    // If this is an input or an indirect output, process the call argument.
    // BasicBlocks are labels, currently appearing only in asm's.
    if (OpInfo.CallOperandVal) {
      if (const BasicBlock *BB = dyn_cast<BasicBlock>(OpInfo.CallOperandVal)) {
        OpInfo.CallOperand = DAG.getBasicBlock(FuncInfo.MBBMap[BB]);
      } else {
        OpInfo.CallOperand = getValue(OpInfo.CallOperandVal);
      }

      OpVT = OpInfo.getCallOperandValEVT(*DAG.getContext(), TLI, TD);
    }

    OpInfo.ConstraintVT = OpVT;

    // Indirect operand accesses access memory.
    if (OpInfo.isIndirect)
      hasMemory = true;
    else {
      for (unsigned j = 0, ee = OpInfo.Codes.size(); j != ee; ++j) {
        TargetLowering::ConstraintType
          CType = TLI.getConstraintType(OpInfo.Codes[j]);
        if (CType == TargetLowering::C_Memory) {
          hasMemory = true;
          break;
        }
      }
    }
  }

  SDValue Chain, Flag;

  // We won't need to flush pending loads if this asm doesn't touch
  // memory and is nonvolatile.
  if (hasMemory || IA->hasSideEffects())
    Chain = getRoot();
  else
    Chain = DAG.getRoot();

  // Second pass over the constraints: compute which constraint option to use
  // and assign registers to constraints that want a specific physreg.
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    // If this is an output operand with a matching input operand, look up the
    // matching input. If their types mismatch, e.g. one is an integer, the
    // other is floating point, or their sizes are different, flag it as an
    // error.
    if (OpInfo.hasMatchingInput()) {
      SDISelAsmOperandInfo &Input = ConstraintOperands[OpInfo.MatchingInput];

      if (OpInfo.ConstraintVT != Input.ConstraintVT) {
        std::pair<unsigned, const TargetRegisterClass*> MatchRC =
          TLI.getRegForInlineAsmConstraint(OpInfo.ConstraintCode,
                                           OpInfo.ConstraintVT);
        std::pair<unsigned, const TargetRegisterClass*> InputRC =
          TLI.getRegForInlineAsmConstraint(Input.ConstraintCode,
                                           Input.ConstraintVT);
        if ((OpInfo.ConstraintVT.isInteger() !=
             Input.ConstraintVT.isInteger()) ||
            (MatchRC.second != InputRC.second)) {
          report_fatal_error("Unsupported asm: input constraint"
                             " with a matching output constraint of"
                             " incompatible type!");
        }
        Input.ConstraintVT = OpInfo.ConstraintVT;
      }
    }

    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, OpInfo.CallOperand, &DAG);

    // If this is a memory input, and if the operand is not indirect, do what we
    // need to to provide an address for the memory input.
    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        !OpInfo.isIndirect) {
      assert((OpInfo.isMultipleAlternative ||
              (OpInfo.Type == InlineAsm::isInput)) &&
             "Can only indirectify direct input operands!");

      // Memory operands really want the address of the value.  If we don't have
      // an indirect input, put it in the constpool if we can, otherwise spill
      // it to a stack slot.
      // TODO: This isn't quite right. We need to handle these according to
      // the addressing mode that the constraint wants. Also, this may take
      // an additional register for the computation and we don't want that
      // either.

      // If the operand is a float, integer, or vector constant, spill to a
      // constant pool entry to get its address.
      const Value *OpVal = OpInfo.CallOperandVal;
      if (isa<ConstantFP>(OpVal) || isa<ConstantInt>(OpVal) ||
          isa<ConstantVector>(OpVal) || isa<ConstantDataVector>(OpVal)) {
        OpInfo.CallOperand = DAG.getConstantPool(cast<Constant>(OpVal),
                                                 TLI.getPointerTy());
      } else {
        // Otherwise, create a stack slot and emit a store to it before the
        // asm.
        Type *Ty = OpVal->getType();
        uint64_t TySize = TLI.getDataLayout()->getTypeAllocSize(Ty);
        unsigned Align  = TLI.getDataLayout()->getPrefTypeAlignment(Ty);
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(TySize, Align, false);
        SDValue StackSlot = DAG.getFrameIndex(SSFI, TLI.getPointerTy());
        Chain = DAG.getStore(Chain, getCurDebugLoc(),
                             OpInfo.CallOperand, StackSlot,
                             MachinePointerInfo::getFixedStack(SSFI),
                             false, false, 0);
        OpInfo.CallOperand = StackSlot;
      }

      // There is no longer a Value* corresponding to this operand.
      OpInfo.CallOperandVal = 0;

      // It is now an indirect operand.
      OpInfo.isIndirect = true;
    }

    // If this constraint is for a specific register, allocate it before
    // anything else.
    if (OpInfo.ConstraintType == TargetLowering::C_Register)
      GetRegistersForValue(DAG, TLI, getCurDebugLoc(), OpInfo);
  }

  // Second pass - Loop over all of the operands, assigning virtual or physregs
  // to register class operands.
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    // C_Register operands have already been allocated, Other/Memory don't need
    // to be.
    if (OpInfo.ConstraintType == TargetLowering::C_RegisterClass)
      GetRegistersForValue(DAG, TLI, getCurDebugLoc(), OpInfo);
  }

  // AsmNodeOperands - The operands for the ISD::INLINEASM node.
  std::vector<SDValue> AsmNodeOperands;
  AsmNodeOperands.push_back(SDValue());  // reserve space for input chain
  AsmNodeOperands.push_back(
          DAG.getTargetExternalSymbol(IA->getAsmString().c_str(),
                                      TLI.getPointerTy()));

  // If we have a !srcloc metadata node associated with it, we want to attach
  // this to the ultimately generated inline asm machineinstr.  To do this, we
  // pass in the third operand as this (potentially null) inline asm MDNode.
  const MDNode *SrcLoc = CS.getInstruction()->getMetadata("srcloc");
  AsmNodeOperands.push_back(DAG.getMDNode(SrcLoc));

  // Remember the HasSideEffect, AlignStack, AsmDialect, MayLoad and MayStore
  // bits as operand 3.
  unsigned ExtraInfo = 0;
  if (IA->hasSideEffects())
    ExtraInfo |= InlineAsm::Extra_HasSideEffects;
  if (IA->isAlignStack())
    ExtraInfo |= InlineAsm::Extra_IsAlignStack;
  // Set the asm dialect.
  ExtraInfo |= IA->getDialect() * InlineAsm::Extra_AsmDialect;

  // Determine if this InlineAsm MayLoad or MayStore based on the constraints.
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];

    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, SDValue());

    // Ideally, we would only check against memory constraints.  However, the
    // meaning of an other constraint can be target-specific and we can't easily
    // reason about it.  Therefore, be conservative and set MayLoad/MayStore
    // for other constriants as well.
    if (OpInfo.ConstraintType == TargetLowering::C_Memory ||
        OpInfo.ConstraintType == TargetLowering::C_Other) {
      if (OpInfo.Type == InlineAsm::isInput)
        ExtraInfo |= InlineAsm::Extra_MayLoad;
      else if (OpInfo.Type == InlineAsm::isOutput)
        ExtraInfo |= InlineAsm::Extra_MayStore;
    }
  }

  AsmNodeOperands.push_back(DAG.getTargetConstant(ExtraInfo,
                                                  TLI.getPointerTy()));

  // Loop over all of the inputs, copying the operand values into the
  // appropriate registers and processing the output regs.
  RegsForValue RetValRegs;

  // IndirectStoresToEmit - The set of stores to emit after the inline asm node.
  std::vector<std::pair<RegsForValue, Value*> > IndirectStoresToEmit;

  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    SDISelAsmOperandInfo &OpInfo = ConstraintOperands[i];

    switch (OpInfo.Type) {
    case InlineAsm::isOutput: {
      if (OpInfo.ConstraintType != TargetLowering::C_RegisterClass &&
          OpInfo.ConstraintType != TargetLowering::C_Register) {
        // Memory output, or 'other' output (e.g. 'X' constraint).
        assert(OpInfo.isIndirect && "Memory output must be indirect operand");

        // Add information to the INLINEASM node to know about this output.
        unsigned OpFlags = InlineAsm::getFlagWord(InlineAsm::Kind_Mem, 1);
        AsmNodeOperands.push_back(DAG.getTargetConstant(OpFlags,
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(OpInfo.CallOperand);
        break;
      }

      // Otherwise, this is a register or register class output.

      // Copy the output from the appropriate register.  Find a register that
      // we can use.
      if (OpInfo.AssignedRegs.Regs.empty()) {
        LLVMContext &Ctx = *DAG.getContext();
        Ctx.emitError(CS.getInstruction(),  
                      "couldn't allocate output register for constraint '" +
                           Twine(OpInfo.ConstraintCode) + "'");
        break;
      }

      // If this is an indirect operand, store through the pointer after the
      // asm.
      if (OpInfo.isIndirect) {
        IndirectStoresToEmit.push_back(std::make_pair(OpInfo.AssignedRegs,
                                                      OpInfo.CallOperandVal));
      } else {
        // This is the result value of the call.
        assert(!CS.getType()->isVoidTy() && "Bad inline asm!");
        // Concatenate this output onto the outputs list.
        RetValRegs.append(OpInfo.AssignedRegs);
      }

      // Add information to the INLINEASM node to know that this register is
      // set.
      OpInfo.AssignedRegs.AddInlineAsmOperands(OpInfo.isEarlyClobber ?
                                           InlineAsm::Kind_RegDefEarlyClobber :
                                               InlineAsm::Kind_RegDef,
                                               false,
                                               0,
                                               DAG,
                                               AsmNodeOperands);
      break;
    }
    case InlineAsm::isInput: {
      SDValue InOperandVal = OpInfo.CallOperand;

      if (OpInfo.isMatchingInputConstraint()) {   // Matching constraint?
        // If this is required to match an output register we have already set,
        // just use its register.
        unsigned OperandNo = OpInfo.getMatchedOperand();

        // Scan until we find the definition we already emitted of this operand.
        // When we find it, create a RegsForValue operand.
        unsigned CurOp = InlineAsm::Op_FirstOperand;
        for (; OperandNo; --OperandNo) {
          // Advance to the next operand.
          unsigned OpFlag =
            cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getZExtValue();
          assert((InlineAsm::isRegDefKind(OpFlag) ||
                  InlineAsm::isRegDefEarlyClobberKind(OpFlag) ||
                  InlineAsm::isMemKind(OpFlag)) && "Skipped past definitions?");
          CurOp += InlineAsm::getNumOperandRegisters(OpFlag)+1;
        }

        unsigned OpFlag =
          cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getZExtValue();
        if (InlineAsm::isRegDefKind(OpFlag) ||
            InlineAsm::isRegDefEarlyClobberKind(OpFlag)) {
          // Add (OpFlag&0xffff)>>3 registers to MatchedRegs.
          if (OpInfo.isIndirect) {
            // This happens on gcc/testsuite/gcc.dg/pr8788-1.c
            LLVMContext &Ctx = *DAG.getContext();
            Ctx.emitError(CS.getInstruction(),  "inline asm not supported yet:"
                          " don't know how to handle tied "
                          "indirect register inputs");
          }

          RegsForValue MatchedRegs;
          MatchedRegs.ValueVTs.push_back(InOperandVal.getValueType());
          EVT RegVT = AsmNodeOperands[CurOp+1].getValueType();
          MatchedRegs.RegVTs.push_back(RegVT);
          MachineRegisterInfo &RegInfo = DAG.getMachineFunction().getRegInfo();
          for (unsigned i = 0, e = InlineAsm::getNumOperandRegisters(OpFlag);
               i != e; ++i)
            MatchedRegs.Regs.push_back
              (RegInfo.createVirtualRegister(TLI.getRegClassFor(RegVT)));

          // Use the produced MatchedRegs object to
          MatchedRegs.getCopyToRegs(InOperandVal, DAG, getCurDebugLoc(),
                                    Chain, &Flag, CS.getInstruction());
          MatchedRegs.AddInlineAsmOperands(InlineAsm::Kind_RegUse,
                                           true, OpInfo.getMatchedOperand(),
                                           DAG, AsmNodeOperands);
          break;
        }

        assert(InlineAsm::isMemKind(OpFlag) && "Unknown matching constraint!");
        assert(InlineAsm::getNumOperandRegisters(OpFlag) == 1 &&
               "Unexpected number of operands");
        // Add information to the INLINEASM node to know about this input.
        // See InlineAsm.h isUseOperandTiedToDef.
        OpFlag = InlineAsm::getFlagWordForMatchingOp(OpFlag,
                                                    OpInfo.getMatchedOperand());
        AsmNodeOperands.push_back(DAG.getTargetConstant(OpFlag,
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(AsmNodeOperands[CurOp+1]);
        break;
      }

      // Treat indirect 'X' constraint as memory.
      if (OpInfo.ConstraintType == TargetLowering::C_Other &&
          OpInfo.isIndirect)
        OpInfo.ConstraintType = TargetLowering::C_Memory;

      if (OpInfo.ConstraintType == TargetLowering::C_Other) {
        std::vector<SDValue> Ops;
        TLI.LowerAsmOperandForConstraint(InOperandVal, OpInfo.ConstraintCode,
                                         Ops, DAG);
        if (Ops.empty()) {
          LLVMContext &Ctx = *DAG.getContext();
          Ctx.emitError(CS.getInstruction(),
                        "invalid operand for inline asm constraint '" +
                        Twine(OpInfo.ConstraintCode) + "'");
          break;
        }

        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType =
          InlineAsm::getFlagWord(InlineAsm::Kind_Imm, Ops.size());
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType,
                                                        TLI.getPointerTy()));
        AsmNodeOperands.insert(AsmNodeOperands.end(), Ops.begin(), Ops.end());
        break;
      }

      if (OpInfo.ConstraintType == TargetLowering::C_Memory) {
        assert(OpInfo.isIndirect && "Operand must be indirect to be a mem!");
        assert(InOperandVal.getValueType() == TLI.getPointerTy() &&
               "Memory operands expect pointer values");

        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType = InlineAsm::getFlagWord(InlineAsm::Kind_Mem, 1);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType,
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(InOperandVal);
        break;
      }

      assert((OpInfo.ConstraintType == TargetLowering::C_RegisterClass ||
              OpInfo.ConstraintType == TargetLowering::C_Register) &&
             "Unknown constraint type!");

      // TODO: Support this.
      if (OpInfo.isIndirect) {
        LLVMContext &Ctx = *DAG.getContext();
        Ctx.emitError(CS.getInstruction(),
                      "Don't know how to handle indirect register inputs yet "
                      "for constraint '" + Twine(OpInfo.ConstraintCode) + "'");
        break;
      }

      // Copy the input into the appropriate registers.
      if (OpInfo.AssignedRegs.Regs.empty()) {
        LLVMContext &Ctx = *DAG.getContext();
        Ctx.emitError(CS.getInstruction(), 
                      "couldn't allocate input reg for constraint '" +
                           Twine(OpInfo.ConstraintCode) + "'");
        break;
      }

      OpInfo.AssignedRegs.getCopyToRegs(InOperandVal, DAG, getCurDebugLoc(),
                                        Chain, &Flag, CS.getInstruction());

      OpInfo.AssignedRegs.AddInlineAsmOperands(InlineAsm::Kind_RegUse, false, 0,
                                               DAG, AsmNodeOperands);
      break;
    }
    case InlineAsm::isClobber: {
      // Add the clobbered value to the operand list, so that the register
      // allocator is aware that the physreg got clobbered.
      if (!OpInfo.AssignedRegs.Regs.empty())
        OpInfo.AssignedRegs.AddInlineAsmOperands(InlineAsm::Kind_Clobber,
                                                 false, 0, DAG,
                                                 AsmNodeOperands);
      break;
    }
    }
  }

  // Finish up input operands.  Set the input chain and add the flag last.
  AsmNodeOperands[InlineAsm::Op_InputChain] = Chain;
  if (Flag.getNode()) AsmNodeOperands.push_back(Flag);

  Chain = DAG.getNode(ISD::INLINEASM, getCurDebugLoc(),
                      DAG.getVTList(MVT::Other, MVT::Glue),
                      &AsmNodeOperands[0], AsmNodeOperands.size());
  Flag = Chain.getValue(1);

  // If this asm returns a register value, copy the result from that register
  // and set it as the value of the call.
  if (!RetValRegs.Regs.empty()) {
    SDValue Val = RetValRegs.getCopyFromRegs(DAG, FuncInfo, getCurDebugLoc(),
                                             Chain, &Flag, CS.getInstruction());

    // FIXME: Why don't we do this for inline asms with MRVs?
    if (CS.getType()->isSingleValueType() && CS.getType()->isSized()) {
      EVT ResultType = TLI.getValueType(CS.getType());

      // If any of the results of the inline asm is a vector, it may have the
      // wrong width/num elts.  This can happen for register classes that can
      // contain multiple different value types.  The preg or vreg allocated may
      // not have the same VT as was expected.  Convert it to the right type
      // with bit_convert.
      if (ResultType != Val.getValueType() && Val.getValueType().isVector()) {
        Val = DAG.getNode(ISD::BITCAST, getCurDebugLoc(),
                          ResultType, Val);

      } else if (ResultType != Val.getValueType() &&
                 ResultType.isInteger() && Val.getValueType().isInteger()) {
        // If a result value was tied to an input value, the computed result may
        // have a wider width than the expected result.  Extract the relevant
        // portion.
        Val = DAG.getNode(ISD::TRUNCATE, getCurDebugLoc(), ResultType, Val);
      }

      assert(ResultType == Val.getValueType() && "Asm result value mismatch!");
    }

    setValue(CS.getInstruction(), Val);
    // Don't need to use this as a chain in this case.
    if (!IA->hasSideEffects() && !hasMemory && IndirectStoresToEmit.empty())
      return;
  }

  std::vector<std::pair<SDValue, const Value *> > StoresToEmit;

  // Process indirect outputs, first output all of the flagged copies out of
  // physregs.
  for (unsigned i = 0, e = IndirectStoresToEmit.size(); i != e; ++i) {
    RegsForValue &OutRegs = IndirectStoresToEmit[i].first;
    const Value *Ptr = IndirectStoresToEmit[i].second;
    SDValue OutVal = OutRegs.getCopyFromRegs(DAG, FuncInfo, getCurDebugLoc(),
                                             Chain, &Flag, IA);
    StoresToEmit.push_back(std::make_pair(OutVal, Ptr));
  }

  // Emit the non-flagged stores from the physregs.
  SmallVector<SDValue, 8> OutChains;
  for (unsigned i = 0, e = StoresToEmit.size(); i != e; ++i) {
    SDValue Val = DAG.getStore(Chain, getCurDebugLoc(),
                               StoresToEmit[i].first,
                               getValue(StoresToEmit[i].second),
                               MachinePointerInfo(StoresToEmit[i].second),
                               false, false, 0);
    OutChains.push_back(Val);
  }

  if (!OutChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, getCurDebugLoc(), MVT::Other,
                        &OutChains[0], OutChains.size());

  DAG.setRoot(Chain);
}

void SelectionDAGBuilder::visitVAStart(const CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VASTART, getCurDebugLoc(),
                          MVT::Other, getRoot(),
                          getValue(I.getArgOperand(0)),
                          DAG.getSrcValue(I.getArgOperand(0))));
}

void SelectionDAGBuilder::visitVAArg(const VAArgInst &I) {
  const DataLayout &TD = *TLI.getDataLayout();
  SDValue V = DAG.getVAArg(TLI.getValueType(I.getType()), getCurDebugLoc(),
                           getRoot(), getValue(I.getOperand(0)),
                           DAG.getSrcValue(I.getOperand(0)),
                           TD.getABITypeAlignment(I.getType()));
  setValue(&I, V);
  DAG.setRoot(V.getValue(1));
}

void SelectionDAGBuilder::visitVAEnd(const CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VAEND, getCurDebugLoc(),
                          MVT::Other, getRoot(),
                          getValue(I.getArgOperand(0)),
                          DAG.getSrcValue(I.getArgOperand(0))));
}

void SelectionDAGBuilder::visitVACopy(const CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VACOPY, getCurDebugLoc(),
                          MVT::Other, getRoot(),
                          getValue(I.getArgOperand(0)),
                          getValue(I.getArgOperand(1)),
                          DAG.getSrcValue(I.getArgOperand(0)),
                          DAG.getSrcValue(I.getArgOperand(1))));
}

/// TargetLowering::LowerCallTo - This is the default LowerCallTo
/// implementation, which just calls LowerCall.
/// FIXME: When all targets are
/// migrated to using LowerCall, this hook should be integrated into SDISel.
std::pair<SDValue, SDValue>
TargetLowering::LowerCallTo(TargetLowering::CallLoweringInfo &CLI) const {
  // Handle all of the outgoing arguments.
  CLI.Outs.clear();
  CLI.OutVals.clear();
  ArgListTy &Args = CLI.Args;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(*this, Args[i].Ty, ValueVTs);
    for (unsigned Value = 0, NumValues = ValueVTs.size();
         Value != NumValues; ++Value) {
      EVT VT = ValueVTs[Value];
      Type *ArgTy = VT.getTypeForEVT(CLI.RetTy->getContext());
      SDValue Op = SDValue(Args[i].Node.getNode(),
                           Args[i].Node.getResNo() + Value);
      ISD::ArgFlagsTy Flags;
      unsigned OriginalAlignment =
        getDataLayout()->getABITypeAlignment(ArgTy);

      if (Args[i].isZExt)
        Flags.setZExt();
      if (Args[i].isSExt)
        Flags.setSExt();
      if (Args[i].isInReg)
        Flags.setInReg();
      if (Args[i].isSRet)
        Flags.setSRet();
      if (Args[i].isByVal) {
        Flags.setByVal();
        PointerType *Ty = cast<PointerType>(Args[i].Ty);
        Type *ElementTy = Ty->getElementType();
        Flags.setByValSize(getDataLayout()->getTypeAllocSize(ElementTy));
        // For ByVal, alignment should come from FE.  BE will guess if this
        // info is not there but there are cases it cannot get right.
        unsigned FrameAlign;
        if (Args[i].Alignment)
          FrameAlign = Args[i].Alignment;
        else
          FrameAlign = getByValTypeAlignment(ElementTy);
        Flags.setByValAlign(FrameAlign);
      }
      if (Args[i].isNest)
        Flags.setNest();
      Flags.setOrigAlign(OriginalAlignment);

      EVT PartVT = getRegisterType(CLI.RetTy->getContext(), VT);
      unsigned NumParts = getNumRegisters(CLI.RetTy->getContext(), VT);
      SmallVector<SDValue, 4> Parts(NumParts);
      ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

      if (Args[i].isSExt)
        ExtendKind = ISD::SIGN_EXTEND;
      else if (Args[i].isZExt)
        ExtendKind = ISD::ZERO_EXTEND;

      getCopyToParts(CLI.DAG, CLI.DL, Op, &Parts[0], NumParts,
                     PartVT, CLI.CS ? CLI.CS->getInstruction() : 0, ExtendKind);

      for (unsigned j = 0; j != NumParts; ++j) {
        // if it isn't first piece, alignment must be 1
        ISD::OutputArg MyFlags(Flags, Parts[j].getValueType(),
                               i < CLI.NumFixedArgs,
                               i, j*Parts[j].getValueType().getStoreSize());
        if (NumParts > 1 && j == 0)
          MyFlags.Flags.setSplit();
        else if (j != 0)
          MyFlags.Flags.setOrigAlign(1);

        CLI.Outs.push_back(MyFlags);
        CLI.OutVals.push_back(Parts[j]);
      }
    }
  }

  // Handle the incoming return values from the call.
  CLI.Ins.clear();
  SmallVector<EVT, 4> RetTys;
  ComputeValueVTs(*this, CLI.RetTy, RetTys);
  for (unsigned I = 0, E = RetTys.size(); I != E; ++I) {
    EVT VT = RetTys[I];
    EVT RegisterVT = getRegisterType(CLI.RetTy->getContext(), VT);
    unsigned NumRegs = getNumRegisters(CLI.RetTy->getContext(), VT);
    for (unsigned i = 0; i != NumRegs; ++i) {
      ISD::InputArg MyFlags;
      MyFlags.VT = RegisterVT.getSimpleVT();
      MyFlags.Used = CLI.IsReturnValueUsed;
      if (CLI.RetSExt)
        MyFlags.Flags.setSExt();
      if (CLI.RetZExt)
        MyFlags.Flags.setZExt();
      if (CLI.IsInReg)
        MyFlags.Flags.setInReg();
      CLI.Ins.push_back(MyFlags);
    }
  }

  SmallVector<SDValue, 4> InVals;
  CLI.Chain = LowerCall(CLI, InVals);

  // Verify that the target's LowerCall behaved as expected.
  assert(CLI.Chain.getNode() && CLI.Chain.getValueType() == MVT::Other &&
         "LowerCall didn't return a valid chain!");
  assert((!CLI.IsTailCall || InVals.empty()) &&
         "LowerCall emitted a return value for a tail call!");
  assert((CLI.IsTailCall || InVals.size() == CLI.Ins.size()) &&
         "LowerCall didn't emit the correct number of values!");

  // For a tail call, the return value is merely live-out and there aren't
  // any nodes in the DAG representing it. Return a special value to
  // indicate that a tail call has been emitted and no more Instructions
  // should be processed in the current block.
  if (CLI.IsTailCall) {
    CLI.DAG.setRoot(CLI.Chain);
    return std::make_pair(SDValue(), SDValue());
  }

  DEBUG(for (unsigned i = 0, e = CLI.Ins.size(); i != e; ++i) {
          assert(InVals[i].getNode() &&
                 "LowerCall emitted a null value!");
          assert(EVT(CLI.Ins[i].VT) == InVals[i].getValueType() &&
                 "LowerCall emitted a value with the wrong type!");
        });

  // Collect the legal value parts into potentially illegal values
  // that correspond to the original function's return values.
  ISD::NodeType AssertOp = ISD::DELETED_NODE;
  if (CLI.RetSExt)
    AssertOp = ISD::AssertSext;
  else if (CLI.RetZExt)
    AssertOp = ISD::AssertZext;
  SmallVector<SDValue, 4> ReturnValues;
  unsigned CurReg = 0;
  for (unsigned I = 0, E = RetTys.size(); I != E; ++I) {
    EVT VT = RetTys[I];
    EVT RegisterVT = getRegisterType(CLI.RetTy->getContext(), VT);
    unsigned NumRegs = getNumRegisters(CLI.RetTy->getContext(), VT);

    ReturnValues.push_back(getCopyFromParts(CLI.DAG, CLI.DL, &InVals[CurReg],
                                            NumRegs, RegisterVT, VT, NULL,
                                            AssertOp));
    CurReg += NumRegs;
  }

  // For a function returning void, there is no return value. We can't create
  // such a node, so we just return a null return value in that case. In
  // that case, nothing will actually look at the value.
  if (ReturnValues.empty())
    return std::make_pair(SDValue(), CLI.Chain);

  SDValue Res = CLI.DAG.getNode(ISD::MERGE_VALUES, CLI.DL,
                                CLI.DAG.getVTList(&RetTys[0], RetTys.size()),
                            &ReturnValues[0], ReturnValues.size());
  return std::make_pair(Res, CLI.Chain);
}

void TargetLowering::LowerOperationWrapper(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  SDValue Res = LowerOperation(SDValue(N, 0), DAG);
  if (Res.getNode())
    Results.push_back(Res);
}

SDValue TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  llvm_unreachable("LowerOperation not implemented for this target!");
}

void
SelectionDAGBuilder::CopyValueToVirtualRegister(const Value *V, unsigned Reg) {
  SDValue Op = getNonRegisterValue(V);
  assert((Op.getOpcode() != ISD::CopyFromReg ||
          cast<RegisterSDNode>(Op.getOperand(1))->getReg() != Reg) &&
         "Copy from a reg to the same reg!");
  assert(!TargetRegisterInfo::isPhysicalRegister(Reg) && "Is a physreg");

  RegsForValue RFV(V->getContext(), TLI, Reg, V->getType());
  SDValue Chain = DAG.getEntryNode();
  RFV.getCopyToRegs(Op, DAG, getCurDebugLoc(), Chain, 0, V);
  PendingExports.push_back(Chain);
}

#include "llvm/CodeGen/SelectionDAGISel.h"

/// isOnlyUsedInEntryBlock - If the specified argument is only used in the
/// entry block, return true.  This includes arguments used by switches, since
/// the switch may expand into multiple basic blocks.
static bool isOnlyUsedInEntryBlock(const Argument *A, bool FastISel) {
  // With FastISel active, we may be splitting blocks, so force creation
  // of virtual registers for all non-dead arguments.
  if (FastISel)
    return A->use_empty();

  const BasicBlock *Entry = A->getParent()->begin();
  for (Value::const_use_iterator UI = A->use_begin(), E = A->use_end();
       UI != E; ++UI) {
    const User *U = *UI;
    if (cast<Instruction>(U)->getParent() != Entry || isa<SwitchInst>(U))
      return false;  // Use not in entry block.
  }
  return true;
}

void SelectionDAGISel::LowerArguments(const BasicBlock *LLVMBB) {
  // If this is the entry block, emit arguments.
  const Function &F = *LLVMBB->getParent();
  SelectionDAG &DAG = SDB->DAG;
  DebugLoc dl = SDB->getCurDebugLoc();
  const DataLayout *TD = TLI.getDataLayout();
  SmallVector<ISD::InputArg, 16> Ins;

  // Check whether the function can return without sret-demotion.
  SmallVector<ISD::OutputArg, 4> Outs;
  GetReturnInfo(F.getReturnType(), F.getAttributes().getRetAttributes(),
                Outs, TLI);

  if (!FuncInfo->CanLowerReturn) {
    // Put in an sret pointer parameter before all the other parameters.
    SmallVector<EVT, 1> ValueVTs;
    ComputeValueVTs(TLI, PointerType::getUnqual(F.getReturnType()), ValueVTs);

    // NOTE: Assuming that a pointer will never break down to more than one VT
    // or one register.
    ISD::ArgFlagsTy Flags;
    Flags.setSRet();
    EVT RegisterVT = TLI.getRegisterType(*DAG.getContext(), ValueVTs[0]);
    ISD::InputArg RetArg(Flags, RegisterVT, true, 0, 0);
    Ins.push_back(RetArg);
  }

  // Set up the incoming argument description vector.
  unsigned Idx = 1;
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I, ++Idx) {
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(TLI, I->getType(), ValueVTs);
    bool isArgValueUsed = !I->use_empty();
    for (unsigned Value = 0, NumValues = ValueVTs.size();
         Value != NumValues; ++Value) {
      EVT VT = ValueVTs[Value];
      Type *ArgTy = VT.getTypeForEVT(*DAG.getContext());
      ISD::ArgFlagsTy Flags;
      unsigned OriginalAlignment =
        TD->getABITypeAlignment(ArgTy);

      if (F.getParamAttributes(Idx).hasAttribute(Attributes::ZExt))
        Flags.setZExt();
      if (F.getParamAttributes(Idx).hasAttribute(Attributes::SExt))
        Flags.setSExt();
      if (F.getParamAttributes(Idx).hasAttribute(Attributes::InReg))
        Flags.setInReg();
      if (F.getParamAttributes(Idx).hasAttribute(Attributes::StructRet))
        Flags.setSRet();
      if (F.getParamAttributes(Idx).hasAttribute(Attributes::ByVal)) {
        Flags.setByVal();
        PointerType *Ty = cast<PointerType>(I->getType());
        Type *ElementTy = Ty->getElementType();
        Flags.setByValSize(TD->getTypeAllocSize(ElementTy));
        // For ByVal, alignment should be passed from FE.  BE will guess if
        // this info is not there but there are cases it cannot get right.
        unsigned FrameAlign;
        if (F.getParamAlignment(Idx))
          FrameAlign = F.getParamAlignment(Idx);
        else
          FrameAlign = TLI.getByValTypeAlignment(ElementTy);
        Flags.setByValAlign(FrameAlign);
      }
      if (F.getParamAttributes(Idx).hasAttribute(Attributes::Nest))
        Flags.setNest();
      Flags.setOrigAlign(OriginalAlignment);

      EVT RegisterVT = TLI.getRegisterType(*CurDAG->getContext(), VT);
      unsigned NumRegs = TLI.getNumRegisters(*CurDAG->getContext(), VT);
      for (unsigned i = 0; i != NumRegs; ++i) {
        ISD::InputArg MyFlags(Flags, RegisterVT, isArgValueUsed,
                              Idx-1, i*RegisterVT.getStoreSize());
        if (NumRegs > 1 && i == 0)
          MyFlags.Flags.setSplit();
        // if it isn't first piece, alignment must be 1
        else if (i > 0)
          MyFlags.Flags.setOrigAlign(1);
        Ins.push_back(MyFlags);
      }
    }
  }

  // Call the target to set up the argument values.
  SmallVector<SDValue, 8> InVals;
  SDValue NewRoot = TLI.LowerFormalArguments(DAG.getRoot(), F.getCallingConv(),
                                             F.isVarArg(), Ins,
                                             dl, DAG, InVals);

  // Verify that the target's LowerFormalArguments behaved as expected.
  assert(NewRoot.getNode() && NewRoot.getValueType() == MVT::Other &&
         "LowerFormalArguments didn't return a valid chain!");
  assert(InVals.size() == Ins.size() &&
         "LowerFormalArguments didn't emit the correct number of values!");
  DEBUG({
      for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
        assert(InVals[i].getNode() &&
               "LowerFormalArguments emitted a null value!");
        assert(EVT(Ins[i].VT) == InVals[i].getValueType() &&
               "LowerFormalArguments emitted a value with the wrong type!");
      }
    });

  // Update the DAG with the new chain value resulting from argument lowering.
  DAG.setRoot(NewRoot);

  // Set up the argument values.
  unsigned i = 0;
  Idx = 1;
  if (!FuncInfo->CanLowerReturn) {
    // Create a virtual register for the sret pointer, and put in a copy
    // from the sret argument into it.
    SmallVector<EVT, 1> ValueVTs;
    ComputeValueVTs(TLI, PointerType::getUnqual(F.getReturnType()), ValueVTs);
    EVT VT = ValueVTs[0];
    EVT RegVT = TLI.getRegisterType(*CurDAG->getContext(), VT);
    ISD::NodeType AssertOp = ISD::DELETED_NODE;
    SDValue ArgValue = getCopyFromParts(DAG, dl, &InVals[0], 1,
                                        RegVT, VT, NULL, AssertOp);

    MachineFunction& MF = SDB->DAG.getMachineFunction();
    MachineRegisterInfo& RegInfo = MF.getRegInfo();
    unsigned SRetReg = RegInfo.createVirtualRegister(TLI.getRegClassFor(RegVT));
    FuncInfo->DemoteRegister = SRetReg;
    NewRoot = SDB->DAG.getCopyToReg(NewRoot, SDB->getCurDebugLoc(),
                                    SRetReg, ArgValue);
    DAG.setRoot(NewRoot);

    // i indexes lowered arguments.  Bump it past the hidden sret argument.
    // Idx indexes LLVM arguments.  Don't touch it.
    ++i;
  }

  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E;
      ++I, ++Idx) {
    SmallVector<SDValue, 4> ArgValues;
    SmallVector<EVT, 4> ValueVTs;
    ComputeValueVTs(TLI, I->getType(), ValueVTs);
    unsigned NumValues = ValueVTs.size();

    // If this argument is unused then remember its value. It is used to generate
    // debugging information.
    if (I->use_empty() && NumValues)
      SDB->setUnusedArgValue(I, InVals[i]);

    for (unsigned Val = 0; Val != NumValues; ++Val) {
      EVT VT = ValueVTs[Val];
      EVT PartVT = TLI.getRegisterType(*CurDAG->getContext(), VT);
      unsigned NumParts = TLI.getNumRegisters(*CurDAG->getContext(), VT);

      if (!I->use_empty()) {
        ISD::NodeType AssertOp = ISD::DELETED_NODE;
        if (F.getParamAttributes(Idx).hasAttribute(Attributes::SExt))
          AssertOp = ISD::AssertSext;
        else if (F.getParamAttributes(Idx).hasAttribute(Attributes::ZExt))
          AssertOp = ISD::AssertZext;

        ArgValues.push_back(getCopyFromParts(DAG, dl, &InVals[i],
                                             NumParts, PartVT, VT,
                                             NULL, AssertOp));
      }

      i += NumParts;
    }

    // We don't need to do anything else for unused arguments.
    if (ArgValues.empty())
      continue;

    // Note down frame index.
    if (FrameIndexSDNode *FI =
        dyn_cast<FrameIndexSDNode>(ArgValues[0].getNode()))
      FuncInfo->setArgumentFrameIndex(I, FI->getIndex());

    SDValue Res = DAG.getMergeValues(&ArgValues[0], NumValues,
                                     SDB->getCurDebugLoc());

    SDB->setValue(I, Res);
    if (!TM.Options.EnableFastISel && Res.getOpcode() == ISD::BUILD_PAIR) {
      if (LoadSDNode *LNode = 
          dyn_cast<LoadSDNode>(Res.getOperand(0).getNode()))
        if (FrameIndexSDNode *FI =
            dyn_cast<FrameIndexSDNode>(LNode->getBasePtr().getNode()))
        FuncInfo->setArgumentFrameIndex(I, FI->getIndex());
    }

    // If this argument is live outside of the entry block, insert a copy from
    // wherever we got it to the vreg that other BB's will reference it as.
    if (!TM.Options.EnableFastISel && Res.getOpcode() == ISD::CopyFromReg) {
      // If we can, though, try to skip creating an unnecessary vreg.
      // FIXME: This isn't very clean... it would be nice to make this more
      // general.  It's also subtly incompatible with the hacks FastISel
      // uses with vregs.
      unsigned Reg = cast<RegisterSDNode>(Res.getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        FuncInfo->ValueMap[I] = Reg;
        continue;
      }
    }
    if (!isOnlyUsedInEntryBlock(I, TM.Options.EnableFastISel)) {
      FuncInfo->InitializeRegForValue(I);
      SDB->CopyToExportRegsIfNeeded(I);
    }
  }

  assert(i == InVals.size() && "Argument register count mismatch!");

  // Finally, if the target has anything special to do, allow it to do so.
  // FIXME: this should insert code into the DAG!
  EmitFunctionEntryCode();
}

/// Handle PHI nodes in successor blocks.  Emit code into the SelectionDAG to
/// ensure constants are generated when needed.  Remember the virtual registers
/// that need to be added to the Machine PHI nodes as input.  We cannot just
/// directly add them, because expansion might result in multiple MBB's for one
/// BB.  As such, the start of the BB might correspond to a different MBB than
/// the end.
///
void
SelectionDAGBuilder::HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB) {
  const TerminatorInst *TI = LLVMBB->getTerminator();

  SmallPtrSet<MachineBasicBlock *, 4> SuccsHandled;

  // Check successor nodes' PHI nodes that expect a constant to be available
  // from this block.
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    const BasicBlock *SuccBB = TI->getSuccessor(succ);
    if (!isa<PHINode>(SuccBB->begin())) continue;
    MachineBasicBlock *SuccMBB = FuncInfo.MBBMap[SuccBB];

    // If this terminator has multiple identical successors (common for
    // switches), only handle each succ once.
    if (!SuccsHandled.insert(SuccMBB)) continue;

    MachineBasicBlock::iterator MBBI = SuccMBB->begin();

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::const_iterator I = SuccBB->begin();
         const PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      // Ignore dead phi's.
      if (PN->use_empty()) continue;

      // Skip empty types
      if (PN->getType()->isEmptyTy())
        continue;

      unsigned Reg;
      const Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);

      if (const Constant *C = dyn_cast<Constant>(PHIOp)) {
        unsigned &RegOut = ConstantsOut[C];
        if (RegOut == 0) {
          RegOut = FuncInfo.CreateRegs(C->getType());
          CopyValueToVirtualRegister(C, RegOut);
        }
        Reg = RegOut;
      } else {
        DenseMap<const Value *, unsigned>::iterator I =
          FuncInfo.ValueMap.find(PHIOp);
        if (I != FuncInfo.ValueMap.end())
          Reg = I->second;
        else {
          assert(isa<AllocaInst>(PHIOp) &&
                 FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(PHIOp)) &&
                 "Didn't codegen value into a register!??");
          Reg = FuncInfo.CreateRegs(PHIOp->getType());
          CopyValueToVirtualRegister(PHIOp, Reg);
        }
      }

      // Remember that this register needs to added to the machine PHI node as
      // the input for this MBB.
      SmallVector<EVT, 4> ValueVTs;
      ComputeValueVTs(TLI, PN->getType(), ValueVTs);
      for (unsigned vti = 0, vte = ValueVTs.size(); vti != vte; ++vti) {
        EVT VT = ValueVTs[vti];
        unsigned NumRegisters = TLI.getNumRegisters(*DAG.getContext(), VT);
        for (unsigned i = 0, e = NumRegisters; i != e; ++i)
          FuncInfo.PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg+i));
        Reg += NumRegisters;
      }
    }
  }
  ConstantsOut.clear();
}
