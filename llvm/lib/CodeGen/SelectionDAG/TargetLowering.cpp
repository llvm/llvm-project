//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// InitLibcallNames - Set default libcall names.
///
static void InitLibcallNames(const char **Names) {
  Names[RTLIB::SHL_I32] = "__ashlsi3";
  Names[RTLIB::SHL_I64] = "__ashldi3";
  Names[RTLIB::SRL_I32] = "__lshrsi3";
  Names[RTLIB::SRL_I64] = "__lshrdi3";
  Names[RTLIB::SRA_I32] = "__ashrsi3";
  Names[RTLIB::SRA_I64] = "__ashrdi3";
  Names[RTLIB::MUL_I32] = "__mulsi3";
  Names[RTLIB::MUL_I64] = "__muldi3";
  Names[RTLIB::SDIV_I32] = "__divsi3";
  Names[RTLIB::SDIV_I64] = "__divdi3";
  Names[RTLIB::UDIV_I32] = "__udivsi3";
  Names[RTLIB::UDIV_I64] = "__udivdi3";
  Names[RTLIB::SREM_I32] = "__modsi3";
  Names[RTLIB::SREM_I64] = "__moddi3";
  Names[RTLIB::UREM_I32] = "__umodsi3";
  Names[RTLIB::UREM_I64] = "__umoddi3";
  Names[RTLIB::NEG_I32] = "__negsi2";
  Names[RTLIB::NEG_I64] = "__negdi2";
  Names[RTLIB::ADD_F32] = "__addsf3";
  Names[RTLIB::ADD_F64] = "__adddf3";
  Names[RTLIB::SUB_F32] = "__subsf3";
  Names[RTLIB::SUB_F64] = "__subdf3";
  Names[RTLIB::MUL_F32] = "__mulsf3";
  Names[RTLIB::MUL_F64] = "__muldf3";
  Names[RTLIB::DIV_F32] = "__divsf3";
  Names[RTLIB::DIV_F64] = "__divdf3";
  Names[RTLIB::REM_F32] = "fmodf";
  Names[RTLIB::REM_F64] = "fmod";
  Names[RTLIB::NEG_F32] = "__negsf2";
  Names[RTLIB::NEG_F64] = "__negdf2";
  Names[RTLIB::POWI_F32] = "__powisf2";
  Names[RTLIB::POWI_F64] = "__powidf2";
  Names[RTLIB::SQRT_F32] = "sqrtf";
  Names[RTLIB::SQRT_F64] = "sqrt";
  Names[RTLIB::SIN_F32] = "sinf";
  Names[RTLIB::SIN_F64] = "sin";
  Names[RTLIB::COS_F32] = "cosf";
  Names[RTLIB::COS_F64] = "cos";
  Names[RTLIB::FPEXT_F32_F64] = "__extendsfdf2";
  Names[RTLIB::FPROUND_F64_F32] = "__truncdfsf2";
  Names[RTLIB::FPTOSINT_F32_I32] = "__fixsfsi";
  Names[RTLIB::FPTOSINT_F32_I64] = "__fixsfdi";
  Names[RTLIB::FPTOSINT_F64_I32] = "__fixdfsi";
  Names[RTLIB::FPTOSINT_F64_I64] = "__fixdfdi";
  Names[RTLIB::FPTOUINT_F32_I32] = "__fixunssfsi";
  Names[RTLIB::FPTOUINT_F32_I64] = "__fixunssfdi";
  Names[RTLIB::FPTOUINT_F64_I32] = "__fixunsdfsi";
  Names[RTLIB::FPTOUINT_F64_I64] = "__fixunsdfdi";
  Names[RTLIB::SINTTOFP_I32_F32] = "__floatsisf";
  Names[RTLIB::SINTTOFP_I32_F64] = "__floatsidf";
  Names[RTLIB::SINTTOFP_I64_F32] = "__floatdisf";
  Names[RTLIB::SINTTOFP_I64_F64] = "__floatdidf";
  Names[RTLIB::UINTTOFP_I32_F32] = "__floatunsisf";
  Names[RTLIB::UINTTOFP_I32_F64] = "__floatunsidf";
  Names[RTLIB::UINTTOFP_I64_F32] = "__floatundisf";
  Names[RTLIB::UINTTOFP_I64_F64] = "__floatundidf";
  Names[RTLIB::OEQ_F32] = "__eqsf2";
  Names[RTLIB::OEQ_F64] = "__eqdf2";
  Names[RTLIB::UNE_F32] = "__nesf2";
  Names[RTLIB::UNE_F64] = "__nedf2";
  Names[RTLIB::OGE_F32] = "__gesf2";
  Names[RTLIB::OGE_F64] = "__gedf2";
  Names[RTLIB::OLT_F32] = "__ltsf2";
  Names[RTLIB::OLT_F64] = "__ltdf2";
  Names[RTLIB::OLE_F32] = "__lesf2";
  Names[RTLIB::OLE_F64] = "__ledf2";
  Names[RTLIB::OGT_F32] = "__gtsf2";
  Names[RTLIB::OGT_F64] = "__gtdf2";
  Names[RTLIB::UO_F32] = "__unordsf2";
  Names[RTLIB::UO_F64] = "__unorddf2";
  Names[RTLIB::O_F32] = "__unordsf2";
  Names[RTLIB::O_F64] = "__unorddf2";
}

/// InitCmpLibcallCCs - Set default comparison libcall CC.
///
static void InitCmpLibcallCCs(ISD::CondCode *CCs) {
  memset(CCs, ISD::SETCC_INVALID, sizeof(ISD::CondCode)*RTLIB::UNKNOWN_LIBCALL);
  CCs[RTLIB::OEQ_F32] = ISD::SETEQ;
  CCs[RTLIB::OEQ_F64] = ISD::SETEQ;
  CCs[RTLIB::UNE_F32] = ISD::SETNE;
  CCs[RTLIB::UNE_F64] = ISD::SETNE;
  CCs[RTLIB::OGE_F32] = ISD::SETGE;
  CCs[RTLIB::OGE_F64] = ISD::SETGE;
  CCs[RTLIB::OLT_F32] = ISD::SETLT;
  CCs[RTLIB::OLT_F64] = ISD::SETLT;
  CCs[RTLIB::OLE_F32] = ISD::SETLE;
  CCs[RTLIB::OLE_F64] = ISD::SETLE;
  CCs[RTLIB::OGT_F32] = ISD::SETGT;
  CCs[RTLIB::OGT_F64] = ISD::SETGT;
  CCs[RTLIB::UO_F32] = ISD::SETNE;
  CCs[RTLIB::UO_F64] = ISD::SETNE;
  CCs[RTLIB::O_F32] = ISD::SETEQ;
  CCs[RTLIB::O_F64] = ISD::SETEQ;
}

TargetLowering::TargetLowering(TargetMachine &tm)
  : TM(tm), TD(TM.getTargetData()) {
  assert(ISD::BUILTIN_OP_END <= 156 &&
         "Fixed size array in TargetLowering is not large enough!");
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));
  memset(LoadXActions, 0, sizeof(LoadXActions));
  memset(&StoreXActions, 0, sizeof(StoreXActions));
  memset(&IndexedModeActions, 0, sizeof(IndexedModeActions));
  memset(&ConvertActions, 0, sizeof(ConvertActions));

  // Set all indexed load / store to expand.
  for (unsigned VT = 0; VT != (unsigned)MVT::LAST_VALUETYPE; ++VT) {
    for (unsigned IM = (unsigned)ISD::PRE_INC;
         IM != (unsigned)ISD::LAST_INDEXED_MODE; ++IM) {
      setIndexedLoadAction(IM, (MVT::ValueType)VT, Expand);
      setIndexedStoreAction(IM, (MVT::ValueType)VT, Expand);
    }
  }

  IsLittleEndian = TD->isLittleEndian();
  UsesGlobalOffsetTable = false;
  ShiftAmountTy = SetCCResultTy = PointerTy = getValueType(TD->getIntPtrType());
  ShiftAmtHandling = Undefined;
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
  memset(TargetDAGCombineArray, 0, array_lengthof(TargetDAGCombineArray));
  maxStoresPerMemset = maxStoresPerMemcpy = maxStoresPerMemmove = 8;
  allowUnalignedMemoryAccesses = false;
  UseUnderscoreSetJmp = false;
  UseUnderscoreLongJmp = false;
  SelectIsExpensive = false;
  IntDivIsCheap = false;
  Pow2DivIsCheap = false;
  StackPointerRegisterToSaveRestore = 0;
  ExceptionPointerRegister = 0;
  ExceptionSelectorRegister = 0;
  SetCCResultContents = UndefinedSetCCResult;
  SchedPreferenceInfo = SchedulingForLatency;
  JumpBufSize = 0;
  JumpBufAlignment = 0;
  IfCvtBlockSizeLimit = 2;

  InitLibcallNames(LibcallRoutineNames);
  InitCmpLibcallCCs(CmpLibcallCCs);
}

TargetLowering::~TargetLowering() {}

/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= 32 &&
         "Too many value types for ValueTypeActions to hold!");

  // Everything defaults to needing one register.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i) {
    NumRegistersForVT[i] = 1;
    RegisterTypeForVT[i] = TransformToType[i] = i;
  }
  // ...except isVoid, which doesn't need any registers.
  NumRegistersForVT[MVT::isVoid] = 0;

  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::i128;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  for (MVT::ValueType ExpandedReg = LargestIntReg + 1;
       MVT::isInteger(ExpandedReg); ++ExpandedReg) {
    NumRegistersForVT[ExpandedReg] = 2*NumRegistersForVT[ExpandedReg-1];
    RegisterTypeForVT[ExpandedReg] = LargestIntReg;
    TransformToType[ExpandedReg] = ExpandedReg - 1;
    ValueTypeActions.setTypeAction(ExpandedReg, Expand);
  }

  // Inspect all of the ValueType's smaller than the largest integer
  // register to see which ones need promotion.
  MVT::ValueType LegalIntReg = LargestIntReg;
  for (MVT::ValueType IntReg = LargestIntReg - 1;
       IntReg >= MVT::i1; --IntReg) {
    if (isTypeLegal(IntReg)) {
      LegalIntReg = IntReg;
    } else {
      RegisterTypeForVT[IntReg] = TransformToType[IntReg] = LegalIntReg;
      ValueTypeActions.setTypeAction(IntReg, Promote);
    }
  }

  // Decide how to handle f64. If the target does not have native f64 support,
  // expand it to i64 and we will be generating soft float library calls.
  if (!isTypeLegal(MVT::f64)) {
    NumRegistersForVT[MVT::f64] = NumRegistersForVT[MVT::i64];
    RegisterTypeForVT[MVT::f64] = RegisterTypeForVT[MVT::i64];
    TransformToType[MVT::f64] = MVT::i64;
    ValueTypeActions.setTypeAction(MVT::f64, Expand);
  }

  // Decide how to handle f32. If the target does not have native support for
  // f32, promote it to f64 if it is legal. Otherwise, expand it to i32.
  if (!isTypeLegal(MVT::f32)) {
    if (isTypeLegal(MVT::f64)) {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::f64];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::f64];
      TransformToType[MVT::f32] = MVT::f64;
      ValueTypeActions.setTypeAction(MVT::f32, Promote);
    } else {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::i32];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::i32];
      TransformToType[MVT::f32] = MVT::i32;
      ValueTypeActions.setTypeAction(MVT::f32, Expand);
    }
  }
  
  // Loop over all of the vector value types to see which need transformations.
  for (MVT::ValueType i = MVT::FIRST_VECTOR_VALUETYPE;
       i <= MVT::LAST_VECTOR_VALUETYPE; ++i) {
    if (!isTypeLegal(i)) {
      MVT::ValueType IntermediateVT, RegisterVT;
      unsigned NumIntermediates;
      NumRegistersForVT[i] =
        getVectorTypeBreakdown(i,
                               IntermediateVT, NumIntermediates,
                               RegisterVT);
      RegisterTypeForVT[i] = RegisterVT;
      TransformToType[i] = MVT::Other; // this isn't actually used
      ValueTypeActions.setTypeAction(i, Expand);
    }
  }
}

const char *TargetLowering::getTargetNodeName(unsigned Opcode) const {
  return NULL;
}

/// getVectorTypeBreakdown - Vector types are broken down into some number of
/// legal first class types.  For example, MVT::v8f32 maps to 2 MVT::v4f32
/// with Altivec or SSE1, or 8 promoted MVT::f64 values with the X86 FP stack.
/// Similarly, MVT::v2i64 turns into 4 MVT::i32 values with both PPC and X86.
///
/// This method returns the number of registers needed, and the VT for each
/// register.  It also returns the VT and quantity of the intermediate values
/// before they are promoted/expanded.
///
unsigned TargetLowering::getVectorTypeBreakdown(MVT::ValueType VT, 
                                                MVT::ValueType &IntermediateVT,
                                                unsigned &NumIntermediates,
                                      MVT::ValueType &RegisterVT) const {
  // Figure out the right, legal destination reg to copy into.
  unsigned NumElts = MVT::getVectorNumElements(VT);
  MVT::ValueType EltTy = MVT::getVectorElementType(VT);
  
  unsigned NumVectorRegs = 1;
  
  // Divide the input until we get to a supported size.  This will always
  // end with a scalar if the target doesn't support vectors.
  while (NumElts > 1 &&
         !isTypeLegal(MVT::getVectorType(EltTy, NumElts))) {
    NumElts >>= 1;
    NumVectorRegs <<= 1;
  }

  NumIntermediates = NumVectorRegs;
  
  MVT::ValueType NewVT = MVT::getVectorType(EltTy, NumElts);
  if (!isTypeLegal(NewVT))
    NewVT = EltTy;
  IntermediateVT = NewVT;

  MVT::ValueType DestVT = getTypeToTransformTo(NewVT);
  RegisterVT = DestVT;
  if (DestVT < NewVT) {
    // Value is expanded, e.g. i64 -> i16.
    return NumVectorRegs*(MVT::getSizeInBits(NewVT)/MVT::getSizeInBits(DestVT));
  } else {
    // Otherwise, promotion or legal types use the same number of registers as
    // the vector decimated to the appropriate level.
    return NumVectorRegs;
  }
  
  return 1;
}

//===----------------------------------------------------------------------===//
//  Optimization Methods
//===----------------------------------------------------------------------===//

/// ShrinkDemandedConstant - Check to see if the specified operand of the 
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
bool TargetLowering::TargetLoweringOpt::ShrinkDemandedConstant(SDOperand Op, 
                                                            uint64_t Demanded) {
  // FIXME: ISD::SELECT, ISD::SELECT_CC
  switch(Op.getOpcode()) {
  default: break;
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1)))
      if ((~Demanded & C->getValue()) != 0) {
        MVT::ValueType VT = Op.getValueType();
        SDOperand New = DAG.getNode(Op.getOpcode(), VT, Op.getOperand(0),
                                    DAG.getConstant(Demanded & C->getValue(), 
                                                    VT));
        return CombineTo(Op, New);
      }
    break;
  }
  return false;
}

/// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
/// DemandedMask bits of the result of Op are ever used downstream.  If we can
/// use this information to simplify Op, create a new simplified DAG node and
/// return true, returning the original and new nodes in Old and New. Otherwise,
/// analyze the expression and return a mask of KnownOne and KnownZero bits for
/// the expression (used to simplify the caller).  The KnownZero/One bits may
/// only be accurate for those bits in the DemandedMask.
bool TargetLowering::SimplifyDemandedBits(SDOperand Op, uint64_t DemandedMask, 
                                          uint64_t &KnownZero,
                                          uint64_t &KnownOne,
                                          TargetLoweringOpt &TLO,
                                          unsigned Depth) const {
  KnownZero = KnownOne = 0;   // Don't know anything.

  // The masks are not wide enough to represent this type!  Should use APInt.
  if (Op.getValueType() == MVT::i128)
    return false;
  
  // Other users may use these bits.
  if (!Op.Val->hasOneUse()) { 
    if (Depth != 0) {
      // If not at the root, Just compute the KnownZero/KnownOne bits to 
      // simplify things downstream.
      TLO.DAG.ComputeMaskedBits(Op, DemandedMask, KnownZero, KnownOne, Depth);
      return false;
    }
    // If this is the root being simplified, allow it to have multiple uses,
    // just set the DemandedMask to all bits.
    DemandedMask = MVT::getIntVTBitMask(Op.getValueType());
  } else if (DemandedMask == 0) {   
    // Not demanding any bits from Op.
    if (Op.getOpcode() != ISD::UNDEF)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::UNDEF, Op.getValueType()));
    return false;
  } else if (Depth == 6) {        // Limit search depth.
    return false;
  }

  uint64_t KnownZero2, KnownOne2, KnownZeroOut, KnownOneOut;
  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getValue() & DemandedMask;
    KnownZero = ~KnownOne & DemandedMask;
    return false;   // Don't fall through, will infinitely loop.
  case ISD::AND:
    // If the RHS is a constant, check to see if the LHS would be zero without
    // using the bits from the RHS.  Below, we use knowledge about the RHS to
    // simplify the LHS, here we're using information from the LHS to simplify
    // the RHS.
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      uint64_t LHSZero, LHSOne;
      TLO.DAG.ComputeMaskedBits(Op.getOperand(0), DemandedMask,
                                LHSZero, LHSOne, Depth+1);
      // If the LHS already has zeros where RHSC does, this and is dead.
      if ((LHSZero & DemandedMask) == (~RHSC->getValue() & DemandedMask))
        return TLO.CombineTo(Op, Op.getOperand(0));
      // If any of the set bits in the RHS are known zero on the LHS, shrink
      // the constant.
      if (TLO.ShrinkDemandedConstant(Op, ~LHSZero & DemandedMask))
        return true;
    }
    
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & ~KnownZero,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
      
    // If all of the demanded bits are known one on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((DemandedMask & ~KnownZero2 & KnownOne)==(DemandedMask & ~KnownZero2))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & ~KnownZero & KnownOne2)==(DemandedMask & ~KnownZero))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((DemandedMask & (KnownZero|KnownZero2)) == DemandedMask)
      return TLO.CombineTo(Op, TLO.DAG.getConstant(0, Op.getValueType()));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask & ~KnownZero2))
      return true;
      
    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case ISD::OR:
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & ~KnownOne, 
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((DemandedMask & ~KnownOne2 & KnownZero) == (DemandedMask & ~KnownOne2))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & ~KnownOne & KnownZero2) == (DemandedMask & ~KnownOne))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((DemandedMask & (~KnownZero) & KnownOne2) == 
        (DemandedMask & (~KnownZero)))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & (~KnownZero2) & KnownOne) == 
        (DemandedMask & (~KnownZero2)))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
          
    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case ISD::XOR:
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((DemandedMask & KnownZero) == DemandedMask)
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((DemandedMask & KnownZero2) == DemandedMask)
      return TLO.CombineTo(Op, Op.getOperand(1));
      
    // If all of the unknown bits are known to be zero on one side or the other
    // (but not both) turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if ((DemandedMask & ~KnownZero & ~KnownZero2) == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::OR, Op.getValueType(),
                                               Op.getOperand(0),
                                               Op.getOperand(1)));
    
    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOneOut = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);
    
    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask) { // all known
      if ((KnownOne & KnownOne2) == KnownOne) {
        MVT::ValueType VT = Op.getValueType();
        SDOperand ANDC = TLO.DAG.getConstant(~KnownOne & DemandedMask, VT);
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::AND, VT, Op.getOperand(0),
                                                 ANDC));
      }
    }
    
    // If the RHS is a constant, see if we can simplify it.
    // FIXME: for XOR, we prefer to force bits to 1 if they will make a -1.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
    
    KnownZero = KnownZeroOut;
    KnownOne  = KnownOneOut;
    break;
  case ISD::SETCC:
    // If we know the result of a setcc has the top bits zero, use this info.
    if (getSetCCResultContents() == TargetLowering::ZeroOrOneSetCCResult)
      KnownZero |= (MVT::getIntVTBitMask(Op.getValueType()) ^ 1ULL);
    break;
  case ISD::SELECT:
    if (SimplifyDemandedBits(Op.getOperand(2), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), DemandedMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
    
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SELECT_CC:
    if (SimplifyDemandedBits(Op.getOperand(3), DemandedMask, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(2), DemandedMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?"); 
    
    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, DemandedMask))
      return true;
      
    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SHL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getValue();
      SDOperand InOp = Op.getOperand(0);

      // If this is ((X >>u C1) << ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the bottom bits (which are shifted
      // out) are never demanded.
      if (InOp.getOpcode() == ISD::SRL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (DemandedMask & ((1ULL << ShAmt)-1)) == 0) {
          unsigned C1 = cast<ConstantSDNode>(InOp.getOperand(1))->getValue();
          unsigned Opc = ISD::SHL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SRL;
          }          
          
          SDOperand NewSA = 
            TLO.DAG.getConstant(Diff, Op.getOperand(1).getValueType());
          MVT::ValueType VT = Op.getValueType();
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }      
      
      if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask >> ShAmt,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      KnownZero <<= SA->getValue();
      KnownOne  <<= SA->getValue();
      KnownZero |= (1ULL << SA->getValue())-1;  // low bits known zero.
    }
    break;
  case ISD::SRL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT::ValueType VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();
      uint64_t TypeMask = MVT::getIntVTBitMask(VT);
      unsigned VTSize = MVT::getSizeInBits(VT);
      SDOperand InOp = Op.getOperand(0);
      
      // If this is ((X << C1) >>u ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the top bits (which are shifted out)
      // are never demanded.
      if (InOp.getOpcode() == ISD::SHL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (DemandedMask & (~0ULL << (VTSize-ShAmt))) == 0) {
          unsigned C1 = cast<ConstantSDNode>(InOp.getOperand(1))->getValue();
          unsigned Opc = ISD::SRL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SHL;
          }          
          
          SDOperand NewSA =
            TLO.DAG.getConstant(Diff, Op.getOperand(1).getValueType());
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }      
      
      // Compute the new bits that are at the top now.
      if (SimplifyDemandedBits(InOp, (DemandedMask << ShAmt) & TypeMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero &= TypeMask;
      KnownOne  &= TypeMask;
      KnownZero >>= ShAmt;
      KnownOne  >>= ShAmt;

      uint64_t HighBits = (1ULL << ShAmt)-1;
      HighBits <<= VTSize - ShAmt;
      KnownZero |= HighBits;  // High bits known zero.
    }
    break;
  case ISD::SRA:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      MVT::ValueType VT = Op.getValueType();
      unsigned ShAmt = SA->getValue();
      
      // Compute the new bits that are at the top now.
      uint64_t TypeMask = MVT::getIntVTBitMask(VT);
      
      uint64_t InDemandedMask = (DemandedMask << ShAmt) & TypeMask;

      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      uint64_t HighBits = (1ULL << ShAmt)-1;
      HighBits <<= MVT::getSizeInBits(VT) - ShAmt;
      if (HighBits & DemandedMask)
        InDemandedMask |= MVT::getIntVTSignBit(VT);
      
      if (SimplifyDemandedBits(Op.getOperand(0), InDemandedMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
      KnownZero &= TypeMask;
      KnownOne  &= TypeMask;
      KnownZero >>= ShAmt;
      KnownOne  >>= ShAmt;
      
      // Handle the sign bits.
      uint64_t SignBit = MVT::getIntVTSignBit(VT);
      SignBit >>= ShAmt;  // Adjust to where it is now in the mask.
      
      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if ((KnownZero & SignBit) || (HighBits & ~DemandedMask) == HighBits) {
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, VT, Op.getOperand(0),
                                                 Op.getOperand(1)));
      } else if (KnownOne & SignBit) { // New bits are known one.
        KnownOne |= HighBits;
      }
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    MVT::ValueType EVT = cast<VTSDNode>(Op.getOperand(1))->getVT();

    // Sign extension.  Compute the demanded bits in the result that are not 
    // present in the input.
    uint64_t NewBits = ~MVT::getIntVTBitMask(EVT) & DemandedMask;
    
    // If none of the extended bits are demanded, eliminate the sextinreg.
    if (NewBits == 0)
      return TLO.CombineTo(Op, Op.getOperand(0));

    uint64_t InSignBit = MVT::getIntVTSignBit(EVT);
    int64_t InputDemandedBits = DemandedMask & MVT::getIntVTBitMask(EVT);
    
    // Since the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InputDemandedBits |= InSignBit;

    if (SimplifyDemandedBits(Op.getOperand(0), InputDemandedBits,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.
    
    // If the input sign bit is known zero, convert this into a zero extension.
    if (KnownZero & InSignBit)
      return TLO.CombineTo(Op, 
                           TLO.DAG.getZeroExtendInReg(Op.getOperand(0), EVT));
    
    if (KnownOne & InSignBit) {    // Input sign bit known set
      KnownOne |= NewBits;
      KnownZero &= ~NewBits;
    } else {                       // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne &= ~NewBits;
    }
    break;
  }
  case ISD::CTTZ:
  case ISD::CTLZ:
  case ISD::CTPOP: {
    MVT::ValueType VT = Op.getValueType();
    unsigned LowBits = Log2_32(MVT::getSizeInBits(VT))+1;
    KnownZero = ~((1ULL << LowBits)-1) & MVT::getIntVTBitMask(VT);
    KnownOne  = 0;
    break;
  }
  case ISD::LOAD: {
    if (ISD::isZEXTLoad(Op.Val)) {
      LoadSDNode *LD = cast<LoadSDNode>(Op);
      MVT::ValueType VT = LD->getLoadedVT();
      KnownZero |= ~MVT::getIntVTBitMask(VT) & DemandedMask;
    }
    break;
  }
  case ISD::ZERO_EXTEND: {
    uint64_t InMask = MVT::getIntVTBitMask(Op.getOperand(0).getValueType());
    
    // If none of the top bits are demanded, convert this into an any_extend.
    uint64_t NewBits = (~InMask) & DemandedMask;
    if (NewBits == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ANY_EXTEND, 
                                               Op.getValueType(), 
                                               Op.getOperand(0)));
    
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero |= NewBits;
    break;
  }
  case ISD::SIGN_EXTEND: {
    MVT::ValueType InVT = Op.getOperand(0).getValueType();
    uint64_t InMask    = MVT::getIntVTBitMask(InVT);
    uint64_t InSignBit = MVT::getIntVTSignBit(InVT);
    uint64_t NewBits   = (~InMask) & DemandedMask;
    
    // If none of the top bits are demanded, convert this into an any_extend.
    if (NewBits == 0)
      return TLO.CombineTo(Op,TLO.DAG.getNode(ISD::ANY_EXTEND,Op.getValueType(),
                                           Op.getOperand(0)));
    
    // Since some of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    uint64_t InDemandedBits = DemandedMask & InMask;
    InDemandedBits |= InSignBit;
    
    if (SimplifyDemandedBits(Op.getOperand(0), InDemandedBits, KnownZero, 
                             KnownOne, TLO, Depth+1))
      return true;
    
    // If the sign bit is known zero, convert this to a zero extend.
    if (KnownZero & InSignBit)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ZERO_EXTEND, 
                                               Op.getValueType(), 
                                               Op.getOperand(0)));
    
    // If the sign bit is known one, the top bits match.
    if (KnownOne & InSignBit) {
      KnownOne  |= NewBits;
      KnownZero &= ~NewBits;
    } else {   // Otherwise, top bits aren't known.
      KnownOne  &= ~NewBits;
      KnownZero &= ~NewBits;
    }
    break;
  }
  case ISD::ANY_EXTEND: {
    uint64_t InMask = MVT::getIntVTBitMask(Op.getOperand(0).getValueType());
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    break;
  }
  case ISD::TRUNCATE: {
    // Simplify the input, using demanded bit information, and compute the known
    // zero/one bits live out.
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    
    // If the input is only used by this truncate, see if we can shrink it based
    // on the known demanded bits.
    if (Op.getOperand(0).Val->hasOneUse()) {
      SDOperand In = Op.getOperand(0);
      switch (In.getOpcode()) {
      default: break;
      case ISD::SRL:
        // Shrink SRL by a constant if none of the high bits shifted in are
        // demanded.
        if (ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(In.getOperand(1))){
          uint64_t HighBits = MVT::getIntVTBitMask(In.getValueType());
          HighBits &= ~MVT::getIntVTBitMask(Op.getValueType());
          HighBits >>= ShAmt->getValue();
          
          if (ShAmt->getValue() < MVT::getSizeInBits(Op.getValueType()) &&
              (DemandedMask & HighBits) == 0) {
            // None of the shifted in bits are needed.  Add a truncate of the
            // shift input, then shift it.
            SDOperand NewTrunc = TLO.DAG.getNode(ISD::TRUNCATE, 
                                                 Op.getValueType(), 
                                                 In.getOperand(0));
            return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL,Op.getValueType(),
                                                   NewTrunc, In.getOperand(1)));
          }
        }
        break;
      }
    }
    
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    uint64_t OutMask = MVT::getIntVTBitMask(Op.getValueType());
    KnownZero &= OutMask;
    KnownOne &= OutMask;
    break;
  }
  case ISD::AssertZext: {
    MVT::ValueType VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    uint64_t InMask = MVT::getIntVTBitMask(VT);
    if (SimplifyDemandedBits(Op.getOperand(0), DemandedMask & InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?"); 
    KnownZero |= ~InMask & DemandedMask;
    break;
  }
  case ISD::ADD:
  case ISD::SUB:
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    // Just use ComputeMaskedBits to compute output bits.
    TLO.DAG.ComputeMaskedBits(Op, DemandedMask, KnownZero, KnownOne, Depth);
    break;
  }
  
  // If we know the value of all of the demanded bits, return this as a
  // constant.
  if ((DemandedMask & (KnownZero|KnownOne)) == DemandedMask)
    return TLO.CombineTo(Op, TLO.DAG.getConstant(KnownOne, Op.getValueType()));
  
  return false;
}

/// computeMaskedBitsForTargetNode - Determine which of the bits specified 
/// in Mask are known to be either zero or one and return them in the 
/// KnownZero/KnownOne bitsets.
void TargetLowering::computeMaskedBitsForTargetNode(const SDOperand Op, 
                                                    uint64_t Mask,
                                                    uint64_t &KnownZero, 
                                                    uint64_t &KnownOne,
                                                    const SelectionDAG &DAG,
                                                    unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");
  KnownZero = 0;
  KnownOne = 0;
}

/// ComputeNumSignBitsForTargetNode - This method can be implemented by
/// targets that want to expose additional information about sign bits to the
/// DAG Combiner.
unsigned TargetLowering::ComputeNumSignBitsForTargetNode(SDOperand Op,
                                                         unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use ComputeNumSignBits if you don't know whether Op"
         " is a target node!");
  return 1;
}


/// SimplifySetCC - Try to simplify a setcc built with the specified operands 
/// and cc. If it is unable to simplify it, return a null SDOperand.
SDOperand
TargetLowering::SimplifySetCC(MVT::ValueType VT, SDOperand N0, SDOperand N1,
                              ISD::CondCode Cond, bool foldBooleans,
                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

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
    if (isa<ConstantSDNode>(N0.Val)) {
      return DAG.FoldSetCC(VT, N0, N1, Cond);
    } else {
      // If the LHS is '(srl (ctlz x), 5)', the RHS is 0/1, and this is an
      // equality comparison, then we're just comparing whether X itself is
      // zero.
      if (N0.getOpcode() == ISD::SRL && (C1 == 0 || C1 == 1) &&
          N0.getOperand(0).getOpcode() == ISD::CTLZ &&
          N0.getOperand(1).getOpcode() == ISD::Constant) {
        unsigned ShAmt = cast<ConstantSDNode>(N0.getOperand(1))->getValue();
        if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
            ShAmt == Log2_32(MVT::getSizeInBits(N0.getValueType()))) {
          if ((C1 == 0) == (Cond == ISD::SETEQ)) {
            // (srl (ctlz x), 5) == 0  -> X != 0
            // (srl (ctlz x), 5) != 1  -> X != 0
            Cond = ISD::SETNE;
          } else {
            // (srl (ctlz x), 5) != 0  -> X == 0
            // (srl (ctlz x), 5) == 1  -> X == 0
            Cond = ISD::SETEQ;
          }
          SDOperand Zero = DAG.getConstant(0, N0.getValueType());
          return DAG.getSetCC(VT, N0.getOperand(0).getOperand(0),
                              Zero, Cond);
        }
      }
      
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
            return DAG.getConstant((C1 & (1ULL << (VSize-1))) != 0, VT);
          case ISD::SETLT:
          case ISD::SETLE:
            // True if the sign bit of C1 isn't set.
            return DAG.getConstant((C1 & (1ULL << (VSize-1))) == 0, VT);
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
        if (!DCI.isCalledByLegalizer())
          DCI.AddToWorklist(ZextOp.Val);
        // Otherwise, make this a use of a zext.
        return DAG.getSetCC(VT, ZextOp, 
                            DAG.getConstant(C1 & (~0ULL>>(64-ExtSrcTyBits)), 
                                            ExtDstTy),
                            Cond);
      } else if ((N1C->getValue() == 0 || N1C->getValue() == 1) &&
                 (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
        
        // SETCC (SETCC), [0|1], [EQ|NE]  -> SETCC
        if (N0.getOpcode() == ISD::SETCC) {
          bool TrueWhenTrue = (Cond == ISD::SETEQ) ^ (N1C->getValue() != 1);
          if (TrueWhenTrue)
            return N0;
          
          // Invert the condition.
          ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
          CC = ISD::getSetCCInverse(CC, 
                               MVT::isInteger(N0.getOperand(0).getValueType()));
          return DAG.getSetCC(VT, N0.getOperand(0), N0.getOperand(1), CC);
        }
        
        if ((N0.getOpcode() == ISD::XOR ||
             (N0.getOpcode() == ISD::AND && 
              N0.getOperand(0).getOpcode() == ISD::XOR &&
              N0.getOperand(1) == N0.getOperand(0).getOperand(1))) &&
            isa<ConstantSDNode>(N0.getOperand(1)) &&
            cast<ConstantSDNode>(N0.getOperand(1))->getValue() == 1) {
          // If this is (X^1) == 0/1, swap the RHS and eliminate the xor.  We
          // can only do this if the top bits are known zero.
          if (DAG.MaskedValueIsZero(N0,
                                    MVT::getIntVTBitMask(N0.getValueType())-1)){
            // Okay, get the un-inverted input value.
            SDOperand Val;
            if (N0.getOpcode() == ISD::XOR)
              Val = N0.getOperand(0);
            else {
              assert(N0.getOpcode() == ISD::AND && 
                     N0.getOperand(0).getOpcode() == ISD::XOR);
              // ((X^1)&1)^1 -> X & 1
              Val = DAG.getNode(ISD::AND, N0.getValueType(),
                                N0.getOperand(0).getOperand(0),
                                N0.getOperand(1));
            }
            return DAG.getSetCC(VT, Val, N1,
                                Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
          }
        }
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
      if ((Cond == ISD::SETGE || Cond == ISD::SETUGE) && C1 == MinVal)
        return DAG.getConstant(1, VT);      // X >= MIN --> true
      if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal)
        return DAG.getConstant(0, VT);      // X > MAX --> false
      if ((Cond == ISD::SETLE || Cond == ISD::SETULE) && C1 == MaxVal)
        return DAG.getConstant(1, VT);      // X <= MAX --> true

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
            if (isPowerOf2_64(AndRHS->getValue())) {
              return DAG.getNode(ISD::SRL, VT, N0,
                             DAG.getConstant(Log2_64(AndRHS->getValue()),
                                             getShiftAmountTy()));
            }
          } else if (Cond == ISD::SETEQ && C1 == AndRHS->getValue()) {
            // (X & 8) == 8  -->  (X & 8) >> 3
            // Perform the xform if C1 is a single bit.
            if (isPowerOf2_64(C1)) {
              return DAG.getNode(ISD::SRL, VT, N0,
                          DAG.getConstant(Log2_64(C1), getShiftAmountTy()));
            }
          }
        }
    }
  } else if (isa<ConstantSDNode>(N0.Val)) {
      // Ensure that the constant occurs on the RHS.
    return DAG.getSetCC(VT, N1, N0, ISD::getSetCCSwappedOperands(Cond));
  }

  if (isa<ConstantFPSDNode>(N0.Val)) {
    // Constant fold or commute setcc.
    SDOperand O = DAG.FoldSetCC(VT, N0, N1, Cond);    
    if (O.Val) return O;
  }

  if (N0 == N1) {
    // We can always fold X == X for integer setcc's.
    if (MVT::isInteger(N0.getValueType()))
      return DAG.getConstant(ISD::isTrueWhenEqual(Cond), VT);
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return DAG.getConstant(ISD::isTrueWhenEqual(Cond), VT);
    if (UOF == unsigned(ISD::isTrueWhenEqual(Cond)))
      return DAG.getConstant(UOF, VT);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    ISD::CondCode NewCond = UOF == 0 ? ISD::SETO : ISD::SETUO;
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
        if (DAG.isCommutativeBinOp(N0.getOpcode())) {
          // If X op Y == Y op X, try other combinations.
          if (N0.getOperand(0) == N1.getOperand(1))
            return DAG.getSetCC(VT, N0.getOperand(1), N1.getOperand(0), Cond);
          if (N0.getOperand(1) == N1.getOperand(0))
            return DAG.getSetCC(VT, N0.getOperand(0), N1.getOperand(1), Cond);
        }
      }
      
      if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(N1)) {
        if (ConstantSDNode *LHSR = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          // Turn (X+C1) == C2 --> X == C2-C1
          if (N0.getOpcode() == ISD::ADD && N0.Val->hasOneUse()) {
            return DAG.getSetCC(VT, N0.getOperand(0),
                              DAG.getConstant(RHSC->getValue()-LHSR->getValue(),
                                N0.getValueType()), Cond);
          }
          
          // Turn (X^C1) == C2 into X == C1^C2 iff X&~C1 = 0.
          if (N0.getOpcode() == ISD::XOR)
            // If we know that all of the inverted bits are zero, don't bother
            // performing the inversion.
            if (DAG.MaskedValueIsZero(N0.getOperand(0), ~LHSR->getValue()))
              return DAG.getSetCC(VT, N0.getOperand(0),
                              DAG.getConstant(LHSR->getValue()^RHSC->getValue(),
                                              N0.getValueType()), Cond);
        }
        
        // Turn (C1-X) == C2 --> X == C1-C2
        if (ConstantSDNode *SUBC = dyn_cast<ConstantSDNode>(N0.getOperand(0))) {
          if (N0.getOpcode() == ISD::SUB && N0.Val->hasOneUse()) {
            return DAG.getSetCC(VT, N0.getOperand(1),
                             DAG.getConstant(SUBC->getValue()-RHSC->getValue(),
                                             N0.getValueType()), Cond);
          }
        }          
      }

      // Simplify (X+Z) == X -->  Z == 0
      if (N0.getOperand(0) == N1)
        return DAG.getSetCC(VT, N0.getOperand(1),
                        DAG.getConstant(0, N0.getValueType()), Cond);
      if (N0.getOperand(1) == N1) {
        if (DAG.isCommutativeBinOp(N0.getOpcode()))
          return DAG.getSetCC(VT, N0.getOperand(0),
                          DAG.getConstant(0, N0.getValueType()), Cond);
        else if (N0.Val->hasOneUse()) {
          assert(N0.getOpcode() == ISD::SUB && "Unexpected operation!");
          // (Z-X) == X  --> Z == X<<1
          SDOperand SH = DAG.getNode(ISD::SHL, N1.getValueType(),
                                     N1, 
                                     DAG.getConstant(1, getShiftAmountTy()));
          if (!DCI.isCalledByLegalizer())
            DCI.AddToWorklist(SH.Val);
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
        if (DAG.isCommutativeBinOp(N1.getOpcode())) {
          return DAG.getSetCC(VT, N1.getOperand(0),
                          DAG.getConstant(0, N1.getValueType()), Cond);
        } else if (N1.Val->hasOneUse()) {
          assert(N1.getOpcode() == ISD::SUB && "Unexpected operation!");
          // X == (Z-X)  --> X<<1 == Z
          SDOperand SH = DAG.getNode(ISD::SHL, N1.getValueType(), N0, 
                                     DAG.getConstant(1, getShiftAmountTy()));
          if (!DCI.isCalledByLegalizer())
            DCI.AddToWorklist(SH.Val);
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
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETNE:  // X != Y   -->  (X^Y)
      N0 = DAG.getNode(ISD::XOR, MVT::i1, N0, N1);
      break;
    case ISD::SETGT:  // X >s Y   -->  X == 0 & Y == 1  -->  X^1 & Y
    case ISD::SETULT: // X <u Y   -->  X == 0 & Y == 1  -->  X^1 & Y
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::AND, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETLT:  // X <s Y   --> X == 1 & Y == 0  -->  Y^1 & X
    case ISD::SETUGT: // X >u Y   --> X == 1 & Y == 0  -->  Y^1 & X
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N1, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::AND, MVT::i1, N0, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETULE: // X <=u Y  --> X == 0 | Y == 1  -->  X^1 | Y
    case ISD::SETGE:  // X >=s Y  --> X == 0 | Y == 1  -->  X^1 | Y
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N0, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::OR, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.Val);
      break;
    case ISD::SETUGE: // X >=u Y  --> X == 1 | Y == 0  -->  Y^1 | X
    case ISD::SETLE:  // X <=s Y  --> X == 1 | Y == 0  -->  Y^1 | X
      Temp = DAG.getNode(ISD::XOR, MVT::i1, N1, DAG.getConstant(1, MVT::i1));
      N0 = DAG.getNode(ISD::OR, MVT::i1, N0, Temp);
      break;
    }
    if (VT != MVT::i1) {
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(N0.Val);
      // FIXME: If running after legalize, we probably can't do this.
      N0 = DAG.getNode(ISD::ZERO_EXTEND, VT, N0);
    }
    return N0;
  }

  // Could not fold it.
  return SDOperand();
}

SDOperand TargetLowering::
PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  // Default implementation: no optimization.
  return SDOperand();
}

//===----------------------------------------------------------------------===//
//  Inline Assembler Implementation Methods
//===----------------------------------------------------------------------===//

TargetLowering::ConstraintType
TargetLowering::getConstraintType(const std::string &Constraint) const {
  // FIXME: lots more standard ones to handle.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'r': return C_RegisterClass;
    case 'm':    // memory
    case 'o':    // offsetable
    case 'V':    // not offsetable
      return C_Memory;
    case 'i':    // Simple Integer or Relocatable Constant
    case 'n':    // Simple Integer
    case 's':    // Relocatable Constant
    case 'X':    // Allow ANY value.
    case 'I':    // Target registers.
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
      return C_Other;
    }
  }
  
  if (Constraint.size() > 1 && Constraint[0] == '{' && 
      Constraint[Constraint.size()-1] == '}')
    return C_Register;
  return C_Unknown;
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void TargetLowering::LowerAsmOperandForConstraint(SDOperand Op,
                                                  char ConstraintLetter,
                                                  std::vector<SDOperand> &Ops,
                                                  SelectionDAG &DAG) {
  switch (ConstraintLetter) {
  default: break;
  case 'i':    // Simple Integer or Relocatable Constant
  case 'n':    // Simple Integer
  case 's':    // Relocatable Constant
  case 'X': {  // Allows any operand.
    // These operands are interested in values of the form (GV+C), where C may
    // be folded in as an offset of GV, or it may be explicitly added.  Also, it
    // is possible and fine if either GV or C are missing.
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op);
    
    // If we have "(add GV, C)", pull out GV/C
    if (Op.getOpcode() == ISD::ADD) {
      C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
      GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(0));
      if (C == 0 || GA == 0) {
        C = dyn_cast<ConstantSDNode>(Op.getOperand(0));
        GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(1));
      }
      if (C == 0 || GA == 0)
        C = 0, GA = 0;
    }
    
    // If we find a valid operand, map to the TargetXXX version so that the
    // value itself doesn't get selected.
    if (GA) {   // Either &GV   or   &GV+C
      if (ConstraintLetter != 'n') {
        int64_t Offs = GA->getOffset();
        if (C) Offs += C->getValue();
        Ops.push_back(DAG.getTargetGlobalAddress(GA->getGlobal(),
                                                 Op.getValueType(), Offs));
        return;
      }
    }
    if (C) {   // just C, no GV.
      // Simple constants are not allowed for 's'.
      if (ConstraintLetter != 's') {
        Ops.push_back(DAG.getTargetConstant(C->getValue(), Op.getValueType()));
        return;
      }
    }
    break;
  }
  }
}

std::vector<unsigned> TargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  return std::vector<unsigned>();
}


std::pair<unsigned, const TargetRegisterClass*> TargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             MVT::ValueType VT) const {
  if (Constraint[0] != '{')
    return std::pair<unsigned, const TargetRegisterClass*>(0, 0);
  assert(*(Constraint.end()-1) == '}' && "Not a brace enclosed constraint?");

  // Remove the braces from around the name.
  std::string RegName(Constraint.begin()+1, Constraint.end()-1);

  // Figure out which register class contains this reg.
  const MRegisterInfo *RI = TM.getRegisterInfo();
  for (MRegisterInfo::regclass_iterator RCI = RI->regclass_begin(),
       E = RI->regclass_end(); RCI != E; ++RCI) {
    const TargetRegisterClass *RC = *RCI;
    
    // If none of the the value types for this register class are valid, we 
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    bool isLegal = false;
    for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
         I != E; ++I) {
      if (isTypeLegal(*I)) {
        isLegal = true;
        break;
      }
    }
    
    if (!isLegal) continue;
    
    for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end(); 
         I != E; ++I) {
      if (StringsEqualNoCase(RegName, RI->get(*I).Name))
        return std::make_pair(*I, RC);
    }
  }
  
  return std::pair<unsigned, const TargetRegisterClass*>(0, 0);
}

//===----------------------------------------------------------------------===//
//  Loop Strength Reduction hooks
//===----------------------------------------------------------------------===//

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool TargetLowering::isLegalAddressingMode(const AddrMode &AM, 
                                           const Type *Ty) const {
  // The default implementation of this implements a conservative RISCy, r+r and
  // r+i addr mode.

  // Allows a sign-extended 16-bit immediate field.
  if (AM.BaseOffs <= -(1LL << 16) || AM.BaseOffs >= (1LL << 16)-1)
    return false;
  
  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;
  
  // Only support r+r, 
  switch (AM.Scale) {
  case 0:  // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (AM.HasBaseReg && AM.BaseOffs)  // "r+r+i" is not allowed.
      return false;
    // Otherwise we have r+r or r+i.
    break;
  case 2:
    if (AM.HasBaseReg || AM.BaseOffs)  // 2*r+r  or  2*r+i is not allowed.
      return false;
    // Allow 2*r as r+r.
    break;
  }
  
  return true;
}

// Magic for divide replacement

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
  } while (p < 128 && (q1 < delta || (q1 == delta && r1 == 0)));
  magu.m = q2 + 1; // resulting magic number
  magu.s = p - 64;  // resulting shift
  return magu;
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand TargetLowering::BuildSDIV(SDNode *N, SelectionDAG &DAG, 
                                    std::vector<SDNode*>* Created) const {
  MVT::ValueType VT = N->getValueType(0);
  
  // Check to see if we can do this.
  if (!isTypeLegal(VT) || (VT != MVT::i32 && VT != MVT::i64))
    return SDOperand();       // BuildSDIV only operates on i32 or i64
  if (!isOperationLegal(ISD::MULHS, VT))
    return SDOperand();       // Make sure the target supports MULHS.
  
  int64_t d = cast<ConstantSDNode>(N->getOperand(1))->getSignExtended();
  ms magics = (VT == MVT::i32) ? magic32(d) : magic64(d);
  
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = DAG.getNode(ISD::MULHS, VT, N->getOperand(0),
                            DAG.getConstant(magics.m, VT));
  // If d > 0 and m < 0, add the numerator
  if (d > 0 && magics.m < 0) { 
    Q = DAG.getNode(ISD::ADD, VT, Q, N->getOperand(0));
    if (Created)
      Created->push_back(Q.Val);
  }
  // If d < 0 and m > 0, subtract the numerator.
  if (d < 0 && magics.m > 0) {
    Q = DAG.getNode(ISD::SUB, VT, Q, N->getOperand(0));
    if (Created)
      Created->push_back(Q.Val);
  }
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0) {
    Q = DAG.getNode(ISD::SRA, VT, Q, 
                    DAG.getConstant(magics.s, getShiftAmountTy()));
    if (Created)
      Created->push_back(Q.Val);
  }
  // Extract the sign bit and add it to the quotient
  SDOperand T =
    DAG.getNode(ISD::SRL, VT, Q, DAG.getConstant(MVT::getSizeInBits(VT)-1,
                                                 getShiftAmountTy()));
  if (Created)
    Created->push_back(T.Val);
  return DAG.getNode(ISD::ADD, VT, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDOperand TargetLowering::BuildUDIV(SDNode *N, SelectionDAG &DAG,
                                    std::vector<SDNode*>* Created) const {
  MVT::ValueType VT = N->getValueType(0);
  
  // Check to see if we can do this.
  if (!isTypeLegal(VT) || (VT != MVT::i32 && VT != MVT::i64))
    return SDOperand();       // BuildUDIV only operates on i32 or i64
  if (!isOperationLegal(ISD::MULHU, VT))
    return SDOperand();       // Make sure the target supports MULHU.
  
  uint64_t d = cast<ConstantSDNode>(N->getOperand(1))->getValue();
  mu magics = (VT == MVT::i32) ? magicu32(d) : magicu64(d);
  
  // Multiply the numerator (operand 0) by the magic value
  SDOperand Q = DAG.getNode(ISD::MULHU, VT, N->getOperand(0),
                            DAG.getConstant(magics.m, VT));
  if (Created)
    Created->push_back(Q.Val);

  if (magics.a == 0) {
    return DAG.getNode(ISD::SRL, VT, Q, 
                       DAG.getConstant(magics.s, getShiftAmountTy()));
  } else {
    SDOperand NPQ = DAG.getNode(ISD::SUB, VT, N->getOperand(0), Q);
    if (Created)
      Created->push_back(NPQ.Val);
    NPQ = DAG.getNode(ISD::SRL, VT, NPQ, 
                      DAG.getConstant(1, getShiftAmountTy()));
    if (Created)
      Created->push_back(NPQ.Val);
    NPQ = DAG.getNode(ISD::ADD, VT, NPQ, Q);
    if (Created)
      Created->push_back(NPQ.Val);
    return DAG.getNode(ISD::SRL, VT, NPQ, 
                       DAG.getConstant(magics.s-1, getShiftAmountTy()));
  }
}
